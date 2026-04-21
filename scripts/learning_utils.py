import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from script_utils import require_file


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 512,
    "stage_embed_dim": 16,
    "dropout": 0.1,
    "num_stages": 10,
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_embedding_archive(path: Path, normalize: bool = False) -> Dict[str, np.ndarray]:
    require_file(path, "Embeddings file")
    with np.load(path, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}

    if "embeddings" in data:
        embeddings = data["embeddings"].astype(np.float32)
        data["embeddings"] = normalize_rows(embeddings) if normalize else embeddings
    if "stage_indices" in data:
        data["stage_indices"] = data["stage_indices"].astype(np.int16)
    return data


def load_clip(model_id: str, device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise RuntimeError(
            "Could not import Hugging Face CLIP modules. "
            "This is often caused by a torch/torchvision or transformers mismatch. "
            f"Original error: {exc}"
        ) from exc

    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize CLIP model/processor. "
            "Confirm model cache or internet access and package compatibility. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    return processor, model


def extract_clip_image_features(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        features = model.get_image_features(**inputs)
        if torch.is_tensor(features):
            return features
    except (AttributeError, TypeError):
        pass

    vision_outputs = model.vision_model(**inputs)
    if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
        features = vision_outputs.pooler_output
    elif hasattr(vision_outputs, "last_hidden_state") and vision_outputs.last_hidden_state is not None:
        features = vision_outputs.last_hidden_state[:, 0, :]
    elif isinstance(vision_outputs, tuple) and vision_outputs and torch.is_tensor(vision_outputs[0]):
        features = vision_outputs[0]
    else:
        raise TypeError(f"Expected tensor features, got {type(vision_outputs)}")

    if hasattr(model, "visual_projection") and model.visual_projection is not None:
        features = model.visual_projection(features)
    return features


def embed_image_with_clip(image_path: Path, model_id: str, device: torch.device) -> np.ndarray:
    require_file(image_path, "Input image")
    processor, model = load_clip(model_id, device)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    inputs = processor(images=[rgb_image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        features = extract_clip_image_features(model, inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

    return features[0].detach().cpu().numpy().astype(np.float32)


class TransitionMLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, stage_embed_dim: int, dropout: float, num_stages: int):
        super().__init__()
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + stage_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, src_embedding: torch.Tensor, src_stage_idx: torch.Tensor) -> torch.Tensor:
        stage_vec = self.stage_embed(src_stage_idx)
        pred_delta = self.net(torch.cat([src_embedding, stage_vec], dim=-1))
        pred = src_embedding + pred_delta
        return torch.nn.functional.normalize(pred, p=2, dim=-1)


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.norm1(self.conv1(x))
        scale_shift = self.cond(cond).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + residual


class PixelDecoderUNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_stages: int,
        base_channels: int = 32,
        stage_embed_dim: int = 16,
        cond_dim: int = 256,
        output_mode: str = "residual",
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if output_mode not in {"direct", "residual"}:
            raise ValueError("output_mode must be 'direct' or 'residual'")
        self.output_mode = output_mode
        self.residual_scale = float(residual_scale)
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(embedding_dim + stage_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = FiLMBlock(3, c1, cond_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMBlock(c2, c2, cond_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.bottleneck = FiLMBlock(c3, c4, cond_dim)
        self.up2 = nn.Conv2d(c4, c3, kernel_size=3, padding=1)
        self.dec2 = FiLMBlock(c3 + c2, c2, cond_dim)
        self.up1 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.dec1 = FiLMBlock(c2 + c1, c1, cond_dim)
        self.out = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(
        self,
        src_image: torch.Tensor,
        target_embedding: torch.Tensor,
        src_stage_idx: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        stage_vec = self.stage_embed(src_stage_idx)
        cond = self.cond_mlp(torch.cat([target_embedding, stage_vec], dim=-1))

        e1 = self.enc1(src_image, cond)
        e2 = self.enc2(self.down1(e1), cond)
        b = self.bottleneck(self.down2(e2), cond)

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)
        raw = self.out(d1)
        if self.output_mode == "direct":
            pred = torch.sigmoid(raw)
            delta = pred - src_image
        else:
            delta = torch.tanh(raw) * self.residual_scale
            pred = torch.clamp(src_image + delta, 0.0, 1.0)
        if return_delta:
            return pred, delta
        return pred


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = pred.shape[1]
    padding = window_size // 2
    kernel = torch.ones((channels, 1, window_size, window_size), dtype=pred.dtype, device=pred.device)
    kernel = kernel / float(window_size * window_size)

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    sigma_pred = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred * mu_pred
    sigma_target = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target * mu_target
    sigma_cross = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred * mu_target

    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_pred * mu_pred + mu_target * mu_target + c1) * (sigma_pred + sigma_target + c2)
    )
    return 1.0 - ssim.mean()


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device, dtype=image.dtype)
    gx = F.conv2d(gray, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def center_crop_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    height = min(a.shape[-2], b.shape[-2])
    width = min(a.shape[-1], b.shape[-1])

    def crop(x: torch.Tensor) -> torch.Tensor:
        top = (x.shape[-2] - height) // 2
        left = (x.shape[-1] - width) // 2
        return x[..., top : top + height, left : left + width]

    return crop(a), crop(b)


def load_model_config(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return dict(DEFAULT_MODEL_CONFIG)

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    model_config = metrics.get("model_config", {})
    return {
        "hidden_dim": int(model_config.get("hidden_dim", DEFAULT_MODEL_CONFIG["hidden_dim"])),
        "stage_embed_dim": int(model_config.get("stage_embed_dim", DEFAULT_MODEL_CONFIG["stage_embed_dim"])),
        "dropout": float(model_config.get("dropout", DEFAULT_MODEL_CONFIG["dropout"])),
        "num_stages": int(model_config.get("num_stages", DEFAULT_MODEL_CONFIG["num_stages"])),
    }


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    require_file(checkpoint_path, "Checkpoint")
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint did not contain a valid state_dict: {checkpoint_path}")
    return state
