import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

try:
    from script_utils import require_file
except ImportError:
    from scripts.script_utils import require_file


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 512,
    "stage_embed_dim": 16,
    "dropout": 0.1,
    "num_stages": 8,
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


def load_dino_extractor(model_id: str, device: torch.device):
    try:
        import timm
        from torchvision import transforms

        model = timm.create_model(model_id, pretrained=True)
        model.eval().to(device)
        cfg = getattr(model, "default_cfg", {})
        size = cfg.get("input_size", (3, 224, 224))
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))

        transform = transforms.Compose(
            [
                transforms.Resize((size[1], size[2])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        def forward(images):
            xs = torch.stack([transform(im) for im in images], dim=0).to(device)
            with torch.no_grad():
                if hasattr(model, "forward_features"):
                    feats = model.forward_features(xs)
                else:
                    feats = model(xs)
            if torch.is_tensor(feats):
                return feats
            if isinstance(feats, dict):
                for key in ("x_norm_clstoken", "cls_token", "pre_logits", "x"):
                    value = feats.get(key)
                    if torch.is_tensor(value):
                        return value
                for value in feats.values():
                    if torch.is_tensor(value):
                        return value
            if isinstance(feats, tuple) and feats and torch.is_tensor(feats[0]):
                return feats[0]
            raise RuntimeError("Unsupported timm model output for features")

        return forward, "timm"
    except Exception:
        pass

    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def forward(images):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state[:, 0, :]
            if isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
                return outputs[0]
            raise RuntimeError("Unsupported HF model outputs for features")

        return forward, "huggingface"
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize DINO model. Install 'timm' or ensure the Hugging Face model id is valid. "
            f"Original error: {exc}"
        ) from exc


def embed_image_with_dino(image_path: Path, model_id: str, device: torch.device) -> np.ndarray:
    require_file(image_path, "Input image")
    extractor, _ = load_dino_extractor(model_id, device)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    features = extractor([rgb_image])
    if not torch.is_tensor(features):
        features = torch.tensor(features)
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features[0].detach().cpu().numpy().astype(np.float32)


def combine_stage_features(image_features: np.ndarray, layer_features: Optional[np.ndarray]) -> np.ndarray:
    image_vec = np.asarray(image_features, dtype=np.float32).reshape(1, -1)
    image_vec = normalize_rows(image_vec)[0]

    if layer_features is None:
        return image_vec.astype(np.float32)

    layer_vec = np.asarray(layer_features, dtype=np.float32).reshape(1, -1)
    layer_vec = normalize_rows(layer_vec)[0]
    combined = np.concatenate([image_vec, layer_vec], axis=0).astype(np.float32)
    combined /= max(float(np.linalg.norm(combined)), 1e-12)
    return combined


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


class TransitionSequenceTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_stages: int,
        max_seq_len: int,
        stage_embed_dim: int,
        stage_specific_output_heads: bool = True,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.max_seq_len = int(max_seq_len)
        self.num_stages = int(num_stages)
        self.stage_specific_output_heads = bool(stage_specific_output_heads)
        self.feature_norm = nn.LayerNorm(embedding_dim)
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim)
        self.stage_proj = nn.Linear(stage_embed_dim, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        if self.stage_specific_output_heads:
            self.output_heads = nn.ModuleList([nn.Linear(model_dim, embedding_dim) for _ in range(num_stages)])
        else:
            self.output_head = nn.Linear(model_dim, embedding_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _apply_output_head(self, hidden: torch.Tensor, stage_idx: torch.Tensor) -> torch.Tensor:
        if not self.stage_specific_output_heads:
            return self.output_head(hidden)

        out = []
        for row_hidden, row_stage in zip(hidden, stage_idx):
            out.append(self.output_heads[int(row_stage.item())](row_hidden))
        return torch.stack(out, dim=0)

    def forward(
        self,
        src_embeddings: torch.Tensor,
        src_stage_idx: torch.Tensor,
        src_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if src_embeddings.ndim != 3:
            raise ValueError(f"Expected src_embeddings shape [batch, seq, dim], got {tuple(src_embeddings.shape)}")
        if src_stage_idx.shape != src_padding_mask.shape:
            raise ValueError("src_stage_idx and src_padding_mask must have matching shapes")
        if src_embeddings.shape[:2] != src_stage_idx.shape:
            raise ValueError("src_embeddings batch/seq dims must match src_stage_idx")
        if src_embeddings.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {src_embeddings.shape[1]} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(src_embeddings.shape[1], device=src_embeddings.device).unsqueeze(0)
        src_embeddings_norm = self.feature_norm(src_embeddings)
        hidden = self.input_proj(src_embeddings_norm)
        hidden = hidden + self.stage_proj(self.stage_embed(src_stage_idx)) + self.pos_embed(positions)
        hidden = self.input_dropout(self.input_norm(hidden))
        hidden = self.encoder(hidden, src_key_padding_mask=src_padding_mask)

        lengths = (~src_padding_mask).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(src_embeddings.shape[0], device=src_embeddings.device)
        last_hidden = hidden[batch_idx, last_idx]
        last_src = src_embeddings[batch_idx, last_idx]
        last_stage = src_stage_idx[batch_idx, last_idx]
        delta = self._apply_output_head(self.readout(last_hidden), last_stage)
        pred = last_src + delta
        return F.normalize(pred, p=2, dim=-1)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric temporal padding")
        padding = dilation * (kernel_size // 2)
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return residual + h


class TransitionSequenceCNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        model_dim: int,
        num_layers: int,
        kernel_size: int,
        dilation_cycle: int,
        dropout: float,
        num_stages: int,
        max_seq_len: int,
        stage_embed_dim: int,
        stage_specific_output_heads: bool = True,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dilation_cycle <= 0:
            raise ValueError("dilation_cycle must be positive")

        self.max_seq_len = int(max_seq_len)
        self.num_stages = int(num_stages)
        self.stage_specific_output_heads = bool(stage_specific_output_heads)
        self.feature_norm = nn.LayerNorm(embedding_dim)
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim)
        self.stage_proj = nn.Linear(stage_embed_dim, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    channels=model_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** (layer_idx % dilation_cycle),
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        if self.stage_specific_output_heads:
            self.output_heads = nn.ModuleList([nn.Linear(model_dim, embedding_dim) for _ in range(num_stages)])
        else:
            self.output_head = nn.Linear(model_dim, embedding_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _apply_output_head(self, hidden: torch.Tensor, stage_idx: torch.Tensor) -> torch.Tensor:
        if not self.stage_specific_output_heads:
            return self.output_head(hidden)

        out = []
        for row_hidden, row_stage in zip(hidden, stage_idx):
            out.append(self.output_heads[int(row_stage.item())](row_hidden))
        return torch.stack(out, dim=0)

    def forward(
        self,
        src_embeddings: torch.Tensor,
        src_stage_idx: torch.Tensor,
        src_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if src_embeddings.ndim != 3:
            raise ValueError(f"Expected src_embeddings shape [batch, seq, dim], got {tuple(src_embeddings.shape)}")
        if src_stage_idx.shape != src_padding_mask.shape:
            raise ValueError("src_stage_idx and src_padding_mask must have matching shapes")
        if src_embeddings.shape[:2] != src_stage_idx.shape:
            raise ValueError("src_embeddings batch/seq dims must match src_stage_idx")
        if src_embeddings.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {src_embeddings.shape[1]} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(src_embeddings.shape[1], device=src_embeddings.device).unsqueeze(0)
        src_embeddings_norm = self.feature_norm(src_embeddings)
        hidden = self.input_proj(src_embeddings_norm)
        hidden = hidden + self.stage_proj(self.stage_embed(src_stage_idx)) + self.pos_embed(positions)
        hidden = self.input_dropout(self.input_norm(hidden))

        valid_mask = (~src_padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = hidden * valid_mask
        hidden = hidden.transpose(1, 2)
        valid_mask_channels = valid_mask.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * valid_mask_channels
        hidden = hidden.transpose(1, 2)

        lengths = (~src_padding_mask).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(src_embeddings.shape[0], device=src_embeddings.device)
        last_hidden = hidden[batch_idx, last_idx]
        last_src = src_embeddings[batch_idx, last_idx]
        last_stage = src_stage_idx[batch_idx, last_idx]
        delta = self._apply_output_head(self.readout(last_hidden), last_stage)
        pred = last_src + delta
        return F.normalize(pred, p=2, dim=-1)


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


class StructuralLayerUNet(nn.Module):
    def __init__(
        self,
        num_stages: int,
        base_channels: int = 32,
        stage_embed_dim: int = 16,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.stage_embed = nn.Embedding(num_stages, stage_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(stage_embed_dim, cond_dim),
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
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, src_image: torch.Tensor, src_stage_idx: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(self.stage_embed(src_stage_idx))

        e1 = self.enc1(src_image, cond)
        e2 = self.enc2(self.down1(e1), cond)
        b = self.bottleneck(self.down2(e2), cond)

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)
        return self.out(d1)


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    gray = image.convert("L").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def compose_white_layer(src_image: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
    if layer.shape[1] == 1:
        layer = layer.repeat(1, src_image.shape[1], 1, 1)
    return torch.maximum(src_image, layer.clamp(0.0, 1.0))


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probs * target, dim=(1, 2, 3))
    denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    return 1.0 - torch.mean((numerator + eps) / denominator)


def focal_bce_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    pos_weight: float = 8.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device),
    )
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return torch.mean(alpha_t * (1.0 - p_t).pow(gamma) * bce)


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


def _adapt_transition_mlp_state_dict(
    state: Dict[str, torch.Tensor],
    model: TransitionMLP,
) -> Dict[str, torch.Tensor]:
    adapted = dict(state)
    model_keys = set(model.state_dict().keys())

    old_to_new = {
        "net.0.weight": "shared.0.weight",
        "net.0.bias": "shared.0.bias",
        "net.3.weight": "shared.3.weight",
        "net.3.bias": "shared.3.bias",
        "net.6.weight": "output_head.weight",
        "net.6.bias": "output_head.bias",
    }
    new_to_old = {new_key: old_key for old_key, new_key in old_to_new.items()}

    if any(key.startswith("shared.") or key.startswith("output_head.") for key in adapted) and any(
        key.startswith("net.") for key in model_keys
    ):
        for new_key, old_key in new_to_old.items():
            if new_key in adapted and old_key not in adapted:
                adapted[old_key] = adapted.pop(new_key)
    elif any(key.startswith("net.") for key in adapted) and any(
        key.startswith("shared.") or key.startswith("output_head.") for key in model_keys
    ):
        for old_key, new_key in old_to_new.items():
            if old_key in adapted and new_key not in adapted:
                adapted[new_key] = adapted.pop(old_key)

    return adapted


def load_transition_mlp_checkpoint(model: TransitionMLP, checkpoint_path: Path, device: torch.device) -> None:
    state = _adapt_transition_mlp_state_dict(load_checkpoint_state(checkpoint_path, device), model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Error loading TransitionMLP checkpoint. "
            f"Missing keys: {list(missing)}. Unexpected keys: {list(unexpected)}."
        )
