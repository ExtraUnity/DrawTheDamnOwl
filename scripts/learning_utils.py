import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
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
