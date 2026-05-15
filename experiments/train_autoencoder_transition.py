from __future__ import annotations

import argparse
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.autoencoder import ConvAutoencoder
from models.latent_transition import SpatialLatentTransition
from scripts.learning_utils import default_device, ssim_loss
from scripts.script_utils import LEARNING_ROOT, ensure_dir, read_csv_rows, require_file, write_csv, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SPLIT_NAMES = ("train", "val", "test")
EPS = 1e-8


@dataclass
class AugmentConfig:
    affine_degrees: float
    affine_translate: float
    affine_scale: float
    horizontal_flip: bool
    brightness_jitter: float
    contrast_jitter: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spatial autoencoder + latent transition model for owl stage pairs.")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "owl_output"), help="Root directory containing owl images.")
    parser.add_argument(
        "--pairs_csv",
        default=str(LEARNING_ROOT / "manifest_transitions.csv"),
        help="CSV with source/target image pairs.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Square input resolution. Must be divisible by 2**num_downsamples.")
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3, help="Load images as grayscale or RGB.")
    parser.add_argument(
        "--num_downsamples",
        type=int,
        default=0,
        help="Autoencoder downsample stages. Set <= 0 to choose a depth that keeps the latent map near 16x16.",
    )
    parser.add_argument("--latent_channels", type=int, default=64, help="Number of channels in the spatial latent map.")
    parser.add_argument("--base_channels", type=int, default=32, help="Base width for the convolutional autoencoder.")
    parser.add_argument("--transition_hidden_channels", type=int, default=64, help="Hidden width for the latent transition CNN.")
    parser.add_argument("--transition_blocks", type=int, default=4, help="Number of residual blocks in the latent transition CNN.")
    parser.add_argument(
        "--stage_embed_dim",
        type=int,
        default=32,
        help="Sinusoidal stage/timestep embedding width for transition and decoder conditioning.",
    )
    parser.add_argument(
        "--transition_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict the next latent as z1 + f(z1) instead of predicting z2 directly.",
    )
    parser.add_argument(
        "--disable_stage_conditioning",
        action="store_true",
        help="Disable explicit source-stage conditioning in the latent transition model.",
    )
    parser.add_argument(
        "--enable_decoder_stage_conditioning",
        action="store_true",
        help="Enable target-stage conditioning in the decoder. Off by default because it can hurt autoencoder reconstruction.",
    )
    parser.add_argument(
        "--disable_decoder_stage_conditioning",
        action="store_true",
        help="Legacy no-op when decoder stage conditioning is not enabled; kept for CLI compatibility.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--epochs_autoencoder", type=int, default=100, help="Number of autoencoder epochs.")
    parser.add_argument("--epochs_transition", type=int, default=100, help="Number of transition epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--latent_noise_std", type=float, default=0.05, help="Gaussian noise std applied to latent codes during autoencoder training.")
    parser.add_argument("--autoencoder_mse_weight", type=float, default=0.0, help="Optional extra MSE weight on top of reconstruction loss.")
    parser.add_argument("--lambda_image", type=float, default=1.0, help="Weight for decoded image L1 during transition training.")
    parser.add_argument("--lambda_foreground", type=float, default=0.0, help="Extra weight on foreground-only L1 for both AE and transition losses.")
    parser.add_argument("--lambda_edge", type=float, default=0.0, help="Extra weight on Sobel edge L1 for both AE and transition losses.")
    parser.add_argument("--foreground_threshold", type=float, default=0.05, help="Foreground mask threshold on the target image.")
    parser.add_argument("--foreground_dilate", type=int, default=0, help="Optional foreground dilation radius in pixels.")
    parser.add_argument(
        "--output_dir",
        default=str(LEARNING_ROOT / "autoencoder_transition"),
        help="Directory for checkpoints, metrics, and visualizations.",
    )
    parser.add_argument("--device", default=default_device(), help="Training device.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split_strategy",
        choices=["auto", "csv_split_column", "identity_random"],
        default="auto",
        help="Use explicit CSV splits, random identity-based splits, or auto-detect.",
    )
    parser.add_argument("--train_fraction", "--train_ratio", dest="train_fraction", type=float, default=0.8, help="Train split fraction for identity-based splitting.")
    parser.add_argument("--val_fraction", "--val_ratio", dest="val_fraction", type=float, default=0.1, help="Validation split fraction for identity-based splitting.")
    parser.add_argument("--test_fraction", "--test_ratio", dest="test_fraction", type=float, default=0.1, help="Test split fraction for identity-based splitting.")
    parser.add_argument("--sample_every", type=int, default=10, help="Save validation sample sheets every N epochs. Set <= 0 to disable periodic sheets.")
    parser.add_argument("--sample_max_rows", type=int, default=6, help="Maximum rows per saved qualitative sheet.")
    parser.add_argument("--horizontal_flip", action="store_true", help="Enable random horizontal flips during training.")
    parser.add_argument("--affine_degrees", type=float, default=5.0, help="Maximum random rotation degrees.")
    parser.add_argument("--affine_translate", type=float, default=0.03, help="Maximum random translation as a fraction of image size.")
    parser.add_argument("--affine_scale", type=float, default=0.05, help="Maximum random isotropic scale jitter.")
    parser.add_argument("--brightness_jitter", type=float, default=0.05, help="Brightness jitter amount for RGB training images.")
    parser.add_argument("--contrast_jitter", type=float, default=0.05, help="Contrast jitter amount for RGB training images.")
    parser.add_argument(
        "--finetune_autoencoder_during_transition",
        action="store_true",
        help="Legacy option: allow encoder/decoder weights to update during the main transition phase.",
    )
    parser.add_argument("--finetune_decoder_epochs", type=int, default=0, help="Extra epochs that jointly fine-tune the transition model and decoder with the encoder frozen.")
    parser.add_argument("--finetune_transition_lr", type=float, default=1e-4, help="Learning rate for transition weights during the decoder fine-tuning stage.")
    parser.add_argument("--finetune_decoder_lr", type=float, default=1e-5, help="Learning rate for decoder weights during the decoder fine-tuning stage.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_num_downsamples(image_size: int, requested_num_downsamples: int) -> int:
    if requested_num_downsamples > 0:
        return int(requested_num_downsamples)

    num_downsamples = 3
    while image_size % (2 ** (num_downsamples + 1)) == 0 and (image_size // (2**num_downsamples)) > 16:
        num_downsamples += 1
    return num_downsamples


def validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    total = train_fraction + val_fraction + test_fraction
    if train_fraction <= 0.0 or val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("Split fractions must be non-negative and train_fraction must be > 0.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")


def resolve_path(raw_path: str, data_dir: Path) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, ROOT / path, data_dir / path])

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path: {raw_path}")


def get_first(row: Mapping[str, str], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return default


def infer_identity_id(row: Mapping[str, str], source_path: str, target_path: str) -> str:
    for key in ("identity_id", "stem", "owl_id", "sample_id"):
        value = row.get(key, "")
        if str(value).strip():
            return str(value).strip()

    for raw_path in (source_path, target_path):
        stem = Path(raw_path).stem
        stem = re.sub(r"_stage\d+$", "", stem)
        if stem:
            return stem
    raise ValueError("Could not infer identity_id from row")


def normalize_stage_value(value: str) -> int | None:
    value = str(value).strip()
    if not value:
        return None
    return int(value)


def load_pair_rows(pairs_csv: Path, data_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(read_csv_rows(pairs_csv, description="Pairs CSV")):
        source_path_raw = get_first(row, ("source_path", "src_image_path", "src_path"))
        target_path_raw = get_first(row, ("target_path", "tgt_image_path", "target_image_path", "tgt_path"))
        if not source_path_raw or not target_path_raw:
            raise KeyError("Pairs CSV must contain source_path/target_path or src_image_path/tgt_image_path columns.")

        source_path = resolve_path(source_path_raw, data_dir)
        target_path = resolve_path(target_path_raw, data_dir)
        require_file(source_path, "Source image")
        require_file(target_path, "Target image")

        rows.append(
            {
                "source_path": source_path,
                "target_path": target_path,
                "source_stage": normalize_stage_value(get_first(row, ("source_stage", "src_stage_idx", "source_stage_idx"))),
                "target_stage": normalize_stage_value(get_first(row, ("target_stage", "tgt_stage_idx", "target_stage_idx"))),
                "identity_id": infer_identity_id(row, source_path_raw, target_path_raw),
                "split": get_first(row, ("split",), default="").lower(),
                "transition_key": get_first(row, ("transition_key",), default=f"pair_{idx:05d}"),
            }
        )
    if not rows:
        raise RuntimeError("No pair rows were loaded.")
    return rows


def count_by_split(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {split_name: sum(1 for row in rows if str(row["split"]) == split_name) for split_name in SPLIT_NAMES}


def has_valid_csv_splits(rows: Sequence[Mapping[str, Any]]) -> bool:
    valid_splits = set(SPLIT_NAMES)
    split_values = [str(row.get("split", "")).strip().lower() for row in rows]
    return all(split_value in valid_splits for split_value in split_values) and all(
        any(row["split"] == split_name for row in rows) for split_name in SPLIT_NAMES
    )


def assign_identity_splits(
    rows: List[Dict[str, Any]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["identity_id"]), []).append(row)

    group_ids = list(grouped.keys())
    if len(group_ids) < 3:
        raise RuntimeError("Need at least 3 identity groups to build train/val/test splits.")

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    num_groups = len(group_ids)
    n_train = int(num_groups * train_fraction)
    n_val = int(num_groups * val_fraction)
    n_test = num_groups - n_train - n_val
    if n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
    if n_test == 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    split_map = {
        "train": group_ids[:n_train],
        "val": group_ids[n_train : n_train + n_val],
        "test": group_ids[n_train + n_val : n_train + n_val + n_test],
    }
    split_of_group = {group_id: split_name for split_name, ids in split_map.items() for group_id in ids}
    for row in rows:
        row["split"] = split_of_group[str(row["identity_id"])]

    return rows, {
        "strategy": "identity_random",
        "group_counts": {name: len(ids) for name, ids in split_map.items()},
        "counts": count_by_split(rows),
    }


def assign_splits(
    rows: List[Dict[str, Any]],
    split_strategy: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if split_strategy == "csv_split_column":
        if not has_valid_csv_splits(rows):
            raise RuntimeError("split_strategy=csv_split_column requires a valid split column covering train/val/test.")
        return rows, {"strategy": "csv_split_column", "counts": count_by_split(rows)}

    if split_strategy == "identity_random":
        return assign_identity_splits(rows, seed, train_fraction, val_fraction, test_fraction)

    if has_valid_csv_splits(rows):
        return rows, {"strategy": "csv_split_column", "counts": count_by_split(rows)}
    return assign_identity_splits(rows, seed, train_fraction, val_fraction, test_fraction)


def split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split_rows_map = {split_name: [] for split_name in SPLIT_NAMES}
    for row in rows:
        split_name = str(row["split"])
        if split_name in split_rows_map:
            split_rows_map[split_name].append(row)
    for split_name, split_rows_list in split_rows_map.items():
        if not split_rows_list:
            raise RuntimeError(f"No rows found for split: {split_name}")
    return split_rows_map


def build_unique_image_rows(pair_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    unique_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {split_name: {} for split_name in SPLIT_NAMES}
    for split_name, rows in pair_splits.items():
        for row in rows:
            image_specs = (
                ("source_path", row["source_stage"]),
                ("target_path", row["target_stage"]),
            )
            for key, stage_idx in image_specs:
                image_path = Path(row[key])
                image_key = str(image_path)
                if image_key not in unique_by_split[split_name]:
                    unique_by_split[split_name][image_key] = {
                        "image_path": image_path,
                        "split": split_name,
                        "stage_idx": stage_idx,
                    }
                elif unique_by_split[split_name][image_key]["stage_idx"] != stage_idx:
                    raise RuntimeError(f"Conflicting stage annotations for image: {image_path}")
    return {split_name: list(items.values()) for split_name, items in unique_by_split.items()}


def load_image_tensor(path: Path, image_size: int, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode).resize((image_size, image_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        return torch.from_numpy(array).unsqueeze(0).contiguous()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def to_display_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def tensor_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(a - b)
    if diff.shape[0] == 1:
        diff = diff.repeat(3, 1, 1)
    return diff


def apply_tensor_affine(
    image: torch.Tensor,
    angle_degrees: float,
    translate_x: float,
    translate_y: float,
    scale: float,
    fill_value: float = 1.0,
) -> torch.Tensor:
    _, height, width = image.shape
    angle = math.radians(float(angle_degrees))
    cos_v = math.cos(angle) / max(float(scale), 1e-6)
    sin_v = math.sin(angle) / max(float(scale), 1e-6)
    tx = 2.0 * float(translate_x) / max(width, 1)
    ty = 2.0 * float(translate_y) / max(height, 1)

    theta = image.new_tensor([[cos_v, -sin_v, tx], [sin_v, cos_v, ty]]).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, image.shape[0], height, width), align_corners=False)
    sampled = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(
        torch.ones((1, 1, height, width), dtype=image.dtype, device=image.device),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled * mask + fill_value * (1.0 - mask)
    return sampled.squeeze(0)


def apply_photometric_jitter(image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    if brightness != 0.0:
        image = image * (1.0 + float(brightness))
    if contrast != 0.0:
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - mean) * (1.0 + float(contrast)) + mean
    return image.clamp(0.0, 1.0)


def apply_augmentation(images: Sequence[torch.Tensor], config: AugmentConfig, channels: int) -> List[torch.Tensor]:
    if not images:
        return []

    height, width = images[0].shape[-2:]
    do_flip = bool(config.horizontal_flip and random.random() < 0.5)
    angle = random.uniform(-config.affine_degrees, config.affine_degrees) if config.affine_degrees > 0.0 else 0.0
    max_tx = config.affine_translate * width
    max_ty = config.affine_translate * height
    translate_x = random.uniform(-max_tx, max_tx) if max_tx > 0.0 else 0.0
    translate_y = random.uniform(-max_ty, max_ty) if max_ty > 0.0 else 0.0
    scale = random.uniform(1.0 - config.affine_scale, 1.0 + config.affine_scale) if config.affine_scale > 0.0 else 1.0
    brightness = 0.0
    contrast = 0.0
    if channels == 3:
        brightness = random.uniform(-config.brightness_jitter, config.brightness_jitter) if config.brightness_jitter > 0.0 else 0.0
        contrast = random.uniform(-config.contrast_jitter, config.contrast_jitter) if config.contrast_jitter > 0.0 else 0.0

    outputs: List[torch.Tensor] = []
    for image in images:
        out = image
        if do_flip:
            out = torch.flip(out, dims=(-1,))
        out = apply_tensor_affine(out, angle, translate_x, translate_y, scale)
        if channels == 3:
            out = apply_photometric_jitter(out, brightness, contrast)
        outputs.append(out.clamp(0.0, 1.0))
    return outputs


class UniqueImageDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        image_size: int,
        channels: int,
        augment_config: AugmentConfig,
        training: bool,
    ):
        self.rows = rows
        self.image_size = image_size
        self.channels = channels
        self.augment_config = augment_config
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image = load_image_tensor(Path(row["image_path"]), self.image_size, self.channels)
        if self.training:
            image = apply_augmentation([image], self.augment_config, self.channels)[0]
        return {
            "image": image,
            "image_path": str(row["image_path"]),
            "stage_idx": -1 if row["stage_idx"] is None else int(row["stage_idx"]),
        }


class StagePairDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        image_size: int,
        channels: int,
        augment_config: AugmentConfig,
        training: bool,
    ):
        self.rows = rows
        self.image_size = image_size
        self.channels = channels
        self.augment_config = augment_config
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        source = load_image_tensor(Path(row["source_path"]), self.image_size, self.channels)
        target = load_image_tensor(Path(row["target_path"]), self.image_size, self.channels)
        if self.training:
            source, target = apply_augmentation([source, target], self.augment_config, self.channels)
        return {
            "source": source,
            "target": target,
            "source_path": str(row["source_path"]),
            "target_path": str(row["target_path"]),
            "source_stage": -1 if row["source_stage"] is None else int(row["source_stage"]),
            "target_stage": -1 if row["target_stage"] is None else int(row["target_stage"]),
            "transition_key": str(row["transition_key"]),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def maybe_tqdm(iterable: Iterable[Any], desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def to_luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def build_foreground_mask(target: torch.Tensor, threshold: float, dilate_radius: int) -> torch.Tensor:
    mask = (target.amax(dim=1, keepdim=True) > float(threshold)).to(dtype=target.dtype)
    if dilate_radius > 0:
        kernel_size = int(dilate_radius) * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=int(dilate_radius))
    return mask


def sobel_edges_general(image: torch.Tensor) -> torch.Tensor:
    gray = to_luminance(image)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device, dtype=image.dtype)
    gx = F.conv2d(gray, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def compute_image_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_threshold: float,
    foreground_dilate: int,
) -> Dict[str, torch.Tensor]:
    mask = build_foreground_mask(target, foreground_threshold, foreground_dilate)
    mask_channels = mask.expand(-1, pred.shape[1], -1, -1)
    abs_diff = torch.abs(pred - target)
    sq_diff = (pred - target).pow(2)
    denom = torch.clamp(mask_channels.sum(), min=EPS)
    edge_pred = sobel_edges_general(pred)
    edge_target = sobel_edges_general(target)

    return {
        "l1": F.l1_loss(pred, target),
        "mse": F.mse_loss(pred, target),
        "ssim": 1.0 - ssim_loss(pred, target),
        "foreground_l1": (abs_diff * mask_channels).sum() / denom,
        "foreground_mse": (sq_diff * mask_channels).sum() / denom,
        "edge_l1": F.l1_loss(edge_pred, edge_target),
        "foreground_fraction": mask.mean(),
    }


def tensor_metrics_to_float(terms: Mapping[str, torch.Tensor], prefix: str = "") -> Dict[str, float]:
    out = {}
    for key, value in terms.items():
        metric_key = f"{prefix}{key}" if prefix else key
        out[metric_key] = float(value.detach().cpu().item())
    return out


def reconstruction_metrics_from_terms(terms: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    metrics = tensor_metrics_to_float(terms)
    metrics["loss"] = metrics["l1"]
    return metrics


def comparison_metrics_from_terms(
    terms: Mapping[str, torch.Tensor],
    latent_mse: torch.Tensor | None,
    cosine: torch.Tensor | None,
) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {
        "image_l1": float(terms["l1"].detach().cpu().item()),
        "image_mse": float(terms["mse"].detach().cpu().item()),
        "image_ssim": float(terms["ssim"].detach().cpu().item()),
        "foreground_l1": float(terms["foreground_l1"].detach().cpu().item()),
        "foreground_mse": float(terms["foreground_mse"].detach().cpu().item()),
        "edge_l1": float(terms["edge_l1"].detach().cpu().item()),
        "foreground_fraction": float(terms["foreground_fraction"].detach().cpu().item()),
    }
    metrics["latent_mse"] = None if latent_mse is None else float(latent_mse.detach().cpu().item())
    metrics["cosine"] = None if cosine is None else float(cosine.detach().cpu().item())
    return metrics


def mean_metrics(metric_rows: Sequence[Mapping[str, float | None]]) -> Dict[str, float | None]:
    if not metric_rows:
        return {}
    keys = list(metric_rows[0].keys())
    out: Dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in metric_rows if row[key] is not None]
        out[key] = None if not values else float(np.mean(values))
    return out


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(path, [{key: row.get(key, "") for key in fieldnames} for row in rows], fieldnames)


def maybe_plot_history(path: Path, rows: List[Dict[str, float]], title: str, y_keys: Sequence[str]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [row["epoch"] for row in rows if "epoch" in row]
    if not epochs:
        return

    plt.figure(figsize=(9, 5))
    for key in y_keys:
        if any(key in row for row in rows):
            values = [row.get(key, float("nan")) for row in rows]
            plt.plot(epochs, values, label=key)
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def set_autoencoder_trainability(model: ConvAutoencoder, train_encoder: bool, train_decoder: bool) -> None:
    encoder_modules = [model.stem, model.encoder_blocks, model.to_latent, model.latent_refine]
    decoder_modules = [model.from_latent, model.decoder_blocks, model.out]
    for module in encoder_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(train_encoder)
    for module in decoder_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(train_decoder)


def get_decoder_parameters(model: ConvAutoencoder) -> List[nn.Parameter]:
    parameters: List[nn.Parameter] = []
    for module in (model.from_latent, model.decoder_blocks, model.out):
        parameters.extend(list(module.parameters()))
    return parameters


def forward_pair_models(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    source: torch.Tensor,
    target: torch.Tensor,
    source_stage: torch.Tensor | None = None,
    target_stage: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    z1 = autoencoder.encode(source)
    z2 = autoencoder.encode(target)
    z2_pred = transition(z1, source_stage)
    source_decode = autoencoder.decode(z1, source_stage)
    target_recon = autoencoder.decode(z2, target_stage)
    transition_pred = autoencoder.decode(z2_pred, target_stage)
    return {
        "z1": z1,
        "z2": z2,
        "z2_pred": z2_pred,
        "source_decode": source_decode,
        "target_recon": target_recon,
        "transition_pred": transition_pred,
    }


def autoencoder_total_loss(terms: Mapping[str, torch.Tensor], mse_weight: float, lambda_foreground: float, lambda_edge: float) -> torch.Tensor:
    return terms["l1"] + float(mse_weight) * terms["mse"] + float(lambda_foreground) * terms["foreground_l1"] + float(lambda_edge) * terms["edge_l1"]


def transition_total_loss(
    latent_loss: torch.Tensor,
    terms: Mapping[str, torch.Tensor],
    lambda_image: float,
    lambda_foreground: float,
    lambda_edge: float,
) -> torch.Tensor:
    return latent_loss + float(lambda_image) * terms["l1"] + float(lambda_foreground) * terms["foreground_l1"] + float(lambda_edge) * terms["edge_l1"]


def save_autoencoder_samples(
    model: ConvAutoencoder,
    dataset: UniqueImageDataset,
    device: torch.device,
    output_path: Path,
    latent_noise_std: float,
    max_samples: int,
) -> None:
    count = min(max_samples, len(dataset))
    if count == 0:
        return

    model.eval()
    tile_size = 160
    label_height = 22
    columns = 3
    canvas = Image.new("RGB", (tile_size * columns, count * (tile_size + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    with torch.no_grad():
        for row_idx in range(count):
            sample = dataset[row_idx]
            image = sample["image"].unsqueeze(0).to(device)
            stage_idx = torch.tensor([int(sample["stage_idx"])], dtype=torch.long, device=device)
            recon = model(image, latent_noise_std=latent_noise_std, stage_idx=stage_idx)[0].cpu()
            images = [sample["image"], recon, tensor_abs_diff(recon, sample["image"])]
            labels = ["input", "recon", "abs diff"]
            y = row_idx * (tile_size + label_height)
            for col_idx, (image_tensor, label) in enumerate(zip(images, labels)):
                pil_image = to_display_pil(image_tensor).resize((tile_size, tile_size), Image.BICUBIC)
                x = col_idx * tile_size
                canvas.paste(pil_image, (x, y))
                draw.text((x + 6, y + tile_size + 4), label, fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def save_transition_samples(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    dataset: StagePairDataset,
    device: torch.device,
    output_path: Path,
    max_samples: int,
) -> None:
    count = min(max_samples, len(dataset))
    if count == 0:
        return

    autoencoder.eval()
    transition.eval()
    tile_size = 160
    label_height = 22
    columns = 6
    canvas = Image.new("RGB", (tile_size * columns, count * (tile_size + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    with torch.no_grad():
        for row_idx in range(count):
            sample = dataset[row_idx]
            source = sample["source"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            source_stage = torch.tensor([int(sample["source_stage"])], dtype=torch.long, device=device)
            target_stage = torch.tensor([int(sample["target_stage"])], dtype=torch.long, device=device)
            outputs = forward_pair_models(autoencoder, transition, source, target, source_stage, target_stage)
            images = [
                sample["source"],
                sample["target"],
                outputs["source_decode"][0].cpu(),
                outputs["transition_pred"][0].cpu(),
                outputs["target_recon"][0].cpu(),
                tensor_abs_diff(outputs["transition_pred"][0].cpu(), sample["target"]),
            ]
            labels = ["source", "target", "decode z1", "transition", "recon z2", "abs diff"]
            y = row_idx * (tile_size + label_height)
            for col_idx, (image_tensor, label) in enumerate(zip(images, labels)):
                pil_image = to_display_pil(image_tensor).resize((tile_size, tile_size), Image.BICUBIC)
                x = col_idx * tile_size
                canvas.paste(pil_image, (x, y))
                draw.text((x + 6, y + tile_size + 4), label, fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def train_autoencoder_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    metrics: List[Dict[str, float]] = []
    for batch in maybe_tqdm(loader, desc=f"ae train {epoch:03d}"):
        image = batch["image"].to(device)
        stage_idx = batch["stage_idx"].to(device)
        recon = model(image, latent_noise_std=args.latent_noise_std, stage_idx=stage_idx)
        terms = compute_image_terms(recon, image, args.foreground_threshold, args.foreground_dilate)
        loss = autoencoder_total_loss(terms, args.autoencoder_mse_weight, args.lambda_foreground, args.lambda_edge)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_metrics = reconstruction_metrics_from_terms(terms)
        batch_metrics["loss"] = float(loss.detach().cpu().item())
        metrics.append(batch_metrics)
    return mean_metrics(metrics)  # type: ignore[return-value]


def evaluate_autoencoder(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    metrics: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in maybe_tqdm(loader, desc="ae eval"):
            image = batch["image"].to(device)
            stage_idx = batch["stage_idx"].to(device)
            recon = model(image, latent_noise_std=0.0, stage_idx=stage_idx)
            terms = compute_image_terms(recon, image, args.foreground_threshold, args.foreground_dilate)
            loss = autoencoder_total_loss(terms, args.autoencoder_mse_weight, args.lambda_foreground, args.lambda_edge)
            batch_metrics = reconstruction_metrics_from_terms(terms)
            batch_metrics["loss"] = float(loss.detach().cpu().item())
            metrics.append(batch_metrics)
    return mean_metrics(metrics)  # type: ignore[return-value]


def train_transition_epoch(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    if args.finetune_autoencoder_during_transition:
        autoencoder.train()
    else:
        autoencoder.eval()
    transition.train()

    metrics: List[Dict[str, float]] = []
    for batch in maybe_tqdm(loader, desc=f"tr train {epoch:03d}"):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        source_stage = batch["source_stage"].to(device)
        target_stage = batch["target_stage"].to(device)

        if args.finetune_autoencoder_during_transition:
            outputs = forward_pair_models(autoencoder, transition, source, target, source_stage, target_stage)
        else:
            with torch.no_grad():
                z1 = autoencoder.encode(source)
                z2 = autoencoder.encode(target)
            z2_pred = transition(z1, source_stage)
            target_pred = autoencoder.decode(z2_pred, target_stage)
            outputs = {"z1": z1, "z2": z2, "z2_pred": z2_pred, "transition_pred": target_pred}

        terms = compute_image_terms(outputs["transition_pred"], target, args.foreground_threshold, args.foreground_dilate)
        latent_loss = F.mse_loss(outputs["z2_pred"], outputs["z2"])
        cosine = F.cosine_similarity(outputs["z2_pred"].flatten(1), outputs["z2"].flatten(1), dim=1).mean()
        total_loss = transition_total_loss(latent_loss, terms, args.lambda_image, args.lambda_foreground, args.lambda_edge)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_metrics = comparison_metrics_from_terms(terms, latent_loss, cosine)
        batch_metrics["loss"] = float(total_loss.detach().cpu().item())
        metrics.append(batch_metrics)  # type: ignore[arg-type]
    return mean_metrics(metrics)  # type: ignore[return-value]


def train_decoder_finetune_epoch(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    autoencoder.train()
    transition.train()
    metrics: List[Dict[str, float]] = []
    for batch in maybe_tqdm(loader, desc=f"ft train {epoch:03d}"):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        source_stage = batch["source_stage"].to(device)
        target_stage = batch["target_stage"].to(device)

        with torch.no_grad():
            z1 = autoencoder.encode(source)
            z2 = autoencoder.encode(target)

        z2_pred = transition(z1, source_stage)
        target_pred = autoencoder.decode(z2_pred, target_stage)
        terms = compute_image_terms(target_pred, target, args.foreground_threshold, args.foreground_dilate)
        latent_loss = F.mse_loss(z2_pred, z2)
        cosine = F.cosine_similarity(z2_pred.flatten(1), z2.flatten(1), dim=1).mean()
        total_loss = transition_total_loss(latent_loss, terms, args.lambda_image, args.lambda_foreground, args.lambda_edge)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_metrics = comparison_metrics_from_terms(terms, latent_loss, cosine)
        batch_metrics["loss"] = float(total_loss.detach().cpu().item())
        metrics.append(batch_metrics)  # type: ignore[arg-type]
    return mean_metrics(metrics)  # type: ignore[return-value]


def evaluate_pair_models(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float | None]]:
    autoencoder.eval()
    transition.eval()
    target_rows: List[Dict[str, float | None]] = []
    source_rows: List[Dict[str, float | None]] = []
    learned_rows: List[Dict[str, float | None]] = []

    with torch.no_grad():
        for batch in maybe_tqdm(loader, desc="pair eval"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            source_stage = batch["source_stage"].to(device)
            target_stage = batch["target_stage"].to(device)
            outputs = forward_pair_models(autoencoder, transition, source, target, source_stage, target_stage)

            target_terms = compute_image_terms(outputs["target_recon"], target, args.foreground_threshold, args.foreground_dilate)
            source_terms = compute_image_terms(outputs["source_decode"], target, args.foreground_threshold, args.foreground_dilate)
            learned_terms = compute_image_terms(outputs["transition_pred"], target, args.foreground_threshold, args.foreground_dilate)

            source_latent_mse = F.mse_loss(outputs["z1"], outputs["z2"])
            source_cosine = F.cosine_similarity(outputs["z1"].flatten(1), outputs["z2"].flatten(1), dim=1).mean()
            learned_latent_mse = F.mse_loss(outputs["z2_pred"], outputs["z2"])
            learned_cosine = F.cosine_similarity(outputs["z2_pred"].flatten(1), outputs["z2"].flatten(1), dim=1).mean()
            learned_loss = transition_total_loss(learned_latent_mse, learned_terms, args.lambda_image, args.lambda_foreground, args.lambda_edge)

            target_metrics = comparison_metrics_from_terms(target_terms, None, None)
            target_metrics["loss"] = float(autoencoder_total_loss(target_terms, args.autoencoder_mse_weight, args.lambda_foreground, args.lambda_edge).detach().cpu().item())
            source_metrics = comparison_metrics_from_terms(source_terms, source_latent_mse, source_cosine)
            source_metrics["loss"] = float(source_terms["l1"].detach().cpu().item())
            learned_metrics = comparison_metrics_from_terms(learned_terms, learned_latent_mse, learned_cosine)
            learned_metrics["loss"] = float(learned_loss.detach().cpu().item())

            target_rows.append(target_metrics)
            source_rows.append(source_metrics)
            learned_rows.append(learned_metrics)

    return {
        "target_autoencoder": mean_metrics(target_rows),
        "source_decode_baseline": mean_metrics(source_rows),
        "learned_transition": mean_metrics(learned_rows),
    }


def save_checkpoint(path: Path, model: nn.Module, extra: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state_dict": model.state_dict(), **extra}, path)


def print_split_summary(pair_splits: Dict[str, List[Dict[str, Any]]], image_splits: Dict[str, List[Dict[str, Any]]]) -> None:
    print("split summary")
    for split_name in SPLIT_NAMES:
        print(f"  {split_name}: pairs={len(pair_splits[split_name])} unique_images={len(image_splits[split_name])}")


def comparison_table_rows(comparison: Mapping[str, Mapping[str, float | None]]) -> List[Tuple[str, Mapping[str, float | None]]]:
    return [
        ("AE target reconstruction", comparison["target_autoencoder"]),
        ("Decode source latent", comparison["source_decode_baseline"]),
        ("Learned latent transition", comparison["learned_transition"]),
    ]


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_comparison_table(split_name: str, comparison: Mapping[str, Mapping[str, float | None]]) -> None:
    print(f"{split_name} comparison")
    print("  model                     image_l1  image_mse  image_ssim  fg_l1    fg_mse   edge_l1  latent_mse cosine")
    for label, row in comparison_table_rows(comparison):
        print(
            f"  {label:<24} "
            f"{format_metric(row.get('image_l1')):>8} "
            f"{format_metric(row.get('image_mse')):>9} "
            f"{format_metric(row.get('image_ssim')):>10} "
            f"{format_metric(row.get('foreground_l1')):>8} "
            f"{format_metric(row.get('foreground_mse')):>8} "
            f"{format_metric(row.get('edge_l1')):>8} "
            f"{format_metric(row.get('latent_mse')):>10} "
            f"{format_metric(row.get('cosine')):>7}"
        )


def save_split_sample_sheets(
    autoencoder: ConvAutoencoder,
    transition: SpatialLatentTransition,
    pair_datasets: Dict[str, StagePairDataset],
    device: torch.device,
    output_dir: Path,
    max_samples: int,
) -> None:
    for split_name in SPLIT_NAMES:
        save_transition_samples(
            autoencoder,
            transition,
            pair_datasets[split_name],
            device,
            output_dir / "samples_transition" / f"{split_name}_best.png",
            max_samples=max_samples,
        )


def main() -> None:
    args = parse_args()
    validate_fractions(args.train_fraction, args.val_fraction, args.test_fraction)
    args.num_downsamples = resolve_num_downsamples(args.image_size, int(args.num_downsamples))
    downsample_factor = 2 ** int(args.num_downsamples)
    if args.image_size % downsample_factor != 0:
        raise ValueError(f"--image_size must be divisible by {downsample_factor} for num_downsamples={args.num_downsamples}.")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    pairs_csv = Path(args.pairs_csv)
    output_dir = ensure_dir(Path(args.output_dir))
    device = torch.device(args.device)

    ensure_dir(output_dir / "samples_autoencoder")
    ensure_dir(output_dir / "samples_transition")
    ensure_dir(output_dir / "curves")

    augment_config = AugmentConfig(
        affine_degrees=float(args.affine_degrees),
        affine_translate=float(args.affine_translate),
        affine_scale=float(args.affine_scale),
        horizontal_flip=bool(args.horizontal_flip),
        brightness_jitter=float(args.brightness_jitter),
        contrast_jitter=float(args.contrast_jitter),
    )

    raw_rows = load_pair_rows(pairs_csv, data_dir)
    rows, split_info = assign_splits(
        raw_rows,
        split_strategy=args.split_strategy,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    pair_splits = split_rows(rows)
    image_splits = build_unique_image_rows(pair_splits)
    source_stages = [int(row["source_stage"]) for row in rows if row["source_stage"] is not None]
    target_stages = [int(row["target_stage"]) for row in rows if row["target_stage"] is not None]
    all_stage_values = source_stages + target_stages
    has_all_stage_annotations = len(source_stages) == len(rows) and len(target_stages) == len(rows)
    use_transition_stage_conditioning = (len(source_stages) == len(rows)) and not bool(args.disable_stage_conditioning)
    use_decoder_stage_conditioning = (
        has_all_stage_annotations
        and bool(args.enable_decoder_stage_conditioning)
        and not bool(args.disable_decoder_stage_conditioning)
    )
    num_stages = (max(all_stage_values) + 1) if all_stage_values else None

    image_datasets = {
        "train": UniqueImageDataset(image_splits["train"], args.image_size, args.channels, augment_config, training=True),
        "val": UniqueImageDataset(image_splits["val"], args.image_size, args.channels, augment_config, training=False),
        "test": UniqueImageDataset(image_splits["test"], args.image_size, args.channels, augment_config, training=False),
    }
    image_eval_datasets = {
        "train": UniqueImageDataset(image_splits["train"], args.image_size, args.channels, augment_config, training=False),
        "val": image_datasets["val"],
        "test": image_datasets["test"],
    }
    pair_datasets = {
        "train": StagePairDataset(pair_splits["train"], args.image_size, args.channels, augment_config, training=True),
        "val": StagePairDataset(pair_splits["val"], args.image_size, args.channels, augment_config, training=False),
        "test": StagePairDataset(pair_splits["test"], args.image_size, args.channels, augment_config, training=False),
    }
    pair_eval_datasets = {
        "train": StagePairDataset(pair_splits["train"], args.image_size, args.channels, augment_config, training=False),
        "val": pair_datasets["val"],
        "test": pair_datasets["test"],
    }

    image_loaders = {
        "train": make_loader(image_datasets["train"], args.batch_size, True, args.num_workers, args.seed + 1, device),
        "val": make_loader(image_datasets["val"], args.batch_size, False, args.num_workers, args.seed + 2, device),
        "test": make_loader(image_datasets["test"], args.batch_size, False, args.num_workers, args.seed + 3, device),
    }
    image_eval_loaders = {
        "train": make_loader(image_eval_datasets["train"], args.batch_size, False, args.num_workers, args.seed + 11, device),
        "val": image_loaders["val"],
        "test": image_loaders["test"],
    }
    pair_loaders = {
        "train": make_loader(pair_datasets["train"], args.batch_size, True, args.num_workers, args.seed + 4, device),
        "val": make_loader(pair_datasets["val"], args.batch_size, False, args.num_workers, args.seed + 5, device),
        "test": make_loader(pair_datasets["test"], args.batch_size, False, args.num_workers, args.seed + 6, device),
    }
    pair_eval_loaders = {
        "train": make_loader(pair_eval_datasets["train"], args.batch_size, False, args.num_workers, args.seed + 12, device),
        "val": pair_loaders["val"],
        "test": pair_loaders["test"],
    }

    print_split_summary(pair_splits, image_splits)
    latent_resolution = args.image_size // downsample_factor
    print(
        f"model setup: num_downsamples={args.num_downsamples} "
        f"latent_resolution={latent_resolution}x{latent_resolution} "
        f"transition_stage_conditioning={'on' if use_transition_stage_conditioning else 'off'} "
        f"decoder_stage_conditioning={'on' if use_decoder_stage_conditioning else 'off'}"
    )

    config = {
        "data_dir": str(data_dir),
        "pairs_csv": str(pairs_csv),
        "output_dir": str(output_dir),
        "split_info": split_info,
        "args": vars(args),
    }
    write_json(output_dir / "config.json", config)

    autoencoder = ConvAutoencoder(
        in_channels=args.channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        num_downsamples=args.num_downsamples,
        num_stages=num_stages,
        stage_embed_dim=args.stage_embed_dim,
        condition_decoder=use_decoder_stage_conditioning,
    ).to(device)
    transition = SpatialLatentTransition(
        latent_channels=args.latent_channels,
        hidden_channels=args.transition_hidden_channels,
        num_blocks=args.transition_blocks,
        residual=args.transition_residual,
        num_stages=num_stages if use_transition_stage_conditioning else None,
        stage_embed_dim=args.stage_embed_dim,
    ).to(device)

    autoencoder_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    autoencoder_history: List[Dict[str, float]] = []
    best_autoencoder_state = None
    best_autoencoder_val_loss = float("inf")
    best_autoencoder_val_l1 = float("inf")
    sample_image_dataset = image_datasets["val"] if len(image_datasets["val"]) > 0 else image_datasets["train"]

    for epoch in range(1, args.epochs_autoencoder + 1):
        train_metrics = train_autoencoder_epoch(autoencoder, image_loaders["train"], autoencoder_optimizer, device, args, epoch)
        val_metrics = evaluate_autoencoder(autoencoder, image_loaders["val"], device, args)
        epoch_row = {
            "epoch": float(epoch),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            "lr": float(autoencoder_optimizer.param_groups[0]["lr"]),
        }
        autoencoder_history.append(epoch_row)

        if float(val_metrics["loss"]) < best_autoencoder_val_loss:
            best_autoencoder_val_loss = float(val_metrics["loss"])
            best_autoencoder_val_l1 = float(val_metrics["l1"])
            best_autoencoder_state = clone_state_dict(autoencoder)

        print(
            f"ae epoch {epoch:03d} "
            f"train_loss={float(train_metrics['loss']):.4f} "
            f"train_l1={float(train_metrics['l1']):.4f} "
            f"train_fg_l1={float(train_metrics['foreground_l1']):.4f} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_l1={float(val_metrics['l1']):.4f} "
            f"val_ssim={float(val_metrics['ssim']):.4f}"
        )

        if epoch == 1 or epoch == args.epochs_autoencoder or (args.sample_every > 0 and epoch % args.sample_every == 0):
            save_autoencoder_samples(
                autoencoder,
                sample_image_dataset,
                device,
                output_dir / "samples_autoencoder" / f"epoch_{epoch:03d}.png",
                latent_noise_std=0.0,
                max_samples=args.sample_max_rows,
            )

    if best_autoencoder_state is None:
        raise RuntimeError("Autoencoder training did not produce a valid checkpoint.")

    autoencoder.load_state_dict(best_autoencoder_state)
    save_checkpoint(
        output_dir / "autoencoder_best.pt",
        autoencoder,
        {"config": config, "best_val_loss": best_autoencoder_val_loss, "best_val_l1": best_autoencoder_val_l1},
    )
    save_autoencoder_samples(
        autoencoder,
        sample_image_dataset,
        device,
        output_dir / "samples_autoencoder" / "best.png",
        latent_noise_std=0.0,
        max_samples=args.sample_max_rows,
    )

    autoencoder_train_metrics = evaluate_autoencoder(autoencoder, image_eval_loaders["train"], device, args)
    autoencoder_val_metrics = evaluate_autoencoder(autoencoder, image_eval_loaders["val"], device, args)
    autoencoder_test_metrics = evaluate_autoencoder(autoencoder, image_eval_loaders["test"], device, args)
    write_history_csv(output_dir / "autoencoder_history.csv", autoencoder_history)
    maybe_plot_history(
        output_dir / "curves" / "autoencoder_history.png",
        autoencoder_history,
        title="Autoencoder Training",
        y_keys=("train_loss", "val_loss", "train_l1", "val_l1", "train_foreground_l1", "val_foreground_l1", "train_edge_l1", "val_edge_l1"),
    )

    if not args.finetune_autoencoder_during_transition:
        set_autoencoder_trainability(autoencoder, train_encoder=False, train_decoder=False)
        autoencoder.eval()

    first_batch = next(iter(pair_loaders["train"]))
    with torch.no_grad():
        latent_shape = tuple(autoencoder.encode(first_batch["source"].to(device)).shape[1:])
    print(f"latent shape: {latent_shape}")

    transition_params: List[nn.Parameter] = list(transition.parameters())
    if args.finetune_autoencoder_during_transition:
        transition_params.extend(autoencoder.parameters())
    transition_optimizer = torch.optim.AdamW(transition_params, lr=args.lr, weight_decay=args.weight_decay)

    transition_history: List[Dict[str, float]] = []
    best_transition_state = None
    best_transition_val_loss = float("inf")
    best_autoencoder_for_transition_state = None

    sample_pair_dataset_val = pair_datasets["val"] if len(pair_datasets["val"]) > 0 else pair_datasets["train"]

    for epoch in range(1, args.epochs_transition + 1):
        train_metrics = train_transition_epoch(autoencoder, transition, pair_loaders["train"], transition_optimizer, device, args, epoch)
        val_comparison = evaluate_pair_models(autoencoder, transition, pair_loaders["val"], device, args)
        val_metrics = val_comparison["learned_transition"]
        epoch_row = {
            "epoch": float(epoch),
            **{f"train_{key}": float(value) for key, value in train_metrics.items() if value is not None},
            **{f"val_{key}": float(value) for key, value in val_metrics.items() if value is not None},
            "lr": float(transition_optimizer.param_groups[0]["lr"]),
        }
        transition_history.append(epoch_row)

        current_val_loss = float(val_metrics["loss"])
        if current_val_loss < best_transition_val_loss:
            best_transition_val_loss = current_val_loss
            best_transition_state = clone_state_dict(transition)
            if args.finetune_autoencoder_during_transition:
                best_autoencoder_for_transition_state = clone_state_dict(autoencoder)

        print(
            f"tr epoch {epoch:03d} "
            f"train_loss={float(train_metrics['loss']):.4f} "
            f"train_latent_mse={float(train_metrics['latent_mse']):.4f} "
            f"train_fg_l1={float(train_metrics['foreground_l1']):.4f} "
            f"val_loss={current_val_loss:.4f} "
            f"val_image_l1={float(val_metrics['image_l1']):.4f} "
            f"val_cosine={float(val_metrics['cosine']):.4f}"
        )

        if epoch == 1 or epoch == args.epochs_transition or (args.sample_every > 0 and epoch % args.sample_every == 0):
            save_transition_samples(
                autoencoder,
                transition,
                sample_pair_dataset_val,
                device,
                output_dir / "samples_transition" / f"val_epoch_{epoch:03d}.png",
                max_samples=args.sample_max_rows,
            )

    if best_transition_state is None:
        raise RuntimeError("Transition training did not produce a valid checkpoint.")

    transition.load_state_dict(best_transition_state)
    if best_autoencoder_for_transition_state is not None:
        autoencoder.load_state_dict(best_autoencoder_for_transition_state)

    finetune_history: List[Dict[str, float]] = []
    best_finetune_transition_state = None
    best_finetune_decoder_state = None
    best_finetune_val_loss = best_transition_val_loss

    if args.finetune_decoder_epochs > 0:
        pre_finetune_transition_state = clone_state_dict(transition)
        pre_finetune_autoencoder_state = clone_state_dict(autoencoder)
        set_autoencoder_trainability(autoencoder, train_encoder=False, train_decoder=True)
        decoder_parameters = get_decoder_parameters(autoencoder)
        finetune_optimizer = torch.optim.AdamW(
            [
                {"params": list(transition.parameters()), "lr": args.finetune_transition_lr},
                {"params": decoder_parameters, "lr": args.finetune_decoder_lr},
            ],
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.finetune_decoder_epochs + 1):
            train_metrics = train_decoder_finetune_epoch(autoencoder, transition, pair_loaders["train"], finetune_optimizer, device, args, epoch)
            val_comparison = evaluate_pair_models(autoencoder, transition, pair_loaders["val"], device, args)
            val_metrics = val_comparison["learned_transition"]
            row = {
                "epoch": float(epoch),
                **{f"train_{key}": float(value) for key, value in train_metrics.items() if value is not None},
                **{f"val_{key}": float(value) for key, value in val_metrics.items() if value is not None},
                "transition_lr": float(finetune_optimizer.param_groups[0]["lr"]),
                "decoder_lr": float(finetune_optimizer.param_groups[1]["lr"]),
            }
            finetune_history.append(row)

            current_val_loss = float(val_metrics["loss"])
            if current_val_loss < best_finetune_val_loss:
                best_finetune_val_loss = current_val_loss
                best_finetune_transition_state = clone_state_dict(transition)
                best_finetune_decoder_state = clone_state_dict(autoencoder)

            print(
                f"ft epoch {epoch:03d} "
                f"train_loss={float(train_metrics['loss']):.4f} "
                f"val_loss={current_val_loss:.4f} "
                f"val_image_l1={float(val_metrics['image_l1']):.4f} "
                f"val_cosine={float(val_metrics['cosine']):.4f}"
            )

        if best_finetune_transition_state is not None and best_finetune_decoder_state is not None:
            transition.load_state_dict(best_finetune_transition_state)
            autoencoder.load_state_dict(best_finetune_decoder_state)
            best_transition_val_loss = best_finetune_val_loss
            best_transition_state = best_finetune_transition_state
            save_checkpoint(
                output_dir / "autoencoder_finetuned_decoder_best.pt",
                autoencoder,
                {"config": config, "best_val_loss": best_finetune_val_loss},
            )
        else:
            transition.load_state_dict(pre_finetune_transition_state)
            autoencoder.load_state_dict(pre_finetune_autoencoder_state)

    save_checkpoint(
        output_dir / "transition_best.pt",
        transition,
        {"config": config, "best_val_loss": best_transition_val_loss, "latent_shape": latent_shape},
    )

    save_split_sample_sheets(autoencoder, transition, pair_eval_datasets, device, output_dir, args.sample_max_rows)
    write_history_csv(output_dir / "transition_history.csv", transition_history)
    maybe_plot_history(
        output_dir / "curves" / "transition_history.png",
        transition_history,
        title="Latent Transition Training",
        y_keys=("train_loss", "val_loss", "train_image_l1", "val_image_l1", "train_foreground_l1", "val_foreground_l1", "train_edge_l1", "val_edge_l1", "train_cosine", "val_cosine"),
    )
    if finetune_history:
        write_history_csv(output_dir / "transition_decoder_finetune_history.csv", finetune_history)
        maybe_plot_history(
            output_dir / "curves" / "transition_decoder_finetune_history.png",
            finetune_history,
            title="Transition + Decoder Fine-Tuning",
            y_keys=("train_loss", "val_loss", "train_image_l1", "val_image_l1", "train_foreground_l1", "val_foreground_l1", "train_edge_l1", "val_edge_l1"),
        )

    comparison_metrics = {
        "train": evaluate_pair_models(autoencoder, transition, pair_eval_loaders["train"], device, args),
        "val": evaluate_pair_models(autoencoder, transition, pair_eval_loaders["val"], device, args),
        "test": evaluate_pair_models(autoencoder, transition, pair_eval_loaders["test"], device, args),
    }
    transition_train_metrics = comparison_metrics["train"]["learned_transition"]
    transition_val_metrics = comparison_metrics["val"]["learned_transition"]
    transition_test_metrics = comparison_metrics["test"]["learned_transition"]

    print_comparison_table("train", comparison_metrics["train"])
    print_comparison_table("val", comparison_metrics["val"])
    print_comparison_table("test", comparison_metrics["test"])

    metrics = {
        "split_info": split_info,
        "num_pairs": {split_name: len(pair_splits[split_name]) for split_name in SPLIT_NAMES},
        "num_unique_images": {split_name: len(image_splits[split_name]) for split_name in SPLIT_NAMES},
        "latent_shape": list(latent_shape),
        "autoencoder": {
            "best_val_loss": best_autoencoder_val_loss,
            "best_val_l1": best_autoencoder_val_l1,
            "train_metrics": autoencoder_train_metrics,
            "val_metrics": autoencoder_val_metrics,
            "test_metrics": autoencoder_test_metrics,
            "history": autoencoder_history,
        },
        "transition": {
            "best_val_loss": best_transition_val_loss,
            "train_metrics": transition_train_metrics,
            "val_metrics": transition_val_metrics,
            "test_metrics": transition_test_metrics,
            "history": transition_history,
            "finetune_history": finetune_history,
        },
        "comparison_metrics": comparison_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)

    print(f"saved autoencoder checkpoint to {output_dir / 'autoencoder_best.pt'}")
    print(f"saved transition checkpoint to {output_dir / 'transition_best.pt'}")
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
