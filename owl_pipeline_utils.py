import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CANVAS_SIZE = 256


STAGE_SPECS = [
    {"idx": 0, "action": "base_ellipses", "folder": "stage_00_base", "layer_name": "base_ellipses"},
    {"idx": 1, "action": "add_outer_contour", "folder": "stage_01_outer_contour", "layer_name": "outer_contour"},
    {"idx": 2, "action": "add_facial_features", "folder": "stage_02_facial_features", "layer_name": "facial_features"},
    {"idx": 3, "action": "add_part_boundaries", "folder": "stage_03_part_boundaries", "layer_name": "part_boundaries"},
    {"idx": 4, "action": "add_inner_contours", "folder": "stage_04_inner_contours", "layer_name": "inner_contours"},
    {"idx": 5, "action": "add_value_regions", "folder": "stage_05_value_regions", "layer_name": "value_regions"},
    {"idx": 6, "action": "add_feather_masses", "folder": "stage_06_feather_masses", "layer_name": "feather_masses"},
    {"idx": 7, "action": "add_fine_texture", "folder": "stage_07_fine_texture", "layer_name": "fine_texture"},
    {"idx": 8, "action": "add_color", "folder": "stage_08_color", "layer_name": "color"},
    {"idx": 9, "action": "add_background", "folder": "stage_09_background", "layer_name": "background"},
]

STAGE_BY_INDEX = {spec["idx"]: spec for spec in STAGE_SPECS}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_all_output_folders(output_root: str) -> None:
    ensure_dir(output_root)
    ensure_dir(os.path.join(output_root, "annotations"))
    for spec in STAGE_SPECS:
        ensure_dir(os.path.join(output_root, spec["folder"]))


def get_image_files(folder: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTENSIONS:
            files.append(path)
    return files


def derive_stem_from_mask_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    if stem.endswith("_mask"):
        stem = stem[:-5]
    return stem


def collect_sample_stems(images_dir: str, masks_dir: str) -> List[str]:
    image_stems = {os.path.splitext(os.path.basename(p))[0] for p in get_image_files(images_dir)}
    mask_stems = {derive_stem_from_mask_name(os.path.basename(p)) for p in get_image_files(masks_dir)}
    return sorted(image_stems.intersection(mask_stems))


def load_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load color image: {path}")
    return img


def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load grayscale image: {path}")
    return img


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def load_binary_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not load mask: {path}")
    return ensure_binary_mask(mask)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_contours_blank(shape: Tuple[int, int], contours: Sequence[np.ndarray], thickness: int = 1) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(canvas, list(contours), -1, 255, thickness)
    return canvas


def overlay_sparse_gray(prev_stage: Optional[np.ndarray], layer: np.ndarray) -> np.ndarray:
    if prev_stage is None:
        return layer.copy()
    return np.maximum(prev_stage, layer)


def overlay_dense_gray(prev_stage: Optional[np.ndarray], layer: np.ndarray) -> np.ndarray:
    if prev_stage is None:
        return layer.copy()
    return np.maximum(prev_stage, layer)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def compose_color_stage(prev_stage: Optional[np.ndarray], color_layer: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if prev_stage is None:
        base = np.zeros_like(color_layer)
    else:
        base = to_bgr(prev_stage)

    out = base.copy()
    if mask is None:
        use_idx = np.where(np.any(color_layer > 0, axis=2))
        out[use_idx] = color_layer[use_idx]
        return out

    inside = mask > 0
    out[inside] = color_layer[inside]
    return out


def serialize_contour(contour: np.ndarray) -> List[List[int]]:
    return contour.reshape(-1, 2).astype(int).tolist()


def serialize_contours(contours: Sequence[np.ndarray]) -> List[List[List[int]]]:
    return [serialize_contour(c) for c in contours]


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def stage_folder(output_root: str, stage_idx: int) -> str:
    return os.path.join(output_root, STAGE_BY_INDEX[stage_idx]["folder"])


def stage_layer_path(output_root: str, stem: str, stage_idx: int) -> str:
    spec = STAGE_BY_INDEX[stage_idx]
    return os.path.join(stage_folder(output_root, stage_idx), f"{stem}_{spec['layer_name']}_layer.png")


def stage_cumulative_path(output_root: str, stem: str, stage_idx: int) -> str:
    return os.path.join(stage_folder(output_root, stage_idx), f"{stem}_stage{stage_idx:02d}.png")


def annotation_path(output_root: str, stem: str, stage_idx: int) -> str:
    return os.path.join(output_root, "annotations", f"{stem}_stage{stage_idx:02d}.json")


def load_previous_stage_if_present(output_root: str, stem: str, stage_idx: int, grayscale: bool = True) -> Optional[np.ndarray]:
    if stage_idx <= 0:
        return None
    prev_path = stage_cumulative_path(output_root, stem, stage_idx - 1)
    if not os.path.isfile(prev_path):
        return None
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    prev = cv2.imread(prev_path, mode)
    return prev


def parse_stage_selection(stage_arg: str) -> List[int]:
    if stage_arg.strip().lower() == "all":
        return [spec["idx"] for spec in STAGE_SPECS]

    items = [s.strip() for s in stage_arg.split(",") if s.strip()]
    stages: List[int] = []
    for item in items:
        idx = int(item)
        if idx not in STAGE_BY_INDEX:
            raise ValueError(f"Unknown stage index: {idx}")
        stages.append(idx)
    return sorted(set(stages))
