"""Prepare image-only owl splits for Christian's staged pipeline.

Expected input layout:

    data/
      train/animal owl/*.jpg
      val/animal owl/*.jpg
      test/animal owl/*.jpg

The script writes a pipeline-friendly layout:

    <output_root>/<split>/images_256/<stem>.png
    <output_root>/<split>/masks_256/<stem>_mask.png

Mask generation is GrabCut-based and includes:
- automatic quality filtering to remove hard-to-segment samples
- optional augmentation to increase dataset size
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_SIZE = 256


def iter_image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []

    files: List[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            files.append(path)
    return files


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def load_mask_image(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {path}")
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def resize_to_square(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    if image.shape[:2] == (target_size, target_size):
        return image
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    height, width = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    mask = cv2.bitwise_or(mask, holes)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def build_prompt_box(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    margin_x = max(1, int(round(width * margin_ratio)))
    margin_y = max(1, int(round(height * margin_ratio)))

    x0 = margin_x
    y0 = margin_y
    x1 = max(x0 + 1, width - margin_x)
    y1 = max(y0 + 1, height - margin_y)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def compute_mask_quality(mask: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
    height, width = image_shape
    image_area = float(height * width)

    fg_pixels = float(np.count_nonzero(mask))
    area_ratio = fg_pixels / image_area if image_area > 0 else 0.0

    border = np.zeros_like(mask, dtype=bool)
    border[:4, :] = True
    border[-4:, :] = True
    border[:, :4] = True
    border[:, -4:] = True
    border_touch = float(np.count_nonzero(mask[border])) / fg_pixels if fg_pixels > 0 else 1.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    num_components = max(0, int(num_labels - 1))
    largest_area = float(stats[1:, cv2.CC_STAT_AREA].max()) if num_labels > 1 else 0.0
    compactness = largest_area / fg_pixels if fg_pixels > 0 else 0.0

    return {
        "area_ratio": float(area_ratio),
        "border_touch": float(border_touch),
        "components": float(num_components),
        "compactness": float(compactness),
    }


def quality_is_acceptable(
    quality: Dict[str, float],
    min_area_ratio: float,
    max_area_ratio: float,
    max_border_touch: float,
    min_compactness: float,
    max_components: int,
) -> bool:
    if quality["area_ratio"] < min_area_ratio:
        return False
    if quality["area_ratio"] > max_area_ratio:
        return False
    if quality["border_touch"] > max_border_touch:
        return False
    if quality["compactness"] < min_compactness:
        return False
    if int(round(quality["components"])) > max_components:
        return False
    return True


def quality_rejection_reason(
    quality: Dict[str, float],
    min_area_ratio: float,
    max_area_ratio: float,
    max_border_touch: float,
    min_compactness: float,
    max_components: int,
) -> str:
    if quality["area_ratio"] < min_area_ratio:
        return "mask_too_small"
    if quality["area_ratio"] > max_area_ratio:
        return "mask_too_large"
    if quality["border_touch"] > max_border_touch:
        return "touches_border_too_much"
    if quality["compactness"] < min_compactness:
        return "too_fragmented"
    if int(round(quality["components"])) > max_components:
        return "too_many_components"
    return "accepted"


def auto_grabcut_mask(image: np.ndarray, margin_ratio: float = 0.08, iterations: int = 5) -> np.ndarray:
    prompt_box = build_prompt_box(image, margin_ratio)
    rect = (
        int(prompt_box[0]),
        int(prompt_box[1]),
        max(1, int(prompt_box[2] - prompt_box[0])),
        max(1, int(prompt_box[3] - prompt_box[1])),
    )

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return clean_mask(binary_mask)


def resolve_augmentation_names(profile: str, augment_list: str) -> List[str]:
    if augment_list.strip():
        values = [item.strip() for item in augment_list.split(",") if item.strip()]
        return values

    if profile == "none":
        return []
    if profile == "flip":
        return ["hflip"]
    if profile == "light":
        return ["hflip", "rot_p8", "rot_m8", "bright_up", "bright_down"]
    if profile == "medium":
        return [
            "hflip",
            "rot_p8",
            "rot_m8",
            "rot_p15",
            "rot_m15",
            "bright_up",
            "bright_down",
            "contrast_up",
            "contrast_down",
            "blur",
        ]
    if profile == "strong":
        return [
            "hflip",
            "rot_p8",
            "rot_m8",
            "rot_p15",
            "rot_m15",
            "bright_up",
            "bright_down",
            "contrast_up",
            "contrast_down",
            "blur",
            "noise",
        ]
    raise ValueError(f"Unknown augmentation profile: {profile}")


def rotate_image_and_mask(image: np.ndarray, mask: np.ndarray, degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)

    image_rot = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_rot = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_rot, clean_mask(mask_rot)


def apply_augmentation(image: np.ndarray, mask: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "hflip":
        return cv2.flip(image, 1), cv2.flip(mask, 1)
    if name == "rot_p8":
        return rotate_image_and_mask(image, mask, 8.0)
    if name == "rot_m8":
        return rotate_image_and_mask(image, mask, -8.0)
    if name == "rot_p15":
        return rotate_image_and_mask(image, mask, 15.0)
    if name == "rot_m15":
        return rotate_image_and_mask(image, mask, -15.0)
    if name == "bright_up":
        aug = cv2.convertScaleAbs(image, alpha=1.0, beta=20)
        return aug, mask
    if name == "bright_down":
        aug = cv2.convertScaleAbs(image, alpha=1.0, beta=-20)
        return aug, mask
    if name == "contrast_up":
        aug = cv2.convertScaleAbs(image, alpha=1.15, beta=0)
        return aug, mask
    if name == "contrast_down":
        aug = cv2.convertScaleAbs(image, alpha=0.85, beta=0)
        return aug, mask
    if name == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0), mask
    if name == "noise":
        noise = np.random.normal(0.0, 8.0, image.shape).astype(np.float32)
        aug = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return aug, mask
    raise ValueError(f"Unknown augmentation: {name}")


def output_paths(output_split_root: Path, stem: str) -> Tuple[Path, Path]:
    images_dir = output_split_root / "images_256"
    masks_dir = output_split_root / "masks_256"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir / f"{stem}.png", masks_dir / f"{stem}_mask.png"


def save_pair(output_split_root: Path, stem: str, image: np.ndarray, mask: np.ndarray, overwrite: bool) -> str:
    image_out, mask_out = output_paths(output_split_root, stem)
    if image_out.exists() and mask_out.exists() and not overwrite:
        return "skipped"

    cv2.imwrite(str(image_out), image)
    cv2.imwrite(str(mask_out), mask)
    return "saved"


def prepare_split(
    input_split_dir: Path,
    output_split_root: Path,
    margin_ratio: float = 0.08,
    augment_profile: str = "none",
    augment_list: str = "",
    max_augmentations_per_image: int = 5,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.92,
    max_border_touch: float = 0.30,
    min_compactness: float = 0.75,
    max_components: int = 3,
    augment_from_existing: bool = False,
    overwrite: bool = False,
) -> Dict[str, int]:
    class_dirs = [path for path in input_split_dir.iterdir() if path.is_dir()]
    image_files: List[Path] = []
    if class_dirs:
        for class_dir in class_dirs:
            image_files.extend(iter_image_files(class_dir))
    else:
        for path in sorted(input_split_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(path)

    counts = {
        "found": len(image_files),
        "saved": 0,
        "augmented_saved": 0,
        "skipped": 0,
        "filtered": 0,
        "errors": 0,
    }

    augmentations = resolve_augmentation_names(augment_profile, augment_list)
    if max_augmentations_per_image >= 0:
        augmentations = augmentations[:max_augmentations_per_image]

    rejected_entries: List[Dict[str, object]] = []

    for image_path in image_files:
        stem = image_path.stem
        image_out, mask_out = output_paths(output_split_root, stem)

        try:
            if augment_from_existing and image_out.exists() and mask_out.exists() and not overwrite:
                image = load_color_image(image_out)
                image = resize_to_square(image, TARGET_SIZE)
                mask = load_mask_image(mask_out)
                counts["skipped"] += 1
            else:
                image = load_color_image(image_path)
                image = resize_to_square(image, TARGET_SIZE)
                mask = auto_grabcut_mask(image, margin_ratio=margin_ratio)

                quality = compute_mask_quality(mask, image.shape[:2])
                if not quality_is_acceptable(
                    quality,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    max_border_touch=max_border_touch,
                    min_compactness=min_compactness,
                    max_components=max_components,
                ):
                    counts["filtered"] += 1
                    rejected_entries.append(
                        {
                            "source": str(image_path),
                            "reason": quality_rejection_reason(
                                quality,
                                min_area_ratio=min_area_ratio,
                                max_area_ratio=max_area_ratio,
                                max_border_touch=max_border_touch,
                                min_compactness=min_compactness,
                                max_components=max_components,
                            ),
                            "quality": quality,
                        }
                    )
                    continue

                save_status = save_pair(output_split_root, stem, image, mask, overwrite=overwrite)
                if save_status == "saved":
                    counts["saved"] += 1
                else:
                    counts["skipped"] += 1

            for aug_name in augmentations:
                aug_image, aug_mask = apply_augmentation(image, mask, aug_name)
                aug_stem = f"{stem}__aug_{aug_name}"
                aug_status = save_pair(output_split_root, aug_stem, aug_image, aug_mask, overwrite=overwrite)
                if aug_status == "saved":
                    counts["augmented_saved"] += 1
                else:
                    counts["skipped"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"[ERROR] {image_path}: {exc}")

    summary_path = output_split_root / "manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2)

    rejected_path = output_split_root / "filtered_out.json"
    with open(rejected_path, "w", encoding="utf-8") as handle:
        json.dump(rejected_entries, handle, indent=2)

    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test owl images for Christian's staged pipeline."
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join("data"),
        help="Root containing train/val/test folders.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("data", "owl_output_prepared"),
        help="Root where split-specific images_256 and masks_256 folders are written.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated list of splits to prepare.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.08,
        help="Margin ratio used to initialize the prompt box.",
    )
    parser.add_argument(
        "--augment-profile",
        choices=["none", "flip", "light", "medium", "strong"],
        default="flip",
        help="Built-in augmentation profile. Use --augment-list to override it.",
    )
    parser.add_argument(
        "--augment-list",
        default="",
        help="Comma-separated explicit augmentations. Example: hflip,rot_p8,bright_up",
    )
    parser.add_argument(
        "--max-augmentations-per-image",
        type=int,
        default=5,
        help="Cap on number of augmentations applied per accepted image (-1 for all).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.03,
        help="Reject masks with foreground area below this ratio.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.92,
        help="Reject masks with foreground area above this ratio.",
    )
    parser.add_argument(
        "--max-border-touch",
        type=float,
        default=0.30,
        help="Reject masks with too much foreground touching image borders.",
    )
    parser.add_argument(
        "--min-compactness",
        type=float,
        default=0.75,
        help="Reject masks where the largest component is too small relative to total foreground.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Reject masks with too many connected foreground components.",
    )
    parser.add_argument(
        "--augment-from-existing",
        action="store_true",
        help="If prepared pairs already exist, reuse them and only create augmentations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        print(f"Input root does not exist: {input_root}")
        return

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        print("No splits selected.")
        return

    overall = {"found": 0, "saved": 0, "augmented_saved": 0, "skipped": 0, "filtered": 0, "errors": 0}
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            print(f"[SKIP] Missing split folder: {split_dir}")
            continue

        output_split_root = output_root / split
        counts = prepare_split(
            split_dir,
            output_split_root,
            margin_ratio=args.margin_ratio,
            augment_profile=args.augment_profile,
            augment_list=args.augment_list,
            max_augmentations_per_image=args.max_augmentations_per_image,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            max_border_touch=args.max_border_touch,
            min_compactness=args.min_compactness,
            max_components=args.max_components,
            augment_from_existing=args.augment_from_existing,
            overwrite=args.overwrite,
        )
        print(
            f"[OK] {split}: found={counts['found']} saved={counts['saved']} augmented_saved={counts['augmented_saved']} skipped={counts['skipped']} filtered={counts['filtered']} errors={counts['errors']}"
        )
        for key in overall:
            if key in counts:
                overall[key] += counts[key]

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    print(f"\nPrepared dataset written to: {output_root}")
    print(f"Summary: {overall}")


if __name__ == "__main__":
    main()
