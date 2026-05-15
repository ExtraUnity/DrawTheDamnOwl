from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_owl_dataset import prepare_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a flat owl image folder into images_256/masks_256 for the staged owl pipeline."
    )
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "data" / "images"),
        help="Directory containing raw owl images. Nested class folders are also supported.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "owl_output"),
        help="Directory where images_256 and masks_256 will be written.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.08,
        help="Margin ratio used to initialize the GrabCut prompt box.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared image/mask pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    counts = prepare_split(
        input_dir,
        output_root,
        margin_ratio=args.margin_ratio,
        augment_profile="none",
        augment_list="",
        max_augmentations_per_image=0,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_border_touch=args.max_border_touch,
        min_compactness=args.min_compactness,
        max_components=args.max_components,
        augment_from_existing=False,
        overwrite=args.overwrite,
    )

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "counts": counts,
    }
    summary_path = output_root / "segmentation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Segmentation complete")
    print(f"Input dir:   {input_dir}")
    print(f"Output root: {output_root}")
    print(f"Counts:      {counts}")


if __name__ == "__main__":
    main()
