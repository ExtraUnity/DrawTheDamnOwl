import argparse
import os
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from owl_pipeline_stages import (
    interior_mask_by_distance,
    stage_00_base_ellipses,
    stage_01_outer_contour,
    stage_02_facial_features,
    stage_04_inner_contours,
    stage_05_value_regions,
    stage_07_fine_texture,
)
from owl_pipeline_utils import (
    CANVAS_SIZE,
    annotation_path,
    collect_sample_stems,
    ensure_all_output_folders,
    keep_largest_component,
    load_binary_mask,
    load_image_color,
    load_previous_stage_if_present,
    parse_stage_selection,
    save_json,
    stage_cumulative_path,
    stage_layer_path,
)


# Carryover policy controls.
EXCLUDE_STAGE_02_03_FROM_LATER = True
EXCLUDE_STAGE_00_FROM_LATER = True
EXCLUDE_STAGE_01_FROM_STAGE5_ONWARD = True
EXCLUDE_STAGE_04_FROM_STAGE7_ONWARD = True
SUPPRESS_EDGE_BAND_FROM_STAGE6_ONWARD = True


STAGE_ACTION = {
    0: "base_ellipses",
    1: "add_outer_contour",
    2: "add_facial_features",
    3: "add_inner_contours",
    4: "add_value_regions",
    5: "add_fine_texture",
}


class OwlStagedPipeline:
    def __init__(self, output_root: str, images_dir: str, masks_dir: str):
        self.output_root = output_root
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        ensure_all_output_folders(output_root)

    def _read_inputs(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        image_path = os.path.join(self.images_dir, f"{stem}.png")
        if not os.path.isfile(image_path):
            candidates = [p for p in os.listdir(self.images_dir) if os.path.splitext(p)[0] == stem]
            if candidates:
                image_path = os.path.join(self.images_dir, candidates[0])

        mask_path = os.path.join(self.masks_dir, f"{stem}_mask.png")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask for stem '{stem}': {mask_path}")

        image = load_image_color(image_path)
        if image.shape[:2] != (CANVAS_SIZE, CANVAS_SIZE):
            image = cv2.resize(image, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA)

        mask = load_binary_mask(mask_path)
        if mask.shape[:2] != (CANVAS_SIZE, CANVAS_SIZE):
            mask = cv2.resize(mask, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = keep_largest_component(mask)
        return image, mask

    def _write_stage_outputs(self, stem: str, stage_idx: int, layer: np.ndarray, cumulative: np.ndarray, meta: Dict) -> None:
        layer_path = stage_layer_path(self.output_root, stem, stage_idx)
        stage_path = stage_cumulative_path(self.output_root, stem, stage_idx)
        ann_path = annotation_path(self.output_root, stem, stage_idx)

        cv2.imwrite(layer_path, layer)
        cv2.imwrite(stage_path, cumulative)

        payload = {
            "image_id": stem,
            "stage_index": stage_idx,
            "stage_action": STAGE_ACTION[stage_idx],
            "layer_file": os.path.basename(layer_path),
            "cumulative_file": os.path.basename(stage_path),
            "metadata": meta,
        }
        save_json(ann_path, payload)

    def _run_single_stage(
        self,
        stem: str,
        image: np.ndarray,
        mask: np.ndarray,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_color: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if stage_idx == 0:
            layer, cumulative, meta = stage_00_base_ellipses(mask)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        if stage_idx == 1:
            layer, cumulative, meta = stage_01_outer_contour(mask, prev_gray)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        if stage_idx == 2:
            layer, cumulative, meta = stage_02_facial_features(image, mask, prev_gray)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        if stage_idx == 3:
            layer, cumulative, meta = stage_04_inner_contours(image, mask, prev_gray)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        if stage_idx == 4:
            layer, cumulative, meta = stage_05_value_regions(image, mask, prev_gray)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        if stage_idx == 5:
            layer, cumulative, meta = stage_07_fine_texture(image, mask, prev_gray)
            self._write_stage_outputs(stem, stage_idx, layer, cumulative, meta)
            return cumulative, prev_color

        raise ValueError(f"Unsupported stage index: {stage_idx}")

    def _restore_skip_state(
        self,
        stem: str,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_color: Optional[np.ndarray],
        prev_gray_clean_later: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        stage_out = stage_cumulative_path(self.output_root, stem, stage_idx)

        if stage_idx <= 7:
            prev_gray = cv2.imread(stage_out, cv2.IMREAD_GRAYSCALE)
            if stage_idx == 1 or (EXCLUDE_STAGE_02_03_FROM_LATER and 4 <= stage_idx <= 7):
                if stage_idx == 1 and EXCLUDE_STAGE_00_FROM_LATER:
                    prev_gray_clean_later = cv2.imread(stage_layer_path(self.output_root, stem, 1), cv2.IMREAD_GRAYSCALE)
                else:
                    prev_gray_clean_later = prev_gray.copy() if prev_gray is not None else None
            return prev_gray, prev_color, prev_gray_clean_later

        prev_color = cv2.imread(stage_out, cv2.IMREAD_COLOR)
        return prev_gray, prev_color, prev_gray_clean_later

    def _ensure_previous_stages_loaded(
        self,
        stem: str,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_color: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if stage_idx > 0 and prev_gray is None and stage_idx <= 5:
            prev_gray = load_previous_stage_if_present(self.output_root, stem, stage_idx, grayscale=True)
        if stage_idx > 8 and prev_color is None:
            prev_color = load_previous_stage_if_present(self.output_root, stem, stage_idx, grayscale=False)
        return prev_gray, prev_color

    def _validate_stage_inputs(
        self,
        stem: str,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_color: Optional[np.ndarray],
    ) -> bool:
        if stage_idx > 0 and stage_idx < 4 and prev_gray is None:
            print(f"[ERROR] {stem} stage {stage_idx:02d}: missing previous grayscale cumulative stage")
            return False
        if stage_idx == 9 and prev_color is None:
            print(f"[ERROR] {stem} stage 09: missing stage 08 color cumulative image")
            return False
        return True

    def _baseline_for_later_stages(
        self,
        stem: str,
        stage_idx: int,
        prev_gray_clean_later: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if prev_gray_clean_later is not None:
            return prev_gray_clean_later

        if stage_idx >= 5:
            prev_chain_path = stage_cumulative_path(self.output_root, stem, stage_idx - 1)
            if os.path.isfile(prev_chain_path):
                return cv2.imread(prev_chain_path, cv2.IMREAD_GRAYSCALE)

        stage01_baseline_path = (
            stage_layer_path(self.output_root, stem, 1)
            if EXCLUDE_STAGE_00_FROM_LATER
            else stage_cumulative_path(self.output_root, stem, 1)
        )
        if os.path.isfile(stage01_baseline_path):
            return cv2.imread(stage01_baseline_path, cv2.IMREAD_GRAYSCALE)

        return None

    def _apply_stage01_suppression(
        self,
        stem: str,
        stage_idx: int,
        stage_prev_gray: Optional[np.ndarray],
        stage01_layer_cache: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not (EXCLUDE_STAGE_01_FROM_STAGE5_ONWARD and stage_idx >= 4 and stage_prev_gray is not None):
            return stage_prev_gray, stage01_layer_cache

        if stage01_layer_cache is None:
            stage01_layer_path = stage_layer_path(self.output_root, stem, 1)
            if os.path.isfile(stage01_layer_path):
                stage01_layer_cache = cv2.imread(stage01_layer_path, cv2.IMREAD_GRAYSCALE)

        if stage01_layer_cache is not None:
            stage_prev_gray = stage_prev_gray.copy()
            stage_prev_gray[stage01_layer_cache > 0] = 0

        return stage_prev_gray, stage01_layer_cache

    def _apply_stage04_suppression(
        self,
        stem: str,
        stage_idx: int,
        stage_prev_gray: Optional[np.ndarray],
        stage04_layer_cache: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not (EXCLUDE_STAGE_04_FROM_STAGE7_ONWARD and stage_idx >= 5 and stage_prev_gray is not None):
            return stage_prev_gray, stage04_layer_cache

        if stage04_layer_cache is None:
            stage04_layer_path = stage_layer_path(self.output_root, stem, 3)
            if os.path.isfile(stage04_layer_path):
                stage04_layer_cache = cv2.imread(stage04_layer_path, cv2.IMREAD_GRAYSCALE)

        if stage04_layer_cache is not None:
            stage_prev_gray = stage_prev_gray.copy()
            smooth_fill = cv2.medianBlur(stage_prev_gray, 5)
            stage_prev_gray[stage04_layer_cache > 0] = smooth_fill[stage04_layer_cache > 0]

        return stage_prev_gray, stage04_layer_cache

    def _apply_edge_band_suppression(self, stage_idx: int, stage_prev_gray: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
        if not (SUPPRESS_EDGE_BAND_FROM_STAGE6_ONWARD and stage_idx >= 6 and stage_prev_gray is not None):
            return stage_prev_gray

        stage_prev_gray = stage_prev_gray.copy()
        inner_mask = interior_mask_by_distance(mask, min_distance_px=6.0)
        edge_band = cv2.subtract(mask, inner_mask)
        smooth_fill = cv2.medianBlur(stage_prev_gray, 5)
        stage_prev_gray[edge_band > 0] = smooth_fill[edge_band > 0]
        return stage_prev_gray

    def _prepare_stage_prev_gray(
        self,
        stem: str,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_gray_clean_later: Optional[np.ndarray],
        stage01_layer_cache: Optional[np.ndarray],
        stage04_layer_cache: Optional[np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        stage_prev_gray = prev_gray

        if EXCLUDE_STAGE_02_03_FROM_LATER and 3 <= stage_idx <= 5:
            clean_baseline = self._baseline_for_later_stages(stem, stage_idx, prev_gray_clean_later)
            if clean_baseline is None:
                return None, prev_gray_clean_later, stage01_layer_cache, stage04_layer_cache
            prev_gray_clean_later = clean_baseline
            stage_prev_gray = clean_baseline

        stage_prev_gray, stage01_layer_cache = self._apply_stage01_suppression(
            stem, stage_idx, stage_prev_gray, stage01_layer_cache
        )
        stage_prev_gray, stage04_layer_cache = self._apply_stage04_suppression(
            stem, stage_idx, stage_prev_gray, stage04_layer_cache
        )
        stage_prev_gray = self._apply_edge_band_suppression(stage_idx, stage_prev_gray, mask)

        return stage_prev_gray, prev_gray_clean_later, stage01_layer_cache, stage04_layer_cache

    def _update_caches_after_stage(
        self,
        stem: str,
        stage_idx: int,
        prev_gray: Optional[np.ndarray],
        prev_gray_clean_later: Optional[np.ndarray],
        stage01_layer_cache: Optional[np.ndarray],
        stage04_layer_cache: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if stage_idx == 1:
            stage01_layer_cache = cv2.imread(stage_layer_path(self.output_root, stem, 1), cv2.IMREAD_GRAYSCALE)
            if EXCLUDE_STAGE_00_FROM_LATER:
                prev_gray_clean_later = stage01_layer_cache
            else:
                prev_gray_clean_later = prev_gray.copy() if prev_gray is not None else None

        if stage_idx == 3:
            stage04_layer_cache = cv2.imread(stage_layer_path(self.output_root, stem, 3), cv2.IMREAD_GRAYSCALE)

        if EXCLUDE_STAGE_02_03_FROM_LATER and 3 <= stage_idx <= 5:
            prev_gray_clean_later = prev_gray.copy() if prev_gray is not None else None

        return prev_gray_clean_later, stage01_layer_cache, stage04_layer_cache

    def run_for_stem(self, stem: str, stages: Sequence[int], overwrite: bool = True) -> bool:
        try:
            image, mask = self._read_inputs(stem)
        except Exception as exc:
            print(f"[ERROR] {stem}: failed loading inputs: {exc}")
            return False

        prev_gray: Optional[np.ndarray] = None
        prev_color: Optional[np.ndarray] = None
        prev_gray_clean_later: Optional[np.ndarray] = None
        stage01_layer_cache: Optional[np.ndarray] = None
        stage04_layer_cache: Optional[np.ndarray] = None

        for stage_idx in sorted(stages):
            stage_out = stage_cumulative_path(self.output_root, stem, stage_idx)
            if os.path.isfile(stage_out) and not overwrite:
                print(f"[SKIP] {stem} stage {stage_idx:02d}: output exists and overwrite disabled")
                prev_gray, prev_color, prev_gray_clean_later = self._restore_skip_state(
                    stem, stage_idx, prev_gray, prev_color, prev_gray_clean_later
                )
                continue

            prev_gray, prev_color = self._ensure_previous_stages_loaded(stem, stage_idx, prev_gray, prev_color)
            if not self._validate_stage_inputs(stem, stage_idx, prev_gray, prev_color):
                return False

            stage_prev_gray, prev_gray_clean_later, stage01_layer_cache, stage04_layer_cache = self._prepare_stage_prev_gray(
                stem,
                stage_idx,
                prev_gray,
                prev_gray_clean_later,
                stage01_layer_cache,
                stage04_layer_cache,
                mask,
            )

            if EXCLUDE_STAGE_02_03_FROM_LATER and 4 <= stage_idx <= 8 and stage_prev_gray is None:
                print(f"[ERROR] {stem} stage {stage_idx:02d}: missing clean baseline (stage 01 layer/cumulative) for later stages")
                return False

            try:
                prev_gray, prev_color = self._run_single_stage(stem, image, mask, stage_idx, stage_prev_gray, prev_color)
                prev_gray_clean_later, stage01_layer_cache, stage04_layer_cache = self._update_caches_after_stage(
                    stem,
                    stage_idx,
                    prev_gray,
                    prev_gray_clean_later,
                    stage01_layer_cache,
                    stage04_layer_cache,
                )
                print(f"[OK] {stem} stage {stage_idx:02d}")
            except Exception as exc:
                print(f"[ERROR] {stem} stage {stage_idx:02d}: {exc}")
                return False

        return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build staged owl-drawing dataset from aligned 256x256 owl images and masks."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join("data", "owl_output"),
        help="Root folder containing images_256 and masks_256 plus output stage folders.",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Stages to run: 'all' or comma-separated indices, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Optional single sample stem to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage outputs. Default is false.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = args.data_root
    images_dir = os.path.join(output_root, "images_256")
    masks_dir = os.path.join(output_root, "masks_256")

    if not os.path.isdir(images_dir):
        print(f"Image folder does not exist: {images_dir}")
        return
    if not os.path.isdir(masks_dir):
        print(f"Mask folder does not exist: {masks_dir}")
        return

    try:
        selected_stages = parse_stage_selection(args.stages)
    except Exception as exc:
        print(f"Invalid --stages argument: {exc}")
        return

    if not selected_stages:
        print("No stages selected.")
        return

    pipeline = OwlStagedPipeline(output_root=output_root, images_dir=images_dir, masks_dir=masks_dir)

    if args.stem is not None:
        stems = [args.stem]
    else:
        stems = collect_sample_stems(images_dir, masks_dir)

    if not stems:
        print("No valid stems found in images_256/masks_256.")
        return

    print(f"Processing {len(stems)} sample(s) with stages: {selected_stages}")
    ok_count = 0
    fail_count = 0

    for stem in stems:
        ok = pipeline.run_for_stem(stem, stages=selected_stages, overwrite=args.overwrite)
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    print("\nDone.")
    print(f"Successful samples: {ok_count}")
    print(f"Failed samples:     {fail_count}")


if __name__ == "__main__":
    main()
