import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence

from script_utils import LEARNING_ROOT, ROOT, ensure_dir, write_csv, write_json

from owl_pipeline_utils import STAGE_SPECS, collect_sample_stems, stage_cumulative_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frame and transition manifests for staged owl learning.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "owl_output"), help="Path to data/owl_output root.")
    parser.add_argument(
        "--output-dir",
        default=str(LEARNING_ROOT),
        help="Directory for generated manifests and QA report.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stem splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Keep stems with missing stages (default drops incomplete stems).",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if train <= 0 or val < 0 or test < 0:
        raise ValueError("Ratios must be non-negative, and train must be > 0.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")


def validate_input_dirs(images_dir: Path, masks_dir: Path) -> None:
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Expected images_256 and masks_256 under --data-root")


def get_missing_stage_indices(data_root: Path, stem: str, stage_indices: Sequence[int]) -> List[int]:
    missing: List[int] = []
    for stage_idx in stage_indices:
        stage_file = Path(stage_cumulative_path(str(data_root), stem, stage_idx))
        if not stage_file.exists():
            missing.append(stage_idx)
    return missing


def validate_split_source_stems(split_source_stems: Sequence[str]) -> None:
    if not split_source_stems:
        raise RuntimeError("No eligible stems found after filtering. Try --allow-incomplete or regenerate stages.")


def split_stems(stems: Sequence[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    stems = list(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)

    train = stems[:n_train]
    val = stems[n_train : n_train + n_val]
    test = stems[n_train + n_val : n_train + n_val + n_test]

    return {"train": train, "val": val, "test": test}


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = data_root / "images_256"
    masks_dir = data_root / "masks_256"

    validate_input_dirs(images_dir, masks_dir)

    stage_indices = [spec["idx"] for spec in STAGE_SPECS]
    stems = collect_sample_stems(str(images_dir), str(masks_dir))

    frame_rows: List[Dict[str, str]] = []
    complete_stems: List[str] = []
    dropped_stems: List[Dict[str, object]] = []

    for stem in stems:
        missing = get_missing_stage_indices(data_root, stem, stage_indices)

        if missing and not args.allow_incomplete:
            dropped_stems.append({"stem": stem, "missing_stage_indices": missing})
            continue

        if not missing:
            complete_stems.append(stem)

        for stage_idx in stage_indices:
            stage_file = Path(stage_cumulative_path(str(data_root), stem, stage_idx))
            if not stage_file.exists():
                continue
            frame_rows.append(
                {
                    "stem": stem,
                    "stage_idx": str(stage_idx),
                    "image_path": str(stage_file),
                    "frame_key": f"{stem}_stage{stage_idx:02d}",
                }
            )

    split_source_stems = complete_stems if not args.allow_incomplete else sorted({row["stem"] for row in frame_rows})
    validate_split_source_stems(split_source_stems)

    split_map = split_stems(split_source_stems, args.train_ratio, args.val_ratio, args.seed)
    split_of_stem = {stem: split_name for split_name, split_stems_list in split_map.items() for stem in split_stems_list}

    for row in frame_rows:
        row["split"] = split_of_stem.get(row["stem"], "")

    frame_rows = [row for row in frame_rows if row["split"]]

    transition_rows: List[Dict[str, str]] = []
    frame_index = {(row["stem"], int(row["stage_idx"])): row for row in frame_rows}
    for stem in split_source_stems:
        split_name = split_of_stem[stem]
        for t in range(0, 9):
            src = frame_index.get((stem, t))
            tgt = frame_index.get((stem, t + 1))
            if src is None or tgt is None:
                continue
            transition_rows.append(
                {
                    "stem": stem,
                    "split": split_name,
                    "src_stage_idx": str(t),
                    "tgt_stage_idx": str(t + 1),
                    "src_image_path": src["image_path"],
                    "tgt_image_path": tgt["image_path"],
                    "transition_key": f"{stem}_stage{t:02d}_to_stage{t+1:02d}",
                }
            )

    ensure_dir(output_dir)
    write_csv(
        output_dir / "manifest_frames.csv",
        frame_rows,
        ["stem", "split", "stage_idx", "image_path", "frame_key"],
    )
    write_csv(
        output_dir / "manifest_transitions.csv",
        transition_rows,
        ["stem", "split", "src_stage_idx", "tgt_stage_idx", "src_image_path", "tgt_image_path", "transition_key"],
    )

    for split_name, split_stems_list in split_map.items():
        split_file = output_dir / f"stems_{split_name}.txt"
        split_file.write_text("\n".join(split_stems_list) + ("\n" if split_stems_list else ""), encoding="utf-8")

    qa = {
        "data_root": str(data_root),
        "allow_incomplete": bool(args.allow_incomplete),
        "num_input_stems": len(stems),
        "num_split_stems": len(split_source_stems),
        "num_dropped_stems": len(dropped_stems),
        "num_frame_rows": len(frame_rows),
        "num_transition_rows": len(transition_rows),
        "split_counts": {k: len(v) for k, v in split_map.items()},
        "dropped_stems": dropped_stems,
    }
    write_json(output_dir / "manifest_qa.json", qa)

    print("Manifest generation complete")
    print(f"Frames:      {len(frame_rows)}")
    print(f"Transitions: {len(transition_rows)}")
    print(f"Splits:      {qa['split_counts']}")
    print(f"Output dir:  {output_dir}")


if __name__ == "__main__":
    main()
