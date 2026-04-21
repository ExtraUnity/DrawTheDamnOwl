import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from learning_utils import default_device, extract_clip_image_features, load_clip
from script_utils import LEARNING_ROOT, ensure_dir, read_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP embeddings for stage frames.")
    parser.add_argument(
        "--manifest-frames",
        default=str(LEARNING_ROOT / "manifest_frames.csv"),
        help="Path to frame manifest CSV from build_manifest.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LEARNING_ROOT / "embeddings"),
        help="Output directory for embedding artifacts",
    )
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model id")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP forward pass")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all", help="Split to embed")
    parser.add_argument("--device", default=default_device(), help="Device for inference")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick smoke runs (0 = all)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings")
    return parser.parse_args()


def read_manifest(path: Path, split: str) -> List[Dict[str, str]]:
    rows = [
        row
        for row in read_csv_rows(path, description="Frame manifest")
        if split == "all" or row["split"] == split
    ]

    if not rows:
        raise RuntimeError("No rows selected from manifest. Check split or manifest content.")

    rows.sort(key=lambda r: (r["stem"], int(r["stage_idx"])))
    return rows


def batch_iter(items: List[Dict[str, str]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_images(batch: List[Dict[str, str]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for row in batch:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_frames)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    rows = read_manifest(manifest_path, args.split)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    device = torch.device(args.device)
    processor, model = load_clip(args.model_id, device)

    all_features: List[np.ndarray] = []
    stems: List[str] = []
    stage_indices: List[int] = []
    image_paths: List[str] = []
    splits: List[str] = []

    with torch.no_grad():
        for batch in batch_iter(rows, args.batch_size):
            images = load_images(batch)
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            features = extract_clip_image_features(model, inputs)
            if not args.no_normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            all_features.append(features.detach().cpu().numpy().astype(np.float32))

            for row in batch:
                stems.append(row["stem"])
                stage_indices.append(int(row["stage_idx"]))
                image_paths.append(row["image_path"])
                splits.append(row["split"])

    embeddings = np.concatenate(all_features, axis=0)
    if embeddings.shape[0] != len(rows):
        raise RuntimeError("Embedding count mismatch with selected rows.")

    stem_arr = np.array(stems, dtype=object)
    stage_arr = np.array(stage_indices, dtype=np.int16)
    path_arr = np.array(image_paths, dtype=object)
    split_arr = np.array(splits, dtype=object)

    split_tag = args.split
    out_npz = output_dir / f"clip_embeddings_{split_tag}.npz"
    np.savez_compressed(
        out_npz,
        embeddings=embeddings,
        stems=stem_arr,
        stage_indices=stage_arr,
        image_paths=path_arr,
        splits=split_arr,
    )

    meta = {
        "model_id": args.model_id,
        "device": str(device),
        "normalize": not args.no_normalize,
        "manifest_frames": str(manifest_path),
        "split": args.split,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "output_npz": str(out_npz),
    }
    out_meta = output_dir / f"clip_embeddings_{split_tag}.json"
    write_json(out_meta, meta)

    print("CLIP embedding extraction complete")
    print(f"Samples:       {meta['num_samples']}")
    print(f"Embedding dim: {meta['embedding_dim']}")
    print(f"Output:        {out_npz}")


if __name__ == "__main__":
    main()
