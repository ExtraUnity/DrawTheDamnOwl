import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from script_utils import ensure_dir, read_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINO-style embeddings for stage frames.")
    parser.add_argument("--manifest-frames", required=True, help="Path to frame manifest CSV from build_manifest.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for embedding artifacts")
    parser.add_argument("--model-id", default="facebook/dinov2-base", help="DINOv2 model id (HF) or timm model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for forward pass")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device for inference")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick smoke runs (0 = all)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(path, description="Frame manifest")
    if not rows:
        raise RuntimeError("No rows in manifest. Check path.")
    rows.sort(key=lambda r: (r["stem"], int(r["stage_idx"])))
    return rows


def load_images(batch: List[Dict[str, str]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for row in batch:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def try_load_timm(model_id: str, device: torch.device):
    try:
        import timm
        from torchvision import transforms

        model = timm.create_model(model_id, pretrained=True)
        model.eval().to(device)
        cfg = getattr(model, "default_cfg", {})
        size = cfg.get("input_size", (3, 224, 224))
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))

        transform = transforms.Compose([
            transforms.Resize((size[1], size[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        def forward(images: List[Image.Image]):
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

        return forward
    except Exception:
        return None


def try_load_hf(model_id: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        def forward(images: List[Image.Image]):
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

        return forward
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_frames)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    rows = read_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    device = torch.device(args.device)

    extractor = try_load_timm(args.model_id, device)
    loader_name = "timm"
    if extractor is None:
        extractor = try_load_hf(args.model_id, device)
        loader_name = "huggingface"

    if extractor is None:
        raise RuntimeError("Failed to initialize DINO model. Install 'timm' or ensure HF model id is valid.")

    print(f"Using {loader_name} loader for model {args.model_id}")

    all_features: List[np.ndarray] = []
    stems: List[str] = []
    stage_indices: List[int] = []
    image_paths: List[str] = []
    splits: List[str] = []

    def batch_iter(items: List[Dict[str, str]], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    for batch in batch_iter(rows, args.batch_size):
        images = load_images(batch)
        feats = extractor(images)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        if not args.no_normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

        all_features.append(feats.detach().cpu().numpy().astype(np.float32))

        for row in batch:
            stems.append(row["stem"])
            stage_indices.append(int(row["stage_idx"]))
            image_paths.append(row["image_path"])
            splits.append(row.get("split", ""))

    embeddings = np.concatenate(all_features, axis=0)

    stem_arr = np.array(stems, dtype=object)
    stage_arr = np.array(stage_indices, dtype=np.int16)
    path_arr = np.array(image_paths, dtype=object)
    split_arr = np.array(splits, dtype=object)

    out_npz = out_dir / f"dino_embeddings_all.npz"
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
        "loader": loader_name,
        "device": str(device),
        "normalize": not args.no_normalize,
        "manifest_frames": str(manifest_path),
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "output_npz": str(out_npz),
    }
    write_json(out_dir / f"dino_embeddings_all.json", meta)

    print("DINO embedding extraction complete")
    print(f"Samples:       {meta['num_samples']}")
    print(f"Embedding dim: {meta['embedding_dim']}")
    print(f"Output:        {out_npz}")


if __name__ == "__main__":
    main()
