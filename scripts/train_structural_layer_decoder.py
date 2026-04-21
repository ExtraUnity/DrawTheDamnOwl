import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from learning_utils import (
    StructuralLayerUNet,
    compose_white_layer,
    default_device,
    dice_loss_from_logits,
    focal_bce_loss_from_logits,
    image_to_tensor,
    mask_to_tensor,
    sobel_edges,
    tensor_to_pil,
)
from script_utils import LEARNING_ROOT, ROOT, ensure_dir, read_csv_rows, require_file, write_json

from owl_pipeline_utils import stage_layer_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a structural layer decoder for early owl drawing stages.")
    parser.add_argument(
        "--transitions-csv",
        default=str(LEARNING_ROOT / "manifest_transitions.csv"),
        help="Path to transition manifest CSV from build_manifest.py",
    )
    parser.add_argument("--data-root", default=str(ROOT / "data" / "owl_output"), help="Path to data/owl_output")
    parser.add_argument(
        "--output-dir",
        default=str(LEARNING_ROOT / "structural_decoder"),
        help="Output directory for structural decoder artifacts",
    )
    parser.add_argument("--device", default=default_device(), help="Training device")
    parser.add_argument("--image-size", type=int, default=256, help="Square training resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--base-channels", type=int, default=32, help="UNet base channel count")
    parser.add_argument("--cond-dim", type=int, default=128, help="Conditioning MLP width")
    parser.add_argument("--stage-embed-dim", type=int, default=16, help="Learned source-stage embedding size")
    parser.add_argument("--min-src-stage", type=int, default=0, help="Minimum source stage to train")
    parser.add_argument("--max-src-stage", type=int, default=3, help="Maximum source stage to train")
    parser.add_argument("--focal-weight", type=float, default=1.0, help="Focal BCE loss weight")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--compose-weight", type=float, default=1.0, help="Composed next-image L1 loss weight")
    parser.add_argument("--edge-weight", type=float, default=0.5, help="Composed edge consistency loss weight")
    parser.add_argument("--pos-weight", type=float, default=12.0, help="Positive pixel weight for focal BCE")
    parser.add_argument("--focal-alpha", type=float, default=0.8, help="Focal BCE alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal BCE gamma")
    parser.add_argument("--threshold", type=float, default=0.35, help="Binary layer threshold for sample previews")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience on validation loss")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(transitions_csv: Path, data_root: Path, min_src_stage: int, max_src_stage: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_csv_rows(transitions_csv, description="Transition manifest"):
        src_stage = int(row["src_stage_idx"])
        tgt_stage = int(row["tgt_stage_idx"])
        if src_stage < min_src_stage or src_stage > max_src_stage:
            continue

        stem = str(row["stem"])
        src_image_path = Path(row["src_image_path"])
        tgt_image_path = Path(row["tgt_image_path"])
        tgt_layer_path = Path(stage_layer_path(str(data_root), stem, tgt_stage))
        require_file(src_image_path, "Source image")
        require_file(tgt_image_path, "Target image")
        require_file(tgt_layer_path, "Target layer image")
        rows.append(
            {
                "stem": stem,
                "split": row["split"],
                "transition_key": row["transition_key"],
                "src_stage_idx": src_stage,
                "tgt_stage_idx": tgt_stage,
                "src_image_path": str(src_image_path),
                "tgt_image_path": str(tgt_image_path),
                "tgt_layer_path": str(tgt_layer_path),
            }
        )

    if not rows:
        raise RuntimeError("No structural transition rows loaded.")
    return rows


def split_rows(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row["split"])
        if split in out:
            out[split].append(row)
    for split_name, split_rows_ in out.items():
        if not split_rows_:
            raise RuntimeError(f"No rows found for split: {split_name}")
    return out


class StructuralLayerDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], image_size: int):
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        with Image.open(str(row["src_image_path"])) as image:
            src_image = image_to_tensor(image, self.image_size)
        with Image.open(str(row["tgt_image_path"])) as image:
            tgt_image = image_to_tensor(image, self.image_size)
        with Image.open(str(row["tgt_layer_path"])) as image:
            layer = mask_to_tensor(image, self.image_size)
        layer = (layer > 0.05).to(dtype=torch.float32)

        return {
            "src_image": src_image,
            "tgt_image": tgt_image,
            "layer": layer,
            "src_stage_idx": torch.tensor(int(row["src_stage_idx"]), dtype=torch.long),
            "tgt_stage_idx": torch.tensor(int(row["tgt_stage_idx"]), dtype=torch.long),
            "transition_key": str(row["transition_key"]),
        }


def compute_loss(
    logits: torch.Tensor,
    src_image: torch.Tensor,
    target_image: torch.Tensor,
    target_layer: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    focal = focal_bce_loss_from_logits(
        logits,
        target_layer,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
    )
    dice = dice_loss_from_logits(logits, target_layer)
    probs = torch.sigmoid(logits)
    composed = compose_white_layer(src_image, probs)
    compose_l1 = F.l1_loss(composed, target_image)
    edge = F.l1_loss(sobel_edges(composed), sobel_edges(target_image))
    total = (
        float(args.focal_weight) * focal
        + float(args.dice_weight) * dice
        + float(args.compose_weight) * compose_l1
        + float(args.edge_weight) * edge
    )
    return total, {
        "loss": float(total.detach().cpu().item()),
        "focal": float(focal.detach().cpu().item()),
        "dice": float(dice.detach().cpu().item()),
        "compose_l1": float(compose_l1.detach().cpu().item()),
        "edge": float(edge.detach().cpu().item()),
    }


def mean_metrics(values: List[Dict[str, float]]) -> Dict[str, float]:
    if not values:
        return {"loss": float("nan"), "focal": float("nan"), "dice": float("nan"), "compose_l1": float("nan"), "edge": float("nan")}
    keys = values[0].keys()
    return {key: float(np.mean([item[key] for item in values])) for key in keys}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    metrics: List[Dict[str, float]] = []
    for batch in loader:
        src_image = batch["src_image"].to(device)
        tgt_image = batch["tgt_image"].to(device)
        layer = batch["layer"].to(device)
        src_stage = batch["src_stage_idx"].to(device)

        logits = model(src_image, src_stage)
        loss, batch_metrics = compute_loss(logits, src_image, tgt_image, layer, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.append(batch_metrics)
    return mean_metrics(metrics)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()
    metrics: List[Dict[str, float]] = []
    by_transition: Dict[str, List[Dict[str, float]]] = {}

    with torch.no_grad():
        for batch in loader:
            src_image = batch["src_image"].to(device)
            tgt_image = batch["tgt_image"].to(device)
            layer = batch["layer"].to(device)
            src_stage = batch["src_stage_idx"].to(device)
            tgt_stage = batch["tgt_stage_idx"].to(device)

            logits = model(src_image, src_stage)
            _, batch_metrics = compute_loss(logits, src_image, tgt_image, layer, args)
            metrics.append(batch_metrics)

            probs = torch.sigmoid(logits)
            pred_binary = (probs >= float(args.threshold)).to(dtype=torch.float32)
            intersection = torch.sum(pred_binary * layer, dim=(1, 2, 3))
            union = torch.sum(pred_binary + layer, dim=(1, 2, 3)) - intersection
            iou = (intersection / torch.clamp(union, min=1.0)).detach().cpu().numpy()
            src_np = src_stage.detach().cpu().numpy()
            tgt_np = tgt_stage.detach().cpu().numpy()
            for idx, iou_value in enumerate(iou):
                key = f"{int(src_np[idx])}_to_{int(tgt_np[idx])}"
                by_transition.setdefault(key, []).append({"iou": float(iou_value)})

    transition_metrics = {key: mean_metrics(values) for key, values in by_transition.items()}
    return mean_metrics(metrics), transition_metrics


def save_sample_sheet(
    model: nn.Module,
    dataset: StructuralLayerDataset,
    device: torch.device,
    output_path: Path,
    threshold: float,
    max_samples: int = 6,
) -> None:
    model.eval()
    count = min(max_samples, len(dataset))
    if count == 0:
        return

    tile_size = 160
    label_height = 22
    sheet = Image.new("RGB", (tile_size * 5, count * (tile_size + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    with torch.no_grad():
        for row_idx in range(count):
            sample = dataset[row_idx]
            src_image = sample["src_image"].unsqueeze(0).to(device)
            src_stage = sample["src_stage_idx"].view(1).to(device)
            logits = model(src_image, src_stage)
            probs = torch.sigmoid(logits)
            pred_layer = (probs >= threshold).to(dtype=torch.float32)
            composed = compose_white_layer(src_image, pred_layer)[0]

            target_layer_rgb = sample["layer"].repeat(3, 1, 1)
            pred_layer_rgb = pred_layer[0].repeat(3, 1, 1)
            images = [
                tensor_to_pil(sample["src_image"]),
                tensor_to_pil(pred_layer_rgb),
                tensor_to_pil(composed),
                tensor_to_pil(sample["tgt_image"]),
                tensor_to_pil(target_layer_rgb),
            ]
            labels = ["source", "pred layer", "composed", "target", "target layer"]
            y = row_idx * (tile_size + label_height)
            for col_idx, (image, label) in enumerate(zip(images, labels)):
                image = image.resize((tile_size, tile_size), Image.BICUBIC)
                x = col_idx * tile_size
                sheet.paste(image, (x, y))
                draw.text((x + 6, y + tile_size + 4), label, fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(Path(args.output_dir))
    device = torch.device(args.device)

    rows = load_rows(Path(args.transitions_csv), Path(args.data_root), args.min_src_stage, args.max_src_stage)
    splits = split_rows(rows)
    datasets = {name: StructuralLayerDataset(split_rows_, args.image_size) for name, split_rows_ in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    max_stage_idx = max(int(row["tgt_stage_idx"]) for row in rows)
    model = StructuralLayerUNet(
        num_stages=max_stage_idx + 1,
        base_channels=args.base_channels,
        stage_embed_dim=args.stage_embed_dim,
        cond_dim=args.cond_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, args)
        val_metrics, _ = evaluate(model, loaders["val"], device, args)
        epoch_summary = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_summary)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_compose_l1={val_metrics['compose_l1']:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid structural checkpoint.")

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "num_stages": max_stage_idx + 1,
            "base_channels": args.base_channels,
            "stage_embed_dim": args.stage_embed_dim,
            "cond_dim": args.cond_dim,
            "image_size": args.image_size,
            "min_src_stage": args.min_src_stage,
            "max_src_stage": args.max_src_stage,
            "threshold": args.threshold,
        },
    }
    torch.save(checkpoint, output_dir / "best_structural_decoder.pt")

    train_metrics, train_by_transition = evaluate(model, loaders["train"], device, args)
    val_metrics, val_by_transition = evaluate(model, loaders["val"], device, args)
    test_metrics, test_by_transition = evaluate(model, loaders["test"], device, args)
    save_sample_sheet(model, datasets["val"], device, output_dir / "val_samples.png", args.threshold)

    summary = {
        "transitions_csv": args.transitions_csv,
        "data_root": args.data_root,
        "device": str(device),
        "model_config": checkpoint["model_config"],
        "loss_weights": {
            "focal": args.focal_weight,
            "dice": args.dice_weight,
            "compose": args.compose_weight,
            "edge": args.edge_weight,
            "pos_weight": args.pos_weight,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
        },
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "best_val_loss": best_val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_by_transition": train_by_transition,
        "val_by_transition": val_by_transition,
        "test_by_transition": test_by_transition,
        "history": history,
    }
    write_json(output_dir / "metrics.json", summary)

    print("Structural layer decoder training complete")
    print(f"Train / Val / Test: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"Best val loss:      {best_val_loss:.4f}")
    print(f"Test loss:          {test_metrics['loss']:.4f}")
    print(f"Output dir:         {output_dir}")


if __name__ == "__main__":
    main()
