# DrawTheDamnOwl Staged Dataset Pipeline

This repository includes a deterministic, modular Python pipeline that converts aligned owl image/mask pairs into progressive drawing stages.

## Inputs

Expected input folders under `data/owl_output`:

- `images_256/<stem>.png`
- `masks_256/<stem>_mask.png`

All samples are assumed to be pre-cropped and aligned to `256x256`.

## Outputs

The pipeline writes active stage folders and JSON annotations:

- `stage_00_base`
- `stage_01_outer_contour`
- `stage_02_facial_features`
- `stage_03_inner_contours`
- `stage_04_value_regions`
- `stage_05_fine_texture`
- `stage_06_color`
- `stage_07_background`
- `annotations`

For each stage and sample stem:

- Layer image: `<stem>_<layer_name>_layer.png`
- Cumulative image: `<stem>_stageXX.png`
- Annotation: `annotations/<stem>_stageXX.json`

## Stage Semantics

1. `base_ellipses` (Stage 00): head/body ellipse scaffold from mask split.
2. `add_outer_contour` (Stage 01): simplified owl silhouette contour.
3. `add_facial_features` (Stage 02): eyes and beak from upper-region heuristics.
4. `add_inner_contours` (Stage 03): medium-scale internal contour structure.
5. `add_value_regions` (Stage 04): coarse tonal zones from quantized grayscale.
6. `add_fine_texture` (Stage 05): high-frequency residual detail layer.
7. `add_color` (Stage 06): chroma transfer while preserving prior structure.
8. `add_background` (Stage 07): outside-mask background and final reconstruction.

## Modeling Note

Structural retrieval-friendly stages:

- `base_ellipses`
- `add_outer_contour`
- `add_facial_features`
- `add_inner_contours`

Dense/generative-oriented stages:

- `add_value_regions`
- `add_fine_texture`

## Main Files

- `data_pipeline.py`: top-level runner and all stage functions.
- `owl_pipeline_utils.py`: reusable helpers for loading, masking, composition, contour serialization, and JSON writing.

## Usage

Run all stages for all matched stems:

```bash
python data_pipeline.py --stages all --overwrite
```

Run a subset of stages:

```bash
python data_pipeline.py --stages 0,1,2,3,4 --overwrite
```

Run only one sample:

```bash
python data_pipeline.py --stages all --stem 2 --overwrite
```

Custom root (if your dataset root differs):

```bash
python data_pipeline.py --data-root data/owl_output --stages all --overwrite
```

## CLIP Baseline Setup

The project now includes a first implementation slice for stage learning:

- Build frame and transition manifests split by owl stem.
- Extract frozen CLIP embeddings for each stage frame and its per-stage layer.
- Run stage-level embedding diagnostics (clusterability and similarity).

Install dependencies:

```bash
pip install numpy torch transformers pillow matplotlib
```

If CLIP loading fails with a `torchvision::nms` or import error, install matching PyTorch and torchvision versions (the issue is usually version mismatch):

```bash
python -m pip install --upgrade torch torchvision
```

Build manifests:

```bash
python scripts/build_manifest.py --data-root data/owl_output --output-dir data/owl_output/learning
```

Extract embeddings for all splits:

```bash
python scripts/extract_clip_embeddings.py \
	--manifest-frames data/owl_output/learning/manifest_frames.csv \
	--output-dir data/owl_output/learning/embeddings \
	--split all
```

Run diagnostics:

```bash
python scripts/embedding_diagnostics.py \
	--embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz \
	--output-dir data/owl_output/learning/diagnostics
```

Train a first latent next-stage baseline:

```bash
python scripts/train_transition_baseline.py \
	--embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz \
	--transitions-csv data/owl_output/learning/manifest_transitions.csv \
	--output-dir data/owl_output/learning/ar_baseline
```

Run latent rollout inference from a rough sketch and retrieve nearest stage images:

```bash
python scripts/infer_transition_rollout.py data/owl_output/stage_00_base/owl_23_stage00.png --checkpoint data/owl_output/learning/ar_baseline/best_model.pt --embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz --manifest-frames data/owl_output/learning/manifest_frames.csv --output-dir data/owl_output/learning/inference
```

Train a structural layer decoder for sparse line-art stages `0->1` through `2->3`:

```bash
python scripts/train_structural_layer_decoder.py \
	--transitions-csv data/owl_output/learning/manifest_transitions.csv \
	--data-root data/owl_output \
	--output-dir data/owl_output/learning/structural_decoder \
	--device cuda \
	--base-channels 32 \
	--cond-dim 128 \
	--batch-size 8
```

Train a dense one-step pixel decoder for appearance stages `3->4` through `6->7`:

```bash
python scripts/train_pixel_decoder.py --transitions-csv data/owl_output/learning/manifest_transitions.csv --latent-checkpoint data/owl_output/learning/ar_baseline/best_model.pt --latent-metrics-json data/owl_output/learning/ar_baseline/metrics.json --output-dir data/owl_output/learning/pixel_decoder --device cpu --min-src-stage 3 --max-src-stage 6 --output-mode residual --change-weight 8.0 --foreground-weight 2.0 --delta-weight 5.0 --delta-change-weight 16.0 --conditioning-mode mix
```

The pixel decoder embedding archive must match the latent checkpoint embedding family. If you use the checked-in `data/owl_output/learning/ar_baseline` artifacts, the script now auto-resolves the matching archive from `metrics.json`.

Run pixel-space rollout with the trained decoder:

```bash
python scripts/infer_transition_rollout.py data/owl_output/stage_00_base/owl_23_stage00.png \
	--render-mode pixel \
	--checkpoint data/owl_output/learning/ar_baseline/best_model.pt \
	--embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz \
	--manifest-frames data/owl_output/learning/manifest_frames.csv \
	--output-dir data/owl_output/learning/inference
```

Primary artifacts:

- `data/owl_output/learning/manifest_frames.csv`
- `data/owl_output/learning/manifest_transitions.csv`
- `data/owl_output/learning/manifest_qa.json`
- `data/owl_output/learning/embeddings/clip_embeddings_all.npz`
- `data/owl_output/learning/diagnostics/embedding_diagnostics.json`

## Notes on Stage Independence

Each stage is implemented as a separate function and can run independently if required prerequisites exist.

- Stages `01-06` need previous grayscale cumulative output.
- Stage `07` needs stage `06` color cumulative output.

When prerequisites are missing, the script prints clear errors and continues with the next sample.


## Code to run DINO embedding and CLIP embedding + diagnostics

# 1. Re-extract CLIP embeddings on combined manifest
uv run --with torch --with transformers --with pillow python scripts/extract_clip_embeddings.py --manifest-frames data/combined_learning_reduced/manifest_frames.csv --output-dir data/combined_learning_reduced/embeddings --model-id openai/clip-vit-base-patch32 --batch-size 32

# 2. Re-extract DINO embeddings on combined manifest
uv run --with timm --with torchvision --with torch --with transformers python scripts/extract_dino_embeddings.py --manifest-frames data/combined_learning_reduced/manifest_frames.csv --output-dir data/combined_learning_reduced/embeddings --model-id facebook/dinov2-base --batch-size 32

# 3. Run diagnostics on DINO embeddings
uv run --with torch --with numpy --with matplotlib python scripts/embedding_diagnostics.py --embeddings-npz data/combined_learning_reduced/embeddings/dino_embeddings_all.npz --output-dir data/combined_learning_reduced/diagnostics_dino

# 4. Run diagnostics on CLIP embeddings
uv run --with torch --with numpy --with matplotlib python scripts/embedding_diagnostics.py --embeddings-npz data/combined_learning_reduced/embeddings/clip_embeddings_all.npz --output-dir data/combined_learning_reduced/diagnostics_clip

# 5. Generate UMAP/heatmaps and retrieval sheets
uv run --with pillow --with matplotlib --with scikit-learn --with numpy python scripts/compare_embeddings_visuals.py --clip data/combined_learning_reduced/embeddings/clip_embeddings_all.npz --dino data/combined_learning_reduced/embeddings/dino_embeddings_all.npz --out-dir data/combined_learning_reduced/visuals --num-seeds 6 --topk 5
