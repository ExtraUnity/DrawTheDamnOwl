# DrawTheDamnOwl Staged Dataset Pipeline

This repository includes a deterministic, modular Python pipeline that converts aligned owl image/mask pairs into progressive drawing stages.

## Inputs

Expected input folders under `data/owl_output`:

- `images_256/<stem>.png`
- `masks_256/<stem>_mask.png`

All samples are assumed to be pre-cropped and aligned to `256x256`.

## Outputs

The pipeline writes stage folders and JSON annotations:

- `stage_00_base`
- `stage_01_outer_contour`
- `stage_02_facial_features`
- `stage_03_part_boundaries`
- `stage_04_inner_contours`
- `stage_05_value_regions`
- `stage_06_feather_masses`
- `stage_07_fine_texture`
- `stage_08_color`
- `stage_09_background`
- `annotations`

For each stage and sample stem:

- Layer image: `<stem>_<layer_name>_layer.png`
- Cumulative image: `<stem>_stageXX.png`
- Annotation: `annotations/<stem>_stageXX.json`

## Stage Semantics

1. `base_ellipses` (Stage 00): head/body ellipse scaffold from mask split.
2. `add_outer_contour` (Stage 01): simplified owl silhouette contour.
3. `add_facial_features` (Stage 02): eyes and beak from upper-region heuristics.
4. `add_part_boundaries` (Stage 03): coarse internal semantic boundaries.
5. `add_inner_contours` (Stage 04): medium-scale internal contour structure.
6. `add_value_regions` (Stage 05): coarse tonal zones from quantized grayscale.
7. `add_feather_masses` (Stage 06): grouped mid-scale texture masses.
8. `add_fine_texture` (Stage 07): high-frequency residual detail layer.
9. `add_color` (Stage 08): chroma transfer while preserving prior structure.
10. `add_background` (Stage 09): outside-mask background and final reconstruction.

## Modeling Note

Structural retrieval-friendly stages:

- `base_ellipses`
- `add_outer_contour`
- `add_facial_features`
- `add_part_boundaries`
- `add_inner_contours`

Dense/generative-oriented stages:

- `add_value_regions`
- `add_feather_masses`
- `add_fine_texture`
- `add_color`
- `add_background`

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
- Extract frozen CLIP embeddings for each stage frame.
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

Train a one-step pixel decoder conditioned on the current stage image, predicted next-stage latent, and stage index:

```bash
python scripts/train_pixel_decoder.py \
	--embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz \
	--transitions-csv data/owl_output/learning/manifest_transitions.csv \
	--latent-checkpoint data/owl_output/learning/ar_baseline/best_model.pt \
	--latent-metrics-json data/owl_output/learning/ar_baseline/metrics.json \
	--output-dir data/owl_output/learning/pixel_decoder \
	--output-mode residual \
	--change-weight 8.0 \
	--foreground-weight 2.0
```

Run pixel-space rollout with the trained decoder:

```bash
python scripts/infer_transition_rollout.py data/owl_output/stage_00_base/owl_23_stage00.png \
	--render-mode pixel \
	--checkpoint data/owl_output/learning/ar_baseline/best_model.pt \
	--pixel-decoder-checkpoint data/owl_output/learning/pixel_decoder/best_pixel_decoder.pt \
	--embeddings-npz data/owl_output/learning/embeddings/clip_embeddings_all.npz \
	--manifest-frames data/owl_output/learning/manifest_frames.csv \
	--output-dir data/owl_output/learning/inference_pixel \
	--use-retrieved-embedding \
	--save-retrieval-fallback
```

Primary artifacts:

- `data/owl_output/learning/manifest_frames.csv`
- `data/owl_output/learning/manifest_transitions.csv`
- `data/owl_output/learning/manifest_qa.json`
- `data/owl_output/learning/embeddings/clip_embeddings_all.npz`
- `data/owl_output/learning/diagnostics/embedding_diagnostics.json`

## Notes on Stage Independence

Each stage is implemented as a separate function and can run independently if required prerequisites exist.

- Stages `01-08` need previous grayscale cumulative output.
- Stage `09` needs stage `08` color cumulative output.

When prerequisites are missing, the script prints clear errors and continues with the next sample.
