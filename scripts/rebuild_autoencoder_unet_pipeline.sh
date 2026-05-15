#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_CMD="${PYTHON_CMD:-python}"
RAW_IMAGES_DIR="${RAW_IMAGES_DIR:-data/images}"
DATA_DIR="${DATA_DIR:-data/owl_output}"
LEARNING_DIR="${LEARNING_DIR:-${DATA_DIR}/learning}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CHANNELS="${CHANNELS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS_AUTOENCODER="${EPOCHS_AUTOENCODER:-100}"
EPOCHS_TRANSITION="${EPOCHS_TRANSITION:-100}"
LR="${LR:-1e-3}"
LATENT_CHANNELS="${LATENT_CHANNELS:-64}"
LATENT_NOISE_STD="${LATENT_NOISE_STD:-0.01}"
LAMBDA_IMAGE="${LAMBDA_IMAGE:-2.0}"
TRANSITION_BASE_CHANNELS="${TRANSITION_BASE_CHANNELS:-64}"
TRANSITION_BOTTLENECK_BLOCKS="${TRANSITION_BOTTLENECK_BLOCKS:-2}"
STAGE_EMBED_DIM="${STAGE_EMBED_DIM:-16}"
FOREGROUND_THRESHOLD="${FOREGROUND_THRESHOLD:-0.05}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE_SEGMENTS="${OVERWRITE_SEGMENTS:-1}"
OVERWRITE_STAGES="${OVERWRITE_STAGES:-1}"

run_step() {
  local name="$1"
  shift
  echo
  echo "============================================================"
  echo "${name}"
  echo "============================================================"
  echo "$PYTHON_CMD $*"
  "$PYTHON_CMD" "$@"
}

SEGMENT_ARGS=(
  scripts/segment_owl_images.py
  --input-dir "$RAW_IMAGES_DIR"
  --output-root "$DATA_DIR"
)
if [[ "$OVERWRITE_SEGMENTS" == "1" ]]; then
  SEGMENT_ARGS+=(--overwrite)
fi

STAGE_ARGS=(
  data_pipeline.py
  --data-root "$DATA_DIR"
  --stages all
)
if [[ "$OVERWRITE_STAGES" == "1" ]]; then
  STAGE_ARGS+=(--overwrite)
fi

TRAIN_ARGS=(
  experiments/train_autoencoder_transition.py
  --data_dir "$DATA_DIR"
  --pairs_csv "$LEARNING_DIR/manifest_transitions.csv"
  --image_size "$IMAGE_SIZE"
  --channels "$CHANNELS"
  --batch_size "$BATCH_SIZE"
  --epochs_autoencoder "$EPOCHS_AUTOENCODER"
  --epochs_transition "$EPOCHS_TRANSITION"
  --lr "$LR"
  --latent_channels "$LATENT_CHANNELS"
  --latent_noise_std "$LATENT_NOISE_STD"
  --lambda_image "$LAMBDA_IMAGE"
  --foreground_threshold "$FOREGROUND_THRESHOLD"
  --transition_model unet
  --transition_base_channels "$TRANSITION_BASE_CHANNELS"
  --transition_bottleneck_blocks "$TRANSITION_BOTTLENECK_BLOCKS"
  --use_stage_conditioning
  --stage_embed_dim "$STAGE_EMBED_DIM"
  --device "$DEVICE"
  --output_dir "$LEARNING_DIR/autoencoder_transition_unet_stagecond"
)

run_step "Segment raw owl images" "${SEGMENT_ARGS[@]}"
run_step "Build staged owl drawings" "${STAGE_ARGS[@]}"
run_step "Build learning manifests" scripts/build_manifest.py --data-root "$DATA_DIR" --output-dir "$LEARNING_DIR"
run_step "Train autoencoder + conditioned latent U-Net" "${TRAIN_ARGS[@]}"

echo
echo "Pipeline rebuild complete."
