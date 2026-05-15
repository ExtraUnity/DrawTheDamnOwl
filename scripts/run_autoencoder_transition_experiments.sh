#!/usr/bin/env bash
set -euo pipefail

# Autoencoder transition experiment sweep runner
# Run this from the repo root:
#   bash scripts/run_autoencoder_transition_experiments.sh
#
# Example latent U-Net + stage conditioning run:
#   python experiments/train_autoencoder_transition.py --data_dir data/owl_output --pairs_csv data/owl_output/learning/manifest_transitions.csv --image_size 256 --channels 1 --batch_size 16 --epochs_autoencoder 100 --epochs_transition 100 --lr 1e-3 --latent_channels 64 --latent_noise_std 0.01 --lambda_image 2.0 --transition_model unet --transition_base_channels 64 --transition_bottleneck_blocks 2 --use_stage_conditioning --stage_embed_dim 16 --device cuda --output_dir data/owl_output/learning/autoencoder_transition_unet_stagecond

PYTHON_CMD="${PYTHON_CMD:-python}"
DATA_DIR="data/owl_output"
PAIRS_CSV="data/owl_output/learning/manifest_transitions.csv"
OUTPUT_ROOT="data/owl_output/learning"
IMAGE_SIZE=256
CHANNELS=1
BATCH_SIZE=16
EPOCHS_AUTOENCODER=100
EPOCHS_TRANSITION=100
LR=1e-3
LATENT_NOISE_STD=0.05
FOREGROUND_THRESHOLD=0.05
TRANSITION_ONLY="${TRANSITION_ONLY:-0}"
AUTOENCODER_CHECKPOINT="${AUTOENCODER_CHECKPOINT:-}"

COMMON_ARGS=(
  --data_dir "$DATA_DIR"
  --pairs_csv "$PAIRS_CSV"
  --image_size "$IMAGE_SIZE"
  --channels "$CHANNELS"
  --batch_size "$BATCH_SIZE"
  --epochs_autoencoder "$EPOCHS_AUTOENCODER"
  --epochs_transition "$EPOCHS_TRANSITION"
  --lr "$LR"
  --latent_noise_std "$LATENT_NOISE_STD"
  --foreground_threshold "$FOREGROUND_THRESHOLD"
)

OPTIONAL_ARGS=()
if [[ "$TRANSITION_ONLY" == "1" ]]; then
  if [[ -z "$AUTOENCODER_CHECKPOINT" ]]; then
    echo
    echo "TRANSITION_ONLY is enabled but AUTOENCODER_CHECKPOINT is not set."
    exit 1
  fi
  OPTIONAL_ARGS=(
    --transition_only
    --autoencoder_checkpoint "$AUTOENCODER_CHECKPOINT"
  )
fi

run_experiment() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "Running ${label}"
  echo "============================================================"
  "$PYTHON_CMD" experiments/train_autoencoder_transition.py "${COMMON_ARGS[@]}" "${OPTIONAL_ARGS[@]}" "$@" || {
    echo
    echo "Experiment ${label} failed."
    exit 1
  }
}

run_experiment "baseline" \
  --latent_channels 64 \
  --lambda_image 1.0 \
  --output_dir "${OUTPUT_ROOT}/autoencoder_transition_baseline"

run_experiment "c128" \
  --latent_channels 128 \
  --lambda_image 1.0 \
  --output_dir "${OUTPUT_ROOT}/autoencoder_transition_c128"

run_experiment "latent U-Net" \
  --latent_channels 64 \
  --lambda_image 2.0 \
  --transition_model unet \
  --transition_base_channels 64 \
  --transition_bottleneck_blocks 2 \
  --output_dir "${OUTPUT_ROOT}/autoencoder_transition_unet"

run_experiment "latent U-Net + stage conditioning" \
  --latent_channels 64 \
  --lambda_image 2.0 \
  --transition_model unet \
  --transition_base_channels 64 \
  --transition_bottleneck_blocks 2 \
  --use_stage_conditioning \
  --stage_embed_dim 16 \
  --output_dir "${OUTPUT_ROOT}/autoencoder_transition_unet_stagecond"

echo
echo "All autoencoder transition experiments completed successfully."
