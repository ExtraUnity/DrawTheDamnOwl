#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_CMD="${PYTHON_CMD:-python}"
RAW_IMAGES_DIR="${RAW_IMAGES_DIR:-data/images}"
DATA_DIR="${DATA_DIR:-data/owl_output}"
LEARNING_DIR="${LEARNING_DIR:-${DATA_DIR}/learning}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-dino}"
DEVICE="${DEVICE:-cuda}"

CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
DINO_MODEL_ID="${DINO_MODEL_ID:-facebook/dinov2-base}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
STAGE_EMBED_DIM="${STAGE_EMBED_DIM:-16}"
DROPOUT="${DROPOUT:-0.1}"
COSINE_WEIGHT="${COSINE_WEIGHT:-1.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.25}"
TEMPERATURE="${TEMPERATURE:-0.05}"
SELECTION_METRIC="${SELECTION_METRIC:-blended}"
PATIENCE="${PATIENCE:-25}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
OVERWRITE_SEGMENTS="${OVERWRITE_SEGMENTS:-1}"
OVERWRITE_STAGES="${OVERWRITE_STAGES:-1}"
ROLLOUT_INPUT_IMAGE="${ROLLOUT_INPUT_IMAGE:-}"
ROLLOUT_START_STAGE="${ROLLOUT_START_STAGE:-0}"
ROLLOUT_END_STAGE="${ROLLOUT_END_STAGE:--1}"
RETRIEVAL_SPLIT="${RETRIEVAL_SPLIT:-train}"

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

case "$EMBEDDING_BACKEND" in
  clip)
    EMBEDDING_SCRIPT="scripts/extract_clip_embeddings.py"
    MODEL_ID="$CLIP_MODEL_ID"
    EMBEDDINGS_NPZ="${LEARNING_DIR}/embeddings/clip_embeddings_all.npz"
    DIAGNOSTICS_DIR="${LEARNING_DIR}/diagnostics_clip"
    BASELINE_DIR="${LEARNING_DIR}/ar_baseline_clip"
    ;;
  dino)
    EMBEDDING_SCRIPT="scripts/extract_dino_embeddings.py"
    MODEL_ID="$DINO_MODEL_ID"
    EMBEDDINGS_NPZ="${LEARNING_DIR}/embeddings/dino_embeddings_all.npz"
    DIAGNOSTICS_DIR="${LEARNING_DIR}/diagnostics_dino"
    BASELINE_DIR="${LEARNING_DIR}/ar_baseline_dino"
    ;;
  *)
    echo "Unsupported EMBEDDING_BACKEND: ${EMBEDDING_BACKEND}" >&2
    exit 1
    ;;
esac

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

EMBED_ARGS=(
  "$EMBEDDING_SCRIPT"
  --manifest-frames "$LEARNING_DIR/manifest_frames.csv"
  --output-dir "$LEARNING_DIR/embeddings"
  --model-id "$MODEL_ID"
  --batch-size "$EMBED_BATCH_SIZE"
  --device "$DEVICE"
)
if [[ "$EMBEDDING_BACKEND" == "clip" ]]; then
  EMBED_ARGS+=(--split all)
fi

TRAIN_ARGS=(
  scripts/train_transition_baseline.py
  --embeddings-npz "$EMBEDDINGS_NPZ"
  --transitions-csv "$LEARNING_DIR/manifest_transitions.csv"
  --manifest-frames "$LEARNING_DIR/manifest_frames.csv"
  --output-dir "$BASELINE_DIR"
  --device "$DEVICE"
  --batch-size "$TRAIN_BATCH_SIZE"
  --epochs "$TRAIN_EPOCHS"
  --lr "$TRAIN_LR"
  --hidden-dim "$HIDDEN_DIM"
  --stage-embed-dim "$STAGE_EMBED_DIM"
  --dropout "$DROPOUT"
  --model-id "$MODEL_ID"
  --cosine-weight "$COSINE_WEIGHT"
  --contrastive-weight "$CONTRASTIVE_WEIGHT"
  --temperature "$TEMPERATURE"
  --selection-metric "$SELECTION_METRIC"
  --patience "$PATIENCE"
  --embedding-backend "$EMBEDDING_BACKEND"
)

run_step "Segment raw owl images" "${SEGMENT_ARGS[@]}"
run_step "Build staged owl drawings" "${STAGE_ARGS[@]}"
run_step "Build learning manifests" scripts/build_manifest.py --data-root "$DATA_DIR" --output-dir "$LEARNING_DIR"
run_step "Extract ${EMBEDDING_BACKEND^^} embeddings" "${EMBED_ARGS[@]}"

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
  run_step "Run embedding diagnostics" scripts/embedding_diagnostics.py --embeddings-npz "$EMBEDDINGS_NPZ" --output-dir "$DIAGNOSTICS_DIR"
fi

run_step "Train MLP latent transition baseline" "${TRAIN_ARGS[@]}"

if [[ -n "$ROLLOUT_INPUT_IMAGE" ]]; then
  run_step "Run retrieval rollout inference" \
    scripts/infer_transition_rollout.py "$ROLLOUT_INPUT_IMAGE" \
    --checkpoint "$BASELINE_DIR/best_model.pt" \
    --metrics-json "$BASELINE_DIR/metrics.json" \
    --embeddings-npz "$EMBEDDINGS_NPZ" \
    --manifest-frames "$LEARNING_DIR/manifest_frames.csv" \
    --output-dir "$BASELINE_DIR/inference_retrieval" \
    --render-mode retrieval \
    --embedding-backend "$EMBEDDING_BACKEND" \
    --model-id "$MODEL_ID" \
    --device "$DEVICE" \
    --start-stage "$ROLLOUT_START_STAGE" \
    --end-stage "$ROLLOUT_END_STAGE" \
    --retrieval-split "$RETRIEVAL_SPLIT"
fi

echo
echo "Embedding retrieval pipeline rebuild complete."
