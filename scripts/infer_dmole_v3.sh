#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_SET_DIR="${PROMPT_SET_DIR:-${PROJECT_ROOT}/data/test_prompts/item}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_ROOT}/models/PixArt-XL-2-512x512}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROJECT_ROOT}/outputs/train/dmole_v3_residual_prototype_router/items_sequential}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/inference/dmole_v3_residual_prototype_router/items_sequential}"
BASE_PROMPT="${BASE_PROMPT:-A photo of a item}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.3625}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

resolve_adapter_run_dir() {
  if [[ -n "${ADAPTER_DIR:-}" ]]; then
    printf '%s\n' "${ADAPTER_DIR}"
    return 0
  fi

  local latest_run
  latest_run="$(find "${ADAPTER_ROOT}" -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1 || true)"
  if [[ -n "${latest_run}" ]]; then
    printf '%s\n' "${latest_run}"
    return 0
  fi

  if [[ -d "${ADAPTER_ROOT}/run" ]]; then
    printf '%s\n' "${ADAPTER_ROOT}/run"
    return 0
  fi

  echo "Could not locate a training run directory under ${ADAPTER_ROOT}" >&2
  exit 1
}

ADAPTER_RUN_DIR="$(resolve_adapter_run_dir)"

declare -a PROMPT_SPECS=(
  "${PROMPT_SET_DIR}/p0h1_dog_dog.txt|task01_p0h1_dog"
  "${PROMPT_SET_DIR}/k5f2_dog_dog3.txt|task02_k5f2_dog3"
  "${PROMPT_SET_DIR}/s5g3_cat_cat2.txt|task03_s5g3_cat2"
  "${PROMPT_SET_DIR}/b9l1_sneaker_shiny_sneaker.txt|task04_b9l1_shiny_sneaker"
)

for PROMPT_SPEC in "${PROMPT_SPECS[@]}"; do
  PROMPT_FILE="${PROMPT_SPEC%%|*}"
  TASK_NAME="${PROMPT_SPEC##*|}"
  OUTPUT_DIR="${INFER_OUTPUT_ROOT}/${TASK_NAME}"

  mkdir -p "${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${PROJECT_ROOT}/infer_dmole_v3.py" \
    --pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    --adapter_dir="${ADAPTER_RUN_DIR}" \
    --prompts_file="${PROMPT_FILE}" \
    --output_dir="${OUTPUT_DIR}" \
    --base_prompt="${BASE_PROMPT}" \
    --num_validation_images=2 \
    --validation_length=20 \
    --seed=42 \
    --router_threshold="${ROUTER_THRESHOLD}" \
    --fp16
done
