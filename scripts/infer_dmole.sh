#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPTS_ROOT="${PROMPTS_ROOT:-${PROJECT_ROOT}/data/test_prompts/item}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-${PROJECT_ROOT}/models/PixArt-XL-2-512x512}"
PROMPTS_FILE1="${PROMPTS_FILE1:-${PROMPTS_ROOT}/p0h1_dog_dog.txt}"
PROMPTS_FILE2="${PROMPTS_FILE2:-${PROMPTS_ROOT}/k5f2_dog_dog3.txt}"
PROMPTS_FILE3="${PROMPTS_FILE3:-${PROMPTS_ROOT}/s5g3_cat_cat2.txt}"
PROMPTS_FILE4="${PROMPTS_FILE4:-${PROMPTS_ROOT}/b9l1_sneaker_shiny_sneaker.txt}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/outputs/train/dmole_without_prior_3/items_sequential/run}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/outputs/inference/dmole_without_prior_3/items_sequential}"

declare -a TASKS=(
  "${PROMPTS_FILE1}|task1_p0h1_dog"
  "${PROMPTS_FILE2}|task2_k5f2_dog"
  "${PROMPTS_FILE3}|task3_s5g3_cat"
  "${PROMPTS_FILE4}|task4_b9l1_sneaker"
)

for TASK in "${TASKS[@]}"; do
  PROMPT_FILE="${TASK%%|*}"
  OUT_FOLDER="${TASK##*|}"
  OUTPUT_DIR="${RESULTS_ROOT}/${OUT_FOLDER}"

  mkdir -p "${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${PROJECT_ROOT}/inference_dmole.py" \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --adapter_dir="${ADAPTER_DIR}" \
    --prompts_file="${PROMPT_FILE}" \
    --output_dir="${OUTPUT_DIR}" \
    --num_validation_images=2 \
    --validation_length=20 \
    --seed=42 \
    --fp16
done
