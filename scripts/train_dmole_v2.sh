#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/item}"
MODEL_NAME="${MODEL_NAME:-${PROJECT_ROOT}/models/PixArt-XL-2-512x512}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/dmole_cross_modal_v2/items_sequential}"
BASE_PROMPT="${BASE_PROMPT:-A photo of a item}"
GPU_IDS="${GPU_IDS:-4,5}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"

INSTANCE_DIRS="${INSTANCE_DIRS:-${DATA_ROOT}/dog,${DATA_ROOT}/dog3,${DATA_ROOT}/cat2,${DATA_ROOT}/shiny_sneaker}"
INSTANCE_PROMPTS="${INSTANCE_PROMPTS:-A photo of p0h1 dog.,A photo of k5f2 dog.,A photo of s5g3 cat.,A photo of b9l1 sneaker.}"
CLASS_DIRS="${CLASS_DIRS:-${DATA_ROOT}/dog_prior_images,${DATA_ROOT}/dog_prior_images,${DATA_ROOT}/cat_prior_images,${DATA_ROOT}/sneaker_prior_images}"
CLASS_PROMPTS="${CLASS_PROMPTS:-A photo of a dog.,A photo of a dog.,A photo of a cat.,A photo of a sneaker.}"

mkdir -p "${OUTPUT_DIR}"

deepspeed --include "localhost:${GPU_IDS}" --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/main_v2.py" \
  --deepspeed "${PROJECT_ROOT}/ds_config/item.json" \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  --base_prompt="${BASE_PROMPT}" \
  --instance_data_dirs="${INSTANCE_DIRS}" \
  --instance_prompts="${INSTANCE_PROMPTS}" \
  --class_data_dirs="${CLASS_DIRS}" \
  --class_prompts="${CLASS_PROMPTS}" \
  --num_class_images=500 \
  --prior_loss_weight=0.02 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --lora_rank=16 \
  --use_dmo_le \
  --param_budget=28 \
  --zcp_sample_ratio=0.01 \
  --router_threshold=0.2 \
  --use_inter_modal_curriculum
