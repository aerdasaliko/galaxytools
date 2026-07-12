#!/bin/bash

set -e  # exit on error

# Inputs
TRAIN_DATASET="$1"
DEV_DATASET="$2"
TEST_DATASET="$3"

MODEL_MAX_LENGTH="$4"
BATCH_TRAIN="$5"
BATCH_EVAL="$6"
GRAD_ACC="$7"
LR="$8"
EPOCHS="$9"
SAVE_STEPS="${10}"
EVAL_STEPS="${11}"
WARMUP_STEPS="${12}"
LOGGING_STEPS="${13}"

PROBLEM_TYPE="${14}"

USE_LORA="${15}"

# LoRA params (optional)
LORA_R="${16:-8}"
LORA_ALPHA="${17:-16}"
LORA_DROPOUT="${18:-0.1}"

OUTPUT_DIR="results/model"
mkdir -p "$OUTPUT_DIR"

# Base training
CMD=(
python train.py # some combination of train-peft and train-peft-regression
  --model_name_or_path "path-to-model"
  --train_data "${TRAIN_DATASET}"
  --dev_data "${DEV_DATASET}"
  --test_data "${TEST_DATASET}"
  --kmer -1 # expose this as well?
  --model_max_length "${MODEL_MAX_LENGTH}"
  --per_device_train_batch_size "${BATCH_TRAIN}"
  --per_device_eval_batch_size "${BATCH_EVAL}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --learning_rate "${LR}"
  --num_train_epochs "${EPOCHS}"
  --fp16
  --save_steps "${SAVE_STEPS}"
  --save_strategy steps
  --output_dir "${OUTPUT_DIR}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --log_level info
  --find_unused_parameters False
  --problem_type "${PROBLEM_TYPE}"
)

# LoRA optional
if [ "${USE_LORA}" = "true" ]; then
    CMD+=(
        --use_lora
        --lora_r "${LORA_R}"
        --lora_alpha "${LORA_ALPHA}"
        --lora_dropout "${LORA_DROPOUT}"
    )
fi

# Run training
"${CMD[@]}"