#! /bin/bash

source /path/to/your/env

export EXP_PATH="/path/to/your/exps"
export EXP_NAME="octomed_7b_pmcvqa8k"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${EXP_PATH}/runs/${EXP_NAME}/log/debug_log_rollouts.$(date +%Y-%m-%d-%H-%M-%S).txt"
export CONFIDENCE_LOG_PATH="${EXP_PATH}/runs/${EXP_NAME}/log/debug_log_confidence.$(date +%Y-%m-%d-%H-%M-%S).txt"

mkdir -p ${EXP_PATH}/runs/${EXP_NAME}/log


QWEN_PATH="OctoMed/OctoMed-7B"
export IMAGE_ROOT="/path/to/image-root/"
HF_DATASET="/path/to/jsonl/pmc_vqa_8k.jsonl"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

cd ${REPO_HOME}/src/r1-v

export WANDB_DISABLED=true


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --use_vllm False \
    --output_dir ${EXP_PATH}/checkpoints/rl/${EXP_NAME} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --deepspeed ${REPO_HOME}/src/r1-v/configs/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0000005 \
    --bf16 true \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control false \
    --attn_implementation flash_attention_2 \
    --max_pixels 802816 \
    --num_train_epochs 1 \
    --run_name ${EXP_NAME} \
    --save_steps 200 \
    --save_only_model true \
    --beta 0 \
    --max_grad_norm 5 \
    --use_care true \
    --ref_ema_decay 0.995 \
    --ref_ema_update_every 10 \
    --bonus_coefficient 0.5 \
    --confidence_upper_bound 0.95 \
    --consistency_margin 0.01 \
    --num_generations 8 \
    --data_seed 42 \
    --report_to tensorboard \

