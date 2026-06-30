#!/bin/bash
# ================================================================================
# Model Growth Script
# ================================================================================
# 
# This script loads Megatron-LM MoE checkpoint and expand it to larger model
# Currently only support weight only (no optimizer state) conversion
#
# Optional args: <iteration> <iteration 2>
#   If provided, the script will load the checkpoint at that iteration.
#   If not provided, it will load the latest checkpoint.
#   If the second iteration is specified, it will use two checkpoints for merging.
#
# Usage:
#   export MEGATRON_PATH="/path/to/megatron-lm"
#   export PROJECT_PATH="/path/to/project"
#   bash ./Launch_offline_checkpoint_growth.sh <iteration> <iteration 2>
# ================================================================================

set -e

MEGATRON_PATH="${MEGATRON_PATH:-./Megatron-LM}"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Configuration paths
MODEL_NAME="${MODEL_NAME:-default_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-${MODEL_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoints/${CHECKPOINT_NAME}}"

ITERATION=$1
SECOND_ITERATION=$2
GROWTH_CHECKPOINT_NAME="${GROWTH_CHECKPOINT_NAME:-growth_model_${ITERATION:-latest}_${SECOND_ITERATION:-$ITERATION}}"
GROWTH_CHECKPOINT_PATH="${GROWTH_CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}}"

echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "GROWTH_CHECKPOINT_PATH: ${GROWTH_CHECKPOINT_PATH}"

# Model and tokenizer paths
PROJECT_PATH="${PROJECT_PATH:-./project}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_PATH}/tokenizer}"

# Load model configuration
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${PROJECT_PATH}/config}"
source ${MODEL_CONFIG_PATH}/model_config.sh
source ${MODEL_CONFIG_PATH}/training_config.sh

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port $(shuf -n 1 -i 10000-65535)
)

LOAD_AND_SAVE_ARGS=(
    --load ${CHECKPOINT_PATH}
    --save ${GROWTH_CHECKPOINT_PATH}
    --ckpt-format torch_dist
    --ckpt-convert-format torch_dist
    --ckpt-convert-save ${GROWTH_CHECKPOINT_PATH}
    --exit-on-missing-checkpoint
    --save-interval 1
    --no-initialization
    --use-cpu-initialization
    --no-load-optim
    --no-save-optim
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_PATH}
    # NOTE: only enable trust-remote-code for tokenizers from sources you trust;
    # it will download and execute arbitrary Python from the HF repo.
    --tokenizer-huggingface-trust-remote-code
)

GROWTH_ARGS=(
    --growth-stack-method interleaved
    --growth-weight-multiplier 1.0
    --growth-ignore-first-num-layers 2
    --growth-ignore-last-num-layers 2
)

# Control growth method using environment variables
if [ "${USE_DEPTH_GROWTH}" = "True" ]; then
    GROWTH_ARGS+=(--do-depth-growth)
fi
if [ "${USE_MOE_WIDTH_GROWTH}" = "True" ]; then
    GROWTH_ARGS+=(--do-moe-width-growth)
fi
if [ "${USE_RANDOM_ROUTER}" = "True" ]; then
    GROWTH_ARGS+=(--growth-use-random-router)
fi

# Add random noise to new experts to break symmetry
if [ "${GROWTH_ADD_EXPERT_NOISE}" = "True" ]; then
    GROWTH_ARGS+=(--growth-add-expert-noise)
    GROWTH_ARGS+=(--growth-expert-noise-std-scaling-factor ${GROWTH_EXPERT_NOISE_STD_SCALING_FACTOR:-0.01})
fi

# Handle checkpoint iterations
if [ ! -z "$SECOND_ITERATION" ]; then
    LOAD_AND_SAVE_ARGS+=(--ckpt-step "$ITERATION")
    GROWTH_ARGS+=(--use-ckpt-merge)
    GROWTH_ARGS+=(--second-ckpt-step "$SECOND_ITERATION")
    
    mkdir -p "${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}"
    echo -n "$SECOND_ITERATION" > "${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}/latest_checkpointed_iteration.txt"
    echo "Using SECOND_ITERATION: $SECOND_ITERATION"
elif [ ! -z "$ITERATION" ]; then
    LOAD_AND_SAVE_ARGS+=(--ckpt-step "$ITERATION")
    
    mkdir -p "${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}"
    echo -n "$ITERATION" > "${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}/latest_checkpointed_iteration.txt"
    echo "Using ITERATION: $ITERATION"
fi

# Execute the conversion and expansion
torchrun ${DISTRIBUTED_ARGS[@]} Offline_checkpoint_growth.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${IMPL_ARGS[@]} \
    ${LOAD_AND_SAVE_ARGS[@]} \
    ${GROWTH_ARGS[@]} 2>&1 | tee ${OUTPUT_DIR}/checkpoints/${GROWTH_CHECKPOINT_NAME}/growth_log.txt
