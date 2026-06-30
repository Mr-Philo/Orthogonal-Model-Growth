#!/bin/bash
# ============================================================
# Example sweep: evaluate several public MoE checkpoints on a
# fixed task set, fan-out across multiple GPUs.
#
# Variables you can override from the env:
#   CACHE_DIR   HuggingFace cache (default: ~/.cache/huggingface)
#   OUTPUT_DIR  Where eval JSONs go (default: ./results)
#   GPUS        Whitespace-separated GPU ids (default: "0 1 2 3 4 5 6 7")
#   MODELS      Whitespace-separated HF ids; quote each
#   USE_GROWTH  "True" to apply interleaving depth growth in-memory
# ============================================================

set -e

CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"

# Default model fleet — replace with whatever you want to sweep.
MODELS_DEFAULT=(
    "deepseek-ai/DeepSeek-V2-Lite"
    "Qwen/Qwen1.5-MoE-A2.7B"
    "Qwen/Qwen3-30B-A3B"
)
if [ -z "${MODELS}" ]; then
    MODELS=("${MODELS_DEFAULT[@]}")
else
    # shellcheck disable=SC2206
    MODELS=($MODELS)
fi

# task -> num_fewshot
declare -A FEWSHOT=(
    ["mmlu"]=5
    ["arc_challenge"]=0
    ["hellaswag"]=0
    ["boolq"]=0
    ["openbookqa"]=0
)

# shellcheck disable=SC2206
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})

GROWTH_FLAGS=""
if [ "${USE_GROWTH}" = "True" ]; then
    GROWTH_FLAGS="--use-model-growth --growth-method interleaving"
fi

i=0
for model in "${MODELS[@]}"; do
    for task in "${!FEWSHOT[@]}"; do
        gpu="${GPUS[$i]}"
        fewshot="${FEWSHOT[$task]}"
        echo "Launching: GPU=${gpu} MODEL=${model} TASK=${task} fewshot=${fewshot} growth=${USE_GROWTH:-False}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            python "$(dirname "$0")/plain_eval.py" \
                --model "${model}" \
                --task "${task}" \
                --num-fewshot "${fewshot}" \
                --cache-dir "${CACHE_DIR}" \
                --output-dir "${OUTPUT_DIR}" \
                ${GROWTH_FLAGS} &
        i=$(( (i + 1) % ${#GPUS[@]} ))
    done
done

wait
echo "All evaluations finished."
