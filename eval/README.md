# Evaluation harness

Thin wrapper around [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
that lets you eval any HuggingFace causal-LM checkpoint, with an
optional in-memory depth-growth pass applied before evaluation. This
is the "grow → eval" closed loop we used for the §3.1 ablations on
public MoE checkpoints.

## Files

| File | Purpose |
|------|---------|
| [`plain_eval.py`](plain_eval.py) | Load a checkpoint, optionally grow it in memory, run `lm_eval.evaluator.simple_evaluate`. |
| [`run_eval.sh`](run_eval.sh) | Example multi-GPU sweep over a few public MoE checkpoints and a small task set. |

## Quick start

```bash
pip install torch transformers lm-eval accelerate

# Plain eval
CUDA_VISIBLE_DEVICES=0 python eval/plain_eval.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --task mmlu --num-fewshot 5 --batch-size 16

# In-memory depth growth (interleaving, k=2) before eval
CUDA_VISIBLE_DEVICES=0 python eval/plain_eval.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --task arc_challenge,boolq,hellaswag,openbookqa,winogrande \
    --batch-size 16 \
    --use-model-growth --growth-method interleaving

# Multi-GPU sweep
bash eval/run_eval.sh
USE_GROWTH=True bash eval/run_eval.sh   # same sweep, with in-memory growth
```

Results land under `--output-dir/<sanitized-model-name>/`.

## Notes

- `--layers-attr` defaults to `model.layers`, which works for most
  LLaMA / Qwen / DeepSeek-style models. Override for other
  architectures (e.g. `model.transformer.h` for GPT-2-style).
- For a *persistent* grown checkpoint (one you can resume training
  from), grow it offline with
  [`../growth/hf_depth_growth.py`](../growth/hf_depth_growth.py) and
  then point `--model` at the saved directory.
- The default `task` list in `run_eval.sh` is a small set of MMLU-style
  multiple-choice tasks; for the longer / harder tasks used in the
  paper (`gsm8k`, `hendrycks_math`, `humaneval`), bump
  `--batch-size` / fewshot per the lm-eval docs.
