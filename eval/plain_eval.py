"""Evaluate a HuggingFace causal-LM checkpoint with lm-evaluation-harness.

Supports two modes:

* **Plain eval** — load the checkpoint as-is.
* **In-memory depth growth** — pass ``--use-model-growth`` to apply
  ``stacking`` or ``interleaving`` depth growth (paper Eq. 1 / Eq. 2)
  on the loaded model *before* running the eval, without writing the
  grown checkpoint to disk. Handy for ablation sweeps.

For a *persistent* grown checkpoint, use
``growth/hf_depth_growth.py`` and point this script at the saved dir.

Example::

    CUDA_VISIBLE_DEVICES=0 python eval/plain_eval.py \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --task mmlu --num-fewshot 5 --batch-size 16 \\
        --cache-dir ~/.cache/huggingface

    # With in-memory interleaving growth:
    CUDA_VISIBLE_DEVICES=0 python eval/plain_eval.py \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --task arc_challenge,boolq,hellaswag,openbookqa,winogrande \\
        --batch-size 16 --cache-dir ~/.cache/huggingface \\
        --use-model-growth --growth-method interleaving

Tasks accept a comma-separated list (forwarded to lm-eval).
Output JSON is written under ``--output-dir/<sanitized-model-name>/``.

Requires:
    pip install lm-eval transformers accelerate
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="HF Hub id or local path of the model to evaluate.")
    p.add_argument("--task", required=True,
                   help="Comma-separated list of lm-eval tasks "
                        "(e.g. 'mmlu' or 'arc_challenge,boolq,hellaswag').")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-dir", default=None,
                   help="HF cache directory (also exported via HF_HOME / "
                        "TRANSFORMERS_CACHE / HUGGINGFACE_HUB_CACHE).")
    p.add_argument("--output-dir", default="results",
                   help="Root directory for evaluation outputs.")
    p.add_argument("--torch-dtype", choices=["float16", "float32", "bfloat16"],
                   default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--use-model-growth", action="store_true",
                   help="Apply in-memory depth growth before evaluation.")
    p.add_argument("--growth-method", choices=["interleaving", "stacking"],
                   default="interleaving",
                   help="Depth growth strategy (only used with "
                        "--use-model-growth).")
    p.add_argument("--growth-factor", type=int, default=2)
    p.add_argument("--layers-attr", default="model.layers",
                   help="Dotted path to the layer ModuleList on the model "
                        "(default works for most LLaMA/Qwen/DeepSeek-style "
                        "models; override for other architectures).")
    return p.parse_args()


def _resolve(obj, dotted: str):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _set_dotted(obj, dotted: str, value) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def apply_in_memory_growth(model, method: str, k: int, layers_attr: str) -> int:
    original = _resolve(model, layers_attr)
    if method == "interleaving":
        new_layers = []
        for layer in original:
            for _ in range(k):
                new_layers.append(copy.deepcopy(layer))
    elif method == "stacking":
        base = [copy.deepcopy(l) for l in original]
        new_layers = [copy.deepcopy(l) for _ in range(k) for l in base]
    else:
        raise ValueError(f"Unknown growth method: {method}")
    new_layers = torch.nn.ModuleList(new_layers)
    _set_dotted(model, layers_attr, new_layers)
    # Keep config in sync so any code that reads ``num_hidden_layers`` works.
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(new_layers)
    print(f"Applied {method} growth (k={k}): {len(original)} -> {len(new_layers)} layers")
    return len(new_layers)


def _json_safe(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main() -> None:
    args = parse_args()
    if args.cache_dir is not None:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.cache_dir, "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.cache_dir, "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.cache_dir, "hub"))
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU {gpu}] model={args.model} task={args.task}")

    out_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )

    if args.use_model_growth:
        apply_in_memory_growth(model, args.growth_method, args.growth_factor,
                               args.layers_attr)

    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=args.batch_size, device="cuda", max_gen_toks=512)

    tasks = args.task.split(",")
    results = evaluator.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=args.num_fewshot,
        confirm_run_unsafe_code=True,
    )

    task_str = "QAs" if len(tasks) > 1 else tasks[0]
    suffix = "ori" if not args.use_model_growth else f"with_growth_{args.growth_method}"
    out_path = os.path.join(out_dir, f"{task_str}_{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"[GPU {gpu}] saved: {out_path}")


if __name__ == "__main__":
    main()
