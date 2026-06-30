"""HuggingFace-side depth growth for off-the-shelf MoE checkpoints.

Companion to the paper: this is a HuggingFace-API variant of the
operator in [`growth/algorithm.py`](algorithm.py). It loads any
`AutoModelForCausalLM` checkpoint, doubles its transformer blocks
(``stacking`` or ``interleaving``, paper Eq. 1 vs Eq. 2), updates
``config.num_hidden_layers``, and saves the grown model with
``save_pretrained``.

Useful for:

* Reproducing the §3.1 result on public MoE checkpoints
  (DeepSeek-V2-Lite, Qwen3-30B-A3B, Mixtral, ...) without going
  through Megatron.
* Pre-staging a grown checkpoint that can then be fed to standard HF
  training stacks (``trl``, ``transformers.Trainer``, ...).

Example::

    python growth/hf_depth_growth.py \\
        --model-name-or-path deepseek-ai/DeepSeek-V2-Lite \\
        --output-dir ./hf_grow/DeepSeek-V2-Lite-interleaving \\
        --method interleaving \\
        --torch-dtype bfloat16

Caveats
-------
* The layer ModuleList is found heuristically (``find_layer_module_list``).
  If your model nests the layers under an unusual attribute, pass
  ``--layers-attr <dotted.path>`` to override.
* Each layer is ``copy.deepcopy``'d before insertion so weights are
  not shared (sharing breaks ``safe_serialization`` and confuses the
  optimizer when resumed).
* For models whose config field is not in the standard list
  (``num_hidden_layers``, ``n_layers``, ``num_layers``, ...) you'll see
  a warning; edit ``config.json`` manually if so.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Common config attribute names for "number of transformer layers".
LAYER_COUNT_FIELDS = (
    "num_hidden_layers", "n_layers", "num_layers", "decoder_layers",
    "encoder_layers", "num_blocks", "n_layer", "num_transformer_layers",
)


def resolve_layers_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """Resolve ``model.<dotted>`` to ``(parent_module, attr_name)`` so the
    caller can ``setattr(parent, attr_name, new_list)`` later."""
    parts = dotted.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def find_layer_module_list(model: nn.Module, max_depth: int = 5
                           ) -> Tuple[nn.Module, str, nn.ModuleList]:
    """Heuristically locate the main transformer-layer ModuleList.

    Returns ``(parent_module, attr_name, module_list)`` so the list can
    be replaced with ``setattr(parent_module, attr_name, new_list)``.
    """
    best: Optional[Tuple[nn.Module, str, nn.ModuleList, int, int]] = None  # depth, length

    def _visit(module: nn.Module, path: str, depth: int) -> None:
        nonlocal best
        if depth > max_depth:
            return
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                hint = name.lower()
                type_hint = type(child[0]).__name__.lower()
                if ("layer" in hint or "block" in hint or hint in {"h", "blk"}
                        or "block" in type_hint or "decoder" in type_hint
                        or "layer" in type_hint or "moe" in type_hint):
                    # Prefer the longest matching ModuleList (skip 1-element
                    # auxiliary lists like Mixtral's pad_token_module_list).
                    if best is None or len(child) > best[4]:
                        best = (module, name, child, depth, len(child))
            _visit(child, child_path, depth + 1)

    _visit(model, "", 0)
    if best is None:
        raise ValueError(
            "Could not auto-detect a transformer-layer ModuleList. "
            "Pass --layers-attr <dotted.path> to override."
        )
    parent, attr, mlist, depth, length = best
    logger.info(f"Detected layer ModuleList: depth={depth}, length={length}, "
                f"type={type(mlist[0]).__name__}")
    return parent, attr, mlist


def stacking_growth(layers: nn.ModuleList, k: int = 2) -> nn.ModuleList:
    """Paper Eq. 1: ``[l1,l2,...,ln, l1,l2,...,ln, ...]``."""
    base = [copy.deepcopy(l) for l in layers]
    return nn.ModuleList([copy.deepcopy(l) for _ in range(k) for l in base])


def interleaving_growth(layers: nn.ModuleList, k: int = 2) -> nn.ModuleList:
    """Paper Eq. 2 (a.k.a. interposition): ``[l1,l1,...,l1, l2,l2,...,l2, ...]``."""
    out = []
    for layer in layers:
        for _ in range(k):
            out.append(copy.deepcopy(layer))
    return nn.ModuleList(out)


def update_config_num_layers(config, new_num_layers: int) -> None:
    """Set whichever of ``num_hidden_layers`` / ``n_layers`` / ... the
    config exposes. Logs a warning if none is found."""
    for field in LAYER_COUNT_FIELDS:
        if hasattr(config, field):
            old = getattr(config, field)
            setattr(config, field, new_num_layers)
            logger.info(f"Updated config.{field}: {old} -> {new_num_layers}")
            return
    logger.warning("No standard layer-count field found on config; edit "
                   "config.json manually after saving if needed.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name-or-path", required=True,
                   help="HF Hub id or local path of the base model.")
    p.add_argument("--output-dir", required=True,
                   help="Where the grown model is written.")
    p.add_argument("--method", choices=["stacking", "interleaving"], required=True,
                   help="stacking = Eq. 1; interleaving = Eq. 2 (preferred for "
                        "converged checkpoints).")
    p.add_argument("--growth-factor", type=int, default=2,
                   help="How many copies of each layer (paper uses 2).")
    p.add_argument("--torch-dtype", choices=["float16", "float32", "bfloat16"],
                   default="bfloat16")
    p.add_argument("--device-map", default="auto",
                   help="HF device_map. Use 'cpu' for low-memory machines.")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache directory (also exported via HF_HOME / "
                        "TRANSFORMERS_CACHE / HUGGINGFACE_HUB_CACHE).")
    p.add_argument("--layers-attr", default=None,
                   help="Override the dotted attribute path to the layer "
                        "ModuleList (e.g. 'model.layers'). If unset, "
                        "the script auto-detects.")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the post-save reload sanity check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cache_dir is not None:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.cache_dir, "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.cache_dir, "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.cache_dir, "hub"))

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.torch_dtype]

    logger.info(f"Loading base model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Locate (and later replace) the transformer-layer ModuleList.
    if args.layers_attr:
        parent, attr = resolve_layers_attr(model, args.layers_attr)
        layers = getattr(parent, attr)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(f"{args.layers_attr} resolves to "
                            f"{type(layers).__name__}, expected nn.ModuleList")
    else:
        parent, attr, layers = find_layer_module_list(model)
    original_num_layers = len(layers)
    logger.info(f"Original layer count: {original_num_layers}")

    grow_fn = {"stacking": stacking_growth, "interleaving": interleaving_growth}[args.method]
    new_layers = grow_fn(layers, k=args.growth_factor)
    new_num_layers = len(new_layers)
    logger.info(f"After {args.method} (k={args.growth_factor}): {new_num_layers} layers")

    setattr(parent, attr, new_layers)
    update_config_num_layers(model.config, new_num_layers)

    logger.info(f"Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")

    if args.skip_verify:
        return
    logger.info("Reload sanity check...")
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output_dir, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )
    if args.layers_attr:
        rp, ra = resolve_layers_attr(reloaded, args.layers_attr)
        rlayers = getattr(rp, ra)
    else:
        _, _, rlayers = find_layer_module_list(reloaded)
    if len(rlayers) != new_num_layers:
        logger.error(f"Reload mismatch: expected {new_num_layers}, got {len(rlayers)}")
    else:
        logger.info(f"Reload OK: {len(rlayers)} layers.")


if __name__ == "__main__":
    main()
