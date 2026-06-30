"""Compute per-layer weight-norm profiles for MoE LLMs.

Reproduces the data behind Figure 2 in the paper — the characteristic
layer-wise weight-norm pattern of converged MoE models. Saves a CSV
of per-expert norms and a PNG/PDF of the per-layer average norm vs.
layer index.

Supports three common MoE expert layouts (auto-dispatched by inspecting
the loaded model's parameter names):

* ``deepseek``-style: ``model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight``
  (DeepSeek-V2/V3, Qwen3-MoE, our 3B/17B Sigma-style models).
* ``mixtral``-style: ``model.layers.{i}.block_sparse_moe.experts.{j}.{w1,w2,w3}.weight``
  (Mixtral, etc.).
* ``gpt_oss``-style: fused per-layer ``mlp.experts.{gate_up_proj,down_proj}``
  tensors with the expert dim baked into shape (OpenAI gpt-oss).

If your model uses a different naming convention, add a small
``make_*_norm_df`` function and dispatch on it.

Example::

    python analysis/layer_norm.py \\
        --hf-modeling-path deepseek-ai/DeepSeek-V2-Lite \\
        --save-path ./out/DeepSeek-V2-Lite

The per-layer Frobenius norms are normalised by ``sqrt(numel)`` so they
are comparable across tensors of different shapes (see
``method.tex:52``).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    import seaborn as sns
except ImportError:  # seaborn is optional
    sns = None


def _normed_fro(param: torch.Tensor) -> float:
    """Frobenius norm divided by sqrt(numel) — comparable across shapes."""
    return torch.linalg.norm(param).item() / param.numel() ** 0.5


def make_sigma_norm_df(model) -> pd.DataFrame:
    """DeepSeek-V2/V3-Lite / Qwen3-MoE-style: per-expert (gate, up, down)."""
    agg = {}
    pat = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
    for name, param in model.named_parameters():
        m = pat.match(name)
        if not m:
            continue
        layer = int(m.group(1)); expert = int(m.group(2)); proj = m.group(3)
        key = (layer, expert)
        agg.setdefault(key, {"layer": layer, "expert": expert,
                             "gate_norm": None, "up_norm": None, "down_norm": None})
        agg[key][f"{proj}_norm"] = _normed_fro(param)
    if not agg:
        raise ValueError("no DeepSeek/Qwen-style expert weights matched")
    return _finalize_df(pd.DataFrame(list(agg.values())),
                        cols=["gate_norm", "up_norm", "down_norm"])


def make_mixtral_norm_df(model) -> pd.DataFrame:
    """Mixtral-style: per-expert (w1, w2, w3) under ``block_sparse_moe``."""
    agg = {}
    pat = re.compile(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight")
    for name, param in model.named_parameters():
        m = pat.match(name)
        if not m:
            continue
        layer = int(m.group(1)); expert = int(m.group(2)); proj = m.group(3)
        key = (layer, expert)
        agg.setdefault(key, {"layer": layer, "expert": expert,
                             "w1_norm": None, "w2_norm": None, "w3_norm": None})
        agg[key][f"{proj}_norm"] = _normed_fro(param)
    if not agg:
        raise ValueError("no Mixtral-style expert weights matched")
    return _finalize_df(pd.DataFrame(list(agg.values())),
                        cols=["w1_norm", "w2_norm", "w3_norm"])


def make_gpt_oss_norm_df(model) -> pd.DataFrame:
    """gpt-oss-style: per-layer fused ``mlp.experts.gate_up_proj`` and
    ``mlp.experts.down_proj`` tensors. The expert dim is collapsed into
    the param itself, so the resulting df has one row per layer."""
    agg = {}
    pat = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")
    for name, param in model.named_parameters():
        m = pat.match(name)
        if not m:
            continue
        layer = int(m.group(1)); proj = m.group(2)
        agg.setdefault(layer, {"layer": layer, "expert": -1,
                               "gate_up_norm": None, "down_norm": None})
        col = "gate_up_norm" if proj == "gate_up_proj" else "down_norm"
        agg[layer][col] = _normed_fro(param)
    if not agg:
        raise ValueError("no gpt-oss-style expert weights matched")
    df = pd.DataFrame(list(agg.values()))
    # gate_up fuses two projections; weight accordingly.
    df["average_norm"] = (2 * df["gate_up_norm"] + df["down_norm"]) / 3
    return _finalize_layer_avg(df)


def _finalize_df(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    df["average_norm"] = df[cols].mean(axis=1, skipna=True)
    return _finalize_layer_avg(df)


def _finalize_layer_avg(df: pd.DataFrame) -> pd.DataFrame:
    layer_sum = df.groupby("layer")["average_norm"].transform("sum")
    df["norm_distribution"] = df["average_norm"] / layer_sum
    layer_avg = (df.groupby("layer")["average_norm"].mean().reset_index()
                 .rename(columns={"average_norm": "layer_avg_norm"}))
    return pd.merge(df, layer_avg, on="layer", how="left")


DISPATCHERS: list[Callable] = [make_sigma_norm_df, make_mixtral_norm_df, make_gpt_oss_norm_df]


def build_df(model) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for fn in DISPATCHERS:
        try:
            return fn(model)
        except Exception as e:
            last_error = e
    print("None of the built-in dispatchers matched. Parameter names follow; "
          "add a new make_*_norm_df dispatcher for this architecture:")
    for name, p in model.named_parameters():
        print(f"  {name}: {tuple(p.shape)}")
    raise RuntimeError("no dispatcher matched this model") from last_error


def load_hf_model(model_path: str, cache_dir: Optional[str]):
    is_local = os.path.isdir(model_path)
    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                     cache_dir=None if is_local else cache_dir)
    load_kwargs["config"] = cfg
    if not is_local and cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    print(f"Loading model: {model_path}")
    return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


def plot_average_norm_per_layer(df: pd.DataFrame, save_path: str) -> None:
    data = (df[["layer", "layer_avg_norm"]]
            .drop_duplicates().sort_values("layer"))
    if data.empty:
        print("no layer_avg_norm data to plot")
        return
    plt.figure(figsize=(2.5, 2.5))
    if sns is not None:
        sns.lineplot(x="layer", y="layer_avg_norm", data=data,
                     marker="o", markersize=6, linestyle="-")
    else:
        plt.plot(data["layer"], data["layer_avg_norm"], marker="o",
                 markersize=6, linestyle="-")
    plt.title("Average norm across all layers", fontsize=6)
    plt.xlabel("Layer ID", fontsize=4)
    plt.ylabel("Average norm of all experts", fontsize=4)
    plt.tick_params(axis="both", which="major", labelsize=4)
    plt.grid(True, which="both", linestyle="--", linewidth=1)
    plt.tight_layout()
    png = os.path.join(save_path, "average_norm_per_layer.png")
    pdf = os.path.join(save_path, "average_norm_per_layer.pdf")
    plt.savefig(png)
    plt.savefig(pdf, dpi=300)
    print(f"Saved {png} and {pdf}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf-modeling-path", required=True,
                   help="HF Hub id or local path of the model.")
    p.add_argument("--save-path", required=True,
                   help="Directory for CSV + plot outputs.")
    p.add_argument("--hf-cache-dir", default=None,
                   help="HF cache directory; used only when loading from the "
                        "Hub.")
    p.add_argument("--save-df", action="store_true", default=True,
                   help="Save the per-expert dataframe as CSV (default on).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    csv_path = os.path.join(args.save_path, "layer_experts_norms.csv")

    if os.path.exists(csv_path):
        print(f"Loading cached df from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        model = load_hf_model(args.hf_modeling_path, args.hf_cache_dir)
        df = build_df(model)
        if args.save_df:
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

    plot_average_norm_per_layer(df, args.save_path)


if __name__ == "__main__":
    main()
