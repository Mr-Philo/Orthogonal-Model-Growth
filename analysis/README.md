# Layer-wise norm analysis

Tool that reproduces the data behind **Figure 2** in the paper — the
characteristic per-layer weight-norm profile that converged MoE LLMs
develop. This profile is the structural signature we exploit when
choosing interposition over stack depth growth (§3.1).

## File

| File | Purpose |
|------|---------|
| [`layer_norm.py`](layer_norm.py) | Load any HuggingFace MoE checkpoint, compute per-expert + per-layer Frobenius norms (normalised by `sqrt(numel)`), save a CSV and a line plot of layer-average norm. |

## Quick start

```bash
pip install torch transformers pandas matplotlib seaborn

python analysis/layer_norm.py \
    --hf-modeling-path deepseek-ai/DeepSeek-V2-Lite \
    --save-path ./out/DeepSeek-V2-Lite
```

Outputs:

```
out/DeepSeek-V2-Lite/
├── layer_experts_norms.csv      # one row per (layer, expert)
├── average_norm_per_layer.png   # the layer-avg-norm line plot
└── average_norm_per_layer.pdf
```

If `layer_experts_norms.csv` already exists in `--save-path`, the
script reuses it (skipping the expensive model load) — handy when
iterating on the plot.

## Supported architectures

Three dispatchers are tried in order; the first that matches the
parameter naming wins:

| Dispatcher | Matches |
|------------|---------|
| `make_sigma_norm_df` | `model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight` — DeepSeek-V2/V3-Lite, Qwen3-MoE, most Sigma-style MoE models. |
| `make_mixtral_norm_df` | `model.layers.{i}.block_sparse_moe.experts.{j}.{w1,w2,w3}.weight` — Mixtral, etc. |
| `make_gpt_oss_norm_df` | `model.layers.{i}.mlp.experts.{gate_up_proj,down_proj}` (fused per-layer tensors) — OpenAI gpt-oss. |

For an unknown layout the script prints all parameter names and exits;
add a fourth `make_*_norm_df` accordingly.

## Models used in the paper

The four panels in `assets/method_norm.png` are produced by running the
script against:

- Our 3B and 17B MoE checkpoints (not public).
- `deepseek-ai/DeepSeek-V2` (and `-Lite` for the appendix).
- `Qwen/Qwen3-30B-A3B-Instruct-2507`.

Run the script on each, then assemble the resulting PNGs into a grid
with your favourite plotting tool.
