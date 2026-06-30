# Reference algorithm

Zero-dependency PyTorch implementation of the orthogonal growth operators
described in [the paper](https://arxiv.org/abs/2510.08008). This module
operates on plain `Dict[str, torch.Tensor]` state dicts so it can be read
in isolation and dropped into any training stack.

For the full Megatron-LM integration that produced the paper's
17B → 70B run, see [`../integration/megatron/`](../integration/megatron/).

## Files

| File | Purpose |
|------|---------|
| [`algorithm.py`](algorithm.py) | The two growth operators and their tensor utilities (Megatron-style state-dict layout). |
| [`demo.py`](demo.py) | Toy state dict (4 layers, 4 experts) exercising both operators. |
| [`hf_depth_growth.py`](hf_depth_growth.py) | HuggingFace-API variant: load any `AutoModelForCausalLM` MoE checkpoint, double its layers (stacking / interleaving), save with `save_pretrained`. |

## Quick start

```bash
pip install torch
python growth/demo.py
```

Expected output (abridged):

```
[source] keys=50, layers=4 (ids 0..3)
[after depth growth (interposition, k=2)] keys=98, layers=8 (ids 0..7)
[after depth growth (stack, k=2)]         keys=98, layers=8 (ids 0..7)
[after width growth (E=4 -> 8, ...)]      router.weight: shape=(8, 8)
```

To grow a public HuggingFace MoE checkpoint instead:

```bash
pip install torch transformers
python growth/hf_depth_growth.py \
    --model-name-or-path deepseek-ai/DeepSeek-V2-Lite \
    --output-dir ./hf_grow/DeepSeek-V2-Lite-interleaving \
    --method interleaving --torch-dtype bfloat16
```

The grown checkpoint can be reloaded with the standard `AutoModelForCausalLM.from_pretrained`
and either further pre-trained or evaluated directly (see [`../eval/`](../eval/)).

## API

### `depth_growth_state_dict(state_dict, *, method, growth_factor=2, ...)`

Grow the model along the depth axis. `method='interposition'` duplicates
each layer in place (paper Eq. 2), `method='stack'` repeats the whole
layer block (paper Eq. 1). With `growth_factor=2` and 4 source layers:

```
source:        l0 l1 l2 l3
interposition: l0 l0 l1 l1 l2 l2 l3 l3
stack:         l0 l1 l2 l3 l0 l1 l2 l3
```

`ignore_first` / `ignore_last` leave a number of edge layers un-grown
(the paper uses 2 on each side for the 3B and 17B runs).

### `moe_width_growth_state_dict(state_dict, *, num_experts, hidden_size, ...)`

Duplicate every expert (and the corresponding router rows / bias) so
that `E -> 2E`. The new experts are initialised by adding a small
Gaussian perturbation `N(0, (alpha * sigma_orig)^2)` (Eq. 4 in the
paper), where `alpha` defaults to `0.01`. After calling this you must
also double the router's `top_k` in your model config.

Supports both per-expert weight layouts (TE grouped GEMM, the default)
and the legacy fused grouped-GEMM tensors (`legacy_grouped_gemm=True`).

### Helpers

- `interleaved_cat(a, b, dim=0)` — alternate two tensors along a dim
  (`[a0, b0, a1, b1, ...]`), used when `interleaved=True` for width
  growth.
- `add_noise_to_tensor(t, std_scaling_factor=0.01)` — the
  symmetry-breaking perturbation.
- `identify_output_layer(key)` — true for attention/MLP/expert output
  projections, used by depth growth's optional `zerofy_output` mode.

## Mapping to the paper

| Paper | Code |
|-------|------|
| §3.1 Depth growth, Eq. 1 (stack) | `depth_growth_state_dict(..., method='stack')` |
| §3.1 Depth growth, Eq. 2 (interposition) | `depth_growth_state_dict(..., method='interposition')` |
| §3.1 "ignore first/last layers" | `ignore_first`, `ignore_last` |
| §3.2 Width growth | `moe_width_growth_state_dict(...)` |
| §3.2 Noisy expert duplication, Eq. 4 | `noise_alpha` (paper's α) |
| §3.2 Random-router baseline | `use_random_router=True` |

## Naming conventions assumed

The reference operators expect Megatron-style keys:

- Layer prefix: `decoder.layers.{i}.` (override with `layer_prefix=`).
- Router: `...mlp.router.weight` (shape `[E, H]`), optional
  `...mlp.router.expert_bias` (shape `[E]`).
- Experts (default, per-expert tensors used by TE grouped GEMM):
  `...mlp.experts.linear_fc1.weight{i}` (`[H, ffn*2]`) and
  `...mlp.experts.linear_fc2.weight{i}` (`[ffn, H]`).
- Experts (legacy fused, set `legacy_grouped_gemm=True`):
  `...mlp.experts.weight1` and `...mlp.experts.weight2`.

If your codebase uses different names, the easiest adaptation is to
rename in/out around the call rather than fork this module.
