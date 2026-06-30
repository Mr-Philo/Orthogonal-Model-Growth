"""Reference implementations of orthogonal model growth for MoE LLMs.

Companion to the paper:
    "Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via
     Orthogonal Growth of Mixture-of-Experts" (ICML 2026, arXiv:2510.08008).

This module operates on plain ``Dict[str, torch.Tensor]`` state dicts and
depends only on ``torch``. It is intentionally framework-agnostic so the
algorithms can be read in isolation and ported into any training stack.
For the production Megatron-LM integration we used in the paper, see
``integration/megatron/Offline_checkpoint_growth.py``.

Two growth operators are provided:

* :func:`depth_growth_state_dict` -- interposition vs. stack depth growth
  (Eq. 1 and Eq. 2 in the paper, Section 3.1).
* :func:`moe_width_growth_state_dict` -- expert duplication with optional
  symmetry-breaking noise (Section 3.2).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch

Tensor = torch.Tensor
StateDict = Dict[str, Tensor]

_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.(.+)")
_TE_FC1_RE = re.compile(r"\.experts\.linear_fc1\.weight(\d+)")
_TE_FC2_RE = re.compile(r"\.experts\.linear_fc2\.weight(\d+)")


# ---------------------------------------------------------------------------
# Low-level tensor utilities
# ---------------------------------------------------------------------------

def interleaved_cat(a: Tensor, b: Tensor, dim: int = 0) -> Tensor:
    """Interleave two tensors along ``dim``: ``[a0, b0, a1, b1, ...]``.

    Used by width growth when one wants the new experts to alternate with
    the original ones rather than be appended at the end.
    """
    if any(sa != sb for i, (sa, sb) in enumerate(zip(a.shape, b.shape)) if i != dim):
        raise ValueError(
            f"interleaved_cat: shapes must match except in dim={dim}, got {a.shape} vs {b.shape}"
        )
    stacked = torch.stack([a, b], dim=dim + 1)
    shape = list(a.shape)
    shape[dim] *= 2
    return stacked.reshape(*shape)


def add_noise_to_tensor(tensor: Tensor, std_scaling_factor: float = 0.01) -> Tensor:
    """Add Gaussian noise ``N(0, (alpha * sigma_orig)^2)`` to ``tensor``.

    This is the symmetry-breaking perturbation from Section 3.2 of the paper.
    ``std_scaling_factor`` corresponds to ``alpha`` in Eq. 4; the default
    ``0.01`` matches the value used in all paper experiments.
    """
    std = tensor.detach().float().std().item() * std_scaling_factor
    noise = torch.normal(mean=0.0, std=std, size=tensor.shape, device=tensor.device,
                         dtype=tensor.dtype)
    return tensor + noise


def identify_output_layer(key: str) -> bool:
    """Return True for output projections that should be zero-initialised
    when an "identity-like" residual is desired for newly copied layers."""
    if_attn_out = key.endswith("self_attention.linear_proj.weight") or \
                  key.endswith("self_attention.linear_proj.bias")
    if_mlp_out = key.endswith("mlp.linear_fc2.weight") or \
                 key.endswith("mlp.linear_fc2.bias")
    if_shared_moe_out = key.endswith("mlp.shared_experts.linear_fc2.weight") or \
                       key.endswith("mlp.shared_experts.linear_fc2.bias")
    if_legacy_grouped_out = key.endswith("mlp.experts.weight2") or \
                           key.endswith("mlp.experts.bias2")
    if_te_grouped_out = key.endswith("mlp.experts.linear_fc2.weight") or \
                       key.endswith("mlp.experts.linear_fc2.bias")
    return (if_attn_out or if_mlp_out or if_shared_moe_out or
            if_legacy_grouped_out or if_te_grouped_out)


# ---------------------------------------------------------------------------
# Depth growth
# ---------------------------------------------------------------------------

def _maybe_zerofy(key: str, tensor: Tensor, zerofy_output: bool) -> Tensor:
    if zerofy_output and identify_output_layer(key):
        return torch.zeros_like(tensor)
    return tensor


def depth_growth_state_dict(
    state_dict: StateDict,
    *,
    method: str = "interposition",
    growth_factor: int = 2,
    ignore_first: int = 0,
    ignore_last: int = 0,
    weight_multiplier: float = 1.0,
    zerofy_output: bool = False,
    layer_prefix: str = "decoder.layers.",
) -> StateDict:
    """Grow a transformer state dict along the depth axis.

    With ``method='interposition'`` (paper Eq. 2) each transformer layer is
    duplicated ``growth_factor`` times in place::

        [l1, l1, ..., l2, l2, ..., ln, ln, ...]

    With ``method='stack'`` (paper Eq. 1) the whole layer block is
    concatenated ``growth_factor`` times::

        [l1, ..., ln, l1, ..., ln, ...]

    The first ``ignore_first`` and last ``ignore_last`` source layers are
    inserted only once (i.e. not duplicated). This matches the policy used
    in the paper to leave embeddings-adjacent and head-adjacent layers
    untouched.

    Args:
        state_dict: source model weights; layer keys must match
            ``{layer_prefix}{idx}.{sub}``.
        method: ``"interposition"`` or ``"stack"``.
        growth_factor: how many times each layer is repeated.
            ``2`` is what the paper uses end-to-end.
        ignore_first / ignore_last: source-layer indices to leave un-grown.
        weight_multiplier: scalar applied to every copied tensor.
        zerofy_output: if True, output projections (attn out, mlp.fc2,
            expert.fc2) of the *newly copied* layers are set to zero,
            yielding an identity-like residual at growth time.

    Returns:
        A new state dict with ``num_layers`` set to the grown count.
    """
    if method not in {"interposition", "stack"}:
        raise ValueError(f"method must be 'interposition' or 'stack', got {method!r}")
    if growth_factor < 1:
        raise ValueError("growth_factor must be >= 1")

    layer_pattern = re.compile(re.escape(layer_prefix) + r"(\d+)\.(.+)")

    # 1. partition source state_dict into per-layer / non-layer
    per_layer: Dict[int, Dict[str, Tensor]] = defaultdict(dict)
    non_layer: StateDict = {}
    for k, v in state_dict.items():
        m = layer_pattern.match(k)
        if m:
            per_layer[int(m.group(1))][m.group(2)] = v
        else:
            non_layer[k] = v

    src_layers = sorted(per_layer.keys())
    n_src = len(src_layers)
    new_state: StateDict = dict(non_layer)
    new_idx = 0

    def _copy_layer(src_layer_idx: int, dst_layer_idx: int, *, zerofy: bool) -> None:
        for sub_key, tensor in per_layer[src_layer_idx].items():
            full_key = f"{layer_prefix}{dst_layer_idx}.{sub_key}"
            new_state[full_key] = _maybe_zerofy(full_key, tensor * weight_multiplier, zerofy)

    if method == "interposition":
        for i in src_layers:
            # always insert the source layer once (no zerofy on the original)
            _copy_layer(i, new_idx, zerofy=False)
            new_idx += 1
            # then insert (growth_factor - 1) duplicates, possibly zerofy'd
            if i < ignore_first or i >= n_src - ignore_last:
                continue
            for _ in range(growth_factor - 1):
                _copy_layer(i, new_idx, zerofy=zerofy_output)
                new_idx += 1
    else:  # "stack"
        # First pass: original block (skip the trailing tail per ignore_last
        # to preserve compatibility with the paper's behaviour).
        for i in src_layers:
            if i >= n_src - ignore_last:
                continue
            _copy_layer(i, new_idx, zerofy=False)
            new_idx += 1
        # Subsequent passes: duplicated blocks, possibly zerofy'd; skip the
        # head per ignore_first.
        for _ in range(growth_factor - 1):
            for i in src_layers:
                if i < ignore_first:
                    continue
                _copy_layer(i, new_idx, zerofy=zerofy_output)
                new_idx += 1

    return new_state


# ---------------------------------------------------------------------------
# Width growth (MoE)
# ---------------------------------------------------------------------------

def moe_width_growth_state_dict(
    state_dict: StateDict,
    *,
    num_experts: int,
    hidden_size: int,
    width_factor: int = 2,
    noise_alpha: float = 0.01,
    interleaved: bool = False,
    use_random_router: bool = False,
    zerofy_expert_bias: bool = False,
    weight_multiplier: float = 1.0,
    legacy_grouped_gemm: bool = False,
) -> StateDict:
    """Grow MoE width by duplicating experts and the router.

    Mirrors Section 3.2 of the paper. Currently only ``width_factor=2``
    is implemented (which is what the paper uses); the parameter is kept
    explicit so future extensions are obvious.

    Layout assumptions:

    * Router keys end with ``mlp.router.weight`` (shape ``[E, H]``) and
      optionally ``mlp.router.expert_bias`` (shape ``[E]``).
    * Per-expert weights follow either of two conventions, selectable
      with ``legacy_grouped_gemm``:

      - Legacy grouped GEMM (single fused tensor per layer):
        ``mlp.experts.weight1`` of shape ``[H, E * ffn_hidden * 2]`` and
        ``mlp.experts.weight2`` of shape ``[E * ffn_hidden, H]``.
      - Per-expert tensors (default):
        ``mlp.experts.linear_fc1.weight{i}`` and
        ``mlp.experts.linear_fc2.weight{i}`` for ``i in [0, E)``.

    Args:
        num_experts: source ``E``.
        hidden_size: model hidden dimension ``H`` (needed only for the
            legacy grouped GEMM reshape).
        width_factor: how many times each expert is replicated. Must be 2.
        noise_alpha: ``alpha`` in Eq. 4. ``0.0`` reproduces exact copies;
            ``0.01`` is the paper's recommendation.
        interleaved: if True, new experts are interleaved with the
            originals along the expert dimension; otherwise appended.
        use_random_router: if True, ignore ``noise_alpha`` for router
            weights and re-initialise router weights from
            ``N(0, 0.02^2)`` and bias from zero. Provided for the random-
            router baseline.
        zerofy_expert_bias: if True, zero out the entire expanded
            ``expert_bias`` (paper does this when ``expert_bias`` was a
            non-zero load-balancing term).
        weight_multiplier: scalar applied to every copied tensor.
        legacy_grouped_gemm: switch between the two weight layouts above.

    Returns:
        A new state dict with ``num_experts`` increased to
        ``num_experts * width_factor``. Callers must remember to also
        scale ``top_k -> top_k * width_factor``.
    """
    if width_factor != 2:
        raise NotImplementedError("Only width_factor=2 is implemented in this reference")
    new_state: StateDict = dict(state_dict)
    new_E = num_experts * width_factor

    for key, value in list(state_dict.items()):
        if "mlp.router" in key:
            base = value * weight_multiplier
            if use_random_router:
                if key.endswith("weight"):
                    new_state[key] = torch.normal(mean=0.0, std=0.02,
                                                  size=(new_E, value.shape[1]))
                else:  # expert_bias
                    new_state[key] = torch.zeros(new_E)
            else:
                second = base.clone()
                if noise_alpha > 0:
                    second = add_noise_to_tensor(second, std_scaling_factor=noise_alpha)
                new_state[key] = (interleaved_cat(base, second, dim=0)
                                  if interleaved else torch.cat([base, second], dim=0))
            if zerofy_expert_bias and key.endswith("expert_bias"):
                new_state[key] = torch.zeros(new_E)

        elif "mlp.experts" in key:
            if legacy_grouped_gemm:
                base = value * weight_multiplier
                second = base.clone()
                if noise_alpha > 0:
                    second = add_noise_to_tensor(second, std_scaling_factor=noise_alpha)
                if key.endswith("weight1"):
                    a = base.view(num_experts, hidden_size, -1)
                    b = second.view(num_experts, hidden_size, -1)
                    merged = (interleaved_cat(a, b, dim=0)
                              if interleaved else torch.cat([a, b], dim=0))
                    new_state[key] = merged.view(hidden_size, -1)
                elif key.endswith("weight2"):
                    a = base.view(num_experts, -1, hidden_size)
                    b = second.view(num_experts, -1, hidden_size)
                    merged = (interleaved_cat(a, b, dim=0)
                              if interleaved else torch.cat([a, b], dim=0))
                    new_state[key] = merged.view(-1, hidden_size)
                else:
                    new_state[key] = base  # bias / other tensors: passthrough
            else:
                m1 = _TE_FC1_RE.search(key)
                m2 = _TE_FC2_RE.search(key)
                if m1 is None and m2 is None:
                    new_state[key] = value * weight_multiplier
                    continue
                if m1 is not None:
                    expert_id = int(m1.group(1))
                    new_key = _TE_FC1_RE.sub(
                        f".experts.linear_fc1.weight{expert_id + num_experts}", key)
                else:
                    expert_id = int(m2.group(1))
                    new_key = _TE_FC2_RE.sub(
                        f".experts.linear_fc2.weight{expert_id + num_experts}", key)
                base = value * weight_multiplier
                second = base.clone()
                if noise_alpha > 0:
                    second = add_noise_to_tensor(second, std_scaling_factor=noise_alpha)
                new_state[key] = base
                new_state[new_key] = second
        else:
            new_state[key] = value * weight_multiplier

    return new_state


__all__ = [
    "interleaved_cat",
    "add_noise_to_tensor",
    "identify_output_layer",
    "depth_growth_state_dict",
    "moe_width_growth_state_dict",
]
