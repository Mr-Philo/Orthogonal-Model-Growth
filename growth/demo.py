"""Minimal runnable demo for ``growth.algorithm``.

Builds a tiny toy MoE state dict (4 layers, 4 experts, ``H=8``,
``ffn=16``) and runs both growth operators on it. No CUDA needed.

Run::

    python growth/demo.py
"""

from __future__ import annotations

import torch

from algorithm import (
    depth_growth_state_dict,
    moe_width_growth_state_dict,
)


def make_toy_state_dict(n_layers: int = 4, num_experts: int = 4,
                        hidden: int = 8, ffn: int = 16) -> dict:
    """Toy state dict with per-expert TE-style weights."""
    torch.manual_seed(0)
    sd = {
        "embedding.word_embeddings.weight": torch.randn(100, hidden),
        "output_layer.weight": torch.randn(100, hidden),
    }
    for li in range(n_layers):
        p = f"decoder.layers.{li}"
        sd[f"{p}.self_attention.linear_qkv.weight"] = torch.randn(3 * hidden, hidden)
        sd[f"{p}.self_attention.linear_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"{p}.mlp.router.weight"] = torch.randn(num_experts, hidden)
        sd[f"{p}.mlp.router.expert_bias"] = torch.zeros(num_experts)
        for ei in range(num_experts):
            sd[f"{p}.mlp.experts.linear_fc1.weight{ei}"] = torch.randn(hidden, ffn * 2)
            sd[f"{p}.mlp.experts.linear_fc2.weight{ei}"] = torch.randn(ffn, hidden)
    return sd


def summarise(name: str, sd: dict) -> None:
    layer_keys = [k for k in sd if k.startswith("decoder.layers.")]
    layer_ids = {int(k.split(".")[2]) for k in layer_keys}
    print(f"\n[{name}] keys={len(sd)}, layers={len(layer_ids)} "
          f"(ids 0..{max(layer_ids)})")
    sample = "decoder.layers.0.mlp.router.weight"
    if sample in sd:
        print(f"  {sample}: shape={tuple(sd[sample].shape)}")
    sample_fc1_keys = [k for k in sd if k.startswith("decoder.layers.0.mlp.experts.linear_fc1.weight")]
    print(f"  per-layer expert fc1 keys at layer 0: {len(sample_fc1_keys)}")


def main() -> None:
    sd = make_toy_state_dict()
    summarise("source", sd)

    grown_depth = depth_growth_state_dict(
        sd, method="interposition", growth_factor=2,
        ignore_first=0, ignore_last=0,
    )
    summarise("after depth growth (interposition, k=2)", grown_depth)

    grown_stack = depth_growth_state_dict(
        sd, method="stack", growth_factor=2,
        ignore_first=0, ignore_last=0,
    )
    summarise("after depth growth (stack, k=2)", grown_stack)

    grown_width = moe_width_growth_state_dict(
        sd, num_experts=4, hidden_size=8,
        width_factor=2, noise_alpha=0.01,
    )
    summarise("after width growth (E=4 -> 8, noise alpha=0.01)", grown_width)

    print("\nOK: all growth operators ran end-to-end on the toy state dict.")


if __name__ == "__main__":
    main()
