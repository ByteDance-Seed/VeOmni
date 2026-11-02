# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint preflight for dynamic RoPE's nonpersistent runtime state."""

import hashlib
import json
import math

import torch


def reject_dynamic_rope_multi_optimizer_training(model):
    """Reject modes whose transient cache history cannot yet be resumed."""
    for name, module in model.named_modules():
        rope_type = getattr(module, "rope_type", None)
        kinds = rope_type.values() if isinstance(rope_type, dict) else (rope_type,)
        if any(isinstance(kind, str) and ("dynamic" in kind or kind == "longrope") for kind in kinds):
            raise RuntimeError(
                f"MultiOptimizer DCP training cannot yet safely resume dynamic RoPE for {name or '<root>'}; "
                "its nonpersistent cache can depend on prior sequence lengths"
            )


def dynamic_rope_checkpoint_contract(model):
    """Reject dynamic caches that a fresh runtime cannot reconstruct exactly.

    HF dynamic RoPE can retain frequencies based on a previous longer input.
    DCP does not save these nonpersistent buffers, so an apparently identical
    model/optimizer restore can otherwise run the next input differently.
    """
    descriptors = []
    for name, module in model.named_modules():
        rope_type = getattr(module, "rope_type", None)
        layers = rope_type.items() if isinstance(rope_type, dict) else [(None, rope_type)]
        for layer_type, kind in layers:
            if not isinstance(kind, str) or ("dynamic" not in kind and kind != "longrope"):
                continue
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            prefix = "" if layer_type is None else f"{layer_type}_"
            try:
                original = int(module.original_max_seq_len)
                cached = int(getattr(module, f"{prefix}max_seq_len_cached", module.max_seq_len_cached))
                inv = getattr(module, f"{prefix}inv_freq")
                initial_inv = getattr(module, f"{prefix}original_inv_freq")
                kwargs = {} if layer_type is None else {"layer_type": layer_type}
                expected_inv, expected_scale = ROPE_INIT_FUNCTIONS[kind](
                    module.config, device=inv.device, seq_len=None if kind == "longrope" else original, **kwargs
                )
                scale = float(
                    getattr(module, f"{prefix}attention_scaling", getattr(module, "attention_scaling", None))
                )
                config = module.config.to_dict()
                # These identify the config source, not its RoPE mathematics.
                config.pop("_name_or_path", None)
                config.pop("transformers_version", None)
                config_hash = hashlib.sha256(
                    json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest()
                valid = (
                    original > 0
                    and cached == original
                    and not inv.requires_grad
                    and not initial_inv.requires_grad
                    and inv.dtype == initial_inv.dtype
                    and torch.equal(inv, initial_inv.to(device=inv.device))
                    and torch.equal(inv, expected_inv.to(device=inv.device, dtype=inv.dtype))
                    and math.isfinite(scale)
                    and scale == float(expected_scale)
                    and (kind != "longrope" or not hasattr(module, f"{prefix}long_inv_freq"))
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot prove dynamic RoPE checkpoint cache for {name or '<root>'}") from exc
            if not valid:
                raise RuntimeError(
                    f"Dynamic RoPE cache for {name or '<root>'} depends on prior sequence lengths; "
                    "its nonpersistent state cannot yet be restored exactly"
                )
            descriptors.append((name, layer_type, kind, original, str(inv.dtype), config_hash))
    return descriptors
