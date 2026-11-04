# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint preflight for dynamic RoPE's nonpersistent runtime state."""

import hashlib
import importlib
import json
import math

from .tensor_bytes import tensor_byte_chunks


def _is_native_qwen3_dynamic(module):
    """Recognize upstream and VeOmni-generated Qwen3 rotary implementations."""
    cls = type(module)
    approved_modules = {
        "transformers.models.qwen3.modeling_qwen3",
        "veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu",
        "veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_npu",
    }
    if cls.__module__ not in approved_modules or cls.__name__ != "Qwen3RotaryEmbedding":
        return False
    return importlib.import_module(cls.__module__).Qwen3RotaryEmbedding is cls and module.rope_type == "dynamic"


def _is_native_qwen3_dynamic_reset_alias(module):
    """Recognize the native Qwen3 reset alias, not custom mutable caches."""
    buffers = module._buffers
    return (
        _is_native_qwen3_dynamic(module)
        and {"inv_freq", "original_inv_freq"} <= module._non_persistent_buffers_set
        and buffers.get("inv_freq") is not None
        and buffers.get("inv_freq") is buffers.get("original_inv_freq")
    )


def _tensor_digest(value):
    digest = hashlib.sha256()
    for chunk in tensor_byte_chunks(value, chunk_bytes=4 << 20):
        digest.update(chunk)
    return digest.hexdigest()


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
                # Native Qwen3 resets by reusing the original buffer. Its
                # forward reads the buffer and growth replaces it, so object
                # identity is not part of its recoverable numerical state.
                if inv is initial_inv and not (layer_type is None and _is_native_qwen3_dynamic_reset_alias(module)):
                    raise RuntimeError("dynamic RoPE reset shares nonpersistent cache buffers")
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
                inv_digest = _tensor_digest(inv)
                initial_digest = _tensor_digest(initial_inv)
                expected_inv = expected_inv.to(dtype=inv.dtype)
                matches_canonical = (
                    tuple(inv.shape) == tuple(expected_inv.shape)
                    and inv_digest == _tensor_digest(expected_inv)
                    and scale == float(expected_scale)
                )
                if not matches_canonical and layer_type is None and _is_native_qwen3_dynamic(module):
                    # HF often builds the original table on CPU before moving it
                    # to an accelerator. Recompute that origin exactly rather
                    # than tolerating device-dependent rounding differences.
                    cpu_inv, cpu_scale = ROPE_INIT_FUNCTIONS[kind](module.config, device="cpu", seq_len=original)
                    cpu_inv = cpu_inv.to(dtype=inv.dtype)
                    matches_canonical = (
                        tuple(inv.shape) == tuple(cpu_inv.shape)
                        and inv_digest == _tensor_digest(cpu_inv)
                        and scale == float(cpu_scale)
                    )
                valid = (
                    original > 0
                    and cached == original
                    and not inv.requires_grad
                    and not initial_inv.requires_grad
                    and inv.dtype == initial_inv.dtype
                    and tuple(inv.shape) == tuple(initial_inv.shape)
                    and inv_digest == initial_digest
                    and matches_canonical
                    and math.isfinite(scale)
                    and (kind != "longrope" or not hasattr(module, f"{prefix}long_inv_freq"))
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot prove dynamic RoPE checkpoint cache for {name or '<root>'}") from exc
            if not valid:
                raise RuntimeError(
                    f"Dynamic RoPE cache for {name or '<root>'} depends on prior sequence lengths; "
                    "its nonpersistent state cannot yet be restored exactly"
                )
            # DCP omits these nonpersistent tables: bind the exact current and
            # reset bytes to the layout so restore rejects another origin even
            # when both individually match a valid canonical reconstruction.
            descriptors.append(
                (
                    name,
                    layer_type,
                    kind,
                    original,
                    str(inv.dtype),
                    config_hash,
                    f"{type(module).__module__}.{type(module).__qualname__}",
                    tuple(inv.shape),
                    scale.hex(),
                    inv_digest,
                    initial_digest,
                )
            )
    return descriptors
