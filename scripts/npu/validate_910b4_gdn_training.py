#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate Qwen3.5 GatedDeltaNet training kernels on Ascend 910B4-1."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


_EXPECTED_TRITON_ARCH = "Ascend910B4"
_SUPPORTED_DEVICE_NAMES = frozenset({"Ascend910B4", "Ascend910B4-1"})
_STRICT_ATOL = 5e-2
_STRICT_RTOL = 5e-2
_STRICT_MIN_TOKENS = 131072
_STRICT_MIN_REFERENCE_TOKENS = 127
_SUCCESS_MARKER = "VEOMNI_910B4_GDN_TRAINING_GATE_OK"
_DIAGNOSTIC_MARKER = "VEOMNI_910B4_GDN_TRAINING_DIAGNOSTIC_OK"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Qwen3.5 config.json used to derive SP-local GatedDeltaNet dimensions.",
    )
    parser.add_argument("--sp-size", type=int, default=8, help="Target Ulysses sequence-parallel size.")
    parser.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="Token count seen by GatedDeltaNet after Ulysses gathers sequence.",
    )
    parser.add_argument(
        "--reference-tokens",
        type=int,
        default=127,
        help="Short, deliberately unaligned token count used for independent references.",
    )
    parser.add_argument("--device", type=int, default=0, help="Visible NPU index used by the gate.")
    parser.add_argument("--atol", type=float, default=_STRICT_ATOL)
    parser.add_argument("--rtol", type=float, default=_STRICT_RTOL)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("atol", "rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if args.sp_size <= 0:
        raise ValueError("--sp-size must be positive")
    if args.reference_tokens > args.tokens:
        raise ValueError("--reference-tokens must not exceed --tokens")
    _segment_boundaries(args.reference_tokens)
    _segment_boundaries(args.tokens)


def _load_gdn_dimensions(config_path: Path, sp_size: int) -> tuple[int, int, int, int, int]:
    with config_path.open() as config_file:
        config = json.load(config_file)
    text_config = config.get("text_config", config)
    names = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    missing = [name for name in names if name not in text_config]
    if missing:
        raise KeyError(f"model config is missing GatedDeltaNet fields: {missing}")

    key_heads = int(text_config["linear_num_key_heads"])
    value_heads = int(text_config["linear_num_value_heads"])
    if key_heads % sp_size or value_heads % sp_size:
        raise ValueError(f"GatedDeltaNet heads ({key_heads}, {value_heads}) must be divisible by SP size {sp_size}")

    local_key_heads = key_heads // sp_size
    local_value_heads = value_heads // sp_size
    if local_value_heads % local_key_heads:
        raise ValueError("SP-local value heads must be divisible by key heads")
    return (
        local_key_heads,
        local_value_heads,
        int(text_config["linear_key_head_dim"]),
        int(text_config["linear_value_head_dim"]),
        int(text_config["linear_conv_kernel_dim"]),
    )


def _segment_boundaries(tokens: int) -> list[int]:
    if tokens < 12:
        raise ValueError("token counts must be at least 12")
    first = tokens // 4
    second = first + tokens // 3
    boundaries = [0, first, second, tokens]
    if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError(f"invalid segment boundaries: {boundaries}")
    return boundaries


def _leaf_clone(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _assert_close(torch, name, actual, expected, *, atol: float, rtol: float) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    if actual_float.device != expected_float.device:
        actual_float = actual_float.cpu()
        expected_float = expected_float.cpu()
    torch.testing.assert_close(
        actual_float,
        expected_float,
        atol=atol,
        rtol=rtol,
        msg=lambda message: f"{name}: {message}",
    )
    return (actual_float - expected_float).abs().max().item()


def _call_causal_conv(causal_conv, x, weight, cu_seqlens):
    return causal_conv(
        x=x,
        weight=weight,
        bias=None,
        activation="silu",
        cu_seqlens=cu_seqlens,
    )[0]


def _call_gdr(chunk_gdr, q, k, v, g, beta, cu_seqlens, key_dim: int):
    return chunk_gdr(
        q,
        k,
        v,
        g,
        beta,
        scale=key_dim**-0.5,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )[0]


def _causal_reference(torch, functional, x, weight, boundaries: list[int]):
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = functional.pad(x[:, start:end].transpose(1, 2).float(), (weight.shape[-1] - 1, 0))
        output = functional.conv1d(segment, weight.float().unsqueeze(1), groups=weight.shape[0])
        outputs.append(functional.silu(output).to(x.dtype).transpose(1, 2))
    return torch.cat(outputs, dim=1)


def _gdr_reference(torch, q, k, v, g, beta, boundaries: list[int]):
    output_dtype = q.dtype
    q_float, k_float = q.float(), k.float()
    q = (q_float * torch.rsqrt((q_float * q_float).sum(dim=-1, keepdim=True) + 1e-6)).to(output_dtype)
    k = (k_float * torch.rsqrt((k_float * k_float).sum(dim=-1, keepdim=True) + 1e-6)).to(output_dtype)
    q, k = q.float(), k.float()
    q = q * (q.shape[-1] ** -0.5)
    v, g, beta = v.float(), g.float(), beta.float()

    segment_outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        state = torch.zeros(
            q.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
        token_outputs = []
        for token in range(start, end):
            q_token = q[:, token]
            k_token = k[:, token]
            v_token = v[:, token]
            state = state * g[:, token].exp().unsqueeze(-1).unsqueeze(-1)
            memory = (state * k_token.unsqueeze(-1)).sum(dim=-2)
            delta = (v_token - memory) * beta[:, token].unsqueeze(-1)
            state = state + k_token.unsqueeze(-1) * delta.unsqueeze(-2)
            token_outputs.append((state * q_token.unsqueeze(-1)).sum(dim=-2))
        segment_outputs.append(torch.stack(token_outputs, dim=1))
    return torch.cat(segment_outputs, dim=1).to(output_dtype)


def _new_gdr_inputs(torch, device, dtype, tokens, heads, key_dim, value_dim):
    return (
        torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype),
        torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype),
        torch.randn(1, tokens, heads, value_dim, device=device, dtype=dtype),
        -torch.nn.functional.softplus(torch.randn(1, tokens, heads, device=device, dtype=torch.float32)),
        torch.rand(1, tokens, heads, device=device, dtype=dtype),
    )


def _run_causal_reference_gate(torch, causal_conv, device, dtype, conv_dim, conv_width, tokens, atol, rtol) -> float:
    boundaries = _segment_boundaries(tokens)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    base_input = torch.randn(1, tokens, conv_dim, device=device, dtype=dtype)
    base_weight = torch.randn(conv_dim, conv_width, device=device, dtype=dtype)
    actual_input, actual_weight = _leaf_clone(base_input), _leaf_clone(base_weight)
    reference_input = base_input.detach().cpu().requires_grad_(True)
    reference_weight = base_weight.detach().cpu().requires_grad_(True)
    actual = _call_causal_conv(causal_conv, actual_input, actual_weight, cu_seqlens)
    expected = _causal_reference(
        torch,
        torch.nn.functional,
        reference_input,
        reference_weight,
        boundaries,
    )
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.cpu())
    return max(
        _assert_close(torch, "causal reference output", actual, expected, atol=atol, rtol=rtol),
        _assert_close(
            torch,
            "causal reference input gradient",
            actual_input.grad,
            reference_input.grad,
            atol=atol,
            rtol=rtol,
        ),
        _assert_close(
            torch,
            "causal reference weight gradient",
            actual_weight.grad,
            reference_weight.grad,
            atol=atol,
            rtol=rtol,
        ),
    )


def _run_gdr_reference_gate(torch, chunk_gdr, device, dtype, tokens, heads, key_dim, value_dim, atol, rtol) -> float:
    boundaries = _segment_boundaries(tokens)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.long, device=device)
    base_tensors = _new_gdr_inputs(torch, device, dtype, tokens, heads, key_dim, value_dim)
    actual_tensors = tuple(_leaf_clone(tensor) for tensor in base_tensors)
    reference_tensors = tuple(_leaf_clone(tensor) for tensor in base_tensors)
    actual = _call_gdr(chunk_gdr, *actual_tensors, cu_seqlens, key_dim)
    expected = _gdr_reference(torch, *reference_tensors, boundaries)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    maxima = [_assert_close(torch, "GDR reference output", actual, expected, atol=atol, rtol=rtol)]
    for name, actual_tensor, reference_tensor in zip(
        ("q", "k", "v", "g", "beta"),
        actual_tensors,
        reference_tensors,
    ):
        maxima.append(
            _assert_close(
                torch,
                f"GDR reference {name} gradient",
                actual_tensor.grad,
                reference_tensor.grad,
                atol=atol,
                rtol=rtol,
            )
        )
    return max(maxima)


def _run_causal_boundary_gate(torch, causal_conv, device, dtype, conv_dim, conv_width, tokens, atol, rtol) -> float:
    boundaries = _segment_boundaries(tokens)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    base_input = torch.randn(1, tokens, conv_dim, device=device, dtype=dtype)
    base_weight = torch.randn(conv_dim, conv_width, device=device, dtype=dtype)
    packed_input, segmented_input = _leaf_clone(base_input), _leaf_clone(base_input)
    packed_weight = _leaf_clone(base_weight)
    packed = _call_causal_conv(causal_conv, packed_input, packed_weight, cu_seqlens)
    segment_weights = []
    segment_outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment_weight = _leaf_clone(base_weight)
        segment_weights.append(segment_weight)
        segment_outputs.append(
            _call_causal_conv(
                causal_conv,
                segmented_input[:, start:end].contiguous(),
                segment_weight,
                torch.tensor([0, end - start], dtype=torch.int32, device=device),
            )
        )
    segmented = torch.cat(segment_outputs, dim=1)
    upstream = torch.randn_like(packed)
    packed_input_grad = torch.autograd.grad(packed, packed_input, upstream, retain_graph=True)[0]
    segmented_input_grad = torch.autograd.grad(segmented, segmented_input, upstream, retain_graph=True)[0]
    maxima = [
        _assert_close(torch, "causal packed output", packed, segmented, atol=atol, rtol=rtol),
        _assert_close(
            torch,
            "causal packed input gradient",
            packed_input_grad,
            segmented_input_grad,
            atol=atol,
            rtol=rtol,
        ),
    ]

    packed_weight_contributions = []
    segmented_weight_contributions = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        packed_contribution = torch.autograd.grad(
            packed[:, start:end],
            packed_weight,
            upstream[:, start:end],
            retain_graph=True,
        )[0]
        segmented_contribution = torch.autograd.grad(
            segment_outputs[index],
            segment_weights[index],
            upstream[:, start:end],
            retain_graph=True,
        )[0]
        maxima.append(
            _assert_close(
                torch,
                f"causal segment {index} weight VJP",
                packed_contribution,
                segmented_contribution,
                atol=atol,
                rtol=rtol,
            )
        )
        packed_weight_contributions.append(packed_contribution.float())
        segmented_weight_contributions.append(segmented_contribution.float())

    maxima.append(
        _assert_close(
            torch,
            "causal FP32-summed weight VJP",
            torch.stack(packed_weight_contributions).sum(dim=0),
            torch.stack(segmented_weight_contributions).sum(dim=0),
            atol=atol,
            rtol=rtol,
        )
    )
    return max(maxima)


def _run_gdr_boundary_gate(torch, chunk_gdr, device, dtype, tokens, heads, key_dim, value_dim, atol, rtol) -> float:
    boundaries = _segment_boundaries(tokens)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.long, device=device)
    base_tensors = _new_gdr_inputs(torch, device, dtype, tokens, heads, key_dim, value_dim)
    packed_tensors = tuple(_leaf_clone(tensor) for tensor in base_tensors)
    segmented_tensors = tuple(_leaf_clone(tensor) for tensor in base_tensors)
    packed = _call_gdr(chunk_gdr, *packed_tensors, cu_seqlens, key_dim)
    segmented = torch.cat(
        [
            _call_gdr(
                chunk_gdr,
                *(tensor[:, start:end].contiguous() for tensor in segmented_tensors),
                torch.tensor([0, end - start], dtype=torch.long, device=device),
                key_dim,
            )
            for start, end in zip(boundaries, boundaries[1:])
        ],
        dim=1,
    )
    upstream = torch.randn_like(packed)
    packed.backward(upstream)
    segmented.backward(upstream)
    maxima = [_assert_close(torch, "GDR packed output", packed, segmented, atol=atol, rtol=rtol)]
    for name, packed_tensor, segmented_tensor in zip(
        ("q", "k", "v", "g", "beta"),
        packed_tensors,
        segmented_tensors,
    ):
        maxima.append(
            _assert_close(
                torch,
                f"GDR packed {name} gradient",
                packed_tensor.grad,
                segmented_tensor.grad,
                atol=atol,
                rtol=rtol,
            )
        )
    return max(maxima)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    strict_profile = (
        args.atol == _STRICT_ATOL
        and args.rtol == _STRICT_RTOL
        and args.tokens >= _STRICT_MIN_TOKENS
        and args.reference_tokens >= _STRICT_MIN_REFERENCE_TOKENS
    )
    triton_arch = os.getenv("TRITON_ASCEND_ARCH")
    if triton_arch != _EXPECTED_TRITON_ARCH:
        raise RuntimeError(
            f"set TRITON_ASCEND_ARCH={_EXPECTED_TRITON_ARCH} before starting this script; got {triton_arch!r}"
        )

    import torch
    import torch_npu  # noqa: F401 - register the torch.npu device namespace

    torch.manual_seed(42)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    device_name = torch.npu.get_device_name(args.device)
    if device_name not in _SUPPORTED_DEVICE_NAMES:
        raise RuntimeError(f"expected an Ascend 910B4-1 device, got {device_name!r}")

    from veomni.ops.kernels.gated_delta_rule._ascend.chunk_gated_delta_rule_mm import (
        chunk_gated_delta_rule,
    )
    from veomni.ops.kernels.gated_delta_rule.npu_causal_conv1d import causal_conv1d
    from veomni.ops.kernels.gated_delta_rule.npu_hardware import get_hidden_state_block_value

    if get_hidden_state_block_value(args.device) != 64:
        raise RuntimeError("Ascend 910B4-1 GDR did not select the BV64 hidden-state launch")

    local_key_heads, local_value_heads, key_dim, value_dim, conv_width = _load_gdn_dimensions(
        args.model_config,
        args.sp_size,
    )
    conv_dim = 2 * local_key_heads * key_dim + local_value_heads * value_dim
    dtype = torch.bfloat16

    causal_reference_max = _run_causal_reference_gate(
        torch,
        causal_conv1d,
        device,
        dtype,
        conv_dim,
        conv_width,
        args.reference_tokens,
        args.atol,
        args.rtol,
    )
    gdr_reference_max = _run_gdr_reference_gate(
        torch,
        chunk_gated_delta_rule,
        device,
        dtype,
        args.reference_tokens,
        local_value_heads,
        key_dim,
        value_dim,
        args.atol,
        args.rtol,
    )
    torch.npu.synchronize()
    torch.npu.empty_cache()

    causal_boundary_max = _run_causal_boundary_gate(
        torch,
        causal_conv1d,
        device,
        dtype,
        conv_dim,
        conv_width,
        args.tokens,
        args.atol,
        args.rtol,
    )
    torch.npu.synchronize()
    torch.npu.empty_cache()
    gdr_boundary_max = _run_gdr_boundary_gate(
        torch,
        chunk_gated_delta_rule,
        device,
        dtype,
        args.tokens,
        local_value_heads,
        key_dim,
        value_dim,
        args.atol,
        args.rtol,
    )
    torch.npu.synchronize()

    print(
        _SUCCESS_MARKER if strict_profile else _DIAGNOSTIC_MARKER,
        f"gate_profile={'strict' if strict_profile else 'diagnostic'}",
        f"device={device_name}",
        f"triton_arch={triton_arch}",
        "gdr_hidden_state_bv=64",
        f"sp={args.sp_size}",
        f"full_tokens_after_ulysses={args.tokens}",
        f"reference_tokens={args.reference_tokens}",
        f"conv={conv_dim}x{conv_width}",
        f"gdr={local_value_heads}x{key_dim}x{value_dim}",
        f"atol={args.atol:.8g}",
        f"rtol={args.rtol:.8g}",
        f"strict_min_tokens={_STRICT_MIN_TOKENS}",
        f"strict_min_reference_tokens={_STRICT_MIN_REFERENCE_TOKENS}",
        f"causal_reference_max_abs={causal_reference_max:.8g}",
        f"gdr_reference_max_abs={gdr_reference_max:.8g}",
        f"causal_boundary_max_abs={causal_boundary_max:.8g}",
        f"gdr_boundary_max_abs={gdr_boundary_max:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
