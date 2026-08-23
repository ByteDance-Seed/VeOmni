#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Gate the production Mojo GDR boundary-state VJP on one Ascend NPU.

The lossless GDN CP implementations split one recurrent sequence across CP
owners.  Their local Mojo calls are therefore correct only if composing two
calls through ``final_state -> initial_state`` is equivalent to one monolithic
call, including gradients from both token outputs and the terminal state.

This diagnostic intentionally bypasses CP communication.  It isolates the
local provider used by the managed 910B Mojo route and compares:

* one packed call with a non-zero initial state for every packed segment; and
* two chunk-aligned packed calls chained through their boundary states.

Both the state-passing normalization mode and KCP's producer-normalized mode
are checked.  The script has a CPU reference provider for host validation; the
production gate must use ``--provider mojo --device npu``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


EXPECTED_XPU_CHUNK_GDR_SHA256 = "e79a21afbb78564451cb135d36c64aea9e7060e925c84f72472a68ec03ae253f"
INPUT_NAMES = ("q", "k", "v", "g", "beta", "h0")


@dataclass(frozen=True)
class Comparison:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    max_abs: float
    reference_linf: float
    normalized_l2: float
    relative_p999: float
    absolute_limit: float
    normalized_l2_limit: float
    ok: bool


def _producer_dtype_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def _reference_provider(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if q.shape[0] != 1 or cu_seqlens.numel() < 2:
        raise ValueError("the reference boundary gate requires a packed batch of size one")
    host_cu = [int(point) for point in cu_seqlens]
    if host_cu[0] != 0 or host_cu[-1] != q.shape[1] or any(b < a for a, b in zip(host_cu, host_cu[1:])):
        raise ValueError("cu_seqlens does not cover the reference input")
    if initial_state.shape[0] != len(host_cu) - 1:
        raise ValueError("initial_state rows do not match packed segments")
    output_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = _producer_dtype_l2norm(q)
        k = _producer_dtype_l2norm(k)
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    scale = q.shape[-1] ** -0.5
    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for segment, (start, end) in enumerate(zip(host_cu, host_cu[1:])):
        state = initial_state[segment : segment + 1].float()
        for token in range(start, end):
            q_t = q[:, token] * scale
            k_t = k[:, token]
            v_t = v[:, token]
            state = state * g[:, token].exp()[..., None, None]
            memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - memory) * beta[:, token].unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        final_states.append(state)
    output = torch.stack(outputs, dim=1).to(output_dtype)
    final_state = torch.cat(final_states, dim=0)
    return output, final_state if output_final_state else None


def _sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_provider(name: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    if name == "reference":
        return _reference_provider, {"provider": "reference"}
    if os.environ.get("NPU_GATED_DELTA_RULE_BACKEND") != "triton":
        raise RuntimeError("the Mojo gate requires NPU_GATED_DELTA_RULE_BACKEND=triton before importing xpu_models")
    from xpu_models.ops.attn_impl.chunk_gdr import chunk_gated_delta_rule_func

    source = inspect.getsourcefile(chunk_gated_delta_rule_func)
    if source is None:
        raise RuntimeError("cannot resolve the xpu_models chunk_gdr source")
    source_sha = _sha256(source)
    if source_sha != EXPECTED_XPU_CHUNK_GDR_SHA256:
        raise RuntimeError(
            "xpu_models chunk_gdr source identity drifted: "
            f"actual={source_sha} expected={EXPECTED_XPU_CHUNK_GDR_SHA256} source={source}"
        )
    metadata: dict[str, Any] = {
        "provider": "mojo",
        "callable": f"{chunk_gated_delta_rule_func.__module__}.{chunk_gated_delta_rule_func.__name__}",
        "xpu_chunk_gdr_source": source,
        "xpu_chunk_gdr_sha256": source_sha,
    }
    for distribution in ("xpu-models", "byted-mojo-opset", "byted-mojo-opset-ext"):
        try:
            metadata[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            metadata[distribution] = None
    try:
        from mojo_opset_ext import mojo_chunk_gated_delta_rule

        mojo_source = inspect.getsourcefile(mojo_chunk_gated_delta_rule)
        if mojo_source is not None:
            metadata["mojo_gdr_source"] = mojo_source
            metadata["mojo_gdr_sha256"] = _sha256(mojo_source)
    except (ImportError, OSError, TypeError):
        metadata["mojo_gdr_source"] = None
        metadata["mojo_gdr_sha256"] = None
    return chunk_gated_delta_rule_func, metadata


def _clone_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)


def _prepare_qk(q: torch.Tensor, k: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mode == "state_kernel_norm":
        return q, k, True
    if mode == "kcp_producer_norm":
        return _producer_dtype_l2norm(q), _producer_dtype_l2norm(k), False
    raise ValueError(f"unsupported normalization mode: {mode}")


def _call(
    provider: Callable[..., Any],
    tensors: tuple[torch.Tensor, ...],
    *,
    mode: str,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, g, beta = tensors
    q, k, use_kernel_norm = _prepare_qk(q, k, mode)
    output, final_state = provider(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_kernel_norm,
    )
    if final_state is None:
        raise RuntimeError("the provider did not return a final state")
    return output, final_state


def _run_monolithic(
    provider: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    *,
    mode: str,
    cu_seqlens: torch.Tensor,
    output_gradient: torch.Tensor,
    final_state_gradient: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
    q, k, v, g, beta, h0 = _clone_inputs(inputs)
    output, final_state = _call(provider, (q, k, v, g, beta), mode=mode, initial_state=h0, cu_seqlens=cu_seqlens)
    gradients = torch.autograd.grad(
        (output, final_state),
        (q, k, v, g, beta, h0),
        grad_outputs=(output_gradient, final_state_gradient),
    )
    return (output.detach(), final_state.detach()), tuple(gradient.detach() for gradient in gradients)


def _run_chained(
    provider: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    *,
    mode: str,
    segment_length: int,
    segments: int,
    split: int,
    output_gradient: torch.Tensor,
    final_state_gradient: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...], torch.Tensor]:
    q, k, v, g, beta, h0 = _clone_inputs(inputs)
    left_indices = torch.tensor(
        [
            index
            for segment in range(segments)
            for index in range(segment * segment_length, segment * segment_length + split)
        ],
        device=q.device,
        dtype=torch.long,
    )
    right_indices = torch.tensor(
        [
            index
            for segment in range(segments)
            for index in range(segment * segment_length + split, (segment + 1) * segment_length)
        ],
        device=q.device,
        dtype=torch.long,
    )
    left_inputs = tuple(tensor.index_select(1, left_indices) for tensor in (q, k, v, g, beta))
    right_inputs = tuple(tensor.index_select(1, right_indices) for tensor in (q, k, v, g, beta))
    left_cu = torch.arange(segments + 1, device=q.device, dtype=torch.int32) * split
    right_length = segment_length - split
    right_cu = torch.arange(segments + 1, device=q.device, dtype=torch.int32) * right_length
    left_output, boundary_state = _call(provider, left_inputs, mode=mode, initial_state=h0, cu_seqlens=left_cu)
    boundary_state.retain_grad()
    right_output, final_state = _call(
        provider, right_inputs, mode=mode, initial_state=boundary_state, cu_seqlens=right_cu
    )
    output_segments = []
    for segment in range(segments):
        output_segments.extend(
            (
                left_output[:, segment * split : (segment + 1) * split],
                right_output[:, segment * right_length : (segment + 1) * right_length],
            )
        )
    output = torch.cat(output_segments, dim=1)
    gradients = torch.autograd.grad(
        (output, final_state),
        (q, k, v, g, beta, h0, boundary_state),
        grad_outputs=(output_gradient, final_state_gradient),
    )
    return (
        (output.detach(), final_state.detach()),
        tuple(gradient.detach() for gradient in gradients[:6]),
        gradients[6].detach(),
    )


def _comparison(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    abs_floor: float,
    relative_limit: float,
    normalized_l2_limit: float,
) -> Comparison:
    reference_f = reference.float()
    candidate_f = candidate.float()
    finite = bool(torch.isfinite(reference_f).all() and torch.isfinite(candidate_f).all())
    difference = (candidate_f - reference_f).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    reference_linf = float(reference_f.abs().max()) if reference_f.numel() else 0.0
    denominator = max(float(torch.linalg.vector_norm(reference_f)), 1e-12)
    normalized_l2 = float(torch.linalg.vector_norm(candidate_f - reference_f)) / denominator
    relative_mask = reference_f.abs() >= max(reference_linf * 1e-5, 1e-6)
    if bool(relative_mask.any()):
        relative = difference[relative_mask] / reference_f[relative_mask].abs()
        relative_p999 = float(torch.quantile(relative, 0.999))
    else:
        relative_p999 = 0.0
    absolute_limit = abs_floor + relative_limit * reference_linf
    ok = finite and max_abs <= absolute_limit and normalized_l2 <= normalized_l2_limit
    return Comparison(
        name=name,
        shape=tuple(reference.shape),
        dtype=str(reference.dtype),
        finite=finite,
        max_abs=max_abs,
        reference_linf=reference_linf,
        normalized_l2=normalized_l2,
        relative_p999=relative_p999,
        absolute_limit=absolute_limit,
        normalized_l2_limit=normalized_l2_limit,
        ok=ok,
    )


def _make_inputs(
    args: argparse.Namespace, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    total_length = args.segments * args.length
    q = torch.randn(1, total_length, args.heads, args.key_dim, generator=generator) * 0.08
    k = torch.randn(1, total_length, args.heads, args.key_dim, generator=generator) * 0.08
    v = torch.randn(1, total_length, args.heads, args.value_dim, generator=generator) * 0.10
    g = -torch.rand(1, total_length, args.heads, generator=generator) * 0.05
    beta = torch.sigmoid(torch.randn(1, total_length, args.heads, generator=generator))
    h0 = torch.randn(args.segments, args.heads, args.key_dim, args.value_dim, generator=generator) * 0.02
    output_gradient = torch.randn(1, total_length, args.heads, args.value_dim, generator=generator) * 0.05
    final_state_gradient = (
        torch.randn(args.segments, args.heads, args.key_dim, args.value_dim, generator=generator) * 0.01
    )
    inputs = (
        q.to(device=device, dtype=torch.bfloat16),
        k.to(device=device, dtype=torch.bfloat16),
        v.to(device=device, dtype=torch.bfloat16),
        g.to(device=device, dtype=torch.float32),
        beta.to(device=device, dtype=torch.bfloat16),
        h0.to(device=device, dtype=torch.float32),
    )
    return (
        inputs,
        output_gradient.to(device=device, dtype=torch.bfloat16),
        final_state_gradient.to(device=device, dtype=torch.float32),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.length <= 0 or args.split <= 0 or args.split >= args.length:
        raise ValueError("length and split must describe two non-empty segments")
    if args.length % 32 or args.split % 32 or (args.length - args.split) % 32:
        raise ValueError("length, split, and suffix length must be multiples of the Mojo TTX chunk size 32")
    if min(args.segments, args.heads, args.key_dim, args.value_dim) <= 0:
        raise ValueError("packed segments, heads, and state dimensions must be positive")
    for name in ("abs_floor_bf16", "abs_floor_fp32", "relative_limit", "normalized_l2_limit"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--device npu requires torch_npu") from exc
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    if args.provider == "mojo" and args.device != "npu":
        raise ValueError("the production Mojo provider must run on NPU")
    device = _device(args)
    provider, provider_identity = _load_provider(args.provider)
    inputs, output_gradient, final_state_gradient = _make_inputs(args, device)
    monolithic_cu = torch.arange(args.segments + 1, device=device, dtype=torch.int32) * args.length
    modes = ("state_kernel_norm", "kcp_producer_norm") if args.mode == "both" else (args.mode,)
    mode_reports: dict[str, Any] = {}
    overall_ok = True
    for mode in modes:
        monolithic, monolithic_gradients = _run_monolithic(
            provider,
            inputs,
            mode=mode,
            cu_seqlens=monolithic_cu,
            output_gradient=output_gradient,
            final_state_gradient=final_state_gradient,
        )
        chained, chained_gradients, boundary_gradient = _run_chained(
            provider,
            inputs,
            mode=mode,
            segment_length=args.length,
            segments=args.segments,
            split=args.split,
            output_gradient=output_gradient,
            final_state_gradient=final_state_gradient,
        )
        _synchronize(device)
        comparisons: list[Comparison] = []
        for name, reference, candidate in (
            ("output", monolithic[0], chained[0]),
            ("final_state", monolithic[1], chained[1]),
            *(
                (f"grad_{name}", reference, candidate)
                for name, reference, candidate in zip(INPUT_NAMES, monolithic_gradients, chained_gradients)
            ),
        ):
            abs_floor = args.abs_floor_bf16 if reference.dtype == torch.bfloat16 else args.abs_floor_fp32
            comparisons.append(
                _comparison(
                    name,
                    reference,
                    candidate,
                    abs_floor=abs_floor,
                    relative_limit=args.relative_limit,
                    normalized_l2_limit=args.normalized_l2_limit,
                )
            )
        nonzero = {
            name: bool(torch.count_nonzero(gradient).item()) for name, gradient in zip(INPUT_NAMES, chained_gradients)
        }
        nonzero["boundary_state"] = bool(torch.count_nonzero(boundary_gradient).item())
        mode_ok = all(comparison.ok for comparison in comparisons) and all(nonzero.values())
        overall_ok = overall_ok and mode_ok
        mode_reports[mode] = {
            "ok": mode_ok,
            "comparisons": [asdict(comparison) for comparison in comparisons],
            "nonzero_gradients": nonzero,
            "boundary_gradient_linf": float(boundary_gradient.float().abs().max()),
        }
    report = {
        "ok": overall_ok,
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "device": str(device),
        "provider_identity": provider_identity,
        "config": {
            "length": args.length,
            "segments": args.segments,
            "total_tokens": args.segments * args.length,
            "split": args.split,
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "seed": args.seed,
            "mode": args.mode,
            "output_final_state": True,
            "nonzero_initial_state": True,
            "nonzero_terminal_state_gradient": True,
        },
        "modes": mode_reports,
    }
    if args.output and int(os.environ.get("RANK", "0")) == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    marker = "AI4SE_MOJO_BOUNDARY_VJP_GATE_OK" if overall_ok else "AI4SE_MOJO_BOUNDARY_VJP_GATE_FAIL"
    print(marker, json.dumps(report, sort_keys=True), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mojo", "reference"), default="mojo")
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument(
        "--mode",
        choices=("both", "state_kernel_norm", "kcp_producer_norm"),
        default="both",
    )
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--split", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026082301)
    parser.add_argument("--abs-floor-bf16", type=float, default=0.25)
    parser.add_argument("--abs-floor-fp32", type=float, default=0.05)
    parser.add_argument("--relative-limit", type=float, default=0.02)
    parser.add_argument("--normalized-l2-limit", type=float, default=0.01)
    parser.add_argument("--output", type=str)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return 0 if run(args)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
