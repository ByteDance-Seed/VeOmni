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

"""Context-parallel forward/backward equivalence for DeepSeek-V4 attention."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter, runner_device_count
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_gpu_compute_capability,
    get_torch_device,
)


_PATCHED_MODULE = "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu"


def _load_dsv4_toy_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"
    return config


def _cuda_device_count() -> int:
    """Devices this suite can run on, which are CUDA devices and nothing else.

    Context parallelism is GPU-only -- ``check_context_parallel_supported``
    refuses it on Ascend -- but these workers build the GPU patched module
    directly instead of going through that gate, so a plain device count would
    let a multi-NPU host start a suite that cannot run there.
    """
    return runner_device_count() if IS_CUDA_AVAILABLE else 0


def _all_devices_sm90_or_above() -> bool:
    """Whether every device a worker might select is SM90 or later.

    Not ``is_sm90_or_above``: that reads the current device, while each worker
    calls ``set_device(rank)`` of its own. On a mixed-capability host the marker
    would pass on device 0 and the kernel would then fail on some other rank.
    """
    count = get_torch_device().device_count() if IS_CUDA_AVAILABLE else 0
    return count > 0 and all(get_gpu_compute_capability(index) >= 90 for index in range(count))


def _use_full_float32_matmuls() -> None:
    """Keep float32 matmuls off the TF32 path so the tolerances here mean something.

    Every check below compares a sharded float32 result against a single-rank
    float32 baseline at ``atol=rtol=1e-4``. TF32 keeps 10 mantissa bits, which
    leaves the two sides ~1e-3 apart on the parameter gradients -- an order of
    magnitude past that tolerance -- and Ampere-and-later devices take the TF32
    path by default. Each rank is a spawned process, so this has to be set per
    worker rather than once at import.
    """
    torch.set_float32_matmul_precision("highest")


def _broadcast_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _build_causal_mask(seq_len: int, sliding_window: int | None, device, dtype) -> torch.Tensor:
    """Copied from tests/parallel/ulysses/test_deepseek_v4_ulysses.py:42-49."""
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    k_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
    causal = k_idx <= q_idx
    if sliding_window is not None:
        causal = causal & (k_idx > q_idx - sliding_window)
    full_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    return full_mask.masked_fill(~causal, torch.finfo(dtype).min)


def _build_packed_causal_mask(seq_len: int, sliding_window: int | None, sample_slices, device, dtype) -> torch.Tensor:
    """``_build_causal_mask`` with attention additionally confined to each packed sample.

    Spelled out here rather than taken from ``isolate_packed_causal_mask_``, for
    the same reason ``_window_starts`` is: a fixture that derives its expectation
    from the code under test cannot contradict it.
    """
    mask = _build_causal_mask(seq_len, sliding_window, device, dtype)
    sample_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    for index, (begin, end) in enumerate(sample_slices):
        sample_ids[begin:end] = index
    crosses_samples = sample_ids.view(-1, 1) != sample_ids.view(1, -1)
    return mask.masked_fill(crosses_samples.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)


def _packed_position_ids(sample_slices, device) -> torch.Tensor:
    """Per-sample positions restarting at every packed boundary."""
    return torch.cat([torch.arange(end - begin, device=device) for begin, end in sample_slices]).view(1, -1)


def _init_position_bias(compressor) -> None:
    """Give ``position_bias`` real values.

    Both compressors declare it with ``torch.empty`` and nothing in these tests
    runs ``_init_weights``, so it would otherwise hold whatever was in memory. A
    single inf there turns every comparison below into a NaN mismatch that says
    nothing about context parallelism.
    """
    with torch.no_grad():
        torch.nn.init.normal_(compressor.position_bias, std=0.02)
        if getattr(compressor, "indexer", None) is not None:
            torch.nn.init.normal_(compressor.indexer.position_bias, std=0.02)


def _init_every_position_bias(model: torch.nn.Module) -> None:
    """``_init_position_bias`` for every compressor and indexer in a whole model."""
    for module in model.modules():
        if getattr(module, "position_bias", None) is not None:
            with torch.no_grad():
                torch.nn.init.normal_(module.position_bias, std=0.02)


def _init_attention_parameters(layer) -> None:
    # A bare Attention skips PreTrainedModel.post_init(): sinks is allocated
    # with torch.empty. Match the model's zero initialization before broadcast;
    # otherwise allocator contents can make both baseline and CP outputs NaN.
    torch.nn.init.zeros_(layer.sinks)
    if layer.compressor is not None:
        _init_position_bias(layer.compressor)


def _make_forward(layer, rotary):
    """The attention call every test shares: both rope variants, then the layer."""

    def forward(hidden_states, position_ids, attention_mask, **kwargs):
        embeddings = {
            name: rotary(hidden_states, position_ids=position_ids, layer_type=name) for name in ("main", "compress")
        }
        output, _ = layer(
            hidden_states,
            position_embeddings=embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return output

    return forward


def _init_cp_attention(
    rank: int,
    world_size: int,
    init_file: str,
    seq_len: int,
    with_compressor: bool,
    layer_idx: int = 0,
    batch_size: int = 1,
    sample_slices=None,
    dtype: torch.dtype = torch.float32,
    dsa_attention_impl: str = "eager",
    *,
    init_dist: bool = True,
):
    device_type = get_device_type()
    if init_dist:
        get_torch_device().set_device(rank)
        _use_full_float32_matmuls()
        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.ops.config import set_ops_config

    cp_state = _init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type, name="cp")
    dp_state = _init_parallel_state(dp_size=world_size, cp_size=1, ulysses_size=1, device_type=device_type, name="dp")
    set_ops_config(
        SimpleNamespace(
            dsa_attention_implementation=dsa_attention_impl,
            dsa_indexer_implementation="eager",
        )
    )

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    layer = dsv4.DeepseekV4Attention(config, layer_idx=layer_idx).to(device=device_type, dtype=dtype)
    if not with_compressor:
        layer.compressor = None
    _init_attention_parameters(layer)
    _broadcast_module(layer)
    layer.train()

    full_hidden = torch.randn(batch_size, seq_len, config.hidden_size, device=device_type, dtype=dtype)
    dist.broadcast(full_hidden, src=0)
    if sample_slices is None:
        full_position_ids = torch.arange(seq_len, device=device_type).view(1, -1)
        full_mask = _build_causal_mask(seq_len, config.sliding_window, device_type, dtype)
    else:
        full_position_ids = _packed_position_ids(sample_slices, device_type)
        full_mask = _build_packed_causal_mask(seq_len, config.sliding_window, sample_slices, device_type, dtype)
    full_position_ids = full_position_ids.repeat(batch_size, 1)
    full_mask = full_mask.repeat(batch_size, 1, 1, 1)

    rotary = dsv4.DeepseekV4RotaryEmbedding(config).to(device=device_type)
    _broadcast_module(rotary)

    return dsv4, layer, _make_forward(layer, rotary), full_hidden, full_position_ids, full_mask, cp_state, dp_state


def _run_attention_cp(
    rank: int,
    world_size: int,
    init_file: str,
    seq_len: int,
    with_compressor: bool,
    layer_idx: int = 0,
    batch_size: int = 1,
    sample_slices=None,
    *,
    init_dist: bool = True,
) -> None:
    from veomni.distributed.parallel_state import clear_parallel_state, use_parallel_state

    _, layer, _forward, full_hidden, full_position_ids, full_mask, cp_state, dp_state = _init_cp_attention(
        rank,
        world_size,
        init_file,
        seq_len,
        with_compressor,
        layer_idx,
        batch_size,
        sample_slices,
        init_dist=init_dist,
    )

    with use_parallel_state(dp_state):
        baseline_input = full_hidden.detach().clone().requires_grad_(True)
        baseline = _forward(baseline_input, full_position_ids, full_mask)
        baseline.sum().backward()
        baseline_grads = {
            name: param.grad.detach().clone() for name, param in layer.named_parameters() if param.grad is not None
        }
        baseline = baseline.detach()
        baseline_input_grad = baseline_input.grad.detach().clone()
        layer.zero_grad(set_to_none=True)

    # Context-parallel: this rank's contiguous slice, real parallel state.
    local_len = seq_len // world_size
    begin = rank * local_len
    local_input = full_hidden[:, begin : begin + local_len].detach().clone().requires_grad_(True)
    with use_parallel_state(cp_state):
        local_output = _forward(local_input, full_position_ids[:, begin : begin + local_len], full_mask)
        local_output.sum().backward()

    summed_grads = {}
    for name, param in layer.named_parameters():
        if param.grad is None:
            continue
        summed = param.grad.detach().clone()
        dist.all_reduce(summed)
        summed_grads[name] = summed

    torch.testing.assert_close(local_output, baseline[:, begin : begin + local_len], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        local_input.grad, baseline_input_grad[:, begin : begin + local_len], rtol=1e-4, atol=1e-4
    )
    for name, summed in summed_grads.items():
        torch.testing.assert_close(summed, baseline_grads[name], rtol=1e-4, atol=1e-4)

    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


class _RecordingDsaAttention:
    """``dsa_attention`` stand-in that records the candidate slots each call implies."""

    def __init__(self) -> None:
        self.built: list[torch.Tensor] = []

    def __call__(self, q, ori_kv, cmp_kv, sinks, position_ids, **kwargs):
        from veomni.ops.kernels.dsa.sparse_indices import build_sparse_attention_indices

        del sinks
        self.built.append(
            build_sparse_attention_indices(
                position_ids=position_ids,
                sliding_window=kwargs["sliding_window"],
                kv_len=ori_kv.shape[1],
                compressed_len=0 if cmp_kv is None else cmp_kv.shape[1],
                candidates=kwargs["candidates"],
                query_offset=kwargs["query_offset"],
                attention_mask=kwargs["attention_mask"],
            )
        )
        return torch.zeros_like(q)


def _run_attention_cp_sparse_indices(
    rank: int, world_size: int, init_file: str, seq_len: int, *, init_dist: bool = True
) -> None:
    from veomni.distributed.parallel_state import clear_parallel_state, use_parallel_state

    _, layer, _forward, full_hidden, full_position_ids, full_mask, cp_state, dp_state = _init_cp_attention(
        rank,
        world_size,
        init_file,
        seq_len,
        with_compressor=False,
        dtype=torch.bfloat16,
        init_dist=init_dist,
    )
    recorder = _RecordingDsaAttention()
    layer.veomni_dsa_attention = recorder

    local_len = seq_len // world_size
    begin = rank * local_len
    with torch.no_grad():
        with use_parallel_state(dp_state):
            _forward(full_hidden, full_position_ids, full_mask)
        (baseline_indices,) = recorder.built
        recorder.built.clear()

        with use_parallel_state(cp_state):
            _forward(
                full_hidden[:, begin : begin + local_len],
                full_position_ids[:, begin : begin + local_len],
                full_mask,
            )
        (local_indices,) = recorder.built

    torch.testing.assert_close(local_indices, baseline_indices[:, begin : begin + local_len], rtol=0, atol=0)
    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


class _FixedIndexer(torch.nn.Module):
    """A Lightning Indexer stand-in that always selects compressed slot 0.

    The CSA compressor calls its indexer unconditionally, and both summarise the
    same windows, so a real one would let an indexer bug masquerade as a
    compressor bug and vice versa. Holding the selection fixed keeps the
    compressor's window compression the only thing these tests can distinguish;
    the indexer has its own parity test below.
    """

    def __init__(self, index_topk: int):
        super().__init__()
        self.index_topk = index_topk

    def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        return torch.zeros(batch, seq_len, self.index_topk, dtype=torch.long, device=hidden_states.device)


def _grad_or_zeros(param: torch.nn.Parameter) -> torch.Tensor:
    """This parameter's gradient, or zeros where the forward never reached it."""
    return torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()


def _assert_close_to_scale(actual: torch.Tensor, expected: torch.Tensor, tolerance: float, name: str) -> None:
    """Compare with a tolerance tied to ``expected``'s own magnitude.

    Whole-model gradients cannot be compared with a fixed ``atol``. Summing one
    128-row forward and summing four 32-row ones reduce in different orders, and
    the resulting error floor scales with the size of the terms being summed --
    not with the element that happens to cancel down to near zero. In the toy
    model the embedding gradient spans ``1e-2`` to ``1e3`` inside a single
    tensor, so ``atol=1e-4`` fails on that cancellation while a fixed ``atol``
    large enough to pass it would no longer catch anything. Scaling each tensor
    by its own maximum makes the tolerance mean "a fraction of this tensor",
    which is the quantity noise here is actually bounded by. The floor of 1 keeps
    a tensor that is entirely rounding noise (the hash routers' gate weights peak
    around ``1e-6``) from being compared against its own noise.
    """
    scale = expected.abs().max().clamp_min(1.0).float()
    torch.testing.assert_close(
        actual.float() / scale,
        expected.float() / scale,
        rtol=tolerance,
        atol=tolerance,
        msg=lambda text: f"{name} (scaled by {float(scale):.4g}): {text}",
    )


def _visible_compressed(candidates, seq_len: int, compressed_len: int) -> torch.Tensor:
    """``[B, S, C]`` bool: the compressed rows each query may read, whatever the slot order."""
    if candidates.topk_indices is not None:
        indices = candidates.topk_indices.long()
        visible = torch.zeros(*indices.shape[:2], compressed_len + 1, dtype=torch.bool, device=indices.device)
        visible.scatter_(-1, torch.where(indices >= 0, indices, compressed_len), True)
        return visible[..., :compressed_len]
    if candidates.range_starts is None:
        return torch.zeros(1, seq_len, compressed_len, dtype=torch.bool)
    entries = torch.arange(compressed_len, device=candidates.range_starts.device)
    starts = candidates.range_starts.reshape(-1, seq_len, 1)
    ends = candidates.range_ends.reshape(-1, seq_len, 1)
    return (entries >= starts) & (entries < ends)


def _window_starts(rate: int, seq_len: int, sample_slices) -> torch.Tensor:
    """The global window starts, spelled out rather than taken from the helper under test."""
    if sample_slices is None:
        return torch.arange(0, seq_len - rate + 1, rate)
    return torch.tensor(
        [start for begin, end in sample_slices for start in range(begin, end - rate + 1, rate)],
        dtype=torch.long,
    )


def _run_compressor_cp(
    rank: int,
    world_size: int,
    init_file: str,
    kind: str,
    seq_len: int,
    sample_slices,
    real_indexer: bool = False,
    *,
    init_dist: bool = True,
) -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    device_type = get_device_type()
    if init_dist:
        get_torch_device().set_device(rank)
        _use_full_float32_matmuls()
        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata
    from veomni.ops.config import set_ops_config

    cp_state = _init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type, name="cp")
    dp_state = _init_parallel_state(dp_size=world_size, cp_size=1, ulysses_size=1, device_type=device_type, name="dp")
    set_ops_config(SimpleNamespace(dsa_attention_implementation="eager", dsa_indexer_implementation="eager"))

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    compressor_class = dsv4.DeepseekV4HCACompressor if kind == "hca" else dsv4.DeepseekV4CSACompressor
    compressor = compressor_class(config).to(device=device_type, dtype=torch.float32)
    _init_position_bias(compressor)
    if kind == "csa" and not real_indexer:
        compressor.indexer = _FixedIndexer(config.index_topk)
    _broadcast_module(compressor)
    compressor.train()

    rate = compressor.compress_rate
    full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type)
    dist.broadcast(full_hidden, src=0)
    q_residual = (
        torch.randn(1, seq_len, config.q_lora_rank, device=device_type)
        if real_indexer
        else torch.zeros(1, seq_len, config.q_lora_rank, device=device_type)
    )
    dist.broadcast(q_residual, src=0)

    if sample_slices is None:
        full_position_ids = torch.arange(seq_len, device=device_type).view(1, -1)
        packed_kwargs = {}
    else:
        full_position_ids = torch.cat(
            [torch.arange(end - begin, device=device_type) for begin, end in sample_slices]
        ).view(1, -1)
        packed_kwargs = {
            "packed_sequence_slices": sample_slices,
            "packed_compression_metadata": build_packed_compression_metadata(
                full_hidden,
                full_position_ids,
                sample_slices,
                (rate,),
            ),
        }

    def _forward(hidden, positions, residual):
        return compressor(
            hidden_states=hidden,
            q_residual=residual,
            position_ids=positions,
            past_key_values=None,
            layer_idx=0,
            **packed_kwargs,
        )

    with use_parallel_state(dp_state):
        baseline_input = full_hidden.detach().clone().requires_grad_(True)
        baseline_kv, baseline_candidates = _forward(baseline_input, full_position_ids, q_residual)
        baseline_kv.sum().backward()
        baseline_grads = {name: _grad_or_zeros(param) for name, param in compressor.named_parameters()}
        baseline_kv = baseline_kv.detach()
        baseline_input_grad = baseline_input.grad.detach().clone()
        compressor.zero_grad(set_to_none=True)

    local_len = seq_len // world_size
    begin = rank * local_len
    local_input = full_hidden[:, begin : begin + local_len].detach().clone().requires_grad_(True)
    with use_parallel_state(cp_state):
        local_kv, local_candidates = _forward(
            local_input,
            full_position_ids[:, begin : begin + local_len],
            q_residual[:, begin : begin + local_len],
        )

        owned = (_window_starts(rate, seq_len, sample_slices) // local_len) == rank
        local_kv[:, :, owned.to(local_kv.device)].sum().backward()
    summed_grads = {}
    for name, param in compressor.named_parameters():
        summed = _grad_or_zeros(param)
        dist.all_reduce(summed)
        summed_grads[name] = summed

    torch.testing.assert_close(local_kv.detach(), baseline_kv, rtol=1e-4, atol=1e-4)
    compressed_len = baseline_kv.shape[2]
    torch.testing.assert_close(
        _visible_compressed(local_candidates, local_len, compressed_len),
        _visible_compressed(baseline_candidates, seq_len, compressed_len)[:, begin : begin + local_len],
    )
    torch.testing.assert_close(
        local_input.grad, baseline_input_grad[:, begin : begin + local_len], rtol=1e-4, atol=1e-4
    )
    for name, summed in summed_grads.items():
        torch.testing.assert_close(summed, baseline_grads[name], rtol=1e-4, atol=1e-4)

    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


# Packed samples for the indexer, misaligned so that a window straddles a shard
# boundary at both cp_size 2 and 4 (rate 4, ``seq_len`` 256):
#
#   * sample 1 starts at 0, so its window starts are multiples of the rate and
#     never straddle a shard edge, which is a multiple of 64 either way;
#   * sample 2 starts at 106, so its starts are 106 + 4k. The window at 126
#     covers 126..129 and crosses the cp_size=2 edge at 128, and the window at
#     190 covers 190..193 and crosses the cp_size=4 edge at 192.
#
# 63 windows in all, against ``index_topk`` 32, so the selection is a real
# ranking rather than "every slot that is causally visible".
_PACKED_INDEXER_SAMPLES = ((0, 106), (106, 256))


def _run_indexer_cp(rank: int, world_size: int, init_file: str, seq_len: int, *, init_dist: bool = True) -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    device_type = get_device_type()
    if init_dist:
        get_torch_device().set_device(rank)
        _use_full_float32_matmuls()
        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata
    from veomni.ops import resolve_op
    from veomni.ops.config import set_ops_config

    cp_state = _init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type, name="cp")
    dp_state = _init_parallel_state(dp_size=world_size, cp_size=1, ulysses_size=1, device_type=device_type, name="dp")
    set_ops_config(SimpleNamespace(dsa_indexer_implementation="tilelang"))

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    indexer = dsv4.DeepseekV4Indexer(config).to(device=device_type, dtype=torch.bfloat16)
    _init_position_bias(indexer)
    _broadcast_module(indexer)

    hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type, dtype=torch.bfloat16)
    q_residual = torch.randn(1, seq_len, config.q_lora_rank, device=device_type, dtype=torch.bfloat16)
    dist.broadcast(hidden, src=0)
    dist.broadcast(q_residual, src=0)

    kernel_runs = []
    indexer_entry = resolve_op("dsa_indexer", "deepseek_v4", "tilelang")
    real_kernel = indexer_entry.wrapper

    def _counting_kernel(*args, **kwargs):
        kernel_runs.append(None)
        return real_kernel(*args, **kwargs)

    local_len = seq_len // world_size
    begin = rank * local_len
    compared = []
    for sample_slices in (None, _PACKED_INDEXER_SAMPLES):
        if sample_slices is None:
            position_ids = torch.arange(seq_len, device=device_type).view(1, -1)
            packed_kwargs = {}
        else:
            position_ids = torch.cat(
                [torch.arange(end - start, device=device_type) for start, end in sample_slices]
            ).view(1, -1)
            packed_kwargs = {
                "packed_sequence_slices": sample_slices,
                "packed_compression_metadata": build_packed_compression_metadata(
                    hidden, position_ids, sample_slices, (indexer.compress_rate,)
                ),
            }

        kernel_runs.clear()
        with patch.object(indexer_entry, "wrapper", _counting_kernel):
            with use_parallel_state(dp_state):
                baseline = indexer(hidden, q_residual, position_ids, None, 0, **packed_kwargs)

            with use_parallel_state(cp_state):
                local = indexer(
                    hidden[:, begin : begin + local_len],
                    q_residual[:, begin : begin + local_len],
                    position_ids[:, begin : begin + local_len],
                    None,
                    0,
                    **packed_kwargs,
                )
        compared.append((local, baseline[:, begin : begin + local_len], len(kernel_runs)))

    for local, expected, kernel_run_count in compared:
        assert kernel_run_count == 2, (
            f"expected the TileLang scorer on both the baseline and the shard, ran {kernel_run_count} time(s)"
        )
        torch.testing.assert_close(local.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0)

    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


class _UnusableGroup:
    """A truthy stand-in for a CP process group that no collective can use.

    ``cp_group=None`` would be worse than useless here: ``gather_outputs``
    resolves a ``None`` group to the global SP group, which is also ``None`` in a
    single-process test, and then returns its input unchanged at
    ``if not group: return x``. The KV all-gather would silently identity-pass and
    a guard moved after it would go unnoticed. This object is truthy, so it gets
    past that check, and has none of a process group's methods, so reaching the
    collective raises ``ValueError: Default process group has not been
    initialized`` instead. That is what lets the tests below pin each guard ahead
    of the all-gather rather than merely somewhere in the forward.
    """


def _cp_state(cp_size: int = 2, cp_rank: int = 0) -> SimpleNamespace:
    """A parallel state claiming CP, with a group that fails loudly if used."""
    return SimpleNamespace(
        ulysses_enabled=False,
        cp_enabled=True,
        cp_group=_UnusableGroup(),
        cp_rank=cp_rank,
        cp_size=cp_size,
    )


def _ulysses_state(ulysses_size: int = 2) -> SimpleNamespace:
    """The other sequence-parallel mode, likewise unusable if a collective is reached."""
    return SimpleNamespace(
        ulysses_enabled=True,
        cp_enabled=False,
        ulysses_group=_UnusableGroup(),
        ulysses_rank=0,
        ulysses_size=ulysses_size,
    )


def _build_local_attention(with_compressor: bool, local_len: int, cp_size: int, layer_idx: int = 0):
    """One rank's fixture on CPU: a local shard plus the full-sequence mask CP requires.

    ``local_len`` must be at least the compressor's rate, which is the halo width,
    so that a guard moved after the halo exchange is caught by the unusable group
    rather than by the compressor's own shard-width check.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    # Layer 0 is HCA and layer 3 is CSA on the toy config; there is no
    # sliding-only layer type, so dropping the compressor is what makes one.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=layer_idx)
    if not with_compressor:
        layer.compressor = None
    _init_attention_parameters(layer)

    hidden = torch.randn(1, local_len, config.hidden_size)
    position_ids = torch.arange(local_len).view(1, -1)
    full_mask = _build_causal_mask(local_len * cp_size, config.sliding_window, "cpu", torch.float32)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config)
    return config, _make_forward(layer, rotary), hidden, position_ids, full_mask


@pytest.mark.parametrize("with_compressor,layer_idx", [(False, 0), (True, 0), (True, 3)])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_attention_fixture_initializes_empty_parameters(monkeypatch, with_compressor, layer_idx):
    """Recycled allocator contents must not decide whether CP fixtures produce NaNs."""
    original_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        return tensor

    with monkeypatch.context() as patcher:
        patcher.setattr(torch, "empty", poisoned_empty)
        _, forward, hidden, positions, mask = _build_local_attention(
            with_compressor, local_len=128, cp_size=1, layer_idx=layer_idx
        )
    hidden.requires_grad_(True)
    no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
        output = forward(hidden, positions, mask)
        assert torch.isfinite(output).all()
        output.sum().backward()
    assert torch.isfinite(hidden.grad).all()


# The three modules that compress windows, and the role each names in its
# refusal. The role is asserted as well as the shared message because a CSA
# compressor whose own guard regressed would still reach its indexer's, and
# "some guard fired" is not what these cases are for.
_WINDOW_COMPRESSORS = {
    "hca": ("DeepseekV4HCACompressor", "DeepSeek V4 HCA compressor"),
    "csa": ("DeepseekV4CSACompressor", "DeepSeek V4 CSA compressor"),
    "indexer": ("DeepseekV4Indexer", "DeepSeek V4 Lightning Indexer"),
}


@pytest.mark.parametrize("kind", list(_WINDOW_COMPRESSORS))
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_deepseek_v4_cp_rejects_a_narrow_shard(kind):
    """A halo comes from one neighbour, so a sub-rate shard cannot work.

    Pinned with the unusable group, which proves the check runs *before* the halo
    exchange rather than merely somewhere in the forward. A rank that raised after
    entering a collective would leave its peers stuck in it -- and since ``rate``
    and the shard width are the same on every rank, either all of them raise here
    or none does, which is the property that keeps this a clean error instead of
    a ten-minute watchdog timeout.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    class_name, role = _WINDOW_COMPRESSORS[kind]
    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    module = getattr(dsv4, class_name)(config)
    local_len = module.compress_rate - 1
    hidden = torch.randn(1, local_len, config.hidden_size)
    q_residual = torch.randn(1, local_len, config.q_lora_rank)
    position_ids = torch.arange(local_len).view(1, -1)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(ValueError, match=f"{role} needs shards at least one compression window wide"):
            module(hidden, q_residual, position_ids, None, 0)


@pytest.mark.parametrize("with_compressor", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_deepseek_v4_attention_cp_rejects_a_kv_cache(with_compressor):
    """Decode would append this rank's shard to a cache the other ranks also gather."""
    from transformers import DynamicCache

    config, forward, hidden, position_ids, full_mask = _build_local_attention(
        with_compressor=with_compressor, local_len=32, cp_size=2
    )
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(NotImplementedError, match="KV cache"):
            forward(hidden, position_ids, full_mask, past_key_values=DynamicCache(config=config))


@pytest.mark.parametrize("with_compressor", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_deepseek_v4_attention_cp_rejects_a_local_length_mask(with_compressor):
    """A shard-width mask would let rank 0 attend everywhere uncaused and later ranks time out."""
    config, forward, hidden, position_ids, _ = _build_local_attention(
        with_compressor=with_compressor, local_len=32, cp_size=2
    )
    local_mask = _build_causal_mask(hidden.shape[1], config.sliding_window, "cpu", torch.float32)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(ValueError, match="full sequence"):
            forward(hidden, position_ids, local_mask)


def _build_toy_model(seq_len: int):
    """A whole toy model on CPU plus one batch of ids, for the model-forward guards."""
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    model = dsv4.DeepseekV4Model(config)
    _init_every_position_bias(model)
    model.eval()
    return model, torch.randint(0, config.vocab_size, (1, seq_len))


@pytest.mark.parametrize("sp_state", [_cp_state, _ulysses_state], ids=["cp", "ulysses"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_deepseek_v4_model_sp_rejects_absent_position_ids(sp_state):
    """A shard cannot invent global positions, so the forward refuses to guess them.

    ``arange(inputs_embeds.shape[1])`` would tell every rank that its tokens
    start at position 0, while ``shard_packed_compression_metadata``, the
    attention forward and the indexer all read ``position_ids`` as global. The
    result would be silently wrong rather than an error: shapes stay
    self-consistent, the sliding-window mask is built from nonsense positions,
    and the indexer's canonical-position check admits the TileLang kernel on rank
    0 alone, so the ranks disagree about causality. Both modes, because the
    fabrication is equally wrong under Ulysses and the guard is one branch.

    Pinned with the unusable group, which proves the refusal lands ahead of the
    ``position_ids`` all-gather rather than merely somewhere in the forward.
    """
    model, input_ids = _build_toy_model(seq_len=32)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=sp_state()):
        with pytest.raises(ValueError, match="requires explicit position_ids"):
            model(input_ids=input_ids)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_deepseek_v4_model_without_sp_still_defaults_position_ids():
    """With sequence parallelism off this rank holds the whole sequence, so ``arange`` is right."""
    model, input_ids = _build_toy_model(seq_len=32)
    no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state), torch.no_grad():
        defaulted = model(input_ids=input_ids).last_hidden_state
        explicit = model(
            input_ids=input_ids, position_ids=torch.arange(input_ids.shape[1]).view(1, -1)
        ).last_hidden_state
    torch.testing.assert_close(defaulted, explicit, rtol=0, atol=0)


_PACKED_SLIDING_SAMPLES = ((0, 70), (70, 128))
_PACKED_COMPRESSOR_FIXTURES = {
    ("csa", 4): (64, ((0, 38), (38, 64))),
    ("csa", 2): (64, ((0, 30), (30, 64))),
    ("hca", 4): (128, ((0, 70), (70, 128))),
    ("hca", 2): (128, ((0, 40), (40, 128))),
}
_ZERO_WINDOW_UNPACKED_SEQ_LEN = {"hca": 80, "csa": 10}
_ZERO_WINDOW_PACKED_SEQ_LEN = 16
_ZERO_WINDOW_PACKED_SAMPLES = ((0, 6), (6, 16))
_PACKED_MODEL_SEQ_LEN = 128
_PACKED_MODEL_SAMPLES = ((0, 70), (70, 128))


def _run_model_cp_packed(
    rank: int, world_size: int, init_file: str, dtype: torch.dtype, tilelang: bool, *, init_dist: bool = True
) -> None:
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    device_type = get_device_type()
    if init_dist:
        get_torch_device().set_device(rank)
        _use_full_float32_matmuls()
        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.ops.config import set_ops_config

    cp_state = _init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type, name="cp")
    dp_state = _init_parallel_state(dp_size=world_size, cp_size=1, ulysses_size=1, device_type=device_type, name="dp")
    if tilelang:
        set_ops_config(SimpleNamespace(dsa_attention_implementation="tilelang", dsa_indexer_implementation="tilelang"))
    else:
        set_ops_config(SimpleNamespace(dsa_attention_implementation="eager", dsa_indexer_implementation="eager"))

    config = _load_dsv4_toy_config()
    torch.manual_seed(0)
    model = dsv4.DeepseekV4Model(config).to(device=device_type, dtype=dtype)
    _init_every_position_bias(model)
    _broadcast_module(model)
    model.train()

    seq_len = _PACKED_MODEL_SEQ_LEN
    cu_seq_lens = torch.tensor([0, *(end for _, end in _PACKED_MODEL_SAMPLES)], device=device_type, dtype=torch.int32)
    position_ids = torch.cat(
        [torch.arange(end - begin, device=device_type) for begin, end in _PACKED_MODEL_SAMPLES]
    ).view(1, seq_len)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device_type)
    dist.broadcast(input_ids, src=0)

    counts = dict.fromkeys(("sparse_attn_tilelang", "v4_lighting_indexer"), 0)

    def _counted(name, real):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return wrapper

    def _forward(ids, positions):
        return model(input_ids=ids, position_ids=positions, cu_seq_lens_q=cu_seq_lens).last_hidden_state

    local_len = seq_len // world_size
    begin = rank * local_len
    with ExitStack() as stack:
        if tilelang:
            from veomni.ops import resolve_op

            attention_entry = resolve_op("dsa_attention", "deepseek_v4", "tilelang")
            indexer_entry = resolve_op("dsa_indexer", "deepseek_v4", "tilelang")
            stack.enter_context(
                patch.object(
                    attention_entry,
                    "wrapper",
                    _counted("sparse_attn_tilelang", attention_entry.wrapper),
                )
            )
            stack.enter_context(
                patch.object(
                    indexer_entry,
                    "wrapper",
                    _counted("v4_lighting_indexer", indexer_entry.wrapper),
                )
            )
        with use_parallel_state(dp_state):
            baseline = _forward(input_ids, position_ids)
            baseline.sum().backward()
            baseline_grads = {name: _grad_or_zeros(param) for name, param in model.named_parameters()}
            baseline = baseline.detach()
            model.zero_grad(set_to_none=True)

        with use_parallel_state(cp_state):
            local = _forward(input_ids[:, begin : begin + local_len], position_ids[:, begin : begin + local_len])
            local.sum().backward()

    summed_grads = {}
    for name, param in model.named_parameters():
        summed = _grad_or_zeros(param)
        dist.all_reduce(summed)
        summed_grads[name] = summed

    expected_layers = config.num_hidden_layers if tilelang else 0
    expected_csa = sum(kind == "compressed_sparse_attention" for kind in config.layer_types) if tilelang else 0
    assert counts["sparse_attn_tilelang"] == 2 * expected_layers, counts
    assert counts["v4_lighting_indexer"] == 2 * expected_csa, counts
    forward_tol, grad_tol = (1e-4, 1e-4) if dtype == torch.float32 else (2**-6, 1e-1)
    _assert_close_to_scale(local.detach(), baseline[:, begin : begin + local_len], forward_tol, "forward")
    for name, summed in summed_grads.items():
        _assert_close_to_scale(summed, baseline_grads[name], grad_tol, name)

    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def _run_cp_collator_contract(rank: int, world_size: int, init_file: str, *, init_dist: bool = True) -> None:
    from veomni.data.data_collator import SequenceParallelCollator
    from veomni.distributed.parallel_state import (
        _init_parallel_state,
        clear_parallel_state,
        get_parallel_state,
        use_parallel_state,
    )

    device_type = get_device_type()
    if init_dist:
        get_torch_device().set_device(rank)
        _use_full_float32_matmuls()
        dist.init_process_group(
            backend=get_dist_comm_backend(),
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
    cp_state = _init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type, name="cp")

    with use_parallel_state(cp_state):
        state = get_parallel_state()
        assert (state.sp_size, state.sp_rank) == (state.cp_size, state.cp_rank), (
            f"CP-only sp/cp mismatch: sp=({state.sp_size}, {state.sp_rank}) cp=({state.cp_size}, {state.cp_rank})"
        )
        assert (state.cp_size, state.cp_rank) == (world_size, rank), (
            f"expected cp=({world_size}, {rank}), got ({state.cp_size}, {state.cp_rank})"
        )

        seq_len = _PACKED_MODEL_SEQ_LEN
        input_ids = torch.arange(seq_len).view(1, seq_len)
        position_ids = torch.cat(
            [torch.arange(end - begin) for begin, end in _PACKED_MODEL_SAMPLES],
        ).view(1, seq_len)
        collated = SequenceParallelCollator()(
            {
                "input_ids": input_ids.clone(),
                "labels": input_ids.clone(),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "position_ids": position_ids.clone(),
            }
        )

        local_len = seq_len // world_size
        begin = rank * local_len
        expected_ids = input_ids[:, begin : begin + local_len]
        assert torch.equal(collated["input_ids"], expected_ids), (
            f"rank {rank} expected rows [{begin}, {begin + local_len}) "
            f"{expected_ids.tolist()}, got {collated['input_ids'].tolist()}"
        )
        expected_positions = position_ids[:, begin : begin + local_len]
        assert torch.equal(collated["position_ids"], expected_positions), (
            f"rank {rank} position_ids must stay global: expected {expected_positions.tolist()}, "
            f"got {collated['position_ids'].tolist()}"
        )
        expected_cu_seq_lens = [0, *(end for _, end in _PACKED_MODEL_SAMPLES)]
        assert collated["cu_seq_lens_q"].tolist() == expected_cu_seq_lens, (
            f"rank {rank} cu_seq_lens_q must span the packed batch: expected {expected_cu_seq_lens}, "
            f"got {collated['cu_seq_lens_q'].tolist()}"
        )
        assert collated["attention_mask"].shape[-1] == seq_len, (
            f"rank {rank} attention_mask must stay full length {seq_len}, got {tuple(collated['attention_mask'].shape)}"
        )

    dist.barrier()
    if init_dist:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def _run_all_cp_scenarios(rank: int, world_size: int, init_file: str) -> None:
    """Fused CP worker running every DeepSeek-V4 CP sub-scenario for ``world_size``."""
    from veomni.distributed.parallel_state import clear_parallel_state

    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    scenarios = [
        lambda: _run_attention_cp(rank, world_size, init_file, 64, False, init_dist=False),
        lambda: _run_attention_cp(
            rank, world_size, init_file, 128, False, 0, 1, _PACKED_SLIDING_SAMPLES, init_dist=False
        ),
        lambda: _run_attention_cp_sparse_indices(rank, world_size, init_file, 64, init_dist=False),
        lambda: _run_attention_cp(rank, world_size, init_file, 128, True, init_dist=False),
        lambda: _run_compressor_cp(rank, world_size, init_file, "hca", 128, None, init_dist=False),
        lambda: _run_compressor_cp(rank, world_size, init_file, "csa", 128, None, init_dist=False),
        lambda: _run_compressor_cp(
            rank,
            world_size,
            init_file,
            "hca",
            _PACKED_COMPRESSOR_FIXTURES[("hca", world_size)][0],
            _PACKED_COMPRESSOR_FIXTURES[("hca", world_size)][1],
            init_dist=False,
        ),
        lambda: _run_compressor_cp(
            rank,
            world_size,
            init_file,
            "csa",
            _PACKED_COMPRESSOR_FIXTURES[("csa", world_size)][0],
            _PACKED_COMPRESSOR_FIXTURES[("csa", world_size)][1],
            init_dist=False,
        ),
        lambda: _run_attention_cp(rank, world_size, init_file, 128, True, 3, init_dist=False),
    ]
    if _all_devices_sm90_or_above():
        scenarios.append(lambda: _run_indexer_cp(rank, world_size, init_file, 256, init_dist=False))

    if world_size == 2:
        scenarios.extend(
            [
                lambda: _run_attention_cp(rank, 2, init_file, 128, True, 0, 2, init_dist=False),
                lambda: _run_compressor_cp(
                    rank, 2, init_file, "hca", _ZERO_WINDOW_UNPACKED_SEQ_LEN["hca"], None, init_dist=False
                ),
                lambda: _run_compressor_cp(
                    rank, 2, init_file, "csa", _ZERO_WINDOW_UNPACKED_SEQ_LEN["csa"], None, init_dist=False
                ),
            ]
        )
    elif world_size == 4:
        scenarios.extend(
            [
                lambda: _run_compressor_cp(
                    rank,
                    4,
                    init_file,
                    "csa",
                    _ZERO_WINDOW_PACKED_SEQ_LEN,
                    _ZERO_WINDOW_PACKED_SAMPLES,
                    True,
                    init_dist=False,
                ),
                lambda: _run_cp_collator_contract(rank, 4, init_file, init_dist=False),
                lambda: _run_model_cp_packed(rank, 4, init_file, torch.float32, False, init_dist=False),
                lambda: _run_attention_cp(rank, 4, init_file, 68, False, init_dist=False),
                lambda: _run_attention_cp(rank, 4, init_file, 160, True, init_dist=False),
            ]
        )
        if _all_devices_sm90_or_above():
            scenarios.append(lambda: _run_model_cp_packed(rank, 4, init_file, torch.bfloat16, True, init_dist=False))

    if os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1":
        scenarios = list(reversed(scenarios))

    try:
        for fn in scenarios:
            fn()
        dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


@pytest.mark.launch_plan(provider=from_parameter("cp_size", device="accelerator"))
@pytest.mark.skipif(_cuda_device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_cp_equivalence(cp_size):
    """Fused CP worker covering all DeepSeek-V4 CP attention, compressor, indexer, and model parity scenarios."""
    if _cuda_device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_all_cp_scenarios,
            args=(cp_size, init_file),
            nprocs=cp_size,
            join=True,
        )
