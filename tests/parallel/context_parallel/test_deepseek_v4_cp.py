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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device, is_sm90_or_above


_PATCHED_MODULE = "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu"


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
):
    """Enter the process group, build the shared layer, and return the fixture.

    ``batch_size`` widens the hidden states, the position ids *and* the mask
    together. It has to be all three: the compressor's ``block_bias`` carries the
    batch dimension and is concatenated onto the mask, so a batch-1 mask beside
    batch-2 hidden states fails on that concatenation instead of on whatever the
    case was written to exercise.

    ``sample_slices`` packs several sequences into the batch row: positions
    restart at each boundary and the mask stops attention crossing one. Without a
    compressor those two are the *whole* of what packing means to this layer --
    the packed kwargs go only to the compressor -- so they are not passed, and
    what the case then covers is the mask narrowing at ``query_offset`` and a
    sliding window reaching back across a shard edge inside one sample.

    ``dtype`` is float32 for the parity cases, which need the headroom to compare
    against a single-rank baseline at ``1e-4``. Only the compact-candidate case
    overrides it, because that path is gated on bfloat16.
    """
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from transformers import AutoConfig

    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type)

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    # Layer 0 is HCA and layer 3 is CSA on the toy config, which has no
    # sliding-only layer type, so drop the compressor the way the Ulysses test
    # does to reach pure sliding MQA.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=layer_idx).to(device=device_type, dtype=dtype)
    if not with_compressor:
        layer.compressor = None
    else:
        _init_position_bias(layer.compressor)
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

    return dsv4, layer, _make_forward(layer, rotary), full_hidden, full_position_ids, full_mask


def _run_attention_cp(
    rank: int,
    world_size: int,
    init_file: str,
    seq_len: int,
    with_compressor: bool,
    layer_idx: int = 0,
    batch_size: int = 1,
    sample_slices=None,
) -> None:
    from veomni.distributed.parallel_state import clear_parallel_state

    _, layer, _forward, full_hidden, full_position_ids, full_mask = _init_cp_attention(
        rank, world_size, init_file, seq_len, with_compressor, layer_idx, batch_size, sample_slices
    )

    # Baseline: whole sequence with the parallel state stubbed out. Re-initialising
    # the state here would contradict the world size the process group was built for.
    no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
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
    local_output = _forward(local_input, full_position_ids[:, begin : begin + local_len], full_mask)
    local_output.sum().backward()

    summed_grads = {}
    for name, param in layer.named_parameters():
        if param.grad is None:
            continue
        summed = param.grad.detach().clone()
        dist.all_reduce(summed)
        summed_grads[name] = summed

    # Every collective is behind us, so a mismatch below fails on all ranks at
    # once instead of leaving the ones that passed waiting in the next one. The
    # asserts used to sit above and inside this loop, which turned any failure
    # into a ten-minute NCCL watchdog timeout with the real message buried under
    # it -- the most expensive failure mode this suite has.
    torch.testing.assert_close(local_output, baseline[:, begin : begin + local_len], rtol=1e-4, atol=1e-4)
    # The KV all-gather's backward sum-reduces before slicing, so the input grad
    # already carries every other rank's contribution and must not be reduced again.
    torch.testing.assert_close(
        local_input.grad, baseline_input_grad[:, begin : begin + local_len], rtol=1e-4, atol=1e-4
    )
    for name, summed in summed_grads.items():
        torch.testing.assert_close(summed, baseline_grads[name], rtol=1e-4, atol=1e-4)

    clear_parallel_state()
    dist.destroy_process_group()


def _stub_sparse_attn_tilelang(query: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    """Stand in for the TileLang kernel, which needs SM90 and is not asserted on here.

    The caller only wants the candidate list the forward built on its way to the
    kernel, so running the kernel would buy nothing but a hardware requirement --
    and this is the one compact-candidate case that fits on the two GPUs CI has.
    ``test_deepseek_v4_model_cp_packed_misaligned_tilelang`` covers the real
    kernel. The shape follows the caller, which transposes to ``[b, s, h, d]``
    before calling because the kernel is layout-specific.
    """
    return torch.zeros_like(query)


def _run_attention_cp_sparse_indices(rank: int, world_size: int, init_file: str, seq_len: int) -> None:
    """The compact candidates a shard builds must be the global build's own rows.

    The eager path ignores ``sparse_topk_indices``, so requesting TileLang is the
    only thing that pins down ``query_offset`` / ``kv_full_len`` /
    ``compressed_len``. Get them wrong and the TileLang kernel silently reads the
    wrong KV rows.

    Unlike the rest of the file this fixture is bfloat16, and it stays bfloat16
    even though the kernel is stubbed out: the attention refuses to build a
    compact list unless the kernel could consume it, so a float32 fixture reaches
    the builder not at all. The assertion is unaffected -- candidates are integer
    indices, compared exactly.
    """
    from veomni.distributed.parallel_state import clear_parallel_state

    dsv4, _, _forward, full_hidden, full_position_ids, full_mask = _init_cp_attention(
        rank, world_size, init_file, seq_len, with_compressor=False, dtype=torch.bfloat16
    )
    dsv4.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))

    built = []
    build_indices = dsv4.build_sparse_attention_indices

    def _record(**kwargs):
        indices = build_indices(**kwargs)
        built.append(indices)
        return indices

    local_len = seq_len // world_size
    begin = rank * local_len
    with (
        torch.no_grad(),
        patch(f"{_PATCHED_MODULE}.build_sparse_attention_indices", _record),
        patch(f"{_PATCHED_MODULE}.sparse_attn_tilelang", _stub_sparse_attn_tilelang),
    ):
        no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
        with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
            _forward(full_hidden, full_position_ids, full_mask)
        # Unpacking asserts the forward built exactly one candidate list.
        (baseline_indices,) = built
        built.clear()

        _forward(
            full_hidden[:, begin : begin + local_len],
            full_position_ids[:, begin : begin + local_len],
            full_mask,
        )
        (local_indices,) = built

    torch.testing.assert_close(local_indices, baseline_indices[:, begin : begin + local_len], rtol=0, atol=0)

    clear_parallel_state()
    dist.destroy_process_group()


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
) -> None:
    """A CP shard's compressor must return the whole globally-ordered compressed KV.

    Every rank compresses only the windows it owns and then all-gathers, so the
    tensor compared here is the *full* baseline result, not a slice of it: a
    dropped or misordered window shows up directly. The backward reduces over
    this rank's own windows only, because summing the replicated array on every
    rank would scale the gradient by ``cp_size``.

    ``real_indexer`` keeps the CSA compressor's own Lightning Indexer instead of
    the fixed stand-in. The default is the stand-in because the two summarise the
    same windows and a real one would let an indexer bug masquerade as a
    compressor bug. The zero-window case wants the opposite: the indexer carries
    its *own* copy of the empty-result construction and its own halo exchange and
    row all-gather, and running it is the only way to drive them. Its selection
    is then observable through ``block_bias``, which is scattered from it.
    """
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    device_type = get_device_type()
    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type)

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
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
    # A real indexer scores against ``q_residual``; zeros would make every query
    # identical and the selection arbitrary.
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
                # The HCA path reads its block bias straight out of the metadata,
                # so this is what makes the query-row slicing observable.
                block_bias_rates=(rate,) if kind == "hca" else (),
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

    no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
        baseline_input = full_hidden.detach().clone().requires_grad_(True)
        baseline_kv, baseline_bias = _forward(baseline_input, full_position_ids, q_residual)
        baseline_kv.sum().backward()
        baseline_grads = {name: _grad_or_zeros(param) for name, param in compressor.named_parameters()}
        baseline_kv = baseline_kv.detach()
        baseline_bias = None if baseline_bias is None else baseline_bias.detach()
        baseline_input_grad = baseline_input.grad.detach().clone()
        compressor.zero_grad(set_to_none=True)

    local_len = seq_len // world_size
    begin = rank * local_len
    local_input = full_hidden[:, begin : begin + local_len].detach().clone().requires_grad_(True)
    local_kv, local_bias = _forward(
        local_input,
        full_position_ids[:, begin : begin + local_len],
        q_residual[:, begin : begin + local_len],
    )

    owned = (_window_starts(rate, seq_len, sample_slices) // local_len) == rank
    local_kv[:, :, owned.to(local_kv.device)].sum().backward()
    # A rank that owns no window never touches ``position_bias`` or ``kv_norm`` and
    # so has no gradient for them, while its peers do. Reducing a zero stand-in
    # instead of skipping keeps every rank in the same collectives.
    summed_grads = {}
    for name, param in compressor.named_parameters():
        summed = _grad_or_zeros(param)
        dist.all_reduce(summed)
        summed_grads[name] = summed

    # Every collective is behind us, so a mismatch below fails on all ranks at
    # once instead of leaving the ones that passed waiting in the next one.
    # Global window order is the whole point of the row all-gather, so compare
    # the entire replicated array rather than this rank's contribution to it.
    torch.testing.assert_close(local_kv.detach(), baseline_kv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        local_bias.detach(), baseline_bias[..., begin : begin + local_len, :], rtol=1e-4, atol=1e-4
    )
    # The gathers sum-reduce in backward, so the local input gradient already
    # carries the halo contributions every neighbour made to it.
    torch.testing.assert_close(
        local_input.grad, baseline_input_grad[:, begin : begin + local_len], rtol=1e-4, atol=1e-4
    )
    for name, summed in summed_grads.items():
        torch.testing.assert_close(summed, baseline_grads[name], rtol=1e-4, atol=1e-4)

    clear_parallel_state()
    dist.destroy_process_group()


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


def _run_indexer_cp(rank: int, world_size: int, init_file: str, seq_len: int) -> None:
    """A CP shard's Lightning Indexer picks the same slots the full forward gives its rows.

    The queries arrive already sharded, but the compressed keys the indexer scores
    them against must stay *global*: a top-k value names a slot in the CSA
    compressor's replicated compressed KV. So the indexer owns and all-gathers
    windows exactly as its enclosing compressor does, and only the query axis is
    local. The packed layout is what forces it to shard the compression metadata
    it is handed, which is global.
    """
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    device_type = get_device_type()
    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type)
    # The TileLang kernel is the production scorer and the only one with a query
    # partitioning of its own; the eager scorer is covered by the CSA layer test.
    dsv4.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    indexer = dsv4.DeepseekV4Indexer(config).to(device=device_type, dtype=torch.bfloat16)
    _init_position_bias(indexer)
    _broadcast_module(indexer)

    hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type, dtype=torch.bfloat16)
    q_residual = torch.randn(1, seq_len, config.q_lora_rank, device=device_type, dtype=torch.bfloat16)
    dist.broadcast(hidden, src=0)
    dist.broadcast(q_residual, src=0)

    # ``use_tilelang`` degrades to the eager scorer rather than failing when the
    # canonical positions it checks do not line up, and the eager scorer reads the
    # global ``position_ids`` and so stays right. Without this count, dropping the
    # query offset entirely would take the kernel out of the comparison and the
    # parity below would still hold, pinning nothing about the CP query rebasing.
    kernel_runs = []
    real_kernel = dsv4.v4_lighting_indexer

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
                # Global metadata alongside a local shard, which is exactly what
                # the CSA compressor hands over: only the indexer knows it is
                # looking at one shard, so only it can do the sharding.
                "packed_compression_metadata": build_packed_compression_metadata(
                    hidden, position_ids, sample_slices, (indexer.compress_rate,)
                ),
            }

        kernel_runs.clear()
        with patch(f"{_PATCHED_MODULE}.v4_lighting_indexer", _counting_kernel):
            no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
            with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
                baseline = indexer(hidden, q_residual, position_ids, None, 0, **packed_kwargs)

            local = indexer(
                hidden[:, begin : begin + local_len],
                q_residual[:, begin : begin + local_len],
                position_ids[:, begin : begin + local_len],
                None,
                0,
                **packed_kwargs,
            )
        compared.append((local, baseline[:, begin : begin + local_len], len(kernel_runs)))

    # Both layouts run their collectives before anything is asserted, so a
    # mismatch fails on every rank at once instead of leaving the ranks that
    # passed inside the next layout's halo exchange.
    for local, expected, kernel_run_count in compared:
        assert kernel_run_count == 2, (
            f"expected the TileLang scorer on both the baseline and the shard, ran {kernel_run_count} time(s)"
        )
        # Sorted, because top-k order follows the scores while it is the
        # selection that addresses the compressed KV. Exact, because these are
        # integer slot ids and any tolerance would hide an off-by-one in the
        # query offset.
        torch.testing.assert_close(local.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0)

    clear_parallel_state()
    dist.destroy_process_group()


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
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    # Layer 0 is HCA and layer 3 is CSA on the toy config; there is no
    # sliding-only layer type, so dropping the compressor is what makes one.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=layer_idx)
    if not with_compressor:
        layer.compressor = None

    hidden = torch.randn(1, local_len, config.hidden_size)
    position_ids = torch.arange(local_len).view(1, -1)
    full_mask = _build_causal_mask(local_len * cp_size, config.sliding_window, "cpu", torch.float32)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config)
    return config, _make_forward(layer, rotary), hidden, position_ids, full_mask


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
def test_deepseek_v4_cp_rejects_a_narrow_shard(kind):
    """A halo comes from one neighbour, so a sub-rate shard cannot work.

    Pinned with the unusable group, which proves the check runs *before* the halo
    exchange rather than merely somewhere in the forward. A rank that raised after
    entering a collective would leave its peers stuck in it -- and since ``rate``
    and the shard width are the same on every rank, either all of them raise here
    or none does, which is the property that keeps this a clean error instead of
    a ten-minute watchdog timeout.
    """
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    class_name, role = _WINDOW_COMPRESSORS[kind]
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
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
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    model = dsv4.DeepseekV4Model(config)
    _init_every_position_bias(model)
    model.eval()
    return model, torch.randint(0, config.vocab_size, (1, seq_len))


@pytest.mark.parametrize("sp_state", [_cp_state, _ulysses_state], ids=["cp", "ulysses"])
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


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_sliding_only(cp_size):
    """A CP shard's sliding-window attention matches the full-sequence forward and backward."""
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 64, False),
            nprocs=cp_size,
            join=True,
        )


# Packed samples for the compressor-free attention cases. Without a compressor
# the packed kwargs have nowhere to go, so what packing means to this layer is
# per-sample positions and a mask that refuses to cross a boundary:
#
#   * the boundary at 70 falls strictly inside a shard at both sizes -- rank 2's
#     [64, 96) at cp_size=4, rank 1's [64, 128) at cp_size=2 -- so in each case
#     one rank holds rows belonging to two different samples;
#   * the sliding window is 32, so the query at 70 sees only itself, while the
#     query at 96 reaches back to 65, across the cp_size=4 shard edge at 96 but
#     not across the sample boundary. Those are the rows where the mask
#     narrowing at ``query_offset`` and the per-sample isolation interact.
_PACKED_SLIDING_SAMPLES = ((0, 70), (70, 128))


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_packed_sliding_only(cp_size):
    """A CP shard of a packed batch matches the full forward and backward without a compressor.

    The parity criterion asks for packed and unpacked at both sizes with and
    without the compressor; this is the compressor-free half of the packed
    column. It is genuinely weaker than the compressor cases -- no window
    arithmetic, no halo, no compressed-row gather -- and what remains is the KV
    all-gather and the mask, which is the whole of the sliding-only CP path.
    """
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 128, False, 0, 1, _PACKED_SLIDING_SAMPLES),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_sparse_indices(cp_size):
    """A CP shard's compact sparse candidates match the full-sequence build's rows."""
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp_sparse_indices,
            args=(cp_size, init_file, 64),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_with_compressor(cp_size):
    """A whole HCA layer under CP matches the full-sequence forward and backward.

    Sequence length 128 keeps every rank holding at least one window at
    ``cp_size=4`` with the toy HCA rate of 32.
    """
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 128, True),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
def test_deepseek_v4_attention_cp_with_compressor_batch_2():
    """The same HCA layer at batch 2, which is where the KV gather's backward broke.

    A compressor gives the gathered ``kv`` exactly one consumer,
    ``torch.cat([kv, compressed_kv], dim=2)``, so ``_Gather.backward`` receives a
    narrowed view of the cat's gradient buffer. ``kv`` is ``[B, 1, S, D]``, whose
    leading dims collapse at batch 1 and make that view contiguous by accident;
    at batch 2 it is not, and the in-place ``dist.all_reduce`` rejects it with
    ``ValueError: Tensors must be contiguous``. Every real DeepSeek-V4 layer has
    a compressor, so without this CP runs at batch 1 only.

    One ``cp_size`` is enough: the view's stride does not depend on how many
    ranks contributed to the buffer it was narrowed out of.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(2, init_file, 128, True, 0, 2),
            nprocs=2,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("kind", ["hca", "csa"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_compressor_cp_unpacked(kind, cp_size):
    """Both compressors rebuild the global compressed KV from per-shard windows."""
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_compressor_cp,
            args=(cp_size, init_file, kind, 128, None),
            nprocs=cp_size,
            join=True,
        )


# Packed lengths that are deliberately not multiples of the compression rate, so
# that a window straddles a shard boundary, a sample boundary falls inside a
# shard, and the ranks own different numbers of windows. Aligned lengths
# exercise none of that. Keyed by ``cp_size`` as well as by rate, because a
# layout tuned to straddle the edges at one shard width need not straddle any at
# another -- the ``cp_size=4`` fixtures cross only at 48 and 96, both of which
# stop being boundaries at ``cp_size=2``.
#
# ``csa`` (rate 4, cp_size=4, L=16): starts
# [0,4,8,12,16,20,24,28,32,38,42,46,50,54,58], counts [4,4,4,3]. The window at
# 46 covers 46..49 and crosses the rank 2/3 edge at 48; rank 2 owns the window
# at 32 whose overlap half lives at 28..31, on rank 1.
#
# ``csa`` (rate 4, cp_size=2, L=32): sample 1 ends at 30, so its last window is
# 24..27 and its 28..29 tail is dropped; sample 2 restarts the stride at 30.
# Starts [0,4,..,24, 30,34,..,58], counts [8,7]. The window at 30 covers 30..33
# and crosses the single edge at 32, and rank 1's first owned window at 34 takes
# its overlap half from 30..33, which spans both ranks.
#
# ``hca`` (rate 32, cp_size=4, L=32): starts [0,32,70], counts [1,1,1,0]. The
# window at 70 covers 70..101 and crosses the rank 2/3 edge at 96, and rank 3
# owns nothing at all. The rate is the halo width, so the shard cannot be
# narrower than 32 here.
#
# ``hca`` (rate 32, cp_size=2, L=64): sample 1 holds one window and drops its
# 32..39 tail; sample 2 starts at 40. Starts [0,40,72], counts [2,1]. The window
# at 40 covers 40..71 and crosses the edge at 64.
_PACKED_COMPRESSOR_FIXTURES = {
    ("csa", 4): (64, ((0, 38), (38, 64))),
    ("csa", 2): (64, ((0, 30), (30, 64))),
    ("hca", 4): (128, ((0, 70), (70, 128))),
    ("hca", 2): (128, ((0, 40), (40, 128))),
}


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("kind", ["hca", "csa"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_compressor_cp_packed_straddling(kind, cp_size):
    """Windows that cross a shard boundary still land in the right global slot."""
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    seq_len, sample_slices = _PACKED_COMPRESSOR_FIXTURES[(kind, cp_size)]
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_compressor_cp,
            args=(cp_size, init_file, kind, seq_len, sample_slices),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
# The TileLang scorer is what this case is about -- it is the only one that
# partitions queries itself, and the run below counts kernel invocations to prove
# it was not quietly swapped for the eager scorer. So there is nothing left to
# assert once the kernel cannot run, and it needs SM90 or later.
@pytest.mark.skipif(not is_sm90_or_above(), reason="the DeepSeek-V4 TileLang kernels need SM90 or later")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_indexer_cp(cp_size):
    """A CP shard's indexer selection matches the full-query result, packed and unpacked."""
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_indexer_cp, args=(cp_size, init_file, 256), nprocs=cp_size, join=True)


# Shard lengths on which the *last* rank owns no compression window at all,
# unpacked. Global window starts run ``0, rate, 2*rate, ...`` only while a whole
# window still fits, so the tail of the sequence carries none: with ``cp_size=2``
# a rank owns nothing whenever no multiple of the rate lands in
# ``[L, 2L - rate]``.
#
#   * ``hca`` (rate 32, L=40): starts [0, 32], both rank 0's, since 64 + 32 > 80.
#   * ``csa`` (rate 4, L=5): starts [0, 4], both rank 0's, since 8 + 4 > 10.
#
# In both the last owned window runs past the shard edge -- 32..63 into rank 1's
# [40, 80), and 4..7 into rank 1's [5, 10) -- so the empty rank still supplies a
# right halo and still has to reach that exchange's backward.
_ZERO_WINDOW_UNPACKED_SEQ_LEN = {"hca": 80, "csa": 10}


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_deepseek_v4_compressor_cp_zero_windows_unpacked(kind):
    """A rank owning no window keeps its peers' backward collectives from hanging.

    This is the regression Task 6 found: the empty compression result has to stay
    attached to both ``kv`` and ``gate``, because a rank that leaves the autograd
    graph never enters the backward of the halo exchange and the row all-gather
    that its peers are blocked in. The parity assertions here are secondary --
    the load-bearing observation is that the backward completes at all -- so a
    regression shows up as a watchdog timeout rather than a mismatch.

    Unpacked, which is a distinct code path from the packed case below: the empty
    result is built in the compressor's own forward rather than inside
    ``compress_packed_windows``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_compressor_cp,
            args=(2, init_file, kind, _ZERO_WINDOW_UNPACKED_SEQ_LEN[kind], None),
            nprocs=2,
            join=True,
        )


# ``csa`` (rate 4, cp_size=4, L=4) with samples ((0, 6), (6, 16)): sample 1
# yields the window at 0 alone (4 + 4 > 6), sample 2 the windows at 6 and 10
# (14 + 4 > 16), so the global starts are [0, 6, 10] and the counts are
# [1, 1, 1, 0]. Rank 3 owns nothing; the window at 10 covers 10..13 and reaches
# two tokens into its shard, so it supplies rank 2's right halo. The window at 6
# straddles the rank 1/2 edge at 8 as well.
#
# L=4 is the toy CSA rate, which is the narrowest shard the guard admits. Nothing
# shorter can produce a zero-window rank at this rate: window starts are at most
# one rate apart inside a sample, so an empty shard needs a sample boundary
# inside it.
_ZERO_WINDOW_PACKED_SEQ_LEN = 16
_ZERO_WINDOW_PACKED_SAMPLES = ((0, 6), (6, 16))


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
def test_deepseek_v4_compressor_cp_zero_windows_packed_with_indexer():
    """The same, packed, with the CSA compressor's real Lightning Indexer attached.

    Three copies of the zero-window path run here at once: the CSA compressor's,
    the indexer's -- it owns no window either, since it compresses the same
    windows at its own head dim -- and ``compress_packed_windows``, which both
    reach on the packed path instead of their own forward.

    The indexer's copy is driven in the forward only, and that is a limit of the
    code rather than of the test: the indexer returns integer top-k indices, so
    nothing downstream of its compression is differentiable and its collectives
    have no backward on *any* rank. It cannot hang the way the compressors can.

    Only three compressed rows exist against ``index_topk`` 32, so every causally
    visible slot is selected and the block-bias comparison does not turn on top-k
    order.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_compressor_cp,
            args=(4, init_file, "csa", _ZERO_WINDOW_PACKED_SEQ_LEN, _ZERO_WINDOW_PACKED_SAMPLES, True),
            nprocs=4,
            join=True,
        )


# The packed fixture for the whole-model cases. ``seq_len`` 128 at cp_size=4
# gives ``L=32``, which is the toy HCA rate: a compressor's halo is one rate
# wide, so 128 is the shortest sequence that can run every layer of this config
# at cp_size=4. Samples ``((0, 70), (70, 128))`` are deliberately misaligned:
#
#   * the sample boundary at 70 falls strictly inside rank 2's shard [64, 96);
#   * HCA (rate 32) starts are [0, 32, 70], so the window at 70 covers 70..101
#     and straddles the rank 2/3 edge at 96 -- and rank 3 owns no HCA window at
#     all, which is the empty-result path;
#   * CSA (rate 4) starts are 0,4,..,64 then 70,74,..,122, so the window at 94
#     covers 94..97 and straddles the same edge from the other rate.
#
# The 31 CSA compressed rows are fewer than ``index_topk`` 32, so every causally
# visible slot is selected and the comparison does not turn on top-k *order*;
# ``test_deepseek_v4_indexer_cp`` is what covers the ranking.
_PACKED_MODEL_SEQ_LEN = 128
_PACKED_MODEL_SAMPLES = ((0, 70), (70, 128))


def _run_model_cp_packed(rank: int, world_size: int, init_file: str, dtype: torch.dtype, tilelang: bool) -> None:
    """A CP shard of a whole packed model matches the single-rank forward and backward.

    This is the end-to-end statement of the contract every layer beneath assumes:
    rank ``r`` receives rows ``[r*L, (r+1)*L)`` of the global sequence, their
    *global* ``position_ids``, and the *full-sequence* ``cu_seq_lens_q``. The
    model forward has to build its packed compression metadata against that
    global sequence rather than against the shard it can see, or every window and
    every per-query compressed range below it describes the wrong tokens.
    ``test_deepseek_v4_cp_collator_shards_contiguously_by_cp_rank`` pins that the
    collator really produces these three things.
    """
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    device_type = get_device_type()
    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type)
    if tilelang:
        # The model forward withholds the dense mask only for bf16 CUDA tensors
        # with the TileLang attention selected, so this is the arm that reaches
        # the compact-candidate path -- and the only place the TileLang indexer
        # runs inside a CSA layer under CP.
        dsv4.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))
        dsv4.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
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

    # Which scorer ran is not observable from the outputs: TileLang and eager
    # agree when both are reachable, and the model forward silently keeps the
    # dense mask when they are not. Counting the launches is what stops a parity
    # pass that never entered the kernel it exists to exercise.
    counts = dict.fromkeys(("sparse_attn_tilelang", "v4_lighting_indexer"), 0)

    def _counted(name):
        real = getattr(dsv4, name)

        def wrapper(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return wrapper

    def _forward(ids, positions):
        return model(input_ids=ids, position_ids=positions, cu_seq_lens_q=cu_seq_lens).last_hidden_state

    local_len = seq_len // world_size
    begin = rank * local_len
    with (
        patch(f"{_PATCHED_MODULE}.sparse_attn_tilelang", _counted("sparse_attn_tilelang")),
        patch(f"{_PATCHED_MODULE}.v4_lighting_indexer", _counted("v4_lighting_indexer")),
    ):
        # Baseline: the whole packed batch with the parallel state stubbed out.
        no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
        with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
            baseline = _forward(input_ids, position_ids)
            baseline.sum().backward()
            baseline_grads = {name: _grad_or_zeros(param) for name, param in model.named_parameters()}
            baseline = baseline.detach()
            model.zero_grad(set_to_none=True)

        # Context-parallel: this rank's contiguous shard beside the full-sequence
        # cu-seqlens, which is what the collator hands the model.
        local = _forward(input_ids[:, begin : begin + local_len], position_ids[:, begin : begin + local_len])
        local.sum().backward()

    summed_grads = {}
    for name, param in model.named_parameters():
        summed = _grad_or_zeros(param)
        dist.all_reduce(summed)
        summed_grads[name] = summed

    # Every collective is behind us, so a mismatch below fails on all ranks at
    # once instead of leaving the ones that passed waiting in the next one.
    expected_layers = config.num_hidden_layers if tilelang else 0
    expected_csa = sum(kind == "compressed_sparse_attention" for kind in config.layer_types) if tilelang else 0
    assert counts["sparse_attn_tilelang"] == 2 * expected_layers, counts
    assert counts["v4_lighting_indexer"] == 2 * expected_csa, counts
    forward_tol, grad_tol = (1e-4, 1e-4) if dtype == torch.float32 else (8e-3, 1e-1)
    _assert_close_to_scale(local.detach(), baseline[:, begin : begin + local_len], forward_tol, "forward")
    for name, summed in summed_grads.items():
        _assert_close_to_scale(summed, baseline_grads[name], grad_tol, name)

    clear_parallel_state()
    dist.destroy_process_group()


def _run_cp_collator_contract(rank: int, world_size: int, init_file: str) -> None:
    """What the collator hands the model under CP is what the model forward assumes.

    The model forward's ``query_offset = cp_rank * local_seq_len`` -- and the same
    expression in the attention forward, the metadata sharding and the indexer --
    is only correct for contiguous equally sized shards indexed by ``cp_rank``,
    carrying global positions and full-sequence cu-seqlens. All three come from
    ``SequenceParallelCollator``, which knows nothing about CP: it keys on
    ``sp_rank`` / ``sp_size``, which resolve to the CP pair only because a
    CP-only mesh flattens ``sp`` onto ``cp``.
    """
    from veomni.data.data_collator import SequenceParallelCollator
    from veomni.distributed.parallel_state import clear_parallel_state, get_parallel_state, init_parallel_state

    device_type = get_device_type()
    get_torch_device().set_device(rank)
    _use_full_float32_matmuls()
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    init_parallel_state(dp_size=1, cp_size=world_size, ulysses_size=1, device_type=device_type)

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
    # Narrowed, not renumbered: every place the design leans on ``position_ids``
    # -- the compressor rope positions, the ``has_previous_window`` gate, the
    # sliding causality test -- reads global positions.
    expected_positions = position_ids[:, begin : begin + local_len]
    assert torch.equal(collated["position_ids"], expected_positions), (
        f"rank {rank} position_ids must stay global: expected {expected_positions.tolist()}, "
        f"got {collated['position_ids'].tolist()}"
    )
    # Derived before the slice, so the model sees the whole packed batch's
    # boundaries next to one shard. That is what ``full_seq_len`` reconstructs.
    expected_cu_seq_lens = [0, *(end for _, end in _PACKED_MODEL_SAMPLES)]
    assert collated["cu_seq_lens_q"].tolist() == expected_cu_seq_lens, (
        f"rank {rank} cu_seq_lens_q must span the packed batch: expected {expected_cu_seq_lens}, "
        f"got {collated['cu_seq_lens_q'].tolist()}"
    )
    # The mask is not sliced either, which is what the attention forward's
    # full-sequence mask check depends on.
    assert collated["attention_mask"].shape[-1] == seq_len, (
        f"rank {rank} attention_mask must stay full length {seq_len}, got {tuple(collated['attention_mask'].shape)}"
    )

    clear_parallel_state()
    dist.destroy_process_group()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
def test_deepseek_v4_cp_collator_shards_contiguously_by_cp_rank():
    """The collator gives rank ``r`` rows ``[r*L, (r+1)*L)``, global positions, global cu-seqlens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_cp_collator_contract, args=(4, init_file), nprocs=4, join=True)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
def test_deepseek_v4_model_cp_packed_misaligned():
    """A whole packed model under CP matches the single-rank forward and backward."""
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_model_cp_packed, args=(4, init_file, torch.float32, False), nprocs=4, join=True)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
@pytest.mark.skipif(not is_sm90_or_above(), reason="the DeepSeek-V4 TileLang kernels need SM90 or later")
def test_deepseek_v4_model_cp_packed_misaligned_tilelang():
    """The same in bf16, where the model withholds the mask and TileLang runs instead.

    Distinct coverage rather than a duplicate: this is the only case that takes
    the mask-free compact-candidate path end to end under CP -- packed sparse
    indices rebased by ``query_offset``, sparse MQA in the kernel, and the
    TileLang Lightning Indexer inside a real CSA layer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_model_cp_packed, args=(4, init_file, torch.bfloat16, True), nprocs=4, join=True)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
def test_deepseek_v4_attention_cp_unpacked_misaligned():
    """A shard length that is a multiple of no compression rate still matches.

    ``seq_len`` 68 gives ``L=17``. The compressor is dropped because the toy HCA
    rate of 32 is wider than that shard, which its own guard refuses; what is
    left is the sliding window, which at 32 reaches two shards back. The shard
    that is narrower than the sliding window is the point of this one;
    ``test_deepseek_v4_attention_cp_unpacked_misaligned_with_compressor`` is the
    misaligned case that runs a compressor.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_attention_cp, args=(4, init_file, 68, False), nprocs=4, join=True)


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="needs 4 devices")
def test_deepseek_v4_attention_cp_unpacked_misaligned_with_compressor():
    """A misaligned shard that is still wide enough to compress.

    ``seq_len`` 160 gives ``L=40``, which is not a multiple of the toy HCA rate
    of 32 but is wider than it, so the narrow-shard guard admits it and the
    compressor runs. Window starts are [0, 32, 64, 96, 128] and the owners are
    [0, 0, 1, 2, 3], so the ranks own unequal numbers of them and the window at
    32 covers 32..63, crossing the rank 0/1 edge at 40 -- unpacked straddling,
    which an aligned length cannot produce because there every window sits
    inside one shard.

    Sequence length is the only free variable here: with four ranks, ``L`` must
    exceed 32 to keep the compressor and must not be a multiple of 32 to be
    misaligned, so 160 is the shortest sequence that satisfies both.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_attention_cp, args=(4, init_file, 160, True), nprocs=4, join=True)


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_with_csa_compressor(cp_size):
    """A whole CSA layer under CP -- compressor and Lightning Indexer -- matches the baseline.

    This is what replaces the guard that used to refuse an indexer-bearing
    compressor under CP. It is the only test that runs the two against each
    other, which is where the shared assumption lives: the indexer's compressed
    array must be the same length, and in the same order, as the compressor's, or
    a top-k value names a different slot on each side.

    Sequence length 128 with the toy CSA rate of 4 gives 32 compressed slots
    against ``index_topk`` 32, so every causally visible slot is selected and the
    block bias this produces does not depend on the top-k *order*. The ranking
    itself is what ``test_deepseek_v4_indexer_cp`` covers.
    """
    if get_torch_device().device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 128, True, 3),
            nprocs=cp_size,
            join=True,
        )
