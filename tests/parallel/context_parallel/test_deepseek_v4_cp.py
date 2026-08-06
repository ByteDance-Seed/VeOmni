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

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


_PATCHED_MODULE = "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu"


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
):
    """Enter the process group, build the shared layer, and return the fixture."""
    device_type = get_device_type()
    get_torch_device().set_device(rank)
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
    layer = dsv4.DeepseekV4Attention(config, layer_idx=layer_idx).to(device=device_type, dtype=torch.float32)
    if not with_compressor:
        layer.compressor = None
    else:
        _init_position_bias(layer.compressor)
    _broadcast_module(layer)
    layer.train()

    full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type, dtype=torch.float32)
    dist.broadcast(full_hidden, src=0)
    full_position_ids = torch.arange(seq_len, device=device_type).view(1, -1)

    rotary = dsv4.DeepseekV4RotaryEmbedding(config).to(device=device_type)
    _broadcast_module(rotary)

    full_mask = _build_causal_mask(seq_len, config.sliding_window, device_type, torch.float32)
    return dsv4, layer, _make_forward(layer, rotary), full_hidden, full_position_ids, full_mask


def _run_attention_cp(
    rank: int,
    world_size: int,
    init_file: str,
    seq_len: int,
    with_compressor: bool,
    layer_idx: int = 0,
) -> None:
    from veomni.distributed.parallel_state import clear_parallel_state

    _, layer, _forward, full_hidden, full_position_ids, full_mask = _init_cp_attention(
        rank, world_size, init_file, seq_len, with_compressor, layer_idx
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

    torch.testing.assert_close(local_output, baseline[:, begin : begin + local_len], rtol=1e-4, atol=1e-4)
    # The KV all-gather's backward sum-reduces before slicing, so the input grad
    # already carries every other rank's contribution and must not be reduced again.
    torch.testing.assert_close(
        local_input.grad, baseline_input_grad[:, begin : begin + local_len], rtol=1e-4, atol=1e-4
    )
    for name, param in layer.named_parameters():
        if param.grad is None:
            continue
        summed = param.grad.detach().clone()
        dist.all_reduce(summed)
        torch.testing.assert_close(summed, baseline_grads[name], rtol=1e-4, atol=1e-4)

    clear_parallel_state()
    dist.destroy_process_group()


def _run_attention_cp_sparse_indices(rank: int, world_size: int, init_file: str, seq_len: int) -> None:
    """The compact candidates a shard builds must be the global build's own rows.

    The eager float32 path ignores ``sparse_topk_indices``, so this is the only
    thing that pins down ``query_offset`` / ``kv_full_len`` / ``compressed_len``.
    Get them wrong and the TileLang kernel silently reads the wrong KV rows.
    """
    from veomni.distributed.parallel_state import clear_parallel_state

    dsv4, _, _forward, full_hidden, full_position_ids, full_mask = _init_cp_attention(
        rank, world_size, init_file, seq_len, with_compressor=False
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
    with torch.no_grad(), patch(f"{_PATCHED_MODULE}.build_sparse_attention_indices", _record):
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


def _window_starts(rate: int, seq_len: int, sample_slices) -> torch.Tensor:
    """The global window starts, spelled out rather than taken from the helper under test."""
    if sample_slices is None:
        return torch.arange(0, seq_len - rate + 1, rate)
    return torch.tensor(
        [start for begin, end in sample_slices for start in range(begin, end - rate + 1, rate)],
        dtype=torch.long,
    )


def _run_compressor_cp(rank: int, world_size: int, init_file: str, kind: str, seq_len: int, sample_slices) -> None:
    """A CP shard's compressor must return the whole globally-ordered compressed KV.

    Every rank compresses only the windows it owns and then all-gathers, so the
    tensor compared here is the *full* baseline result, not a slice of it: a
    dropped or misordered window shows up directly. The backward reduces over
    this rank's own windows only, because summing the replicated array on every
    rank would scale the gradient by ``cp_size``.
    """
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    device_type = get_device_type()
    get_torch_device().set_device(rank)
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
    if kind == "csa":
        compressor.indexer = _FixedIndexer(config.index_topk)
    _broadcast_module(compressor)
    compressor.train()

    rate = compressor.compress_rate
    full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type)
    dist.broadcast(full_hidden, src=0)
    q_residual = torch.zeros(1, seq_len, config.q_lora_rank, device=device_type)

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


def test_deepseek_v4_indexer_cp_rejects_a_narrow_shard():
    """The indexer's halo comes from one neighbour, so a sub-rate shard cannot work.

    Pinned with the unusable group, which proves the check runs *before* the halo
    exchange rather than merely somewhere in the forward. A rank that raised after
    entering a collective would leave its peers stuck in it.
    """
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    indexer = dsv4.DeepseekV4Indexer(config)
    local_len = indexer.compress_rate - 1
    hidden = torch.randn(1, local_len, config.hidden_size)
    q_residual = torch.randn(1, local_len, config.q_lora_rank)
    position_ids = torch.arange(local_len).view(1, -1)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(ValueError, match="one compression window wide"):
            indexer(hidden, q_residual, position_ids, None, 0)


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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_sliding_only(cp_size):
    """A CP shard's sliding-window attention matches the full-sequence forward and backward."""
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 64, False),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_sparse_indices(cp_size):
    """A CP shard's compact sparse candidates match the full-sequence build's rows."""
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp_sparse_indices,
            args=(cp_size, init_file, 64),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_attention_cp_with_compressor(cp_size):
    """A whole HCA layer under CP matches the full-sequence forward and backward.

    Sequence length 128 keeps every rank holding at least one window at
    ``cp_size=4`` with the toy HCA rate of 32.
    """
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 128, True),
            nprocs=cp_size,
            join=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("kind", ["hca", "csa"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_compressor_cp_unpacked(kind, cp_size):
    """Both compressors rebuild the global compressed KV from per-shard windows."""
    if torch.cuda.device_count() < cp_size:
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
# exercise none of that.
#
# ``csa`` (rate 4, cp_size=4, L=16): starts
# [0,4,8,12,16,20,24,28,32,38,42,46,50,54,58], counts [4,4,4,3]. The window at
# 46 covers 46..49 and crosses the rank 2/3 edge at 48; rank 2 owns the window
# at 32 whose overlap half lives at 28..31, on rank 1.
#
# ``hca`` (rate 32, cp_size=4, L=32): starts [0,32,70], counts [1,1,1,0]. The
# window at 70 covers 70..101 and crosses the rank 2/3 edge at 96, and rank 3
# owns nothing at all. The rate is the halo width, so the shard cannot be
# narrower than 32 here.
_PACKED_COMPRESSOR_FIXTURES = {
    "csa": (64, ((0, 38), (38, 64))),
    "hca": (128, ((0, 70), (70, 128))),
}


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 devices")
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_deepseek_v4_compressor_cp_packed_straddling(kind):
    """Windows that cross a shard boundary still land in the right global slot."""
    seq_len, sample_slices = _PACKED_COMPRESSOR_FIXTURES[kind]
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_compressor_cp,
            args=(4, init_file, kind, seq_len, sample_slices),
            nprocs=4,
            join=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deepseek_v4_indexer_cp(cp_size):
    """A CP shard's indexer selection matches the full-query result, packed and unpacked."""
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(_run_indexer_cp, args=(cp_size, init_file, 256), nprocs=cp_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
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
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"needs {cp_size} devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_attention_cp,
            args=(cp_size, init_file, 128, True, 3),
            nprocs=cp_size,
            join=True,
        )
