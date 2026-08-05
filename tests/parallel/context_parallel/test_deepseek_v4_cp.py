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


def _init_cp_attention(rank: int, world_size: int, init_file: str, seq_len: int, with_compressor: bool):
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
    # Layer 0 is HCA on the toy config, which has no sliding-only layer type, so
    # drop the compressor the way the Ulysses test does to reach pure sliding MQA.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=0).to(device=device_type, dtype=torch.float32)
    if not with_compressor:
        layer.compressor = None
    _broadcast_module(layer)
    layer.train()

    full_hidden = torch.randn(1, seq_len, config.hidden_size, device=device_type, dtype=torch.float32)
    dist.broadcast(full_hidden, src=0)
    full_position_ids = torch.arange(seq_len, device=device_type).view(1, -1)

    rotary = dsv4.DeepseekV4RotaryEmbedding(config).to(device=device_type)
    _broadcast_module(rotary)

    full_mask = _build_causal_mask(seq_len, config.sliding_window, device_type, torch.float32)
    return dsv4, layer, _make_forward(layer, rotary), full_hidden, full_position_ids, full_mask


def _run_attention_cp(rank: int, world_size: int, init_file: str, seq_len: int, with_compressor: bool) -> None:
    from veomni.distributed.parallel_state import clear_parallel_state

    _, layer, _forward, full_hidden, full_position_ids, full_mask = _init_cp_attention(
        rank, world_size, init_file, seq_len, with_compressor
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


def _cp_state(cp_size: int = 2, cp_rank: int = 0) -> SimpleNamespace:
    """A parallel state claiming CP without a process group.

    Every guard below raises before the KV all-gather, so ``cp_group=None`` is
    reached only if a guard has been moved after the collective.
    """
    return SimpleNamespace(ulysses_enabled=False, cp_enabled=True, cp_group=None, cp_rank=cp_rank, cp_size=cp_size)


def _build_local_attention(with_compressor: bool, local_len: int, cp_size: int):
    """One rank's fixture on CPU: a local shard plus the full-sequence mask CP requires."""
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    torch.manual_seed(0)
    # Layer 0 is HCA on the toy config; dropping the compressor is what makes it
    # sliding-only, so keeping it is exactly the layer kind CP must refuse.
    layer = dsv4.DeepseekV4Attention(config, layer_idx=0)
    if not with_compressor:
        layer.compressor = None

    hidden = torch.randn(1, local_len, config.hidden_size)
    position_ids = torch.arange(local_len).view(1, -1)
    full_mask = _build_causal_mask(local_len * cp_size, config.sliding_window, "cpu", torch.float32)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config)
    return config, _make_forward(layer, rotary), hidden, position_ids, full_mask


def test_deepseek_v4_attention_cp_rejects_a_compressor_layer():
    """A compressor would summarise the local shard only, so CP must refuse it until Task 6."""
    _, forward, hidden, position_ids, full_mask = _build_local_attention(with_compressor=True, local_len=16, cp_size=2)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(NotImplementedError, match="compressor path"):
            forward(hidden, position_ids, full_mask)


def test_deepseek_v4_attention_cp_rejects_a_kv_cache():
    """Decode would append this rank's shard to a cache the other ranks also gather."""
    from transformers import DynamicCache

    config, forward, hidden, position_ids, full_mask = _build_local_attention(
        with_compressor=False, local_len=16, cp_size=2
    )
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=_cp_state()):
        with pytest.raises(NotImplementedError, match="KV cache"):
            forward(hidden, position_ids, full_mask, past_key_values=DynamicCache(config=config))


def test_deepseek_v4_attention_cp_rejects_a_local_length_mask():
    """A shard-width mask would let rank 0 attend everywhere uncaused and later ranks time out."""
    config, forward, hidden, position_ids, _ = _build_local_attention(with_compressor=False, local_len=16, cp_size=2)
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
