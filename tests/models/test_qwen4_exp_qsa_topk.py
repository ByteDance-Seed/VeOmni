# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Finite-mask QSA selection semantics for Qwen4-Exp NPU vs GPU generated modeling."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import from_parameter
from veomni.utils.device import IS_NPU_AVAILABLE


def _load_qwen4_exp_modeling(backend: str):
    return importlib.import_module(
        f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}"
    )


def test_compact_qsa_npu_finite_mask_matches_gpu_packed_selection(monkeypatch):
    """Verify NPU QSA indexer masks invisible blocks with -1.0 (not -inf) and matches GPU selection."""
    gpu_modeling = _load_qwen4_exp_modeling("gpu")
    npu_modeling = _load_qwen4_exp_modeling("npu")

    torch.manual_seed(38)
    hidden = torch.randn(1, 25, 3)
    indexer = SimpleNamespace(
        index_kv_heads=1,
        index_n_heads=2,
        index_head_dim=2,
        compress_ratio=4,
        block_topk=2,
        token_budget=8,
        index_qk_proj=torch.nn.Linear(3, 6, bias=False),
        q_layernorm=torch.nn.Identity(),
        k_layernorm=torch.nn.Identity(),
        veomni_rope=lambda q, k, cos, sin, **kwargs: (q, k),
    )
    positions = (torch.ones(1, 25, 2), torch.zeros(1, 25, 2))
    boundaries = torch.tensor([0, 9, 25], dtype=torch.int32)

    expected = gpu_modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )

    orig_topk = torch.Tensor.topk
    topk_inputs: list[torch.Tensor] = []

    def recording_topk(self: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        topk_inputs.append(self.detach().clone())
        return orig_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    monkeypatch.setattr(torch.Tensor, "topk", recording_topk)
    actual = npu_modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )

    assert topk_inputs, "Expected Qwen4ExpTextQSAIndexer.forward to invoke Tensor.topk"
    for scores in topk_inputs:
        assert torch.isfinite(scores).all(), "NPU QSA indexer must not pass -inf to topk"
        assert (scores == -1.0).any(), "Expected invisible blocks to be masked with -1.0"
        assert (scores[scores != -1.0] >= 0.0).all(), "Unmasked ReLU block scores must be non-negative"

    # Random continuous scores have no cutoff ties in this fixture. Compare
    # token sets, since TopK does not specify the order of equal masked entries.
    torch.testing.assert_close(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
    for query in range(25):
        selected = actual[0, query]
        selected = selected[selected >= 0]
        start = 0 if query < 9 else 9
        assert ((selected >= start) & (selected <= query)).all()
    assert (actual[0, 0] >= 0).sum() == 1  # No complete block; only the causal tail.
    assert (actual[0, 9] >= 0).sum() == 1  # The next packed example starts fresh.


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend"))],
)
@pytest.mark.parametrize("key_length", [511, 513, 29166])
@pytest.mark.parametrize("quantized", [False, True])
def test_qsa_sparse_topk_boundary(device_type: str, key_length: int, quantized: bool):
    """Verify -1.0 finite masking matches -inf selection across TopKV2 boundary sizes."""
    rows, k = 143, min(512, key_length)
    scores = torch.rand((1, rows, key_length), generator=torch.Generator().manual_seed(38))
    if quantized:
        scores = (scores * 4).floor()  # Exact zeros and ties at the selection cutoff.
    visible = torch.zeros_like(scores, dtype=torch.bool)
    for row in range(rows - 6):
        count = min((1951 + row) // 4, key_length)
        visible[0, row, key_length - count :] = True
    visible[0, -3:, -1] = True
    # The preceding three rows are all masked; the last three have one block.
    reference = scores.masked_fill(~visible, float("-inf")).sort(dim=-1, descending=True).values[..., :k]
    device_scores, device_visible = scores.to(device_type), visible.to(device_type)
    masked = device_scores.masked_fill(~device_visible, -1.0)
    picked = masked.topk(k, dim=-1).indices.cpu()
    valid = visible.gather(-1, picked)
    values = scores.gather(-1, picked).masked_fill(~valid, float("-inf"))
    torch.testing.assert_close(values.sort(dim=-1, descending=True).values, reference, rtol=0, atol=0)
    assert torch.equal(valid.sum(dim=-1), visible.sum(dim=-1).clamp(max=k))
    assert not valid[0, -6:-3].any()


def _native_indexer_worker(rank, world_size, endpoint):
    """Compare the real NPU INT8 gather against a bool-gather oracle on packed shards."""
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.utils.device import get_torch_device

    get_torch_device().set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group(
        "hccl", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    try:
        _init_parallel_state(ulysses_size=world_size, device_type="npu")
        module = _load_qwen4_exp_modeling("npu")
        torch.manual_seed(38)
        indexer = SimpleNamespace(
            index_kv_heads=1,
            index_n_heads=2,
            index_head_dim=4,
            compress_ratio=4,
            block_topk=2,
            token_budget=8,
            index_qk_proj=torch.nn.Linear(6, 12, bias=False).to(device=device, dtype=torch.bfloat16),
            q_layernorm=torch.nn.Identity(),
            k_layernorm=torch.nn.Identity(),
            veomni_rope=lambda q, k, cos, sin, **kwargs: (q, k),
        )
        gather = torch.Tensor.gather
        bool_calls = []

        def bool_reference(tensor, dim, indices, *args, **kwargs):
            """Substitute only the candidate's integer visibility gather."""
            if tensor.dtype == torch.int8:
                bool_calls.append(tensor.shape)
                return gather(tensor.bool(), dim, indices, *args, **kwargs).to(torch.int8)
            return gather(tensor, dim, indices, *args, **kwargs)

        length = 64 // world_size
        for lengths in ((3, 20, 41), (13, 51), (1, 63)):
            bool_calls.clear()
            torch.manual_seed(41)
            full = torch.randn(2, 64, 6).to(device=device, dtype=torch.bfloat16)
            hidden = full[:, rank * length : (rank + 1) * length].contiguous()
            positions = (torch.ones(2, length, 4, device=device), torch.zeros(2, length, 4, device=device))
            packed_lengths = lengths * 2
            bounds = [0]
            for count in packed_lengths:
                bounds.append(bounds[-1] + count)
            cu = torch.tensor(bounds, device=device, dtype=torch.int32)
            with torch.no_grad():
                actual = module.Qwen4ExpTextQSAIndexer.forward(
                    indexer, hidden, positions, None, None, cu, packed_seq_lens=packed_lengths
                )
                with patch.object(torch.Tensor, "gather", bool_reference):
                    reference = module.Qwen4ExpTextQSAIndexer.forward(
                        indexer, hidden, positions, None, None, cu, packed_seq_lens=packed_lengths
                    )
            assert bool_calls, f"The bool reference must exercise the NPU-only gather branch for {lengths}"
            assert torch.equal(actual, reference)
            assert actual.dtype == torch.int32
            for batch in range(2):
                for local_query in range(length):
                    query = rank * length + local_query
                    segment_start = max(b for b in bounds if b <= batch * 64 + query) - batch * 64
                    selected = actual[batch, local_query]
                    selected = selected[selected >= 0]
                    assert bool(((selected >= segment_start) & (selected <= query)).all())
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.launch_plan(provider=from_parameter("world_size"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Native NPU indexer parity requires Ascend")
def test_native_int8_indexer_matches_bool_packed_reference(world_size):
    """Run real HCCL plus NPU indexer ops for non-aligned packed multi-batch layouts."""
    from tests.tools.launch_utils import spawn_with_rendezvous
    from veomni.utils.device import get_torch_device

    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires {world_size} Ascend devices")
    spawn_with_rendezvous(_native_indexer_worker, args=(world_size,), nprocs=world_size)
