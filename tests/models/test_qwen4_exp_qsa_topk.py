# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Finite-mask QSA selection semantics for Qwen4-Exp NPU vs GPU generated modeling."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import from_parameter
from veomni.ops import resolve_op
from veomni.utils.device import IS_NPU_AVAILABLE


def _load_qwen4_exp_modeling():
    return importlib.import_module("veomni.models.transformers.qwen4_exp.modeling_qwen4_exp")


def test_compact_qsa_finite_mask_matches_inf_packed_selection(monkeypatch):
    """Verify QSA indexer masks invisible blocks with finite -1.0 and matches -inf selection."""
    modeling = _load_qwen4_exp_modeling()

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
        veomni_qsa_indexer=resolve_op("qsa_indexer", "standard", "eager").wrapper,
    )
    positions = (torch.ones(1, 25, 2), torch.zeros(1, 25, 2))
    boundaries = torch.tensor([0, 9, 25], dtype=torch.int32)

    def inf_selection(scores: torch.Tensor, visible: torch.Tensor, top_count: int):
        top_blocks = scores.masked_fill(~visible, float("-inf")).topk(top_count, dim=-1).indices
        return top_blocks, visible.gather(-1, top_blocks)

    indexer.veomni_qsa_indexer = inf_selection
    expected = modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )
    indexer.veomni_qsa_indexer = resolve_op("qsa_indexer", "standard", "eager").wrapper

    orig_topk = torch.Tensor.topk
    topk_inputs: list[torch.Tensor] = []

    def recording_topk(self: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        topk_inputs.append(self.detach().clone())
        return orig_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    monkeypatch.setattr(torch.Tensor, "topk", recording_topk)
    actual = modeling.Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=boundaries,
    )

    assert topk_inputs, "Expected Qwen4ExpTextQSAIndexer.forward to invoke Tensor.topk"
    for scores in topk_inputs:
        assert torch.isfinite(scores).all(), "QSA indexer must not pass -inf to topk"
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


def _native_indexer_worker(rank, world_size, endpoint):
    """Compare the real NPU INT8 gather against a bool-gather oracle on packed shards."""
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.models.transformers.qwen4_exp.qsa_metadata import prepare_qsa_metadata
    from veomni.utils.device import get_torch_device

    get_torch_device().set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group(
        "hccl", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    try:
        _init_parallel_state(ulysses_size=world_size, device_type="npu")
        module = _load_qwen4_exp_modeling()
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
            veomni_qsa_indexer=resolve_op("qsa_indexer", "standard", "eager").wrapper,
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
            metadata = prepare_qsa_metadata(2, length, 4, rank, world_size, device, cu, packed_lengths)
            with torch.no_grad():
                actual = module.Qwen4ExpTextQSAIndexer.forward(
                    indexer,
                    hidden,
                    positions,
                    None,
                    None,
                    cu,
                    packed_seq_lens=packed_lengths,
                    qsa_metadata=metadata,
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
