import importlib.util
import math
import os
import sys
from contextlib import nullcontext
from functools import partial
from importlib.machinery import ModuleSpec
from types import SimpleNamespace


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer


_real_find_spec = importlib.util.find_spec


def _find_spec_without_torch_npu(name: str, package: str | None = None) -> ModuleSpec | None:
    if name == "torch_npu":
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_torch_npu  # type: ignore[assignment]
try:
    import veomni.trainer.callbacks.channel_loss_callback as channel_loss_module
    from veomni.arguments.arguments_types import ChannelLossConfig, ChunkMBSConfig
    from veomni.models.seed_omni.foundation.qwen3_moe_foundation.modeling_qwen3_moe_foundation import (
        Qwen3MoeFoundationModel,
    )
    from veomni.models.seed_omni.modeling_seed_omni import SeedOmniModel
    from veomni.models.transformers.qwen2_5_omni.generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from veomni.models.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_gpu import (
        Qwen3OmniMoeForConditionalGeneration,
    )
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.base_rl_trainer import BaseRLTrainer
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.callbacks.channel_loss_callback import (
        ChannelLossCallback,
        ChannelLossComputer,
        ChannelLossMetadataError,
    )
    from veomni.trainer.dit_trainer import DiTTrainer
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer
    from veomni.utils.constants import IGNORE_INDEX
finally:
    importlib.util.find_spec = _real_find_spec  # type: ignore[assignment]


class _DummyOpSlot:
    def __init__(self, kernel):
        self._kernel = kernel
        self.use_non_eager_impl = True

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)


class _TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, use_cache=False):
        return SimpleNamespace(loss=(self.weight * x).sum())


def _expected_segment(logits, labels, start, end):
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous().view(-1)
    losses = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)), shifted, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = shifted[start : end + 1] != IGNORE_INDEX
    return losses[start : end + 1][valid].sum().item(), valid.sum().item()


def test_channel_loss_computer_aggregates_packed_logits_by_source():
    torch.manual_seed(0)
    vocab_size = 32
    labels = torch.tensor([[10, 11, 12, IGNORE_INDEX, 21]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    logits = torch.randn(1, 5, vocab_size)

    computer = ChannelLossComputer()
    computer._source_ids = [0, 1]
    computer._position_ids = position_ids

    result = computer._compute_channel_loss(
        logits=logits,
        labels=labels,
        vocab_size=vocab_size,
        hidden_states=None,
        weights=None,
        attention_mask=None,
        shift_labels=None,
        ignore_index=IGNORE_INDEX,
    )

    loss_0, tokens_0 = _expected_segment(logits, labels, 0, 2)
    loss_1, tokens_1 = _expected_segment(logits, labels, 3, 4)
    assert result == [
        {"source_id": 0, "loss_sum": pytest.approx(loss_0), "token_count": tokens_0},
        {"source_id": 1, "loss_sum": pytest.approx(loss_1), "token_count": tokens_1},
    ]


def test_channel_loss_computer_hidden_states_path_matches_logits_path():
    torch.manual_seed(1)
    vocab_size = 16
    hidden_size = 8
    labels = torch.tensor([[3, 4, IGNORE_INDEX, 7]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    hidden_states = torch.randn(1, 4, hidden_size)
    weights = torch.randn(vocab_size, hidden_size)
    logits = F.linear(hidden_states, weights)

    logits_computer = ChannelLossComputer()
    logits_computer._source_ids = ["a", "b"]
    logits_computer._position_ids = position_ids
    logits_result = logits_computer._compute_channel_loss(
        logits=logits,
        labels=labels,
        vocab_size=vocab_size,
        hidden_states=None,
        weights=None,
        attention_mask=None,
        shift_labels=None,
        ignore_index=IGNORE_INDEX,
    )

    hs_computer = ChannelLossComputer()
    hs_computer._source_ids = ["a", "b"]
    hs_computer._position_ids = position_ids
    hs_result = hs_computer._compute_channel_loss(
        logits=None,
        labels=labels,
        vocab_size=vocab_size,
        hidden_states=hidden_states,
        weights=weights,
        attention_mask=None,
        shift_labels=None,
        ignore_index=IGNORE_INDEX,
    )

    assert hs_result == logits_result


def test_channel_loss_wrapper_forwards_original_call_unchanged():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def loss_function(
            self,
            logits,
            labels,
            vocab_size,
            num_items_in_batch=None,
            ignore_index=IGNORE_INDEX,
            **kwargs,
        ):
            self.calls.append((logits, labels, vocab_size, num_items_in_batch, ignore_index, kwargs))
            return "main-loss"

    model = DummyModel()
    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])
    num_items = object()

    try:
        computer.install(model)
        with computer.capture():
            result = model.loss_function(logits, labels, 8, num_items, -1, passthrough=True)
        assert result == "main-loss"
        assert model.calls == [(logits, labels, 8, num_items, -1, {"passthrough": True})]
        assert computer._result
        assert computer._result[0]["source_id"] == 0
    finally:
        computer.uninstall()


def test_channel_loss_opslot_wrapper_handles_positional_loss_args():
    def original_kernel(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=IGNORE_INDEX, **kwargs):
        return "main-loss", logits, None

    module = sys.modules[__name__]
    old_slot = getattr(module, "veomni_causal_lm_loss", None)
    module.veomni_causal_lm_loss = _DummyOpSlot(original_kernel)

    class DummyModel(torch.nn.Module):
        pass

    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])

    try:
        computer.install(DummyModel())
        assert computer._wrapped_opslots
        with computer.capture():
            result = module.veomni_causal_lm_loss(logits, labels, 8)
        assert result[0] == "main-loss"
        assert result[1] is logits
        assert result[2] is None
        assert computer._result
        assert computer._result[0]["source_id"] == 0
    finally:
        computer.uninstall()
        if old_slot is None:
            delattr(module, "veomni_causal_lm_loss")
        else:
            module.veomni_causal_lm_loss = old_slot


@pytest.mark.parametrize("signature_kind", ["liger_partial", "chunk_dispatch"])
def test_channel_loss_extracts_fused_inputs_from_variadic_loss_signatures(signature_kind):
    from veomni.ops.kernels.cross_entropy import ForCausalLMLoss, _chunk_loss_dispatch

    if signature_kind == "liger_partial":

        def fake_fused_cross_entropy(*args, **kwargs):
            return torch.tensor(0.0), None

        loss_fn = partial(ForCausalLMLoss, cross_entropy_fn=fake_fused_cross_entropy)
    else:
        loss_fn = _chunk_loss_dispatch

    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[1, 2, 3]])
    hidden_states = torch.randn(1, 3, 4)
    weights = torch.randn(8, 4)

    with computer.capture():
        computer._observe_opslot_call(
            loss_fn,
            (),
            {
                "logits": None,
                "labels": labels,
                "vocab_size": 8,
                "hidden_states": hidden_states,
                "weights": weights,
            },
        )

    assert computer._result
    assert computer._result[0]["source_id"] == 0
    assert computer._result[0]["token_count"].item() == 2


def test_channel_loss_unwraps_native_lora_model_for_eager_loss():
    from veomni.lora.config import VeOmniLoraConfig
    from veomni.lora.model import VeOmniLoraModel

    class EagerLossModel(torch.nn.Module):
        def loss_function(self, logits, labels, vocab_size, **kwargs):
            return "main-loss"

        def forward(self, logits, labels, vocab_size):
            return self.loss_function(logits, labels, vocab_size)

    inner_model = EagerLossModel()
    lora_model = VeOmniLoraModel(
        inner_model,
        VeOmniLoraConfig(target_modules=["unused"]),
        inject=False,
    )
    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])

    try:
        computer.install(lora_model)
        assert computer._model_ref is inner_model
        with computer.capture():
            result = lora_model(logits, labels, 8)
        assert result == "main-loss"
        assert computer._result
        assert computer._result[0]["source_id"] == 0
    finally:
        computer.uninstall()


def test_channel_loss_unwraps_seed_omni_foundation_loss():
    class EagerFoundationModel(torch.nn.Module):
        def loss_function(self, logits, labels, vocab_size, **kwargs):
            return "foundation-loss"

        def forward(self, logits, labels, vocab_size):
            return self.loss_function(logits, labels, vocab_size)

    class LightweightSeedOmniModel(SeedOmniModel):
        def __init__(self, foundation):
            torch.nn.Module.__init__(self)
            self.foundation = foundation

        def forward(self, *args, **kwargs):
            return self.foundation(*args, **kwargs)

    foundation = EagerFoundationModel()
    model = LightweightSeedOmniModel(foundation)
    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])

    try:
        computer.install(model)
        assert computer._model_ref is foundation
        with computer.capture():
            result = model(logits, labels, 8)
        assert result == "foundation-loss"
        assert computer._result
        assert computer._result[0]["source_id"] == 0
    finally:
        computer.uninstall()


def test_channel_loss_rejects_seed_omni_qwen3_moe_direct_loss():
    class LightweightSeedOmniModel(SeedOmniModel):
        def __init__(self, foundation):
            torch.nn.Module.__init__(self)
            self.foundation = foundation

    foundation = object.__new__(Qwen3MoeFoundationModel)
    torch.nn.Module.__init__(foundation)
    model = LightweightSeedOmniModel(foundation)
    computer = ChannelLossComputer()

    try:
        with pytest.raises(ValueError, match="Qwen3MoeFoundationModel.*computes causal-LM loss directly"):
            computer.install(model)
    finally:
        computer.uninstall()


@pytest.mark.parametrize(
    "model_cls",
    [Qwen2_5OmniForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration],
)
def test_channel_loss_unwraps_omni_thinker_loss(model_cls):
    class EagerThinkerModel(torch.nn.Module):
        def loss_function(self, logits, labels, vocab_size, **kwargs):
            return "thinker-loss"

        def forward(self, logits, labels, vocab_size):
            return self.loss_function(logits, labels, vocab_size)

    thinker = EagerThinkerModel()
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.thinker = thinker
    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])

    try:
        computer.install(model)
        assert computer._model_ref is thinker
        with computer.capture():
            result = model(logits=logits, labels=labels, vocab_size=8)
        assert result == "thinker-loss"
        assert computer._result
        assert computer._result[0]["source_id"] == 0
    finally:
        computer.uninstall()


def test_channel_loss_opslot_dispatch_is_scoped_and_reference_counted():
    def original_kernel(logits, labels, vocab_size, **kwargs):
        return "main-loss", logits, None

    module = sys.modules[__name__]
    old_slot = getattr(module, "veomni_causal_lm_loss", None)
    slot = _DummyOpSlot(original_kernel)
    module.veomni_causal_lm_loss = slot

    class DummyModel(torch.nn.Module):
        pass

    first = ChannelLossComputer()
    second = ChannelLossComputer()
    for index, computer in enumerate((first, second)):
        computer._source_ids = [index]
        computer._position_ids = torch.tensor([[0, 1, 2]])

    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])

    try:
        first.install(DummyModel())
        dispatcher = slot._kernel
        second.install(DummyModel())
        assert slot._kernel is dispatcher

        slot(logits, labels, 8)
        assert first._result is None
        assert second._result is None

        with first.capture():
            slot(logits, labels, 8)
        assert first._result and first._result[0]["source_id"] == 0
        assert second._result is None

        with second.capture():
            slot(logits, labels, 8)
        assert second._result and second._result[0]["source_id"] == 1

        first.uninstall()
        assert slot._kernel is dispatcher
        second.uninstall()
        assert slot._kernel is original_kernel
    finally:
        first.uninstall()
        second.uninstall()
        if old_slot is None:
            delattr(module, "veomni_causal_lm_loss")
        else:
            module.veomni_causal_lm_loss = old_slot


def test_channel_loss_sp_reduce_preserves_batch_order_for_multiple_sources(monkeypatch):
    rank1_metadata = [
        {"batch_idx": 0, "sp_rank": 1, "local_order": 0, "is_start": False},
        {"batch_idx": 1, "sp_rank": 1, "local_order": 0, "is_start": False},
    ]
    gather_inputs = []

    def fake_all_gather_object(gathered, local_metadata, group=None):
        gathered[0] = local_metadata
        gathered[1] = rank1_metadata

    def fake_all_gather(gathered, local_values, group=None):
        gathered[0].copy_(local_values)
        if local_values.dtype == torch.float32:
            gathered[1].copy_(torch.tensor([5.0, 7.0]))
        else:
            gathered[1].copy_(torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.int64))
        gather_inputs.append((local_values.dtype, tuple(local_values.shape)))

    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 10.0, 20.0]),
        labels_flat=torch.tensor([1, 1, 1, 1]),
        attention_mask_flat=None,
        source_ids=["batch0", "batch1"],
        positions=torch.tensor([[0, 1], [0, 1]]),
        seq_len=4,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "batch0", "loss_sum": 8.0, "token_count": 3},
        {"source_id": "batch1", "loss_sum": 37.0, "token_count": 3},
    ]
    assert gather_inputs == [(torch.float32, (2,)), (torch.int64, (2, 3))]


def test_channel_loss_sp_reduce_handles_empty_local_rank(monkeypatch):
    rank1_metadata = [{"batch_idx": 0, "sp_rank": 1, "local_order": 0, "is_start": True}]

    def fake_all_gather_object(gathered, local_metadata, group=None):
        assert local_metadata == []
        gathered[0] = []
        gathered[1] = rank1_metadata

    def fake_all_gather(gathered, local_values, group=None):
        gathered[0].copy_(local_values)
        if local_values.dtype == torch.float32:
            gathered[1].copy_(torch.tensor([4.0]))
        else:
            gathered[1].copy_(torch.tensor([[2, 0, 0]], dtype=torch.int64))

    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._reduce_sp(
        local_segments=[],
        source_ids=["remote"],
        device=torch.device("cpu"),
        strict=True,
    )

    assert result == [{"source_id": "remote", "loss_sum": 4.0, "token_count": 2}]


def test_channel_loss_sp_reduce_filters_uneven_remote_padding_segments(monkeypatch):
    rank1_metadata = [
        {"batch_idx": 0, "sp_rank": 1, "local_order": 0, "is_start": True},
        {"batch_idx": 0, "sp_rank": 1, "local_order": 1, "is_start": True},
    ]

    def fake_all_gather_object(gathered, local_metadata, group=None):
        assert len(local_metadata) == 1
        gathered[0] = local_metadata
        gathered[1] = rank1_metadata

    def fake_all_gather(gathered, local_values, group=None):
        gathered[0].copy_(local_values)
        if local_values.dtype == torch.float32:
            gathered[1].zero_()
        else:
            gathered[1].copy_(torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int64))

    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0]),
        labels_flat=torch.tensor([1, 1]),
        attention_mask_flat=torch.ones(2, dtype=torch.long),
        source_ids=["doc"],
        positions=torch.tensor([[0, 1]]),
        seq_len=2,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 3.0, "token_count": 2}]


def test_channel_loss_sp_aggregation_does_not_extract_segment_scalars(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)

    def fail_item(tensor):
        raise AssertionError(f"SP segment aggregation extracted a scalar with shape={tuple(tensor.shape)}")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", fail_item)
        result = ChannelLossComputer._aggregate_sp(
            per_token_loss=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            labels_flat=torch.ones(6, dtype=torch.long),
            attention_mask_flat=torch.ones(6, dtype=torch.long),
            source_ids=["a", "b", "c"],
            positions=torch.tensor([[0, 1, 0, 1, 0, 1]]),
            seq_len=6,
            ignore_index=IGNORE_INDEX,
            strict=True,
        )

    assert result == [
        {"source_id": "a", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "b", "loss_sum": 7.0, "token_count": 2},
        {"source_id": "c", "loss_sum": 11.0, "token_count": 2},
    ]


def test_channel_loss_segment_count_mismatch_skips_or_raises():
    kwargs = dict(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        labels_flat=torch.tensor([1, 1, 1, 1]),
        attention_mask_flat=None,
        source_ids=["only_one_source"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        ignore_index=IGNORE_INDEX,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_ignores_pad_to_length_position_zero_segments():
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(5, dtype=torch.long),
        source_ids=["doc"],
        position_ids=torch.tensor([[0, 1, 2, 0, 0]]),
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 6.0, "token_count": 3}]


def test_channel_loss_filters_padding_tail_per_batch_row():
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.tensor([1.0, 2.0, 100.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]),
        source_ids=["row0", "row1"],
        position_ids=torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]),
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "row0", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "row1", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_preserves_zero_supervision_tail_when_later_row_has_padding():
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        position_ids=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "row0_normal", "loss_sum": 6.0, "token_count": 3},
        {"source_id": "row0_zero", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "row1_normal", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_compute_uses_position_aligned_mask_for_padding_evidence(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "_is_sp_enabled", lambda: False)
    computer = ChannelLossComputer()
    computer.strict = True
    computer._source_ids = ["row0_normal", "row0_zero", "row1_normal"]
    computer._position_ids = torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]])
    labels = torch.tensor([[1, 2, 3, 4], [1, 2, IGNORE_INDEX, IGNORE_INDEX]])
    logits = torch.randn(2, 4, 8)

    result = computer._compute_channel_loss(
        logits=logits,
        labels=labels,
        vocab_size=8,
        hidden_states=None,
        weights=None,
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        shift_labels=None,
        ignore_index=IGNORE_INDEX,
    )

    assert [item["source_id"] for item in result] == ["row0_normal", "row0_zero", "row1_normal"]
    assert [item["token_count"].item() for item in result] == [3, 0, 1]


def test_channel_loss_rejects_ambiguous_per_row_padding_without_mask_evidence():
    kwargs = dict(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(8, dtype=torch.long),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        position_ids=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        ignore_index=IGNORE_INDEX,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_keeps_zero_supervision_segment_before_tail_padding():
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.tensor([100.0, 2.0, 100.0]),
        labels_flat=torch.tensor([IGNORE_INDEX, 1, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(3, dtype=torch.long),
        source_ids=["one_token", "normal"],
        position_ids=torch.tensor([[0, 0, 0]]),
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "one_token", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "normal", "loss_sum": 2.0, "token_count": 1},
    ]


def test_channel_loss_sp_ignores_padding_only_position_zero_segments(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=None,
        source_ids=["doc"],
        positions=torch.tensor([[0, 1, 0, 0]]),
        seq_len=4,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 3.0, "token_count": 2}]


def test_channel_loss_sp_filters_padding_tail_per_batch_row(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 100.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]),
        source_ids=["row0", "row1"],
        positions=torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "row0", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "row1", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_preserves_zero_supervision_tail_when_later_row_has_padding(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        positions=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "row0_normal", "loss_sum": 6.0, "token_count": 3},
        {"source_id": "row0_zero", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "row1_normal", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_rejects_ambiguous_per_row_padding_without_mask_evidence(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)
    kwargs = dict(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(8, dtype=torch.long),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        positions=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
    )

    assert ChannelLossComputer._aggregate_sp(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_sp(**kwargs, strict=True)


def test_channel_loss_sp_keeps_zero_supervision_segment_before_tail_padding(monkeypatch):
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(channel_loss_module, "get_unified_sequence_parallel_world_size", lambda: 1)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([100.0, 2.0, 100.0]),
        labels_flat=torch.tensor([IGNORE_INDEX, 1, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(3, dtype=torch.long),
        source_ids=["one_token", "normal"],
        positions=torch.tensor([[0, 0, 0]]),
        seq_len=3,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [
        {"source_id": "one_token", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "normal", "loss_sum": 2.0, "token_count": 1},
    ]


def test_channel_loss_logits_path_computes_ce_in_chunks(monkeypatch):
    calls = []
    original_cross_entropy = F.cross_entropy

    def spy_cross_entropy(input, target, **kwargs):
        calls.append(input.shape[0])
        return original_cross_entropy(input, target, **kwargs)

    monkeypatch.setattr(channel_loss_module.F, "cross_entropy", spy_cross_entropy)
    monkeypatch.setattr(ChannelLossComputer, "EAGER_CHUNK_SIZE", 2)

    logits = torch.randn(1, 5, 8)
    labels_flat = torch.tensor([1, 2, 3, 4, 5])

    result = ChannelLossComputer()._per_token_ce(
        logits=logits,
        labels_flat=labels_flat,
        vocab_size=8,
        hidden_states=None,
        weights=None,
        ignore_index=IGNORE_INDEX,
    )

    assert result.shape == (5,)
    assert calls == [2, 2, 1]


def test_channel_loss_config_validates_sampling_interval():
    assert ChannelLossConfig().interval == 10
    with pytest.raises(ValueError, match="interval must be at least 1"):
        ChannelLossConfig(interval=0)


def test_channel_loss_computer_aggregates_step_results_locally():
    computer = ChannelLossComputer()
    computer.begin_step([], [], [])
    computer._result = [
        {"source_id": "a", "loss_sum": torch.tensor(2.0), "token_count": torch.tensor(1)},
        {"source_id": "a", "loss_sum": torch.tensor(4.0), "token_count": torch.tensor(2)},
        {"source_id": "b", "loss_sum": torch.tensor(3.0), "token_count": torch.tensor(1)},
    ]

    computer.after_micro_step()

    assert set(computer.step_totals) == {"a", "b"}
    assert computer.step_totals["a"][0].item() == 6.0
    assert computer.step_totals["a"][1].item() == 3


def test_base_forward_backward_allows_missing_channel_loss_callback():
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            enable_batch_invariant_mode=False,
            local_rank=0,
        )
    )
    trainer.model = _TinyLossModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = 1
    trainer.micro_batches_token_len = 1
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()})

    loss, loss_dict = BaseTrainer.forward_backward_step(trainer, {"x": torch.tensor(2.0)})

    assert loss.item() == 2.0
    assert loss_dict["loss"].item() == 2.0
    assert trainer.model.weight.grad.item() == 2.0


def test_base_forward_backward_strips_channel_metadata_after_preforward():
    cfg = ChannelLossConfig(enable=True)
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            channel_loss=cfg,
            enable_batch_invariant_mode=False,
            local_rank=0,
        )
    )
    trainer.model = _TinyLossModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = 1
    trainer.micro_batches_token_len = 1
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()})
    trainer.channel_loss_callback = ChannelLossCallback(trainer)
    preforward_seen = {}

    def preforward(micro_batch):
        preforward_seen["has_source_metadata"] = "ds_idx" in micro_batch and "cur_token_num" in micro_batch
        return BaseTrainer.preforward(trainer, micro_batch)

    trainer.preforward = preforward
    micro_batch = {
        "x": torch.tensor(2.0),
        "ds_idx": torch.tensor([3]),
        "source_name": ["train/a"],
        "cur_token_num": torch.tensor([1]),
    }

    trainer.channel_loss_callback.on_step_begin(trainer.state, micro_batches=[micro_batch])
    loss, loss_dict = BaseTrainer.forward_backward_step(trainer, micro_batch)

    assert preforward_seen["has_source_metadata"]
    assert loss.item() == 2.0
    assert loss_dict["loss"].item() == 2.0
    assert trainer.model.weight.grad.item() == 2.0


def test_base_forward_backward_composes_channel_loss_and_chunk_mbs_contexts(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs
    import veomni.trainer.base as base_trainer_module

    capture_states = []
    range_states = []
    checkpoint_calls = []
    observation_calls = []

    class CheckpointedDecoderLayer(GradientCheckpointingLayer):
        def __init__(self, capture_probe):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.capture_probe = capture_probe

        def forward(self, hidden_states, **kwargs):
            capture_states.append(self.capture_probe())
            range_states.append(chunk_mbs._chunk_mbs_ranges.get())
            return self.proj(hidden_states)

    class CheckpointedLossModel(torch.nn.Module):
        _no_split_modules = ["CheckpointedDecoderLayer"]

        def __init__(self, capture_probe):
            super().__init__()
            self.layers = torch.nn.ModuleList([CheckpointedDecoderLayer(capture_probe)])
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.loss_calls = 0

        def gradient_checkpointing_enable(self, checkpoint_func=None, gradient_checkpointing_kwargs=None):
            if checkpoint_func is None:
                checkpoint_func = partial(checkpoint, **(gradient_checkpointing_kwargs or {}))
            for layer in self.layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = checkpoint_func

        def loss_function(self, logits, labels, vocab_size, **kwargs):
            self.loss_calls += 1
            return F.cross_entropy(logits[..., :-1, :].flatten(0, 1), labels[..., 1:].flatten())

        def forward(
            self,
            x,
            labels,
            position_ids,
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
            use_cache=False,
        ):
            hidden_states = self.layers[0](
                x,
                position_ids=position_ids,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
            )
            logits = self.lm_head(hidden_states)
            return SimpleNamespace(loss=self.loss_function(logits, labels, logits.shape[-1]))

    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            channel_loss=ChannelLossConfig(enable=True, interval=1),
            chunk_mbs_config=ChunkMBSConfig(enable=True, chunk_mbs=1),
            enable_batch_invariant_mode=False,
            local_rank=0,
        )
    )
    trainer.model = CheckpointedLossModel(lambda: trainer.channel_loss_callback.computer.capture_active)
    trainer.model.gradient_checkpointing_enable(
        lambda function, *args, **kwargs: (
            checkpoint_calls.append(None) or checkpoint(function, *args, use_reentrant=False, **kwargs)
        )
    )
    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    monkeypatch.setattr(base_trainer_module, "use_parallel_state", lambda _: nullcontext())
    chunk_mbs.apply_chunk_mbs(trainer.model, trainer.args.train.chunk_mbs_config)
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = 1
    trainer.micro_batches_token_len = 1
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()})
    trainer.channel_loss_callback = ChannelLossCallback(trainer)
    original_compute_side_channel = trainer.channel_loss_callback.computer.compute_side_channel

    def record_observation(*args, **kwargs):
        observation_calls.append(None)
        return original_compute_side_channel(*args, **kwargs)

    monkeypatch.setattr(trainer.channel_loss_callback.computer, "compute_side_channel", record_observation)
    cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    micro_batch = {
        "x": torch.randn(1, 4, 4, requires_grad=True),
        "labels": torch.tensor([[0, 1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": 2,
        "max_length_k": 2,
        "ds_idx": torch.tensor([3, 4]),
        "source_name": ["train/a", "train/b"],
        "cur_token_num": torch.tensor([2, 2]),
    }

    try:
        trainer.channel_loss_callback.on_train_begin(trainer.state)
        trainer.channel_loss_callback.on_step_begin(trainer.state, micro_batches=[micro_batch])
        BaseTrainer.forward_backward_step(trainer, micro_batch)
    finally:
        trainer.channel_loss_callback.on_train_end(trainer.state)

    assert checkpoint_calls == [None, None]
    assert capture_states == [True, True, False, False]
    assert all(ranges == trainer._chunk_mbs_ranges for ranges in range_states)
    assert observation_calls == [None]
    assert trainer.model.loss_calls == 1
    assert set(trainer.channel_loss_callback.computer.step_totals) == {3, 4}
    assert chunk_mbs._chunk_mbs_ranges.get() is None


def test_dpo_forward_backward_scopes_channel_loss_to_policy_model():
    cfg = ChannelLossConfig(enable=True, interval=1)
    state = TrainerState(global_step=1)
    policy_model = object()
    reference_model = object()
    base = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(channel_loss=cfg, enable_batch_invariant_mode=False),
            dpo_config=SimpleNamespace(
                beta=0.1,
                label_smoothing=0.0,
                loss_type="sigmoid",
                reference_free=False,
            ),
        ),
        state=state,
        model=policy_model,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
    )
    base.channel_loss_callback = ChannelLossCallback(base)
    preforward_seen = {}

    def preforward(micro_batch):
        preforward_seen["metadata"] = "ds_idx" in micro_batch and "source_name" in micro_batch
        return micro_batch

    base.preforward = preforward
    trainer = object.__new__(TextDPOTrainer)
    trainer.base = base
    trainer.reference_model = reference_model
    forward_calls = []

    def concatenated_forward(model, micro_batch):
        forward_calls.append(
            {
                "model": model,
                "capture_active": base.channel_loss_callback.computer.capture_active,
                "has_metadata": "ds_idx" in micro_batch or "source_name" in micro_batch,
            }
        )
        if model is reference_model:
            return torch.tensor([0.2]), torch.tensor([0.1])
        return torch.tensor([0.3], requires_grad=True), torch.tensor([0.1], requires_grad=True)

    trainer.concatenated_forward = concatenated_forward
    micro_batch = {
        "labels": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "ds_idx": torch.tensor([3]),
        "source_name": ["train/a"],
    }
    base.channel_loss_callback.on_step_begin(state, micro_batches=[micro_batch])

    loss, _ = TextDPOTrainer.forward_backward_step(trainer, micro_batch)

    assert loss.item() > 0
    assert preforward_seen["metadata"]
    assert forward_calls == [
        {"model": reference_model, "capture_active": False, "has_metadata": False},
        {"model": policy_model, "capture_active": True, "has_metadata": False},
    ]
    assert base.channel_loss_callback.computer._micro_step == 1
    assert base.channel_loss_callback.computer._source_ids == []


def test_dpo_channel_loss_emits_policy_totals():
    class DpoLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.loss_calls = 0

        def loss_function(self, logits, labels, vocab_size, **kwargs):
            self.loss_calls += 1
            return logits.sum() * 0

        def forward(self, labels=None, **kwargs):
            seq_len = labels.size(-1) if labels is not None else 8
            logits = torch.randn(1, seq_len, 8)
            if labels is not None:
                self.loss_function(logits=logits, labels=labels, vocab_size=8)
            log_probs = self.scale * (-torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(0) / 10)
            return SimpleNamespace(fused_linear_aux=SimpleNamespace(log_probs=log_probs))

    cfg = ChannelLossConfig(enable=True, interval=1)
    state = TrainerState(global_step=1)
    policy_model = DpoLossModel()
    reference_model = DpoLossModel()
    base = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(channel_loss=cfg, enable_batch_invariant_mode=False),
            dpo_config=SimpleNamespace(
                beta=0.1,
                label_smoothing=0.0,
                loss_type="sigmoid",
                reference_free=False,
                average_log_prob=False,
            ),
        ),
        state=state,
        model=policy_model,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        preforward=lambda micro_batch: micro_batch,
    )
    base.channel_loss_callback = ChannelLossCallback(base)
    step_begin_args = {}

    def on_step_begin(micro_batches=None, **kwargs):
        step_begin_args["source_repeat"] = kwargs.get("source_repeat", 1)
        base.channel_loss_callback.on_step_begin(
            state,
            micro_batches=micro_batches,
            **kwargs,
        )

    base.on_step_begin = on_step_begin
    trainer = object.__new__(TextDPOTrainer)
    trainer.base = base
    trainer.reference_model = reference_model
    trainer.post_forward = SimpleNamespace(compute_seqlens_func=lambda micro_batch: [2, 2, 2, 2])
    trainer.sp_enabled = False
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[1, 2, IGNORE_INDEX, 4, IGNORE_INDEX, 6, IGNORE_INDEX, 7]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]]),
        "ds_idx": torch.tensor([3, 4]),
        "source_name": ["train/a", "train/b"],
    }

    try:
        base.channel_loss_callback.computer.install(policy_model)
        TextDPOTrainer.on_step_begin(trainer, micro_batches=[micro_batch])
        assert step_begin_args == {"source_repeat": 2}
        assert base.channel_loss_callback.computer._per_mb_source_ids == [[3, 3, 4, 4]]
        TextDPOTrainer.forward_backward_step(trainer, micro_batch)

        assert policy_model.loss_calls == 1
        assert reference_model.loss_calls == 1
        assert set(base.channel_loss_callback.computer.step_totals) == {3, 4}
        assert base.channel_loss_callback.computer.step_totals[3][1].item() == 2
        assert base.channel_loss_callback.computer.step_totals[4][1].item() == 2
        assert base.channel_loss_callback.computer.source_names == {3: "train/a", 4: "train/b"}
        assert policy_model.scale.grad is not None
        assert reference_model.scale.grad is None
    finally:
        base.channel_loss_callback.computer.uninstall()


def test_dit_rejects_channel_loss_before_initialization():
    args = SimpleNamespace(train=SimpleNamespace(channel_loss=ChannelLossConfig(enable=True)))
    with pytest.raises(ValueError, match="causal-LM trainers"):
        DiTTrainer(args)


def test_channel_loss_rejects_classification_objective():
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(channel_loss=ChannelLossConfig(enable=True)),
            data=SimpleNamespace(data_type="classification"),
        )
    )

    with pytest.raises(ValueError, match="causal-LM objectives.*classification"):
        ChannelLossCallback(trainer)


def test_channel_loss_rejects_base_rl_trainer():
    trainer = object.__new__(BaseRLTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(channel_loss=ChannelLossConfig(enable=True)),
        data=SimpleNamespace(data_type="conversation"),
    )

    with pytest.raises(ValueError, match="BaseRLTrainer.*packs metadata during preforward"):
        ChannelLossCallback(trainer)


def test_channel_loss_resolve_source_name_handles_missing_environ_meter():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)

    assert callback._resolve_source_name(7) == "source_7"


def test_channel_loss_metric_name_collisions_remain_distinct():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)
    totals = {1: (6.0, 2), 2: (9.0, 3)}
    source_names = {1: "train/a", 2: "train_a"}

    metrics = callback._build_metrics(totals, source_names)

    assert metrics["channel_loss/source-i-1__train_a"] == 3.0
    assert metrics["channel_loss/source-i-2__train_a"] == 3.0
    assert metrics["channel_tokens/source-i-1__train_a"] == 2.0
    assert metrics["channel_tokens/source-i-2__train_a"] == 3.0
    assert len(metrics) == 6

    callback.config.strict = True
    assert callback._build_metrics(totals, source_names) == metrics


def test_channel_loss_metric_name_is_source_qualified_from_first_emission():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)

    metrics = callback._build_metrics({1: (6.0, 2)}, {1: "train/a"})

    assert "channel_loss/source-i-1__train_a" in metrics
    assert "channel_loss/train_a" not in metrics


def test_channel_loss_metric_name_collision_registry_persists_across_steps():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)

    first = callback._build_metrics({1: (6.0, 2)}, {1: "train/a"})
    second = callback._build_metrics({2: (9.0, 3)}, {2: "train_a"})
    third = callback._build_metrics({1: (3.0, 1)}, {1: "train/a"})

    assert "channel_loss/source-i-1__train_a" in first
    assert "channel_loss/source-i-2__train_a" in second
    assert "channel_loss/source-i-1__train_a" in third

    strict_cfg = ChannelLossConfig(enable=True, strict=True)
    strict_trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=strict_cfg)), environ_meter=None
    )
    strict_callback = ChannelLossCallback(strict_trainer)
    strict_first = strict_callback._build_metrics({1: (6.0, 2)}, {1: "train/a"})
    strict_second = strict_callback._build_metrics({2: (9.0, 3)}, {2: "train_a"})
    assert "channel_loss/source-i-1__train_a" in strict_first
    assert "channel_loss/source-i-2__train_a" in strict_second


def test_channel_loss_source_registry_round_trips_across_resume():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)
    callback._build_metrics({1: (6.0, 2)}, {1: "train/a"})

    resumed = ChannelLossCallback(trainer)
    resumed.load_state_dict(callback.state_dict())
    metrics = resumed._build_metrics({2: (9.0, 3)}, {2: "train_a"})

    assert "channel_loss/train_a" not in metrics
    assert "channel_loss/source-i-2__train_a" in metrics


def test_channel_loss_callback_strips_metadata_after_preforward():
    cfg = ChannelLossConfig(enable=True, interval=1)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    state = TrainerState(global_step=1)
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "ds_idx": torch.tensor([3]),
        "source_name": ["train/a"],
        "cur_token_num": torch.tensor([3]),
    }

    callback.on_step_begin(state, micro_batches=[micro_batch])
    callback.on_micro_step_begin(state, micro_batch=micro_batch)

    assert "ds_idx" in micro_batch
    assert "source_name" in micro_batch
    assert "cur_token_num" in micro_batch
    callback.strip_model_inputs(micro_batch)

    assert "ds_idx" not in micro_batch
    assert "source_name" not in micro_batch
    assert "cur_token_num" not in micro_batch
    assert "position_ids" in micro_batch
    assert callback.computer._source_ids == [3]
    assert callback.computer.source_names == {3: "train/a"}


def test_channel_loss_callback_samples_steps_but_always_strips_metadata():
    cfg = ChannelLossConfig(enable=True, interval=2)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    micro_batch = {
        "position_ids": torch.tensor([[0, 1]]),
        "ds_idx": torch.tensor([3]),
        "source_name": ["train/a"],
    }

    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[micro_batch])
    callback.on_micro_step_begin(TrainerState(global_step=1), micro_batch=micro_batch)
    with callback.model_forward_context():
        assert not callback.computer.capture_active
    assert callback.computer._source_ids == []
    callback.strip_model_inputs(micro_batch)
    assert "ds_idx" not in micro_batch
    assert "source_name" not in micro_batch

    sampled_batch = {
        "position_ids": torch.tensor([[0, 1]]),
        "ds_idx": torch.tensor([4]),
        "source_name": ["train/b"],
    }
    callback.on_step_begin(TrainerState(global_step=2), micro_batches=[sampled_batch])
    callback.on_micro_step_begin(TrainerState(global_step=2), micro_batch=sampled_batch)
    with callback.model_forward_context():
        assert callback.computer.capture_active
    assert callback.computer._source_ids == [4]
    callback.on_micro_step_end(TrainerState(global_step=2))


def test_channel_loss_callback_updates_trainer_metrics():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    callback.computer.source_names = {0: "source/a", 1: "source-b"}
    callback.computer.step_totals = {
        0: (torch.tensor(6.0), torch.tensor(3)),
        1: (torch.tensor(3.0), torch.tensor(1)),
    }
    callback._collect_step = True

    callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert math.isclose(trainer.step_env_metrics["channel_loss/source-i-0__source_a"], 2.0)
    assert math.isclose(trainer.step_env_metrics["channel_loss/source-i-1__source-b"], 3.0)
    assert math.isclose(trainer.step_env_metrics["channel_loss_weighted/source-i-0__source_a"], 1.5)
    assert math.isclose(trainer.step_env_metrics["channel_tokens/source-i-1__source-b"], 1.0)
    assert trainer.step_train_metrics == trainer.step_env_metrics


def test_channel_loss_callback_reduces_compact_totals_and_syncs_names(monkeypatch):
    cfg = ChannelLossConfig(enable=True, interval=1)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        device=torch.device("cpu"),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.source_names = {0: "source/a"}
    callback.computer.step_totals = {0: (torch.tensor(6.0), torch.tensor(3))}
    group = object()
    observed = {}

    monkeypatch.setattr(channel_loss_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, dp_group=group))
    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(gathered, local_metadata, group=None):
        observed["metadata"] = local_metadata
        gathered[0] = local_metadata
        gathered[1] = [(1, "source-b")]

    def fake_all_reduce(value, group=None):
        assert group is not None
        if value.dtype.is_floating_point:
            value.add_(torch.tensor([0.0, 3.0]))
        else:
            value.add_(torch.tensor([0, 1]))

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_reduce", fake_all_reduce)

    callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert observed["metadata"] == [(0, "source/a")]
    assert trainer.step_env_metrics["channel_loss/source-i-0__source_a"] == 2.0
    assert trainer.step_env_metrics["channel_loss/source-i-1__source-b"] == 3.0
    assert trainer.step_env_metrics["channel_tokens/source-i-1__source-b"] == 1.0


def test_channel_loss_callback_strict_rejects_cross_rank_name_mismatch(monkeypatch):
    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        device=torch.device("cpu"),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.source_names = {0: "source-a"}
    callback.computer.step_totals = {0: (torch.tensor(1.0), torch.tensor(1))}

    monkeypatch.setattr(
        channel_loss_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, dp_group=object())
    )
    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(gathered, local_metadata, group=None):
        gathered[0] = local_metadata
        gathered[1] = [(0, "source-b")]

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(ChannelLossMetadataError, match="inconsistent names"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)


def test_channel_loss_callback_strict_requires_source_id():
    cfg = ChannelLossConfig(enable=True, strict=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    with pytest.raises(ValueError, match="no configured source ID"):
        callback.on_step_begin(TrainerState(global_step=1), micro_batches=[{"input_ids": torch.tensor([[1]])}])
