import asyncio
import importlib.util
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from importlib.machinery import ModuleSpec
from threading import Event
from types import SimpleNamespace


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous


_real_find_spec = importlib.util.find_spec


def _find_spec_without_torch_npu(name: str, package: str | None = None) -> ModuleSpec | None:
    if name == "torch_npu":
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_torch_npu  # type: ignore[assignment]
try:
    import veomni.distributed.parallel_state as parallel_state_module
    import veomni.trainer.callbacks.base as callback_base_module
    import veomni.trainer.callbacks.channel_loss_callback as channel_loss_module
    from veomni.arguments.arguments_types import ChannelLossConfig
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


def _stub_parallel_state(monkeypatch, **overrides):
    """Let a hand-built trainer run without a registered ParallelState.

    ``forward_backward_step`` enters ``use_parallel_state`` and ``Callback``
    caches ``get_parallel_state()``, both of which need a real registry these
    single-process tests never populate.
    """
    state = SimpleNamespace(sp_enabled=False, sp_group=None, sp_size=1, dp_size=1, dp_group=None, **overrides)
    import veomni.trainer.base as base_trainer_module
    import veomni.trainer.text_dpo_trainer as dpo_trainer_module

    # Each trainer imports the context manager into its own namespace.
    for module in (base_trainer_module, dpo_trainer_module):
        monkeypatch.setattr(module, "use_parallel_state", lambda _: nullcontext())
    monkeypatch.setattr(callback_base_module, "get_parallel_state", lambda: state)
    return state


def _sp_state(monkeypatch, *, sp_size, sp_rank=0):
    """Stand-in for the ParallelState the SP collectives read.

    ``sp_group`` is an opaque sentinel: the computer only forwards it to the
    collectives, which these tests fake out, and reads ``None`` as "SP off".
    """
    group = object() if sp_size > 1 else None
    if group is not None:
        monkeypatch.setattr(channel_loss_module.dist, "get_rank", lambda _group=None: sp_rank)
    return SimpleNamespace(sp_enabled=sp_size > 1, sp_group=group, sp_size=sp_size)


class _TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.parallel_state = object()

    def forward(self, x, use_cache=False):
        return SimpleNamespace(loss=(self.weight * x).sum())


def _expected_segment(logits, labels, start, end):
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous().view(-1)
    losses = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)), shifted, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = shifted[start : end + 1] != IGNORE_INDEX
    return losses[start : end + 1][valid].sum().item(), valid.sum().item()


def _channel_features():
    return [
        {
            "input_ids": torch.tensor([], dtype=torch.long),
            "labels": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "ds_idx": 0,
        },
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "labels": torch.tensor([0, 1, 2, 3]),
            "attention_mask": torch.tensor([1, 0, 1, 1]),
            "position_ids": torch.tensor([0, 1, 0, 1]),
            "ds_idx": 1,
        },
    ]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_uses_original_feature_boundaries_without_changing_training(monkeypatch):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    baseline = data_collator.MainCollator()(_channel_features())
    observed = data_collator.MainCollator(channel_loss_source_id_keys=tuple(cfg.source_id_keys))(_channel_features())
    proof = observed[data_collator.CHANNEL_LOSS_PROVENANCE]
    assert proof["source_cu"] == [0, 0, 4]
    assert observed["cu_seq_lens_q"].tolist() == [0, 2, 4]
    for key in baseline:
        if isinstance(baseline[key], torch.Tensor):
            torch.testing.assert_close(observed[key], baseline[key], rtol=0, atol=0)
        else:
            assert observed[key] == baseline[key]

    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[observed])
    assert data_collator.CHANNEL_LOSS_PROVENANCE not in observed
    callback.computer.before_micro_step()
    callback.computer._original_loss_fn = lambda logits, labels, vocab_size: F.cross_entropy(
        logits.reshape(-1, vocab_size),
        F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    logits = torch.randn(1, 4, 5, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    with callback.computer.capture():
        loss = callback.computer._wrapped_loss_fn(logits, observed["labels"], 5)
    reference_loss = callback.computer._original_loss_fn(reference, baseline["labels"], 5)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)
    result = callback.computer._result
    assert [(item["source_id"], item["token_count"].item()) for item in result] == [(0, 0), (1, 3)]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_private_provenance_is_removed_on_unsampled_step(monkeypatch):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    cfg = ChannelLossConfig(enable=True, interval=2)
    batch = data_collator.MainCollator(channel_loss_source_id_keys=tuple(cfg.source_id_keys))(_channel_features())
    callback = ChannelLossCallback(SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg))))
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch])
    assert data_collator.CHANNEL_LOSS_PROVENANCE not in batch


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_metadata_hook_token_reordering(monkeypatch):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)

    def reorder(batch, _sp_pad):
        batch["input_ids"] = batch["input_ids"].flip(-1)

    cfg = ChannelLossConfig(enable=True, interval=1)
    batch = data_collator.MainCollator(
        channel_loss_source_id_keys=tuple(cfg.source_id_keys), metadata_collate_func=reorder
    )(_channel_features())
    assert batch[data_collator.CHANNEL_LOSS_PROVENANCE] == {"error": "unsupported source layout"}
    callback = ChannelLossCallback(SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg))))
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch])
    callback.computer.before_micro_step()
    callback.computer._original_loss_fn = lambda logits, labels, vocab_size: logits.sum()
    logits = torch.ones(1, 4, 5, requires_grad=True)
    with callback.computer.capture():
        loss = callback.computer._wrapped_loss_fn(logits, batch["labels"], 5)
    loss.backward()
    assert callback.computer._result is None
    assert torch.equal(logits.grad, torch.ones_like(logits))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_diagnostic_snapshot_failure_preserves_collation(monkeypatch):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    hook_calls = []

    def hook(batch, _sp_pad):
        hook_calls.append(True)

    baseline = data_collator.MainCollator(metadata_collate_func=hook)(_channel_features())
    cfg = ChannelLossConfig(enable=True, interval=1)

    def fail_clone(*args, **kwargs):
        raise RuntimeError("injected observer-only snapshot failure")

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    observed = data_collator.MainCollator(
        channel_loss_source_id_keys=tuple(cfg.source_id_keys), metadata_collate_func=hook
    )(_channel_features())
    assert observed.pop(data_collator.CHANNEL_LOSS_PROVENANCE) == {"error": "unsupported source layout"}
    assert len(hook_calls) == 2
    for key in baseline:
        if isinstance(baseline[key], torch.Tensor):
            torch.testing.assert_close(observed[key], baseline[key], rtol=0, atol=0)
        else:
            assert observed[key] == baseline[key]


def test_channel_loss_uses_shift_labels_when_sequence_parallel_is_on():
    torch.manual_seed(2)
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    shift_labels = torch.tensor([[4, 5, 6]], dtype=torch.long)

    def run(sp_enabled: bool, shift: torch.Tensor | None):
        computer = ChannelLossComputer(parallel_state=SimpleNamespace(sp_enabled=sp_enabled, sp_group=None, sp_size=1))
        computer._source_ids = [0]
        computer._position_ids = torch.arange(3).unsqueeze(0)
        result = computer._compute_channel_loss(
            logits=logits,
            labels=labels,
            vocab_size=8,
            hidden_states=None,
            weights=None,
            attention_mask=None,
            shift_labels=shift,
            ignore_index=IGNORE_INDEX,
        )
        return result[0]["loss_sum"], result[0]["token_count"]

    with_shift = run(True, shift_labels)
    without_sp = run(False, shift_labels)
    without_shift = run(True, None)
    torch.testing.assert_close(with_shift[0], without_sp[0])
    assert with_shift[1] == without_sp[1] == 3
    assert not torch.allclose(with_shift[0], without_shift[0])


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


def test_channel_loss_extracts_fused_inputs_from_models_loss_partial():
    from veomni.models.loss_utils import ForCausalLMLoss

    ce_calls = []

    def fake_ce(hidden_states, labels, weights, **kwargs):
        ce_calls.append((hidden_states, weights))
        del labels, kwargs
        return hidden_states.sum() * 0 + weights.sum() * 0

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = partial(ForCausalLMLoss, op=fake_ce)

    model = DummyModel()
    computer = ChannelLossComputer()
    computer._source_ids = [0]
    computer._position_ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[1, 2, 3]])
    hidden_states = torch.randn(1, 3, 4)
    weights = torch.randn(8, 4)

    try:
        computer.install(model)
        with computer.capture():
            model.loss_function(
                logits=None,
                labels=labels,
                vocab_size=8,
                hidden_states=hidden_states,
                weights=weights,
            )

        assert len(ce_calls) == 1
        assert ce_calls[0][1] is weights
        assert computer._result
        assert computer._result[0]["source_id"] == 0
        assert computer._result[0]["token_count"].item() == 2
    finally:
        computer.uninstall()


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

    sp_state = _sp_state(monkeypatch, sp_size=2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
        per_token_loss=torch.tensor([1.0, 2.0, 10.0, 20.0]),
        labels_flat=torch.tensor([1, 1, 1, 1]),
        attention_mask_flat=None,
        source_ids=["batch0", "batch1"],
        positions=torch.tensor([[0, 1], [0, 1]]),
        packed_global_cu_seqlens=[0, 4, 8],
        tail_padding_length=0,
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

    sp_state = _sp_state(monkeypatch, sp_size=2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._reduce_sp(
        parallel_state=sp_state,
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

    sp_state = _sp_state(monkeypatch, sp_size=2)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
        per_token_loss=torch.tensor([1.0, 2.0]),
        labels_flat=torch.tensor([1, 1]),
        attention_mask_flat=torch.ones(2, dtype=torch.long),
        source_ids=["doc"],
        positions=torch.tensor([[0, 1]]),
        packed_global_cu_seqlens=[0, 2, 4],
        tail_padding_length=2,
        seq_len=2,
        ignore_index=IGNORE_INDEX,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 3.0, "token_count": 2}]


def test_channel_loss_sp_aggregation_does_not_extract_segment_scalars(monkeypatch):
    sp_state = _sp_state(monkeypatch, sp_size=1)

    def fail_item(tensor):
        raise AssertionError(f"SP segment aggregation extracted a scalar with shape={tuple(tensor.shape)}")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", fail_item)
        result = ChannelLossComputer._aggregate_sp(
            parallel_state=sp_state,
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
    computer = ChannelLossComputer(parallel_state=_sp_state(monkeypatch, sp_size=1))
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
    sp_state = _sp_state(monkeypatch, sp_size=1)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
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
    sp_state = _sp_state(monkeypatch, sp_size=1)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
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
    sp_state = _sp_state(monkeypatch, sp_size=1)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
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
    sp_state = _sp_state(monkeypatch, sp_size=1)
    kwargs = dict(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(8, dtype=torch.long),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        positions=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        parallel_state=sp_state,
    )

    assert ChannelLossComputer._aggregate_sp(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_sp(**kwargs, strict=True)


def test_channel_loss_sp_keeps_zero_supervision_segment_before_tail_padding(monkeypatch):
    sp_state = _sp_state(monkeypatch, sp_size=1)

    result = ChannelLossComputer._aggregate_sp(
        parallel_state=sp_state,
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


def test_base_forward_backward_allows_missing_channel_loss_callback(monkeypatch):
    _stub_parallel_state(monkeypatch)
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            enable_batch_invariant_mode=False,
            local_rank=0,
        ),
    )
    trainer.model = _TinyLossModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = 1
    trainer.micro_batches_token_len = 1
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()}, {})

    loss, loss_dict, _ = BaseTrainer.forward_backward_step(trainer, {"x": torch.tensor(2.0)})

    assert loss.item() == 2.0
    assert loss_dict["loss"].item() == 2.0
    assert trainer.model.weight.grad.item() == 2.0


def test_base_forward_backward_strips_channel_metadata_after_preforward(monkeypatch):
    _stub_parallel_state(monkeypatch)
    cfg = ChannelLossConfig(enable=True)
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            channel_loss=cfg,
            enable_batch_invariant_mode=False,
            local_rank=0,
        ),
    )
    trainer.model = _TinyLossModel()
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.micro_batch_token_len = 1
    trainer.micro_batches_token_len = 1
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()}, {})
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
    loss, loss_dict, _ = BaseTrainer.forward_backward_step(trainer, micro_batch)

    assert preforward_seen["has_source_metadata"]
    assert loss.item() == 2.0
    assert loss_dict["loss"].item() == 2.0
    assert trainer.model.weight.grad.item() == 2.0


def test_dpo_forward_backward_scopes_channel_loss_to_policy_model(monkeypatch):
    _stub_parallel_state(monkeypatch)
    cfg = ChannelLossConfig(enable=True, interval=1)
    state = TrainerState(global_step=1)
    policy_model = SimpleNamespace(parallel_state=object())
    reference_model = SimpleNamespace(parallel_state=object())
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
    trainer.policy_model = policy_model
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


@pytest.mark.parametrize("compact", [False, True])
def test_dpo_channel_loss_emits_policy_totals(monkeypatch, compact):
    _stub_parallel_state(monkeypatch)

    class DpoLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.loss_calls = 0
            self.parallel_state = object()

        def loss_function(self, logits, labels, vocab_size, **kwargs):
            self.loss_calls += 1
            return logits.sum() * 0

        def forward(self, labels=None, **kwargs):
            seq_len = labels.size(-1) if labels is not None else 8
            logits = torch.randn(1, seq_len, 8)
            if labels is not None:
                self.loss_function(
                    logits=logits[:, :4] if compact else logits,
                    labels=labels[:, :4] if compact else labels,
                    vocab_size=8,
                )
            if compact:
                rows = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
                log_probs = -torch.arange(1, 5, dtype=torch.float32).index_select(0, rows).unsqueeze(0) / 10
            else:
                log_probs = -torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(0) / 10
            log_probs = self.scale * log_probs
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
    trainer = object.__new__(TextDPOTrainer)
    trainer.base = base
    trainer.policy_model = policy_model
    trainer.reference_model = reference_model
    trainer.state = state
    trainer.channel_loss_callback = base.channel_loss_callback
    base._callbacks = [base.channel_loss_callback]
    base.on_step_begin = lambda micro_batches=None, **kwargs: BaseTrainer.on_step_begin(
        base, micro_batches=micro_batches, **kwargs
    )
    trainer.post_forward = SimpleNamespace(compute_seqlens_func=lambda micro_batch: [2, 2, 2, 2])
    trainer.sp_enabled = False
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[1, 2, IGNORE_INDEX, 4, IGNORE_INDEX, 6, IGNORE_INDEX, 7]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]]),
        "cu_seq_lens_q": torch.tensor([0, 2, 4, 6, 8]),
        "_veomni_channel_loss_source_provenance": {
            "source_ids": [3, 4],
            "source_cu": [0, 4, 8],
            "input_mask": torch.ones(8, dtype=torch.bool),
            "tail": 0,
            "sp_size": 1,
            "sp_rank": 0,
        },
        "ds_idx": torch.tensor([3, 4]),
        "source_name": ["train/a", "train/b"],
    }

    try:
        base.channel_loss_callback.computer.install(policy_model)
        TextDPOTrainer.on_step_begin(trainer, micro_batches=[micro_batch])
        assert base.channel_loss_callback.computer._per_mb_source_ids == [[3, 4]]
        loss, _ = TextDPOTrainer.forward_backward_step(trainer, micro_batch)

        assert policy_model.loss_calls == 1
        assert reference_model.loss_calls == 1
        assert set(base.channel_loss_callback.computer.step_totals) == {3, 4}
        assert base.channel_loss_callback.computer.step_totals[3][1].item() == 2
        assert base.channel_loss_callback.computer.step_totals[4][1].item() == 2
        torch.testing.assert_close(
            base.channel_loss_callback.computer.step_totals[3][0], torch.tensor(0.2 if compact else 0.4)
        )
        torch.testing.assert_close(
            base.channel_loss_callback.computer.step_totals[4][0], torch.tensor(0.6 if compact else 1.2)
        )
        assert base.channel_loss_callback.computer.source_names == {3: "train/a", 4: "train/b"}
        assert policy_model.scale.grad is not None
        assert reference_model.scale.grad is None
        observed_grad = policy_model.scale.grad.detach().clone()
        policy_model.scale.grad = None
        base.channel_loss_callback.computer.uninstall()
        trainer.channel_loss_callback = None
        baseline, _ = TextDPOTrainer.forward_backward_step(trainer, micro_batch)
        torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
        torch.testing.assert_close(observed_grad, policy_model.scale.grad, rtol=0, atol=0)
    finally:
        base.channel_loss_callback.computer.uninstall()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_dpo_shared_prefix_reordering_uses_expanded_policy_log_probs(monkeypatch):
    from veomni.data.data_collator import CHANNEL_LOSS_PROVENANCE
    from veomni.models.transformers.qwen3_5.shared_prefix import build_shared_prefix_plan

    _stub_parallel_state(monkeypatch)
    prefix = list(range(64))
    ids = torch.tensor([prefix + [100, 101] + prefix + [102, 103]])
    positions = torch.tensor([list(range(66)) * 2])
    cu = torch.tensor([0, 66, 132], dtype=torch.int32)
    plan = build_shared_prefix_plan(ids, cu, positions)
    assert plan is not None
    labels = torch.tensor([prefix + [100, 101] + [IGNORE_INDEX] * 64 + [102, 103]])
    target = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    _, compact_labels, inverse = plan.lm_rows(target)
    assert compact_labels.numel() == target.numel() == 132
    assert not torch.equal(inverse, torch.arange(132))

    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    callback = ChannelLossCallback(SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg))))
    batch = {
        "labels": labels,
        "position_ids": positions,
        "cu_seq_lens_q": cu,
        "ds_idx": torch.tensor([0]),
        CHANNEL_LOSS_PROVENANCE: {
            "source_ids": [0],
            "source_cu": [0, 132],
            "input_mask": torch.ones(132, dtype=torch.bool),
            "tail": 0,
            "sp_size": 1,
            "sp_rank": 0,
        },
    }
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch], source_repeat=2)
    callback.computer.before_micro_step()

    def compact_loss(logits, labels, vocab_size):
        return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX)

    callback.computer._original_loss_fn = compact_loss
    torch.manual_seed(3)
    logits = torch.randn(132, 106, requires_grad=True)
    baseline_logits = logits.detach().clone().requires_grad_()
    with callback.computer.capture():
        loss = callback.computer._wrapped_loss_fn(logits.unsqueeze(0), compact_labels.unsqueeze(0), 106)
        per_token = F.cross_entropy(logits, compact_labels, ignore_index=IGNORE_INDEX, reduction="none")
        expanded = per_token.index_select(0, inverse).unsqueeze(0)
        callback.computer.observe_dpo_log_probs(-expanded, labels, batch["cu_seq_lens_q"])
    baseline = compact_loss(baseline_logits.unsqueeze(0), compact_labels.unsqueeze(0), 106)
    loss.backward()
    baseline.backward()
    torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, baseline_logits.grad, rtol=0, atol=0)
    valid = target.ne(IGNORE_INDEX)
    valid[0, torch.tensor([65, 131])] = False
    result = callback.computer._result[0]
    assert result["token_count"].item() == valid.sum().item() == 67
    torch.testing.assert_close(result["loss_sum"], expanded.detach()[valid].sum())


def test_dit_rejects_channel_loss_before_initialization():
    args = SimpleNamespace(
        train=SimpleNamespace(channel_loss=ChannelLossConfig(enable=True)),
    )
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
    group = object()
    observed = {}

    # Patched before construction: Callback.__init__ caches the parallel state,
    # so the DP collectives never consult the ambient one again.
    monkeypatch.setattr(callback_base_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, dp_group=group))
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.source_names = {0: "source/a"}
    callback.computer.step_totals = {0: (torch.tensor(6.0), torch.tensor(3))}

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
    monkeypatch.setattr(
        callback_base_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, dp_group=object())
    )
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.source_names = {0: "source-a"}
    callback.computer.step_totals = {0: (torch.tensor(1.0), torch.tensor(1))}

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


def _test_parallel_state(
    *,
    sp_enabled=False,
    sp_group=None,
    sp_size=1,
    sp_rank=0,
    dp_size=1,
    dp_group=None,
):
    return SimpleNamespace(
        sp_enabled=sp_enabled,
        sp_group=sp_group,
        sp_size=sp_size,
        sp_rank=sp_rank,
        dp_size=dp_size,
        dp_group=dp_group,
    )


def _install_test_parallel_state(monkeypatch, parallel_state=None):
    parallel_state = parallel_state or _test_parallel_state()
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(channel_loss_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setitem(parallel_state_module._PARALLEL_STATE_REGISTRY, "base", parallel_state)
    monkeypatch.setattr(callback_base_module, "get_parallel_state", lambda: parallel_state)
    return parallel_state


@pytest.mark.parametrize("implementation", ["eager", "chunk_loss"])
@pytest.mark.parametrize("bias_kind", ["full", "factored"])
@pytest.mark.parametrize("explicit_shift", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_preserves_full_and_factored_bias(
    monkeypatch, implementation, bias_kind, explicit_shift, strict, dtype
):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    _install_test_parallel_state(monkeypatch)
    monkeypatch.setattr(ChannelLossComputer, "EAGER_CHUNK_SIZE", 3)
    model = torch.nn.Module()
    model.loss_function = partial(ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", implementation))
    torch.manual_seed(125)
    hidden = torch.randn(2, 7, 3, dtype=dtype, requires_grad=True)
    weight = torch.randn(7, 3, dtype=dtype, requires_grad=True)
    bias = torch.randn(2, 7, 2 if bias_kind == "factored" else 7, dtype=dtype, requires_grad=True)
    tensors = [hidden, weight, bias]
    if bias_kind == "factored":
        tensors.append(torch.randn(7, 2, dtype=dtype, requires_grad=True))
    labels = torch.arange(14).reshape(2, 7) % 7
    labels[:, 3] = IGNORE_INDEX
    targets = labels if explicit_shift else F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:].contiguous()
    kwargs = dict(labels=labels, vocab_size=7, chunk_size=3)
    if explicit_shift:
        kwargs["shift_labels"] = targets

    def compute(inputs):
        return model.loss_function(
            **kwargs,
            hidden_states=inputs[0],
            weights=inputs[1],
            logits_bias=inputs[2],
            logits_bias_weight=inputs[3] if bias_kind == "factored" else None,
        )[0]

    reference_inputs = [t.detach().clone().requires_grad_() for t in tensors]
    reference = compute(reference_inputs)
    reference.backward()
    computer = ChannelLossComputer()
    computer.strict = strict
    computer._source_ids = ["first", "second"]
    computer._position_ids = torch.arange(7).expand(2, -1)
    try:
        computer.install(model)
        with computer.capture():
            actual = compute(tensors)
        actual.backward()
    finally:
        computer.uninstall()

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for tensor, expected in zip(tensors, reference_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, expected.grad, rtol=0, atol=0)
    logits = F.linear(hidden.detach(), weight.detach()).float()
    bias_logits = F.linear(bias.detach(), tensors[3].detach()) if bias_kind == "factored" else bias.detach()
    per_token = F.cross_entropy(
        (logits + bias_logits.float()).reshape(-1, 7), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).reshape(2, 7)
    assert [entry["source_id"] for entry in computer._result] == ["first", "second"]
    for row, entry in enumerate(computer._result):
        torch.testing.assert_close(entry["loss_sum"], per_token[row].sum(), rtol=1e-6, atol=1e-6)
        assert entry["token_count"].item() == targets[row].ne(IGNORE_INDEX).sum().item()


def test_channel_loss_chunk_kernel_reuses_main_projection(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    _install_test_parallel_state(monkeypatch)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = partial(ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"))

    model = DummyModel()
    torch.manual_seed(7)
    hidden = torch.randn(1, 17, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.arange(17).unsqueeze(0) % 7
    ref_hidden, ref_weight = (t.detach().clone().requires_grad_() for t in (hidden, weight))
    kwargs = dict(logits=None, labels=labels, vocab_size=7, chunk_size=8)
    reference, _, _ = model.loss_function(**kwargs, hidden_states=ref_hidden, weights=ref_weight)
    reference.backward()
    computer = ChannelLossComputer()
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(17).unsqueeze(0)
    projections = []
    original_linear = F.linear

    def linear(x, w, bias=None):
        projections.append(tuple(x.shape))
        return original_linear(x, w, bias)

    monkeypatch.setattr(F, "linear", linear)
    try:
        computer.install(model)
        with computer.capture():
            actual, _, _ = model.loss_function(**kwargs, hidden_states=hidden, weights=weight)
        actual.backward()
    finally:
        computer.uninstall()
    assert projections == [(8, 4), (8, 4), (1, 4)]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=0, atol=0)
    assert computer._result[0]["token_count"].item() == 16
    assert computer._result[0]["loss_sum"].item() == pytest.approx(actual.item() * 16, rel=1e-6)


def test_wrapped_chunk_loss_accepts_shift_labels_without_raw_labels(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    _install_test_parallel_state(monkeypatch)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = partial(ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"))

    model = Model()
    computer = ChannelLossComputer()
    computer.strict = True
    computer._source_ids = ["shift-only"]
    computer._position_ids = torch.arange(7).unsqueeze(0)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, -100, 5, 6, 7]])
    try:
        computer.install(model)
        with computer.capture():
            loss, _, _ = model.loss_function(hidden_states=hidden, weights=weight, shift_labels=targets)
        loss.backward()
        assert computer._result[0]["token_count"].item() == 6
        assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
        assert computer._result[0]["loss_sum"].item() == pytest.approx(loss.item() * 6, rel=1e-6)
    finally:
        computer.uninstall()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_headwise_cp_uses_local_packed_boundaries():
    state = _test_parallel_state(sp_enabled=True)
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.arange(1, 11, dtype=torch.float32),
        labels_flat=torch.tensor(
            [1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        ),
        attention_mask_flat=torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 1, 1]),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 0, 0, 1, 2, 0, 0, 0]]),
        packed_cu_seqlens=[0, 4, 8, 10],
        packed_global_cu_seqlens=[0, 4, 8, 10],
        packed_global_physical_seq_len=10,
        tail_padding_length=2,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=state,
        strict=True,
    )
    assert [item["source_id"] for item in result] == ["first", "second"]
    assert [item["loss_sum"].item() for item in result] == [3, 18]
    assert [item["token_count"].item() for item in result] == [2, 3]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("failure", ["missing_positions", "supervised_tail", "stale_cu"])
def test_channel_loss_sp_rejects_ambiguous_metadata(failure):
    state = _test_parallel_state(sp_enabled=True)
    labels = torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX])
    positions = torch.tensor([[0, 1, 0, 0]])
    cu, tail = [0, 2, 4], 2
    if failure == "missing_positions":
        positions = None
    elif failure == "supervised_tail":
        labels[-1] = 2
    else:
        cu = [0, 2, 5]
    with pytest.raises(ChannelLossMetadataError):
        ChannelLossComputer._aggregate_by_source(
            per_token_loss=torch.ones(4),
            labels_flat=labels,
            attention_mask_flat=None,
            source_ids=["source"],
            position_ids=positions,
            packed_cu_seqlens=cu,
            packed_global_cu_seqlens=[0, 2, 4],
            packed_global_physical_seq_len=4,
            tail_padding_length=tail,
            ignore_index=IGNORE_INDEX,
            sp_enabled=True,
            parallel_state=state,
            strict=True,
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "cu", [[], torch.tensor([], dtype=torch.int64), torch.tensor([[0, 2, 4]], dtype=torch.int32), [[0, 2, 4]]]
)
def test_channel_loss_rejects_malformed_cu(cu):
    with pytest.raises(ChannelLossMetadataError, match="invalid packed CU"):
        ChannelLossComputer._aggregate_sp(
            per_token_loss=torch.ones(4),
            labels_flat=torch.ones(4, dtype=torch.long),
            attention_mask_flat=None,
            source_ids=["first", "second"],
            positions=torch.tensor([[0, 1, 0, 1]]),
            packed_cu_seqlens=cu,
            seq_len=4,
            ignore_index=IGNORE_INDEX,
            parallel_state=_test_parallel_state(sp_enabled=True),
            strict=True,
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_headwise_cp_keeps_empty_source():
    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.empty(0),
        labels_flat=torch.empty(0, dtype=torch.long),
        attention_mask_flat=torch.empty(0, dtype=torch.long),
        source_ids=["empty"],
        positions=torch.empty((1, 0), dtype=torch.long),
        packed_cu_seqlens=[0, 0],
        seq_len=0,
        ignore_index=IGNORE_INDEX,
        parallel_state=_test_parallel_state(sp_enabled=True),
        strict=True,
    )
    assert len(result) == 1 and result[0]["source_id"] == "empty"
    assert result[0]["loss_sum"].item() == result[0]["token_count"].item() == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_stale_local_cu_without_global_mask():
    state = _test_parallel_state(sp_enabled=True, sp_size=2)
    state.cp_size = 2
    with pytest.raises(ChannelLossMetadataError, match="global geometry"):
        ChannelLossComputer._aggregate_sp(
            per_token_loss=torch.ones(4),
            labels_flat=torch.ones(4, dtype=torch.long),
            attention_mask_flat=None,
            source_ids=["first", "second"],
            positions=torch.tensor([[0, 1, 0, 1]]),
            packed_cu_seqlens=[0, 1, 4],
            packed_global_cu_seqlens=[0, 4, 8],
            seq_len=4,
            ignore_index=IGNORE_INDEX,
            parallel_state=state,
            strict=True,
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ulysses_global_spans_are_intersected_per_batch_row():
    points = [0, 3, 6, 8, 11, 14, 16]
    assert channel_loss_module._ulysses_local_span_lengths(points, 2, 4, 0, 2) == [3, 1, 0, 3, 1, 0]
    assert channel_loss_module._ulysses_local_span_lengths(points, 2, 4, 1, 2) == [0, 2, 2, 0, 2, 2]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_accumulates_multiple_loss_calls():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._position_ids = torch.tensor([[0, 1]])
    computer._original_loss_fn = lambda logits, labels, vocab_size, shift_labels: F.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    labels = torch.tensor([[1, 2]])
    logits = [torch.randn(1, 2, 4) for _ in range(2)]
    with computer.capture():
        for value in logits:
            computer._wrapped_loss_fn(value, labels, 4, labels)
    computer.after_micro_step()
    expected = sum(F.cross_entropy(value.reshape(-1, 4), labels.reshape(-1), reduction="sum") for value in logits)
    loss_sum, token_count = computer.step_totals["source"]
    torch.testing.assert_close(loss_sum, expected)
    assert token_count.item() == 4


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("strict", [False, True])
def test_channel_loss_discards_partially_reduced_observations(monkeypatch, strict):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state(sp_enabled=True))
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._capture_attempts = 2
    computer._pending_sp = [{"segments": [], "device": torch.device("cpu")} for _ in range(2)]
    results = iter([[{"source_id": "source", "loss_sum": torch.tensor(1.0), "token_count": torch.tensor(1)}], []])
    monkeypatch.setattr(computer, "_reduce_sp", lambda *args: next(results))
    context = pytest.raises(ChannelLossMetadataError, match="no source-aligned") if strict else nullcontext()
    with context:
        computer._finish_capture()
    assert computer._result is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_strict_rejects_missing_non_sp_observation():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.tensor([[0, 1]])
    computer._original_loss_fn = lambda labels: torch.tensor(1.0)
    with pytest.raises(ChannelLossMetadataError, match="no source-aligned"):
        with computer.capture():
            computer._wrapped_loss_fn(labels=torch.tensor([[1, 2]]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_concurrent_capture_on_same_computer():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    entered, release = Event(), Event()

    def worker():
        with computer.capture():
            entered.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        assert entered.wait(10)
        try:
            with pytest.raises(RuntimeError, match="Concurrent capture"), computer.capture():
                pass
        finally:
            release.set()
        future.result(timeout=10)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_async_child_capture_on_same_computer():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())

    async def run():
        async def child():
            with pytest.raises(RuntimeError, match="Concurrent capture"), computer.capture():
                pass

        with computer.capture():
            await asyncio.gather(child(), child())
            with computer.capture():
                pass

    asyncio.run(run())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_async_child_direct_and_expired_context():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._position_ids = torch.tensor([[0, 1]])
    computer._original_loss_fn = lambda logits, labels, vocab_size, shift_labels: torch.tensor(1.0)
    logits, labels = torch.randn(1, 2, 4), torch.tensor([[1, 2]])

    async def run():
        release = asyncio.Event()

        async def child():
            with pytest.raises(RuntimeError, match="Concurrent capture"):
                computer._wrapped_loss_fn(logits, labels, 4, labels)
            await release.wait()
            assert not computer.capture_active
            computer._wrapped_loss_fn(logits, labels, 4, labels)
            assert computer._result is None
            with computer.capture():
                assert computer.capture_active

        with computer.capture():
            task = asyncio.create_task(child())
            await asyncio.sleep(0)
        release.set()
        await task

    asyncio.run(run())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_async_child_without_sources_runs_main_loss():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._original_loss_fn = lambda labels: torch.tensor(1.0)
    computer._dpo_segments = True

    async def run():
        async def child():
            assert computer._wrapped_loss_fn(labels=torch.tensor([[1]])).item() == 1.0
            computer.observe_dpo_log_probs(torch.zeros(1, 1), torch.ones(1, 1), None)

        with computer.capture():
            await asyncio.create_task(child())
            assert computer._result is None

    asyncio.run(run())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_dpo_loss_wrapper_async_child_with_sources_uses_main_loss_only():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._original_loss_fn = lambda labels: torch.tensor(1.0)
    computer._dpo_segments = True
    computer._source_ids = [0]

    async def run():
        async def child():
            assert computer._wrapped_loss_fn(labels=torch.tensor([[1]])).item() == 1.0

        with computer.capture():
            await asyncio.create_task(child())
            computer._source_ids = []

    asyncio.run(run())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("nested", [False, True])
def test_channel_loss_discards_observation_when_forward_fails(nested):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._position_ids = torch.tensor([[0, 1]])
    computer._original_loss_fn = lambda logits, labels, vocab_size, shift_labels: torch.tensor(1.0)
    labels = torch.tensor([[1, 2]])
    logits = torch.randn(1, 2, 4)
    outer = computer.capture() if nested else nullcontext()
    with outer:
        with pytest.raises(RuntimeError, match="model failure"):
            with computer.capture():
                computer._wrapped_loss_fn(logits, labels, 4, labels)
                raise RuntimeError("model failure")
        assert computer._result is None


def _gloo_channel_provenance_worker(rank, world_size, scenario, endpoint):
    import veomni.data.data_collator as data_collator

    global_rank = rank
    try:
        torch.distributed.init_process_group(
            "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=40)
        )
        sp_size = 2 if scenario in {"two_stripes", "strict_two_stripes"} else world_size
        dp_size = world_size // sp_size
        dpo = scenario in {"dpo_cp", "dpo_observer_error", "dpo_global_cu_mismatch", "dpo_attention_cu_mismatch"}
        use_cp = scenario in {
            "cp_tail",
            "dpo_cp",
            "dpo_observer_error",
            "dpo_global_cu_mismatch",
            "dpo_attention_cu_mismatch",
            "two_stripes",
            "strict_two_stripes",
            "mismatched_mask",
            "mismatched_global_cu",
        }
        state = parallel_state_module._init_parallel_state(
            dp_size=dp_size,
            dp_shard_size=dp_size,
            cp_size=sp_size if use_cp else 1,
            ulysses_size=1 if use_cp else sp_size,
            device_type="cpu",
            name="channel-provenance-test",
        )
        source_base = 10 * (global_rank // sp_size) if scenario in {"two_stripes", "strict_two_stripes"} else 0
        if scenario == "empty_internal_reset":
            positions = [[], [0, 1, 0, 1]]
        elif dpo:
            positions = [[0, 1, 2, 0, 1], [0, 0, 1, 2]]
        elif scenario == "ulysses_no_tail":
            positions = [[0, 1, 2], [0, 1, 2, 3, 4]]
        else:
            positions = [[0, 1, 2], [0, 1, 2, 3]]

        def features():
            return [
                {
                    "input_ids": torch.arange(len(pos), dtype=torch.long) % 5,
                    "labels": torch.arange(len(pos), dtype=torch.long) % 5,
                    "position_ids": torch.tensor(pos, dtype=torch.long),
                    "attention_mask": torch.tensor([int(i % 3 != 1) for i in range(len(pos))]),
                    "ds_idx": source_base + index,
                }
                for index, pos in enumerate(positions)
            ]

        cfg = ChannelLossConfig(enable=True, interval=1, strict=scenario == "strict_two_stripes")
        batch = data_collator.MainCollator(channel_loss_source_id_keys=tuple(cfg.source_id_keys))(features())
        proof = batch[data_collator.CHANNEL_LOSS_PROVENANCE]
        if (scenario == "mismatched_mask" and state.sp_rank == 1) or (
            scenario == "strict_two_stripes" and global_rank == 3
        ):
            proof["input_mask"][0] = ~proof["input_mask"][0]
        if scenario in {"mismatched_global_cu", "dpo_global_cu_mismatch"} and state.sp_rank == 1:
            key = "linear_attn_cu_seqlens_list_q"
            if key not in batch:
                key = "linear_attn_cu_seq_lens_q"
            cu = batch[key]
            cu = cu.clone() if isinstance(cu, torch.Tensor) else list(cu)
            cu[1] -= 1
            batch[key] = cu
        if scenario == "dpo_attention_cu_mismatch" and state.sp_rank == 1:
            batch["cu_seq_lens_q"] = batch["cu_seq_lens_q"].clone()
            batch["cu_seq_lens_q"][1] -= 1
        trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
        callback = ChannelLossCallback(trainer)
        callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch], source_repeat=2 if dpo else 1)
        callback.computer.before_micro_step()

        def training_loss(logits, labels, vocab_size, shift_labels=None):
            return F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX)

        callback.computer._original_loss_fn = training_loss
        labels = batch["labels"]
        logits = torch.zeros(*labels.shape, 5, requires_grad=True)
        reference = logits.detach().clone().requires_grad_()
        failed = False
        try:
            with callback.computer.capture():
                loss = callback.computer._wrapped_loss_fn(logits, labels, 5, shift_labels=labels)
                if dpo:
                    token_loss = F.cross_entropy(
                        logits.reshape(-1, 5), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
                    ).reshape_as(labels)
                    log_probs = -token_loss
                    if scenario == "dpo_observer_error" and state.sp_rank == 1:

                        class FaultyLogProbs:
                            def detach(self):
                                raise RuntimeError("injected observer-only allocation failure")

                        log_probs = FaultyLogProbs()
                    callback.computer.observe_dpo_log_probs(log_probs, labels, batch["cu_seq_lens_q"])
        except ChannelLossMetadataError:
            if scenario != "strict_two_stripes":
                raise
            failed = True
        if scenario == "strict_two_stripes":
            assert failed
        expected_loss = training_loss(reference, labels, 5, shift_labels=labels)
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)
        result = callback.computer._result
        if scenario in {
            "mismatched_mask",
            "mismatched_global_cu",
            "dpo_global_cu_mismatch",
            "dpo_attention_cu_mismatch",
            "dpo_observer_error",
            "strict_two_stripes",
        }:
            assert result is None
            return

        # Recreate the exact global shifted training labels independently of the
        # source observer, then mask DPO chosen/rejected segment ends.
        raw_labels = [torch.arange(len(pos), dtype=torch.long) % 5 for pos in positions]
        for raw in raw_labels[1:]:
            if raw.numel():
                raw[0] = IGNORE_INDEX
        shifted = F.pad(torch.cat(raw_labels), (0, 1), value=IGNORE_INDEX)[1:]
        if dpo:
            shifted[torch.tensor([2, 4, 5, 8])] = IGNORE_INDEX
        expected = []
        start = 0
        for pos in positions:
            expected.append(int((shifted[start : start + len(pos)] != IGNORE_INDEX).sum()))
            start += len(pos)
        assert [item["source_id"] for item in result] == [source_base, source_base + 1]
        assert [int(item["token_count"].item()) for item in result] == expected
        for item, count in zip(result, expected):
            torch.testing.assert_close(item["loss_sum"], torch.tensor(count * math.log(5)), rtol=1e-6, atol=1e-6)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        parallel_state_module.clear_parallel_state()


@pytest.mark.parametrize(
    ("scenario", "world_size"),
    [
        ("empty_internal_reset", 2),
        ("ulysses_no_tail", 2),
        ("cp_tail", 2),
        ("dpo_cp", 2),
        ("mismatched_mask", 2),
        ("mismatched_global_cu", 2),
        ("dpo_global_cu_mismatch", 2),
        ("dpo_attention_cu_mismatch", 2),
        ("dpo_observer_error", 2),
        ("two_stripes", 4),
        ("strict_two_stripes", 4),
    ],
)
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_real_gloo_channel_provenance(scenario, world_size):
    spawn_with_rendezvous(_gloo_channel_provenance_worker, args=(world_size, scenario), nprocs=world_size)


def _gloo_packed_loss_worker(rank, world_size, scenario, endpoint):
    global_rank = rank
    try:
        torch.distributed.init_process_group(
            "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=40)
        )
        sp_size = 2 if scenario == "two_stripes" else world_size
        dp_size = world_size // sp_size
        use_cp = scenario in (
            "headwise",
            "headwise_tail",
            "cu_mismatch",
            "layout_mismatch",
            "global_cu_mismatch",
            "cp_contiguous",
            "cp_global_contiguous",
            "cp_global_tail",
            "cp_mixed_geometry",
            "cp_shared_stale",
            "cp_no_global",
            "cp_only_global",
            "cp_only_global_strict",
            "cp_asymmetric",
            "cp_stale_contiguous",
            "cp_no_local",
            "two_stripes",
        )
        state = parallel_state_module._init_parallel_state(
            dp_size=dp_size,
            dp_shard_size=dp_size,
            cp_size=sp_size if use_cp else 1,
            ulysses_size=1 if use_cp else sp_size,
            device_type="cpu",
            name="channel-loss-test",
        )
        rank, world_size = state.sp_rank, state.sp_size
        if scenario == "two_stripes":
            expected_group = [global_rank // 2 * 2, global_rank // 2 * 2 + 1]
            assert torch.distributed.get_process_group_ranks(state.sp_group) == expected_group
        if scenario == "strict_step_missing":
            cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
            trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
            callback = ChannelLossCallback(trainer)
            batch = {} if rank == 0 else {"ds_idx": 1}
            with pytest.raises(ValueError, match="no configured source ID"):
                callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch])
            return
        computer = ChannelLossComputer(parallel_state=state)
        computer.strict = scenario in (
            "headwise",
            "headwise_tail",
            "ulysses",
            "ulysses_local_cu",
            "ulysses_local_tail",
            "cp_contiguous",
            "strict_missing",
            "cp_only_global_strict",
            "bad_global_no_local_strict",
        )
        if scenario in (
            "ulysses",
            "ulysses_local_cu",
            "ulysses_local_tail",
            "ulysses_stale_local",
            "cp_contiguous",
            "cp_global_contiguous",
            "cp_global_tail",
            "ulysses_no_global",
            "ulysses_no_global_stale",
        ):
            computer._source_ids = ["first", "second", "third"]
            tail = scenario in ("ulysses_local_tail", "cp_global_tail")
            full_labels = torch.tensor(
                [0, 1, IGNORE_INDEX, 2, 3, 4, 1, 0] + ([IGNORE_INDEX, IGNORE_INDEX] if tail else [])
            )
            full_positions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1] + ([0, 0] if tail else []))
            full_logits = torch.arange(full_labels.numel() * 5, dtype=torch.float32).reshape(-1, 5) / 19
            width = full_labels.numel() // world_size
            start, end = rank * width, (rank + 1) * width
            labels = full_labels[start:end]
            logits = full_logits[start:end].clone().requires_grad_()
            computer._position_ids = full_positions[start:end].reshape(1, -1)
            if scenario == "cp_global_tail":
                computer._packed_cu_seqlens = [0, 3, 6, 8, 10]
                computer._packed_global_cu_seqlens = [0, 3, 6, 8, 10]
                computer._tail_padding_length = 2
            elif scenario == "ulysses_local_tail":
                computer._packed_cu_seqlens = [0, 3, 5, 5, 5] if rank == 0 else [0, 0, 1, 3, 5]
                computer._packed_global_cu_seqlens = [0, 3, 6, 8, 10]
                computer._tail_padding_length = 2
            elif scenario in (
                "ulysses_local_cu",
                "ulysses_stale_local",
                "cp_contiguous",
                "ulysses_no_global",
                "ulysses_no_global_stale",
            ):
                computer._packed_cu_seqlens = [0, 3, 4, 4] if rank == 0 else [0, 0, 2, 4]
                if scenario in ("ulysses_stale_local", "ulysses_no_global_stale") and rank == 1:
                    computer._packed_cu_seqlens = [0, 0, 1, 4]
                if scenario not in ("ulysses_no_global", "ulysses_no_global_stale"):
                    computer._packed_global_cu_seqlens = [0, 3, 6, 8]
            else:
                computer._packed_cu_seqlens = [0, 3, 6, 8]
                computer._packed_global_cu_seqlens = [0, 3, 6, 8]
                if scenario == "ulysses":
                    computer._tail_padding_length = 0
            tokens = F.cross_entropy(full_logits, full_labels, ignore_index=IGNORE_INDEX, reduction="none")
            expected = torch.stack((tokens[:3].sum(), tokens[3:6].sum(), tokens[6:].sum()))
            expected_counts = [2, 3, 2]
        elif scenario == "headwise_tail":
            computer._source_ids = ["first", "second", "empty"]
            computer._position_ids = torch.zeros((1, 8), dtype=torch.long)
            computer._packed_cu_seqlens = [0, 2, 4, 4, 8]
            computer._packed_global_cu_seqlens = [0, 3, 5, 5, 16]
            computer._packed_global_physical_seq_len = 32
            computer._tail_padding_length = 11
            labels = torch.tensor(
                [
                    0 if rank < 3 else IGNORE_INDEX,
                    IGNORE_INDEX,
                    1 if rank < 2 else IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                ]
            )
            logits = (torch.arange(40, dtype=torch.float32).reshape(8, 5) / 19 + rank / 13).requires_grad_()
            expected = torch.zeros(3)
            for other_rank in range(world_size):
                other_labels = torch.tensor(
                    [
                        0 if other_rank < 3 else IGNORE_INDEX,
                        IGNORE_INDEX,
                        1 if other_rank < 2 else IGNORE_INDEX,
                        IGNORE_INDEX,
                        IGNORE_INDEX,
                        IGNORE_INDEX,
                        IGNORE_INDEX,
                        IGNORE_INDEX,
                    ]
                )
                other_logits = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 19 + other_rank / 13
                tokens = F.cross_entropy(other_logits, other_labels, ignore_index=IGNORE_INDEX, reduction="none")
                expected += torch.stack((tokens[:2].sum(), tokens[2:4].sum(), tokens[4:4].sum()))
            expected_counts = [3, 2, 0]
        else:
            computer._source_ids = (
                ["first", "second"] if rank != 0 or scenario not in ("missing", "strict_missing") else []
            )
            computer._position_ids = torch.tensor([[0, 1, 0, 0]])
            computer._packed_cu_seqlens = [0, 2, 4]
            if use_cp and scenario != "cp_no_global":
                computer._packed_global_cu_seqlens = [0, 2 * world_size, 4 * world_size]
                computer._packed_global_physical_seq_len = 4 * world_size
            labels = torch.tensor([0, 1, 2, IGNORE_INDEX])
            logits = (torch.arange(20, dtype=torch.float32).reshape(4, 5) / 19 + rank / 13).requires_grad_()
            expected = torch.zeros(2)
            for other_rank in range(world_size):
                other_logits = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 19 + other_rank / 13
                tokens = F.cross_entropy(other_logits, labels, ignore_index=IGNORE_INDEX, reduction="none")
                expected += torch.stack((tokens[:2].sum(), tokens[2:].sum()))
            expected_counts = [2 * world_size, world_size]
        if scenario == "side_failure" and rank == 0:

            def fail_ce(*args, **kwargs):
                raise RuntimeError("injected rank-local observer failure")

            computer._per_token_ce = fail_ce
        if scenario == "layout_mismatch" and rank == 0:
            computer._packed_cu_seqlens = [0, 4, 8]
            computer._packed_global_cu_seqlens = [0, 4, 8]
        if scenario == "cu_mismatch" and rank == 0:
            computer._packed_cu_seqlens = [0, 1, 4]
        if scenario == "global_cu_mismatch":
            computer._packed_global_cu_seqlens = [0, 3, 6] if rank == 0 else [0, 4, 8]
            computer._packed_global_physical_seq_len = 8
        if scenario == "cp_mixed_geometry":
            computer._packed_cu_seqlens = [0, 2, 4] if rank == 0 else [0, 0, 4]
            computer._packed_global_cu_seqlens = [0, 4, 8]
            computer._packed_global_physical_seq_len = 8
        if scenario == "cp_shared_stale":
            computer._position_ids = torch.tensor([[0, 1, 2, 3]])
        if scenario == "cp_only_global":
            computer._packed_cu_seqlens = [0, 4, 8]
            computer._position_ids = torch.tensor([[0, 3, 0, 3] if rank == 0 else [1, 2, 1, 2]])
        if scenario == "cp_only_global_strict" and rank == 0:
            computer._packed_cu_seqlens = [0, 4, 8]
        if scenario == "cp_asymmetric":
            computer._packed_cu_seqlens = [0, 4, 16]
            computer._packed_global_cu_seqlens = [0, 4, 16]
            computer._packed_global_physical_seq_len = 16
            computer._position_ids = torch.tensor(
                [[0, 1, 2, 3, 0, 1, 2, 3] if rank == 0 else [1, 2, 3, 4, 5, 6, 7, 8]]
            )
            labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
            logits = (torch.arange(40, dtype=torch.float32).reshape(8, 5) / 19 + rank / 13).requires_grad_()
        if scenario == "cp_stale_contiguous":
            computer._packed_cu_seqlens = [0, 3, 4] if rank == 0 else [0, 0, 4]
            computer._packed_global_cu_seqlens = [0, 3, 8]
            computer._packed_global_physical_seq_len = 8
            computer._position_ids = torch.tensor([[0, 1, 2, 3]])
        if scenario in (
            "ulysses_empty_tail_no_global",
            "ulysses_empty_tail_global_missing",
            "ulysses_empty_tail_global_explicit",
            "ulysses_empty_tail_local_explicit",
            "ulysses_empty_tail_unknown",
            "ulysses_empty_tail_local_missing",
        ):
            computer._source_ids = ["empty", "real"]
            computer._position_ids = torch.tensor([[0, 1] if rank == 0 else [2, 0]])
            computer._tail_padding_length = 1
            all_labels = torch.tensor([0, 1, 2, IGNORE_INDEX])
            all_logits = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 19
            labels = all_labels[rank * 2 : rank * 2 + 2]
            logits = all_logits[rank * 2 : rank * 2 + 2].clone().requires_grad_()
            if scenario == "ulysses_empty_tail_no_global":
                computer._packed_cu_seqlens = [0, 0, 2]
            elif scenario == "ulysses_empty_tail_global_missing":
                computer._packed_cu_seqlens = [0, 3, 4]
                computer._packed_global_cu_seqlens = [0, 3, 4]
            elif scenario == "ulysses_empty_tail_unknown":
                computer._packed_cu_seqlens = [0, 3, 4]
                computer._packed_global_cu_seqlens = [0, 3, 4]
                computer._tail_padding_length = None
            elif scenario == "ulysses_empty_tail_local_missing":
                computer._packed_cu_seqlens = [0, 2, 2] if rank == 0 else [0, 1, 2]
                computer._packed_global_cu_seqlens = [0, 3, 4]
            elif scenario == "ulysses_empty_tail_global_explicit":
                computer._packed_cu_seqlens = [0, 0, 3, 4]
                computer._packed_global_cu_seqlens = [0, 0, 3, 4]
            else:
                computer._packed_cu_seqlens = [0, 0, 2, 2] if rank == 0 else [0, 0, 1, 2]
                computer._packed_global_cu_seqlens = [0, 0, 3, 4]
                computer._packed_global_physical_seq_len = 4
                tokens = F.cross_entropy(all_logits, all_labels, ignore_index=IGNORE_INDEX, reduction="none")
                expected = torch.stack((tokens[:0].sum(), tokens[:3].sum()))
                expected_counts = [0, 3]
        if scenario in ("ulysses_supervised_tail", "ulysses_valid_tail"):
            computer._source_ids = ["first", "second"]
            computer._position_ids = torch.tensor([[0, 1, 2, 0] if rank == 0 else [1, 2, 0, 0]])
            computer._packed_cu_seqlens = [0, 3, 6, 8]
            computer._packed_global_cu_seqlens = [0, 3, 6, 8]
            computer._tail_padding_length = 2
            all_labels = torch.tensor([0, 1, 2, 3, 4, 0, IGNORE_INDEX, IGNORE_INDEX])
            if scenario == "ulysses_supervised_tail":
                all_labels[-2:] = torch.tensor([1, 2])
            all_logits = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 19
            labels = all_labels[rank * 4 : rank * 4 + 4]
            logits = all_logits[rank * 4 : rank * 4 + 4].clone().requires_grad_()
            tokens = F.cross_entropy(all_logits, all_labels, ignore_index=IGNORE_INDEX, reduction="none")
            expected = torch.stack((tokens[:3].sum(), tokens[3:6].sum()))
            expected_counts = [3, 3]
        if scenario == "cp_no_local":
            computer._packed_cu_seqlens = None
            computer._packed_global_cu_seqlens = None
        if scenario in ("bad_global_no_local", "bad_global_no_local_strict"):
            computer._packed_cu_seqlens = None
            computer._packed_global_cu_seqlens = torch.tensor([[0, 8]]) if rank == 0 else [0, 8]
        reference = logits.detach().clone().requires_grad_()
        reference_loss = F.cross_entropy(reference, labels, ignore_index=IGNORE_INDEX, reduction="sum")

        def main_loss(logits, labels, vocab_size, shift_labels):
            return F.cross_entropy(
                logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
            )

        computer._original_loss_fn = main_loss
        if scenario in ("strict_missing", "bad_global_no_local_strict", "cp_only_global_strict"):
            with pytest.raises(
                ChannelLossMetadataError, match="SP ranks disagree|no causal-loss|global packed CU|CP requires"
            ):
                with computer.capture():
                    computer._wrapped_loss_fn(logits.unsqueeze(0), labels.unsqueeze(0), 5, labels.unsqueeze(0))
            return
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits.unsqueeze(0), labels.unsqueeze(0), 5, labels.unsqueeze(0))
        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
        torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)
        if scenario in (
            "missing",
            "side_failure",
            "layout_mismatch",
            "cu_mismatch",
            "global_cu_mismatch",
            "ulysses_stale_local",
            "ulysses_no_global",
            "ulysses_no_global_stale",
            "ulysses_empty_tail_no_global",
            "ulysses_empty_tail_global_missing",
            "ulysses_empty_tail_global_explicit",
            "ulysses_empty_tail_unknown",
            "ulysses_empty_tail_local_missing",
            "ulysses_supervised_tail",
            "cp_mixed_geometry",
            "cp_no_global",
            "cp_shared_stale",
            "cp_only_global",
            "cp_asymmetric",
            "cp_stale_contiguous",
            "cp_no_local",
            "cp_global_contiguous",
            "cp_global_tail",
            "bad_global_no_local",
        ):
            assert computer._result is None
            return
        assert [item["source_id"] for item in computer._result] == computer._source_ids
        torch.testing.assert_close(torch.stack([item["loss_sum"] for item in computer._result]), expected)
        assert [item["token_count"].item() for item in computer._result] == expected_counts
    finally:
        if torch.distributed.is_initialized():
            parallel_state_module.clear_parallel_state()
            torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
@pytest.mark.parametrize(
    "world_size,scenario",
    [
        (2, "headwise"),
        (4, "headwise"),
        (4, "headwise_tail"),
        (4, "two_stripes"),
        (2, "ulysses"),
        (2, "ulysses_local_cu"),
        (2, "ulysses_local_tail"),
        (2, "ulysses_no_global"),
        (2, "ulysses_no_global_stale"),
        (2, "ulysses_empty_tail_no_global"),
        (2, "ulysses_empty_tail_global_missing"),
        (2, "ulysses_empty_tail_global_explicit"),
        (2, "ulysses_empty_tail_local_explicit"),
        (2, "ulysses_empty_tail_unknown"),
        (2, "ulysses_empty_tail_local_missing"),
        (2, "ulysses_supervised_tail"),
        (2, "ulysses_valid_tail"),
        (2, "cp_contiguous"),
        (2, "cp_global_contiguous"),
        (2, "cp_global_tail"),
        (2, "ulysses_stale_local"),
        (4, "ulysses"),
        (2, "missing"),
        (2, "strict_missing"),
        (2, "strict_step_missing"),
        (2, "side_failure"),
        (2, "layout_mismatch"),
        (2, "cu_mismatch"),
        (2, "global_cu_mismatch"),
        (2, "cp_mixed_geometry"),
        (2, "cp_shared_stale"),
        (2, "cp_no_global"),
        (2, "cp_only_global"),
        (2, "cp_only_global_strict"),
        (2, "cp_asymmetric"),
        (2, "cp_stale_contiguous"),
        (2, "cp_no_local"),
        (2, "bad_global_no_local"),
        (2, "bad_global_no_local_strict"),
    ],
)
def test_channel_loss_real_gloo_packed_parity(world_size, scenario):
    spawn_with_rendezvous(_gloo_packed_loss_worker, args=(world_size, scenario), nprocs=world_size)
