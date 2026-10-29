import importlib.util
import math
import os
from contextlib import nullcontext
from functools import partial
from importlib.machinery import ModuleSpec
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


def test_channel_loss_uses_shift_labels_when_sequence_parallel_is_on():
    torch.manual_seed(2)
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    shift_labels = torch.tensor([[4, 5, 6]], dtype=torch.long)

    def run(sp_enabled: bool, shift: torch.Tensor | None):
        computer = ChannelLossComputer(parallel_state=SimpleNamespace(sp_enabled=sp_enabled))
        computer._source_ids = [0]
        computer._position_ids = None
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


def test_dpo_channel_loss_emits_policy_totals(monkeypatch):
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
        "ds_idx": torch.tensor([3, 4]),
        "source_name": ["train/a", "train/b"],
    }

    try:
        base.channel_loss_callback.computer.install(policy_model)
        TextDPOTrainer.on_step_begin(trainer, micro_batches=[micro_batch])
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


def _test_parallel_state():
    return SimpleNamespace(sp_enabled=False, sp_group=None, sp_size=1, sp_rank=0, dp_size=1, dp_group=None)


def _install_test_parallel_state(monkeypatch):
    state = _test_parallel_state()
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(channel_loss_module, "get_parallel_state", lambda: state)
    monkeypatch.setitem(parallel_state_module._PARALLEL_STATE_REGISTRY, "base", state)
    monkeypatch.setattr(callback_base_module, "get_parallel_state", lambda: state)
    return state


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_chunk_capture_error_recomputes_metrics_and_replaces_previous_result():
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._result = [{"source_id": "stale"}]

    def loss_fn(logits, labels, vocab_size):
        chunk_loss._CAPTURE.get()["error"] = "observer failed"
        return logits.square().sum()

    computer._original_loss_fn = loss_fn
    labels = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 4, requires_grad=True)
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, labels, 4)
    loss.backward()
    assert computer._result[0]["source_id"] == "source"
    assert computer._result[0]["token_count"].item() == 2
    torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("override", ["compute_side_channel", "_compute_channel_loss"])
def test_channel_loss_preserves_legacy_observer_overrides(override):
    calls = []

    def side_channel(self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100):
        calls.append("side")

    def compute(self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index):
        calls.append("compute")
        return []

    method = side_channel if override == "compute_side_channel" else compute
    custom_computer = type("LegacyComputer", (ChannelLossComputer,), {override: method})
    computer = custom_computer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    loss.backward()
    assert calls == ["side" if override == "compute_side_channel" else "compute"]
    torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)


def test_loss_call_binding_does_not_treat_weight_as_ignore_index():
    def loss_fn(logits, labels, vocab_size, hidden_states, weights, *args):
        pass

    values = (torch.zeros(2), torch.ones(2), 3, torch.ones(2, 4), torch.ones(3, 4))
    bound = channel_loss_module._bind_loss_call_args(loss_fn, values, {})
    assert bound["weights"] is values[-1]
    assert "ignore_index" not in bound


@pytest.mark.parametrize("strict", [False, True])
def test_legacy_observer_exception_keeps_upstream_propagation(strict):
    class LegacyComputer(ChannelLossComputer):
        def compute_side_channel(
            self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100
        ):
            raise RuntimeError("observer only")

    computer = LegacyComputer(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="observer only"), computer.capture():
        computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    assert computer._result is None


@pytest.mark.parametrize("override", ["compute_side_channel", "_compute_channel_loss"])
@pytest.mark.parametrize("bias_kind", ["full", "factored"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("variadic", [False, True])
def test_legacy_observer_without_bias_support_never_reports_unbiased_metrics(override, bias_kind, strict, variadic):
    calls = []

    def side(self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100):
        calls.append(1)

    def compute(self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index):
        calls.append(1)
        return []

    def side_variadic(
        self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100, **kwargs
    ):
        calls.append(1)

    def compute_variadic(
        self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index, **kwargs
    ):
        calls.append(1)
        return []

    methods = {
        "compute_side_channel": side_variadic if variadic else side,
        "_compute_channel_loss": compute_variadic if variadic else compute,
    }
    legacy = type("LegacyComputer", (ChannelLossComputer,), {override: methods[override]})
    computer = legacy(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    hidden = torch.randn(1, 3, 2, requires_grad=True)
    weight = torch.randn(4, 2)
    bias = torch.randn(1, 3, 4 if bias_kind == "full" else 2)
    bias_weight = torch.randn(4, 2) if bias_kind == "factored" else None

    def loss_fn(hidden_states, labels, weights, logits_bias=None, logits_bias_weight=None):
        scores = F.linear(hidden_states, weights)
        scores = scores + (
            F.linear(logits_bias, logits_bias_weight) if logits_bias_weight is not None else logits_bias
        )
        return scores.square().sum()

    computer._original_loss_fn = loss_fn
    labels = torch.tensor([[0, 1, 2]])
    with pytest.raises(ChannelLossMetadataError, match="logits bias") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(hidden, labels, weight, logits_bias=bias, logits_bias_weight=bias_weight)
    if not strict:
        loss.backward()
        assert torch.isfinite(hidden.grad).all()
    assert not calls and computer._result is None


def test_modern_observer_override_receives_supported_bias():
    seen = []

    class ModernComputer(ChannelLossComputer):
        def compute_side_channel(
            self,
            logits,
            labels,
            vocab_size,
            hidden_states,
            weights,
            shift_labels=None,
            ignore_index=-100,
            logits_bias=None,
            logits_bias_weight=None,
        ):
            seen.append((logits_bias, logits_bias_weight))

    computer = ModernComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size, logits_bias=None: logits.square().sum()
    bias = torch.randn(1, 2, 3)
    with computer.capture():
        loss = computer._wrapped_loss_fn(
            torch.ones(1, 2, 3, requires_grad=True), torch.tensor([[0, 1]]), 3, logits_bias=bias
        )
    loss.backward()
    assert seen == [(bias, None)]


@pytest.mark.parametrize("strict", [False, True])
def test_legacy_compute_exception_respects_strict_mode(strict):
    class LegacyComputer(ChannelLossComputer):
        def _compute_channel_loss(
            self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index
        ):
            raise RuntimeError("legacy compute failed")

    computer = LegacyComputer(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="legacy compute failed") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    if not strict:
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    assert computer._result is None


def test_positional_only_bias_is_not_claimed_as_supported():
    def observer(logits_bias=None, /, **kwargs):
        pass

    bias = torch.ones(1)
    supported = channel_loss_module._supported_observer_kwargs(observer, {"logits_bias": bias, "defer_sp": True})
    assert "logits_bias" not in supported and "defer_sp" not in supported


def _gloo_chunk_capture_fallback_worker(rank, world_size, endpoint):
    from datetime import timedelta

    from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        state = SimpleNamespace(sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size)
        computer = ChannelLossComputer(parallel_state=state)
        computer._source_ids = ["source"]
        computer._position_ids = torch.tensor([[rank * 2, rank * 2 + 1]])
        labels = torch.tensor([[0, 1]]) if rank == 0 else torch.tensor([[2, IGNORE_INDEX]])
        logits = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).requires_grad_()

        def loss_fn(logits, labels, vocab_size):
            capture = chunk_loss._CAPTURE.get()
            if rank == 0:
                capture["error"] = "observer failed"
            else:
                capture["per_token"] = F.cross_entropy(
                    logits.detach().reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).reshape_as(labels)
            return logits.square().sum()

        computer._original_loss_fn = loss_fn
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, labels, 4)
        loss.backward()
        assert computer._result[0]["token_count"].item() == 3
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_gloo_chunk_capture_fallback_keeps_ranks_in_same_collective():
    spawn_with_rendezvous(_gloo_chunk_capture_fallback_worker, args=(2,), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_compiled_chunk_loss_falls_back_to_detached_source_observation():
    from veomni.ops import resolve_op

    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[0, 1, 2, 3]])
    shift = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    compiled = torch.compile(op, backend="eager", fullgraph=True)
    with torch.no_grad():
        reference = op(hidden, shift, weight, chunk_size=3)
        # Compile while unsampled, then enable observation on the same graph.
        torch.testing.assert_close(compiled(hidden, shift, weight, chunk_size=3), reference)

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(4).unsqueeze(0)

    def model_loss(logits, labels, vocab_size, hidden_states, weights):
        shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
        return compiled(hidden_states, shifted, weights, chunk_size=3)

    computer._original_loss_fn = model_loss
    with torch.no_grad(), computer.capture():
        actual = computer._wrapped_loss_fn(None, labels, 7, hidden, weight)
    torch.testing.assert_close(actual, reference)
    expected = F.cross_entropy(F.linear(hidden, weight).reshape(-1, 7), shift.reshape(-1), reduction="none")
    torch.testing.assert_close(computer._result[0]["loss_sum"], expected.sum())
    assert computer._result[0]["token_count"].item() == 3


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
