import importlib.util
import math
import os
import queue
import time
import traceback
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from importlib.machinery import ModuleSpec
from types import SimpleNamespace


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec


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
    from veomni.distributed.parallel_state import use_parallel_state
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
        ChannelLossCollectiveError,
        ChannelLossComputer,
        ChannelLossMetadataError,
    )
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
        self.parallel_state = "base"

    def forward(self, x, use_cache=False):
        return SimpleNamespace(loss=(self.weight * x).sum())


def _expected_segment(logits, labels, start, end):
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous().view(-1)
    losses = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)), shifted, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = shifted[start : end + 1] != IGNORE_INDEX
    return losses[start : end + 1][valid].sum().item(), valid.sum().item()


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


def test_channel_loss_uses_shift_labels_when_sequence_parallel_is_on():
    torch.manual_seed(2)
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    shift_labels = torch.tensor([[4, 5, 6]], dtype=torch.long)

    def run(sp_enabled: bool, shift: torch.Tensor | None):
        computer = ChannelLossComputer(
            parallel_state=SimpleNamespace(sp_enabled=sp_enabled, sp_group=None, sp_size=1, sp_rank=0)
        )
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
        {
            "source_id": 0,
            "loss_sum": pytest.approx(loss_0),
            "token_count": tokens_0,
        },
        {
            "source_id": 1,
            "loss_sum": pytest.approx(loss_1),
            "token_count": tokens_1,
        },
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


def test_channel_loss_computer_counts_data_metrics_with_sequence_parallel():
    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.ones(4),
        labels_flat=torch.tensor([1, 2, 3, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 1, 0]),
        source_ids=["repoqa", "other"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=_test_parallel_state(sp_enabled=True),
        strict=True,
        include_data_stats=True,
    )

    assert [item["sample_count"].item() for item in result] == [1, 1]
    assert [item["input_token_count"].item() for item in result] == [2, 1]
    assert [item["token_count"].item() for item in result] == [2, 1]


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
    assert not computer.strict_observation_errors
    assert [entry["source_id"] for entry in computer._result] == ["first", "second"]
    for row, entry in enumerate(computer._result):
        torch.testing.assert_close(entry["loss_sum"], per_token[row].sum(), rtol=1e-6, atol=1e-6)
        assert entry["token_count"].item() == targets[row].ne(IGNORE_INDEX).sum().item()


@pytest.mark.parametrize("bias_kind", ["full", "factored"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_unsupported_bias_without_false_metrics(monkeypatch, bias_kind):
    from veomni.models.loss_utils import ForCausalLMLoss

    _install_test_parallel_state(monkeypatch)

    def unsupported_op(*args, **kwargs):
        pytest.fail("An unsupported bias backend must be rejected before execution")

    unsupported_op.impl = "liger_kernel"
    model = torch.nn.Module()
    model.loss_function = partial(ForCausalLMLoss, op=unsupported_op)
    computer = ChannelLossComputer()
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(5).unsqueeze(0)
    kwargs = dict(
        labels=torch.zeros(1, 5, dtype=torch.long),
        hidden_states=torch.randn(1, 5, 3),
        weights=torch.randn(7, 3),
        logits_bias=torch.randn(1, 5, 2 if bias_kind == "factored" else 7),
    )
    if bias_kind == "factored":
        kwargs["logits_bias_weight"] = torch.randn(7, 2)
    try:
        computer.install(model)
        with pytest.raises(NotImplementedError, match="logits_bias is only supported"), computer.capture():
            model.loss_function(**kwargs)
    finally:
        computer.uninstall()
    assert not computer._result


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


@pytest.mark.parametrize("capture_count", [0, 2])
def test_channel_loss_capture_is_scoped_to_instance_loss(monkeypatch, capture_count):
    from veomni.trainer.callbacks.channel_loss_callback import get_chunk_loss_consumer

    _install_test_parallel_state(monkeypatch)

    class Model(torch.nn.Module):
        def loss_function(self, logits, labels, vocab_size):
            consumer = get_chunk_loss_consumer()
            assert consumer is not None
            for _ in range(capture_count):
                consumer(torch.zeros_like(labels, dtype=torch.float32))
            return "main-loss"

    computer = ChannelLossComputer()
    computer._source_ids = ["a"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    model = Model()
    original = model.loss_function
    computer.install(model)
    with computer.capture():
        assert model.loss_function(torch.randn(1, 3, 8), torch.tensor([[1, 2, 3]]), 8) == "main-loss"
    computer.uninstall()
    assert model.loss_function == original
    assert get_chunk_loss_consumer() is None
    assert bool(computer._result) == (capture_count == 0)


def test_chunk_capture_restores_nested_scope_and_is_thread_local():
    import threading

    from veomni.trainer.callbacks.channel_loss_callback import capture_chunk_loss_per_token, get_chunk_loss_consumer

    first, second = [], []
    barrier = threading.Barrier(2)
    errors = []

    def worker(output):
        try:
            with capture_chunk_loss_per_token(output.append):
                barrier.wait(timeout=5)
                get_chunk_loss_consumer()(torch.tensor(1))
                with capture_chunk_loss_per_token(second.append):
                    assert get_chunk_loss_consumer() == second.append
                assert get_chunk_loss_consumer() == output.append
        except Exception as error:
            errors.append(error)

    worker_thread = threading.Thread(target=worker, args=(first,))
    worker_thread.start()
    with capture_chunk_loss_per_token(second.append):
        barrier.wait(timeout=5)
        assert get_chunk_loss_consumer() == second.append
    worker_thread.join(timeout=5)
    assert not worker_thread.is_alive()
    assert not errors
    assert len(first) == 1 and not second
    assert get_chunk_loss_consumer() is None


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
    group = object()
    parallel_state = _test_parallel_state(sp_enabled=True, sp_group=group, sp_size=2)
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
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "batch0", "loss_sum": 8.0, "token_count": 3},
        {"source_id": "batch1", "loss_sum": 37.0, "token_count": 3},
    ]
    assert gather_inputs == [(torch.float32, (2,)), (torch.int64, (2, 3))]


def test_channel_loss_sp_reduce_handles_empty_local_rank(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True, sp_group=object(), sp_size=2)
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

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_gather", fake_all_gather)

    result = ChannelLossComputer._reduce_sp(
        local_segments=[],
        source_ids=["remote"],
        device=torch.device("cpu"),
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [{"source_id": "remote", "loss_sum": 4.0, "token_count": 2}]


def test_channel_loss_sp_reduce_filters_uneven_remote_padding_segments(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True, sp_group=object(), sp_size=2)
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
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 3.0, "token_count": 2}]


def test_channel_loss_sp_aggregation_does_not_extract_segment_scalars(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True)

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
            parallel_state=parallel_state,
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
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
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
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=None,
        source_ids=["doc"],
        positions=torch.tensor([[0, 1, 0, 0]]),
        seq_len=4,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [{"source_id": "doc", "loss_sum": 3.0, "token_count": 2}]


def test_channel_loss_sp_filters_padding_tail_per_batch_row(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 100.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]),
        source_ids=["row0", "row1"],
        positions=torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "row0", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "row1", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_uses_rank_local_cu_for_headwise_padding():
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.arange(1, 11, dtype=torch.float32),
        labels_flat=torch.tensor(
            [1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        ),
        # FlashAttention keeps the known pad-to-length tail visible (mask=1),
        # while labels and positions still identify it as synthetic padding.
        attention_mask_flat=torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 1, 1]),
        source_ids=["first", "second"],
        # Per-sample CP padding and the synthetic pad-to-length tail all use
        # zero positions. Position-only segmentation would find six sources.
        position_ids=torch.tensor([[0, 1, 0, 0, 0, 1, 2, 0, 0, 0]]),
        packed_cu_seqlens=[0, 4, 8, 10],
        packed_global_cu_seqlens=[0, 4, 8, 10],
        packed_global_physical_seq_len=10,
        tail_padding_length=2,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
        strict=True,
        include_data_stats=True,
    )

    assert result == [
        {
            "source_id": "first",
            "loss_sum": 3.0,
            "token_count": 2,
            "sample_count": 1,
            "input_token_count": 2,
        },
        {
            "source_id": "second",
            "loss_sum": 18.0,
            "token_count": 3,
            "sample_count": 1,
            "input_token_count": 3,
        },
    ]


def test_channel_loss_sp_validates_production_headwise_tail_geometry():
    parallel_state = _test_parallel_state(sp_enabled=True, sp_size=4)

    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.arange(1, 9, dtype=torch.float32),
        labels_flat=torch.tensor([1, 1, 1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 1, 1, 1, 0, 1, 0]),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0]]),
        packed_cu_seqlens=[0, 2, 4, 8],
        packed_global_cu_seqlens=[0, 3, 5, 16],
        packed_global_physical_seq_len=32,
        tail_padding_length=11,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "first", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "second", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_rejects_stale_global_cu_for_headwise_tail():
    parallel_state = _test_parallel_state(sp_enabled=True)
    kwargs = dict(
        per_token_loss=torch.arange(1, 11, dtype=torch.float32),
        labels_flat=torch.tensor(
            [1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1, 1, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        ),
        attention_mask_flat=torch.ones(10, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 0, 0, 1, 2, 0, 0, 0]]),
        packed_cu_seqlens=[0, 4, 8, 10],
        packed_global_cu_seqlens=[0, 2, 8, 10],
        packed_global_physical_seq_len=10,
        tail_padding_length=2,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="do not describe the same headwise microbatch"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_rank_local_cu_rejects_unproven_extra_source():
    parallel_state = _test_parallel_state(sp_enabled=True)

    kwargs = dict(
        per_token_loss=torch.arange(1, 7, dtype=torch.float32),
        labels_flat=torch.tensor([1, 1, 1, 1, 1, 1]),
        attention_mask_flat=torch.ones(6, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1, 0, 1]]),
        packed_cu_seqlens=[0, 2, 4, 6],
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_rank_local_cu_rejects_unsupervised_one_token_without_tail_provenance():
    parallel_state = _test_parallel_state(sp_enabled=True)

    kwargs = dict(
        per_token_loss=torch.arange(1, 6, dtype=torch.float32),
        labels_flat=torch.tensor([1, 1, 1, 1, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(5, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1, 0]]),
        packed_cu_seqlens=[0, 2, 4, 5],
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_rank_local_cu_rejects_inconsistent_tail_provenance():
    parallel_state = _test_parallel_state(sp_enabled=True)

    kwargs = dict(
        per_token_loss=torch.arange(1, 6, dtype=torch.float32),
        labels_flat=torch.tensor([1, 1, 1, 1, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(5, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1, 0]]),
        packed_cu_seqlens=[0, 2, 4, 5],
        packed_global_cu_seqlens=[0, 2, 4, 5],
        packed_global_physical_seq_len=5,
        tail_padding_length=2,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="tail_padding_length"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_rejects_malformed_local_cu_instead_of_falling_back_to_positions():
    parallel_state = _test_parallel_state(sp_enabled=True)

    kwargs = dict(
        per_token_loss=torch.arange(1, 5, dtype=torch.float32),
        labels_flat=torch.ones(4, dtype=torch.long),
        attention_mask_flat=torch.ones(4, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        packed_cu_seqlens=[0, 2, 1, 4],
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="invalid packed CU metadata"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_valid_global_cu_falls_back_for_standard_ulysses_slice():
    parallel_state = _test_parallel_state(sp_enabled=True, sp_size=2)

    result = ChannelLossComputer._aggregate_by_source(
        per_token_loss=torch.arange(1, 5, dtype=torch.float32),
        labels_flat=torch.ones(4, dtype=torch.long),
        attention_mask_flat=torch.ones(4, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        packed_cu_seqlens=[0, 4, 8],
        packed_global_cu_seqlens=[0, 4, 8],
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "first", "loss_sum": 3.0, "token_count": 2},
        {"source_id": "second", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_rejects_stale_cu_endpoint_instead_of_falling_back_to_positions():
    parallel_state = _test_parallel_state(sp_enabled=True)

    kwargs = dict(
        per_token_loss=torch.arange(1, 5, dtype=torch.float32),
        labels_flat=torch.ones(4, dtype=torch.long),
        attention_mask_flat=torch.ones(4, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        packed_cu_seqlens=[0, 2, 5],
        packed_global_cu_seqlens=[0, 2, 5],
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="standard-Ulysses global provenance"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


def test_channel_loss_sp_preserves_zero_supervision_tail_when_later_row_has_padding(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        positions=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "row0_normal", "loss_sum": 6.0, "token_count": 3},
        {"source_id": "row0_zero", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "row1_normal", "loss_sum": 7.0, "token_count": 2},
    ]


def test_channel_loss_sp_rejects_ambiguous_per_row_padding_without_mask_evidence(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True)
    kwargs = dict(
        per_token_loss=torch.tensor([1.0, 2.0, 3.0, 100.0, 3.0, 4.0, 100.0, 100.0]),
        labels_flat=torch.tensor([1, 1, 1, IGNORE_INDEX, 1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(8, dtype=torch.long),
        source_ids=["row0_normal", "row0_zero", "row1_normal"],
        positions=torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        seq_len=8,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
    )

    assert ChannelLossComputer._aggregate_sp(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="source metadata count"):
        ChannelLossComputer._aggregate_sp(**kwargs, strict=True)


def test_channel_loss_sp_keeps_zero_supervision_segment_before_tail_padding(monkeypatch):
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([100.0, 2.0, 100.0]),
        labels_flat=torch.tensor([IGNORE_INDEX, 1, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(3, dtype=torch.long),
        source_ids=["one_token", "normal"],
        positions=torch.tensor([[0, 0, 0]]),
        seq_len=3,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [
        {"source_id": "one_token", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "normal", "loss_sum": 2.0, "token_count": 1},
    ]


def test_channel_loss_sp_capture_reuses_snapshot():
    parallel_state = _test_parallel_state(sp_enabled=True)
    computer = ChannelLossComputer(parallel_state=parallel_state)
    computer.strict = True
    computer._source_ids = ["source-a"]
    computer._position_ids = torch.tensor([[0, 1]])

    with computer.capture():
        assert computer._capture_topology["sp_rank"] == 0
        parallel_state.sp_rank = 7
        computer.compute_side_channel(
            logits=torch.randn(1, 2, 4),
            labels=torch.tensor([[1, 2]]),
            vocab_size=4,
            hidden_states=None,
            weights=None,
        )
        assert computer._pending_observations[0]["segments"][0]["sp_rank"] == 0

    assert computer.strict_observation_errors == []
    assert computer._micro_step_observation_count == 1
    assert computer._result is not None


def test_channel_loss_world_topology_validates_sp_group_partition():
    statuses = [
        {
            "enabled": True,
            "sp_size": 2,
            "sp_rank": rank % 2,
            "members": (0, 1) if rank < 2 else (2, 3),
            "topology_error": "",
        }
        for rank in range(4)
    ]
    assert channel_loss_module._validate_channel_sp_topologies(statuses) is None
    statuses[1] = {**statuses[1], "members": (1, 2)}
    assert channel_loss_module._validate_channel_sp_topologies(statuses) is not None


@pytest.mark.parametrize("field", ["source", "missing", "attempts", "layout", "error", "observation_source"])
def test_channel_loss_sp_group_rejects_mismatched_observations(field):
    obs = {
        "source_digest": "source-a-hash",
        "mode": "packed_cu",
        "data": False,
        "segment_count": 1,
        "valid_rows": True,
    }
    local = {
        "source_count": 1,
        "source_digest": obs["source_digest"],
        "attempts": 1,
        "errors": [],
        "observations": [obs],
    }
    remote_obs = dict(obs)
    remote = {**local, "observations": [remote_obs]}
    assert channel_loss_module._validate_channel_sp_group([local, remote]) is None
    if field == "source":
        remote["source_digest"] = "source-b-hash"
    elif field == "missing":
        remote["source_count"] = 0
    elif field == "attempts":
        remote["attempts"] = 2
    elif field == "layout":
        remote_obs["mode"] = "positions"
    elif field == "error":
        remote["errors"] = [("generic", "observer failed")]
    else:
        remote_obs["source_digest"] = "source-b-hash"
    assert channel_loss_module._validate_channel_sp_group([local, remote]) is not None


@pytest.mark.parametrize("loss_value", [float("nan"), float("inf"), -float("inf"), 0.125])
def test_channel_loss_sp_packet_preserves_float_values_and_int64_counts(loss_value):
    import pickle

    huge_count = 2**54 + 3
    segment = {
        "batch_idx": 0,
        "sp_rank": 0,
        "local_order": 0,
        "is_start": True,
        "has_explicit_padding_mask": torch.tensor(False),
        "has_only_zero_positions": False,
        "loss_sum": torch.tensor(loss_value, dtype=torch.float32),
        "token_count": torch.tensor(huge_count, dtype=torch.int64),
        "sample_count": torch.tensor(huge_count + 1, dtype=torch.int64),
        "input_token_count": torch.tensor(huge_count + 2, dtype=torch.int64),
    }
    observation = {
        "segments": [segment],
        "source_ids": ["source-a"],
        "device": torch.device("cpu"),
        "data": True,
        "mode": "packed_cu",
    }
    serialized = channel_loss_module._serialize_channel_sp_observation(observation)
    decoded = pickle.loads(pickle.dumps([serialized]))
    result = channel_loss_module._materialize_channel_sp_segments([decoded], 0, device=torch.device("cpu"))[0]
    torch.testing.assert_close(result["loss_sum"], segment["loss_sum"], rtol=0, atol=0, equal_nan=True)
    assert result["token_count"].item() == huge_count
    assert result["sample_count"].item() == huge_count + 1
    assert result["input_token_count"].item() == huge_count + 2


def test_channel_loss_failed_observation_exchange_aborts_unsafe_process_group(monkeypatch):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source-a"]
    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "get_rank", lambda *a: 0)
    monkeypatch.setattr(channel_loss_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        channel_loss_module.dist,
        "all_gather_object",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("exchange failed")),
    )
    with pytest.raises(ChannelLossCollectiveError, match="process group is unsafe"):
        with computer.capture():
            pass
    assert computer._result is None


def test_channel_loss_failed_sp_payload_exchange_aborts_unsafe_process_group(monkeypatch):
    group = object()
    state = _test_parallel_state(sp_enabled=True, sp_group=group, sp_size=2, sp_rank=0)
    computer = ChannelLossComputer(parallel_state=state)
    computer._source_ids = ["source-a"]
    segment = {
        "batch_idx": 0,
        "sp_rank": 0,
        "local_order": 0,
        "is_start": True,
        "has_explicit_padding_mask": torch.tensor(False),
        "has_only_zero_positions": False,
        "loss_sum": torch.tensor(1.0),
        "token_count": torch.tensor(1),
    }
    observation = {
        "segments": [segment],
        "source_ids": ["source-a"],
        "device": torch.device("cpu"),
        "data": False,
        "mode": "packed_cu",
    }
    monkeypatch.setattr(computer, "_compute_channel_loss", lambda **kwargs: observation)
    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "get_rank", lambda *args: 0)
    monkeypatch.setattr(channel_loss_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(channel_loss_module.dist, "get_process_group_ranks", lambda value: [0, 1])
    calls = []

    def exchange(outputs, value, group=None):
        calls.append(group)
        if group is None:
            outputs[0] = value
            outputs[1] = {**value, "sp_rank": 1}
        else:
            raise RuntimeError("SP payload failure")

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", exchange)
    with pytest.raises(ChannelLossCollectiveError, match="SP observation exchange failed"):
        with computer.capture():
            computer.compute_side_channel(
                logits=torch.randn(1, 2, 4),
                labels=torch.tensor([[1, 2]]),
                vocab_size=4,
                hidden_states=None,
                weights=None,
            )
    assert calls == [None, group]


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


def test_channel_loss_multiple_observations_accumulate_all_totals(monkeypatch):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source-a"]
    computer._position_ids = torch.tensor([[0, 1]])
    results = iter(
        [
            [
                {
                    "source_id": "source-a",
                    "loss_sum": torch.tensor(2.0),
                    "token_count": torch.tensor(1),
                    "sample_count": torch.tensor(1),
                    "input_token_count": torch.tensor(2),
                }
            ],
            [
                {
                    "source_id": "source-a",
                    "loss_sum": torch.tensor(5.0),
                    "token_count": torch.tensor(2),
                    "sample_count": torch.tensor(1),
                    "input_token_count": torch.tensor(3),
                }
            ],
        ]
    )
    monkeypatch.setattr(computer, "_compute_channel_loss", lambda **kwargs: next(results))
    with computer.capture():
        for _ in range(2):
            computer.compute_side_channel(
                logits=torch.randn(1, 2, 4),
                labels=torch.tensor([[1, 2]]),
                vocab_size=4,
                hidden_states=None,
                weights=None,
            )
    computer.after_micro_step()

    loss_sum, token_count = computer.step_totals["source-a"]
    sample_count, input_token_count, label_token_count = computer.step_data_totals["source-a"]
    assert (loss_sum.item(), token_count.item()) == (7.0, 3)
    assert (sample_count.item(), input_token_count.item(), label_token_count.item()) == (2, 5, 3)


def test_channel_loss_strict_success_plus_failure_discards_partial_evidence(monkeypatch):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer._source_ids = ["source-a"]
    successful = [
        {
            "source_id": "source-a",
            "loss_sum": torch.tensor(2.0),
            "token_count": torch.tensor(1),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(2),
        }
    ]
    calls = 0

    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return successful
        raise RuntimeError("second observation failed")

    monkeypatch.setattr(computer, "_compute_channel_loss", observe)
    with pytest.raises(ChannelLossMetadataError, match="second observation failed"):
        with computer.capture():
            for _ in range(2):
                computer.compute_side_channel(
                    logits=torch.randn(1, 2, 4),
                    labels=torch.tensor([[1, 2]]),
                    vocab_size=4,
                    hidden_states=None,
                    weights=None,
                )

    assert computer._result is None
    assert computer._micro_step_observation_count == 0
    assert "second observation failed" in computer.strict_observation_errors[0][1]


def test_channel_loss_strict_failure_blocks_later_capture_evidence(monkeypatch):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer._source_ids = ["source-a"]
    successful = [
        {
            "source_id": "source-a",
            "loss_sum": torch.tensor(2.0),
            "token_count": torch.tensor(1),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(2),
        }
    ]
    outcomes = iter([successful, RuntimeError("middle capture failed"), successful])

    def observe(**kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(computer, "_compute_channel_loss", observe)
    with pytest.raises(ChannelLossMetadataError, match="middle capture failed"):
        for _ in range(3):
            with computer.capture():
                computer.compute_side_channel(
                    logits=torch.randn(1, 2, 4),
                    labels=torch.tensor([[1, 2]]),
                    vocab_size=4,
                    hidden_states=None,
                    weights=None,
                )

    assert computer._result is None
    assert computer._micro_step_observation_count == 0
    assert computer._micro_step_failed
    computer.after_micro_step()
    assert computer.step_totals == {}
    assert computer.step_data_totals == {}


def test_channel_loss_computer_aggregates_step_results_locally():
    computer = ChannelLossComputer()
    computer.begin_step([], [], [])
    computer._result = [
        {
            "source_id": "a",
            "loss_sum": torch.tensor(2.0),
            "token_count": torch.tensor(1),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(3),
        },
        {
            "source_id": "a",
            "loss_sum": torch.tensor(4.0),
            "token_count": torch.tensor(2),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(4),
        },
        {
            "source_id": "b",
            "loss_sum": torch.tensor(3.0),
            "token_count": torch.tensor(1),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(2),
        },
    ]

    computer.after_micro_step()

    assert set(computer.step_totals) == {"a", "b"}
    assert computer.step_totals["a"][0].item() == 6.0
    assert computer.step_totals["a"][1].item() == 3
    assert tuple(value.item() for value in computer.step_data_totals["a"]) == (2, 7, 3)


def test_channel_loss_begin_step_rejects_parallel_metadata_list_mismatch():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())

    with pytest.raises(ChannelLossMetadataError, match="per-microbatch metadata list lengths"):
        computer.begin_step(
            [["a"]],
            [torch.tensor([[0]])],
            [torch.ones(1, 1, dtype=torch.long)],
            per_mb_packed_global_physical_seq_lens=[],
        )


def test_base_step_begin_allows_missing_channel_loss_callback():
    calls = []

    class RecordingCallback:
        def on_step_begin(self, state, micro_batches=None, **kwargs):
            calls.append(micro_batches)

    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.channel_loss_callback = None
    trainer._callbacks = [RecordingCallback()]
    micro_batches = [{"input_ids": torch.tensor([1, 2])}]

    BaseTrainer.on_step_begin(trainer, micro_batches=micro_batches)

    assert calls == [micro_batches]


def test_base_step_begin_skips_unregistered_channel_loss_callback():
    calls = []

    class RecordingCallback:
        def __init__(self, name):
            self.name = name

        def on_step_begin(self, state, micro_batches=None, **kwargs):
            calls.append(self.name)

    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.channel_loss_callback = RecordingCallback("unregistered-channel")
    trainer._callbacks = [RecordingCallback("registered")]

    BaseTrainer.on_step_begin(trainer, micro_batches=[{"input_ids": torch.tensor([1, 2])}])

    assert calls == ["registered"]


@pytest.mark.parametrize("callback", [None], ids=["missing"])
def test_base_forward_backward_allows_missing_channel_loss_callback(monkeypatch, callback):
    _install_test_parallel_state(monkeypatch)
    trainer = object.__new__(BaseTrainer)
    trainer.channel_loss_callback = callback
    trainer._callbacks = []
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
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()}, {})

    loss, loss_dict, _ = BaseTrainer.forward_backward_step(trainer, {"x": torch.tensor(2.0)})

    assert loss.item() == 2.0
    assert loss_dict["loss"].item() == 2.0
    assert trainer.model.weight.grad.item() == 2.0


def test_base_forward_backward_strips_channel_metadata_after_preforward(monkeypatch):
    _install_test_parallel_state(monkeypatch)
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
    trainer.postforward = lambda outputs, micro_batch: (outputs.loss, {"loss": outputs.loss.detach()}, {})
    trainer.channel_loss_callback = ChannelLossCallback(trainer)
    trainer._callbacks = [trainer.channel_loss_callback]
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
    _install_test_parallel_state(monkeypatch)
    cfg = ChannelLossConfig(enable=True, interval=1)
    state = TrainerState(global_step=1)
    policy_model = SimpleNamespace(parallel_state="base")
    reference_model = SimpleNamespace(parallel_state="base")
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
    base._callbacks = [base.channel_loss_callback]
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


def test_dpo_channel_loss_real_base_dispatch_repeats_metadata_and_emits_policy_totals(monkeypatch):
    _install_test_parallel_state(monkeypatch)

    class DpoLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.parallel_state = "base"
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
    base = object.__new__(BaseTrainer)
    base.args = SimpleNamespace(
        train=SimpleNamespace(channel_loss=cfg, enable_batch_invariant_mode=False),
        dpo_config=SimpleNamespace(
            beta=0.1,
            label_smoothing=0.0,
            loss_type="sigmoid",
            reference_free=False,
            average_log_prob=False,
        ),
    )
    base.state = state
    base.model = policy_model
    base.model_fwd_context = nullcontext()
    base.model_bwd_context = nullcontext()
    base.preforward = lambda micro_batch: micro_batch
    base.channel_loss_callback = ChannelLossCallback(base)
    base._callbacks = [base.channel_loss_callback]
    trainer = object.__new__(TextDPOTrainer)
    trainer.base = base
    trainer.policy_model = policy_model
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
    pytest.importorskip("torchvision")
    from veomni.trainer.dit_trainer import DiTTrainer

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


def test_channel_loss_metadata_prefers_rank_local_router_mask_and_cu():
    cfg = ChannelLossConfig(enable=True, interval=1)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    global_attention_mask = torch.ones(1, 16, dtype=torch.long)
    local_router_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    local_cu = [0, 2, 4]

    (
        source_ids,
        source_names,
        position_ids,
        attention_mask,
        packed_cu_seqlens,
        packed_global_cu_seqlens,
        packed_global_physical_seq_len,
        tail_padding_length,
    ) = callback._extract_metadata(
        {
            "ds_idx": torch.tensor([3, 4]),
            "source_name": ["train/a", "train/b"],
            "labels": torch.tensor([[1, 2, IGNORE_INDEX, IGNORE_INDEX]]),
            "position_ids": torch.tensor([[0, 1, 0, 0]]),
            "attention_mask": global_attention_mask,
            "router_attention_mask": local_router_mask,
            "cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32),
            "cu_seqlens_list_q": local_cu,
            "linear_attn_cu_seqlens_list_q": [0, 8, 16],
            "tail_padding_length": torch.tensor(8, dtype=torch.int32),
        }
    )

    assert source_ids == [3, 4]
    assert source_names == ["train/a", "train/b"]
    assert position_ids.shape == (1, 4)
    assert attention_mask is local_router_mask
    assert packed_cu_seqlens is local_cu
    assert packed_global_cu_seqlens == [0, 8, 16]
    assert packed_global_physical_seq_len == 16
    assert tail_padding_length == 8


def test_channel_loss_sp_rank_local_cu_keeps_empty_source():
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.empty(0),
        labels_flat=torch.empty(0, dtype=torch.long),
        attention_mask_flat=torch.empty(0, dtype=torch.long),
        source_ids=["empty"],
        positions=torch.empty((1, 0), dtype=torch.long),
        packed_cu_seqlens=[0, 0],
        seq_len=0,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [{"source_id": "empty", "loss_sum": 0.0, "token_count": 0}]


def test_channel_loss_sp_rank_local_cu_keeps_multiple_empty_sources():
    parallel_state = _test_parallel_state(sp_enabled=True)

    prepared = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.empty(0),
        labels_flat=torch.empty(0, dtype=torch.long),
        attention_mask_flat=torch.empty(0, dtype=torch.long),
        source_ids=["empty-a", "empty-b"],
        positions=torch.empty((1, 0), dtype=torch.long),
        packed_cu_seqlens=[0, 0, 0],
        seq_len=0,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
        defer_reduce=True,
    )

    assert len(prepared["segments"]) == 2
    result = ChannelLossComputer._merge_sp_segments(
        prepared["segments"], prepared["source_ids"], strict=True, include_data_stats=False
    )
    assert result == [
        {"source_id": "empty-a", "loss_sum": 0.0, "token_count": 0},
        {"source_id": "empty-b", "loss_sum": 0.0, "token_count": 0},
    ]


def test_channel_loss_sp_rank_local_cu_drops_proven_synthetic_tail():
    parallel_state = _test_parallel_state(sp_enabled=True)

    result = ChannelLossComputer._aggregate_sp(
        per_token_loss=torch.tensor([2.0, 3.0, 4.0, 5.0]),
        labels_flat=torch.tensor([1, 1, IGNORE_INDEX, IGNORE_INDEX]),
        attention_mask_flat=torch.ones(4, dtype=torch.long),
        source_ids=["source"],
        positions=torch.tensor([[0, 1, 0, 0]]),
        packed_cu_seqlens=[0, 2, 4],
        packed_global_cu_seqlens=[0, 2, 3],
        packed_global_physical_seq_len=4,
        tail_padding_length=1,
        seq_len=4,
        ignore_index=IGNORE_INDEX,
        parallel_state=parallel_state,
        strict=True,
    )

    assert result == [{"source_id": "source", "loss_sum": 5.0, "token_count": 2}]


def test_channel_loss_metadata_rejects_misaligned_router_mask():
    cfg = ChannelLossConfig(enable=True, interval=1)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    global_attention_mask = torch.ones(1, 4, dtype=torch.long)

    _, _, _, attention_mask, _, _, _, _ = callback._extract_metadata(
        {
            "ds_idx": torch.tensor([3]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": global_attention_mask,
            "router_attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )

    assert attention_mask is global_attention_mask


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
    callback.parallel_state = _test_parallel_state(dp_size=2, dp_group=group)
    observed = {}

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
        elif value.ndim == 1:
            value.add_(torch.tensor([0, 1]))
        else:
            assert value.shape == (2, 3)
            value.add_(torch.tensor([[0, 0, 0], [1, 4, 1]]))

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
    callback.parallel_state = _test_parallel_state(dp_size=2, dp_group=object())

    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "all_reduce", lambda value, group=None: None)

    def fake_all_gather_object(gathered, local_metadata, group=None):
        gathered[0] = local_metadata
        gathered[1] = [(0, "source-b")]

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(ChannelLossMetadataError, match="inconsistent names"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)


def _strict_reduction_callback(*, source_name="source-a", with_totals=True):
    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        device=torch.device("cpu"),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.source_names = {0: source_name}
    if with_totals:
        callback.computer.step_totals = {0: (torch.tensor(1.0), torch.tensor(1))}
    return callback, trainer


def _mock_strict_reduction_collectives(
    monkeypatch,
    callback,
    *,
    remote_metadata=None,
    fail_postflight=False,
):
    dp_group = object()
    parallel_state = _test_parallel_state(dp_size=2, dp_group=dp_group)
    callback.parallel_state = parallel_state
    callback.computer.parallel_state = parallel_state
    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)
    events = []
    world_calls = 0

    def fake_all_gather_object(gathered, local_metadata, group=None):
        assert group is dp_group
        events.append("group-metadata")
        gathered[0] = local_metadata
        gathered[1] = local_metadata if remote_metadata is None else remote_metadata

    def fake_all_reduce(value, group=None):
        nonlocal world_calls
        if group is None:
            world_calls += 1
            events.append(f"world-{world_calls}")
            if fail_postflight and world_calls == 2:
                value.fill_(1)
        elif value.dtype.is_floating_point:
            events.append("group-loss")
        elif value.ndim == 1:
            events.append("group-token")
        else:
            events.append("group-data")

    monkeypatch.setattr(channel_loss_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(channel_loss_module.dist, "all_reduce", fake_all_reduce)
    return events


def test_channel_loss_callback_strict_finishes_group_collectives_before_world_name_failure(monkeypatch):
    callback, trainer = _strict_reduction_callback()
    events = _mock_strict_reduction_collectives(
        monkeypatch,
        callback,
        remote_metadata=[(0, "source-b")],
    )

    with pytest.raises(ChannelLossMetadataError, match="inconsistent names"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert events == [
        "world-1",
        "group-metadata",
        "group-loss",
        "group-token",
        "world-2",
    ]
    assert trainer.step_env_metrics == {}


def test_channel_loss_callback_strict_world_syncs_remote_group_reduction_failure(monkeypatch):
    callback, trainer = _strict_reduction_callback()
    events = _mock_strict_reduction_collectives(monkeypatch, callback, fail_postflight=True)

    with pytest.raises(ChannelLossMetadataError, match="at least one rank"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert events == [
        "world-1",
        "group-metadata",
        "group-loss",
        "group-token",
        "world-2",
    ]
    assert trainer.step_env_metrics == {}


def test_channel_loss_callback_strict_world_syncs_no_totals_after_group_metadata(monkeypatch):
    callback, _ = _strict_reduction_callback(with_totals=False)
    callback.computer._per_mb_source_ids = [[0]]
    events = _mock_strict_reduction_collectives(monkeypatch, callback)

    with pytest.raises(ChannelLossMetadataError, match="no causal-LM CE totals"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert events == ["world-1", "group-metadata", "world-2"]


def test_channel_loss_callback_strict_defers_cross_step_registry_conflict(monkeypatch):
    callback, _ = _strict_reduction_callback(source_name="source-b")
    callback._source_registry = {0: "source-a"}
    events = _mock_strict_reduction_collectives(monkeypatch, callback)

    with pytest.raises(ChannelLossMetadataError, match="inconsistent names across sampled steps"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert events == [
        "world-1",
        "group-metadata",
        "group-loss",
        "group-token",
        "world-2",
    ]


def test_channel_loss_callback_strict_requires_source_id():
    cfg = ChannelLossConfig(enable=True, strict=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    with pytest.raises(ValueError, match="no configured source ID"):
        callback.on_step_begin(TrainerState(global_step=1), micro_batches=[{"input_ids": torch.tensor([[1]])}])


def test_channel_loss_callback_strict_syncs_missing_source_id_globally(monkeypatch):
    cfg = ChannelLossConfig(enable=True, strict=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        device=torch.device("cpu"),
    )
    callback = ChannelLossCallback(trainer)
    local_batch = {"position_ids": torch.tensor([[0, 1]]), "ds_idx": torch.tensor([3])}
    observed_groups = []

    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)

    def fake_all_reduce(value, group=None):
        observed_groups.append(group)
        value.fill_(1)

    monkeypatch.setattr(channel_loss_module.dist, "all_reduce", fake_all_reduce)

    with pytest.raises(ValueError, match="no configured source ID"):
        callback.on_step_begin(TrainerState(global_step=1), micro_batches=[local_batch])

    assert observed_groups == [None]


def test_channel_loss_callback_strict_raises_observer_failure_before_backward(monkeypatch):
    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    batch = {"position_ids": torch.tensor([[0, 1]]), "ds_idx": torch.tensor([3])}
    state = TrainerState(global_step=1)
    callback.on_step_begin(state, micro_batches=[batch])
    callback.on_micro_step_begin(state, micro_batch=batch)
    monkeypatch.setattr(
        callback.computer,
        "_compute_channel_loss",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )

    with pytest.raises(ChannelLossMetadataError, match="observer failed"):
        with callback.model_forward_context():
            callback.computer.compute_side_channel(
                logits=torch.randn(1, 2, 4),
                labels=torch.tensor([[1, 2]]),
                vocab_size=4,
                hidden_states=None,
                weights=None,
            )
    assert callback.computer.strict_observation_errors


def test_strict_observer_failure_prevents_real_train_step_optimizer_update(monkeypatch):
    import veomni.trainer.base as base_module

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer._source_ids = ["source-a"]
    monkeypatch.setattr(
        computer,
        "_compute_channel_loss",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )
    calls = []
    weight = torch.nn.Parameter(torch.tensor(2.0))
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=0)
    trainer.model = SimpleNamespace(
        optimizer=SimpleNamespace(step=lambda: calls.append("optimizer"), zero_grad=lambda: calls.append("zero")),
        lr_scheduler=SimpleNamespace(step=lambda: calls.append("scheduler")),
    )
    trainer._reset_async_activation_offload_if_enabled = lambda *args: None
    trainer.on_step_begin = lambda **kwargs: None
    trainer.sync_before_train_step = lambda: None
    trainer.model_reshard = lambda *args: None
    trainer._configure_hsdp_allreduce = lambda *args: None
    monkeypatch.setattr(base_module, "count_loss_token", lambda *args: 1)
    monkeypatch.setattr(base_module, "reduce_global_loss_token", lambda value: value)
    monkeypatch.setattr(base_module, "mark_compile_step_begin", lambda *args: None)

    def forward_backward_step(micro_batch):
        with computer.capture():
            loss = weight * 2
            computer.compute_side_channel(
                logits=torch.randn(1, 2, 4),
                labels=torch.tensor([[1, 2]]),
                vocab_size=4,
                hidden_states=None,
                weights=None,
            )
        loss.backward()
        return loss, {}, {}

    trainer.forward_backward_step = forward_backward_step
    with pytest.raises(ChannelLossMetadataError, match="observer failed"):
        BaseTrainer.train_step(trainer, iter([[{"labels": torch.tensor([[1, 2]])}]]))
    assert calls == []
    assert weight.item() == 2.0
    assert weight.grad is None


def test_channel_loss_callback_strict_syncs_missing_observation_before_group_collectives(monkeypatch):
    cfg = ChannelLossConfig(enable=True, interval=1, strict=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)),
        device=torch.device("cpu"),
        step_train_metrics={},
        step_env_metrics={},
    )
    callback = ChannelLossCallback(trainer)
    callback._collect_step = True
    callback.computer.step_totals = {0: (torch.tensor(1.0), torch.tensor(1))}
    observed_groups = []

    monkeypatch.setattr(channel_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)

    def fake_all_reduce(value, group=None):
        observed_groups.append(group)
        value.fill_(1)

    monkeypatch.setattr(channel_loss_module.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        callback,
        "_reduce_step_totals",
        lambda *args, **kwargs: pytest.fail("group collective entered after world preflight failure"),
    )

    with pytest.raises(ChannelLossMetadataError, match="sampled micro-step.*no causal-LM CE observation"):
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert observed_groups == [None]


def _observe_real_sp_channel_loss(computer):
    computer.compute_side_channel(
        logits=torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 10,
        labels=torch.tensor([[1, 2]]),
        vocab_size=8,
        hidden_states=None,
        weights=None,
    )


def _gloo_sp_preflight_worker(rank, world_size, init_file, result_queue):
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        parallel_state = _test_parallel_state(
            sp_enabled=True,
            sp_group=torch.distributed.group.WORLD,
            sp_size=world_size,
            sp_rank=rank,
        )
        singleton_groups = [
            torch.distributed.new_group(ranks=[singleton_rank]) for singleton_rank in range(world_size)
        ]

        class _BrokenParallelState:
            @property
            def sp_enabled(self):
                raise RuntimeError("cached parallel-state descriptor resolution failed")

        for scenario in (
            "valid",
            "source",
            "missing",
            "generic",
            "count",
            "metadata",
            "malformed",
            "source_digest_failure",
            "oversized_source_id",
            "descriptor_resolution_failure",
            "descriptor_mismatch",
            "strict_mismatch",
            "outer_descriptor_disabled",
            "outer_descriptor_group",
            "outer_descriptor_world",
        ):
            scenario_state = parallel_state
            if scenario == "descriptor_resolution_failure":
                scenario_state = _BrokenParallelState()
            elif scenario == "outer_descriptor_disabled" and rank == 1:
                scenario_state = _test_parallel_state()
            elif scenario == "outer_descriptor_group" and rank == 1:
                scenario_state = _test_parallel_state(
                    sp_enabled=True,
                    sp_group=singleton_groups[rank],
                    sp_size=1,
                    sp_rank=0,
                )
            elif scenario == "outer_descriptor_world" and rank == 1:
                scenario_state = _test_parallel_state(
                    sp_enabled=True,
                    sp_group=torch.distributed.group.WORLD,
                    sp_size=world_size + 1,
                    sp_rank=rank,
                )

            computer = ChannelLossComputer(parallel_state=scenario_state)
            computer.strict = not (scenario == "strict_mismatch" and rank == 1)
            computer._source_ids = ["source-a"]
            computer._position_ids = torch.tensor([[0, 1]]) if rank == 0 else torch.tensor([[2, 3]])
            original_serialize = channel_loss_module._serialize_channel_sp_observation
            original_digest = channel_loss_module._source_signature_digest

            if scenario == "source" and rank == 1:
                computer._source_ids = ["source-b"]
            elif scenario == "oversized_source_id" and rank == 1:
                computer._source_ids = [10**4300]
            elif scenario == "missing" and rank == 1:
                computer._source_ids = []
            elif scenario == "generic" and rank == 1:
                computer._compute_channel_loss = lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError("rank-local CE preparation failed")
                )
            elif scenario == "metadata" and rank == 1:
                computer._position_ids = None
            elif scenario == "malformed" and rank == 1:

                def fail_ready_payload(*args, **kwargs):
                    raise RuntimeError("rank-local malformed segment stack")

                channel_loss_module._serialize_channel_sp_observation = fail_ready_payload
            elif scenario == "source_digest_failure" and rank == 1:
                channel_loss_module._source_signature_digest = lambda *_: (_ for _ in ()).throw(
                    RuntimeError("source digest failed on this rank")
                )

            raised = False
            try:
                try:
                    with computer.capture():
                        if computer._source_ids:
                            _observe_real_sp_channel_loss(computer)
                            if scenario == "count" and rank == 1:
                                _observe_real_sp_channel_loss(computer)
                            if scenario == "descriptor_mismatch" and rank == 1:
                                observation = computer._pending_observations[0]
                                observation["source_ids"] = ["changed-after-capture"]
                except ChannelLossMetadataError:
                    raised = True
            finally:
                channel_loss_module._serialize_channel_sp_observation = original_serialize
                channel_loss_module._source_signature_digest = original_digest

            if scenario == "valid":
                assert not raised
                assert computer._result is not None
                assert computer._micro_step_observation_count == 1
                assert computer.strict_observation_errors == []
                assert computer._result[0]["source_id"] == "source-a"
                assert computer._result[0]["token_count"].item() == 4
            else:
                assert raised
                assert computer._result is None
                assert computer._micro_step_observation_count == 0
                if computer.strict:
                    assert computer.strict_observation_errors
                if scenario == "descriptor_resolution_failure":
                    assert (
                        "cached parallel-state descriptor resolution failed"
                        in (computer.strict_observation_errors[0][1])
                    )
            torch.distributed.barrier()
        result_queue.put((rank, "ok"))
    except Exception:
        result_queue.put((rank, traceback.format_exc()))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _gloo_two_stripe_worker(rank, world_size, init_file, result_queue):
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        stripe_groups = [
            torch.distributed.new_group(ranks=[0, 1]),
            torch.distributed.new_group(ranks=[2, 3]),
        ]
        stripe_idx = rank // 2
        local_sp_rank = rank % 2
        parallel_state = _test_parallel_state(
            sp_enabled=True,
            sp_group=stripe_groups[stripe_idx],
            sp_size=2,
            sp_rank=local_sp_rank,
        )
        config = ChannelLossConfig(enable=True, interval=1, strict=True)
        trainer = SimpleNamespace(
            args=SimpleNamespace(train=SimpleNamespace(channel_loss=config)),
            device=torch.device("cpu"),
            step_train_metrics={},
            step_env_metrics={},
        )
        for scenario in ("missing", "reconstruction_failure"):
            with use_parallel_state(parallel_state):
                callback = ChannelLossCallback(trainer)
            batch = {
                "position_ids": torch.tensor([[0, 1]]) if local_sp_rank == 0 else torch.tensor([[2, 3]]),
                "ds_idx": torch.tensor([3]),
            }
            state = TrainerState(global_step=1)
            callback.on_step_begin(state, micro_batches=[batch])
            callback.on_micro_step_begin(state, micro_batch=batch)
            if scenario == "reconstruction_failure" and rank == 3:
                callback.computer._merge_sp_segments = lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("rank-local reconstruction failed")
                )
            with pytest.raises(ChannelLossMetadataError):
                with callback.model_forward_context():
                    if scenario != "missing" or rank != 3:
                        _observe_real_sp_channel_loss(callback.computer)
            assert trainer.step_env_metrics == {}
            torch.distributed.barrier()
        result_queue.put((rank, "ok"))
    except Exception:
        result_queue.put((rank, traceback.format_exc()))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _gloo_packed_channel_parity_worker(rank, world_size, init_file, result_queue):
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        state = _test_parallel_state(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = ChannelLossComputer(parallel_state=state)
        computer.strict = True
        computer._source_ids = ["a", "b", "c"]
        computer._position_ids = torch.tensor([[0, 1, 2, 0]]) if rank == 0 else torch.tensor([[1, 2, 0, 1]])
        computer._packed_cu_seqlens = [0, 3, 4, 4] if rank == 0 else [0, 0, 2, 4]
        all_logits = torch.arange(40, dtype=torch.float32).reshape(8, 5) / 19
        all_labels = torch.tensor([0, 1, IGNORE_INDEX, 2, 3, 4, 1, 0])
        local_logits = all_logits[rank * 4 : (rank + 1) * 4].clone().requires_grad_()
        reference_logits = local_logits.detach().clone().requires_grad_()
        local_labels = all_labels[rank * 4 : (rank + 1) * 4]
        training_loss = F.cross_entropy(local_logits, local_labels, ignore_index=IGNORE_INDEX)
        reference_loss = F.cross_entropy(reference_logits, local_labels, ignore_index=IGNORE_INDEX)
        exchange_kinds = []
        original_all_gather = torch.distributed.all_gather_object

        def record_exchange(output, value, group=None):
            if isinstance(value, dict):
                assert group is None
                assert "source_ids" not in value
                assert all("segments" not in observation for observation in value["observations"])
                exchange_kinds.append("world-status")
            elif isinstance(value, bytes):
                assert isinstance(value, bytes) and group is state.sp_group
                exchange_kinds.append("sp-payload")
            else:
                assert group is None and value is None
                exchange_kinds.append("world-strict-result")
            return original_all_gather(output, value, group=group)

        torch.distributed.all_gather_object = record_exchange
        with computer.capture():
            computer.compute_side_channel(
                logits=local_logits.reshape(1, 4, 5),
                labels=local_labels.reshape(1, 4),
                vocab_size=5,
                hidden_states=None,
                weights=None,
            )
        torch.distributed.all_gather_object = original_all_gather
        assert exchange_kinds == ["world-status", "sp-payload", "world-strict-result"]
        training_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(training_loss, reference_loss, rtol=0, atol=0)
        torch.testing.assert_close(local_logits.grad, reference_logits.grad, rtol=0, atol=0)
        assert not computer.strict_observation_errors
        assert [item["source_id"] for item in computer._result] == ["a", "b", "c"]
        per_token = F.cross_entropy(all_logits, all_labels, ignore_index=IGNORE_INDEX, reduction="none")
        for item, start, end in zip(computer._result, [0, 3, 6], [3, 6, 8], strict=True):
            expected_count = all_labels[start:end].ne(IGNORE_INDEX).sum()
            torch.testing.assert_close(item["loss_sum"], per_token[start:end].sum(), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(item["token_count"], expected_count, rtol=0, atol=0)
        result_queue.put((rank, "ok"))
    except Exception:
        result_queue.put((rank, traceback.format_exc()))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_gloo_workers(worker, *, world_size, init_file, timeout_seconds=120):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, world_size, str(init_file), result_queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        process.join(max(deadline - time.monotonic(), 0))
    hung = [process for process in processes if process.is_alive()]
    if hung:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)
        pytest.fail(f"distributed channel-loss test timed out on {len(hung)} process(es)")

    results = []
    for _ in range(world_size):
        try:
            results.append(result_queue.get(timeout=3))
        except queue.Empty:
            break
    failures = [detail for _, detail in results if detail != "ok"]
    exit_codes = [process.exitcode for process in processes]
    assert len(results) == world_size, (results, exit_codes)
    assert not failures, "\n".join(failures)
    assert exit_codes == [0] * world_size


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_real_gloo_sp_preflight_avoids_divergent_payload_collectives(tmp_path):
    _run_gloo_workers(
        _gloo_sp_preflight_worker,
        world_size=2,
        init_file=tmp_path / "sp-preflight-init",
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_real_gloo_two_sp_stripes_propagate_strict_failure_worldwide(tmp_path):
    _run_gloo_workers(
        _gloo_two_stripe_worker,
        world_size=4,
        init_file=tmp_path / "sp-stripes-init",
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_real_gloo_packed_loss_and_training_gradient_parity(tmp_path):
    _run_gloo_workers(_gloo_packed_channel_parity_worker, world_size=2, init_file=tmp_path / "sp-packed-init")


@pytest.mark.parametrize("strict", [False, True])
def test_chunk_observer_failure_preserves_main_loss_and_gradients(monkeypatch, strict):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    _install_test_parallel_state(monkeypatch)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = partial(ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"))

    model = Model()
    torch.manual_seed(17)
    hidden = torch.randn(1, 8, 4, requires_grad=True)
    weight = torch.randn(8, 4, requires_grad=True)
    labels = torch.arange(8).unsqueeze(0)
    ref_hidden, ref_weight = (t.detach().clone().requires_grad_() for t in (hidden, weight))
    kwargs = dict(labels=labels, chunk_size=2)
    reference, _, _ = model.loss_function(**kwargs, hidden_states=ref_hidden, weights=ref_weight)
    reference.backward()
    original_ce = F.cross_entropy
    observer_calls = []

    def fail_observer(*args, **kwargs):
        if kwargs.get("reduction") == "none":
            observer_calls.append(None)
            raise RuntimeError("injected detached observer failure")
        return original_ce(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", fail_observer)
    computer = ChannelLossComputer()
    computer.strict = strict
    computer._source_ids = ["a"]
    computer._position_ids = torch.arange(8).unsqueeze(0)
    monkeypatch.setattr(computer, "compute_side_channel", lambda *a, **kw: pytest.fail("unexpected CE retry"))
    try:
        computer.install(model)
        if strict:
            with pytest.raises(ChannelLossMetadataError, match="injected detached observer failure"):
                with computer.capture():
                    actual, _, _ = model.loss_function(**kwargs, hidden_states=hidden, weights=weight)
        else:
            with computer.capture():
                actual, _, _ = model.loss_function(**kwargs, hidden_states=hidden, weights=weight)
        actual.backward()
    finally:
        computer.uninstall()
    assert observer_calls == [None]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=0, atol=0)
    assert not computer._result
    if strict:
        assert computer.strict_observation_errors


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_capture_forward_failure_discards_observations_without_collectives(monkeypatch, nested, error_type):
    _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer()
    monkeypatch.setattr(channel_loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        channel_loss_module.dist, "all_gather_object", lambda *a, **kw: pytest.fail("collective on failed forward")
    )
    monkeypatch.setattr(computer, "_finalize_capture", lambda: pytest.fail("finalized failed forward"))
    error = error_type("injected training forward failure")
    with pytest.raises(error_type) as captured:
        with computer.capture():
            computer._pending_observations = [{"partial": torch.ones(1)}]
            if nested:
                with computer.capture():
                    raise error
            raise error
    assert captured.value is error
    assert not computer.capture_active
    assert not computer._pending_observations
    assert not computer._capture_errors
    assert computer._capture_topology is None


@pytest.mark.parametrize("fail_observer", [False, True])
def test_chunk_observer_with_offload_hooks_preserves_loss_and_gradients(monkeypatch, fail_observer):
    from torch.autograd.graph import saved_tensors_hooks

    from veomni.ops import resolve_op
    from veomni.trainer.callbacks.channel_loss_callback import capture_chunk_loss_per_token

    torch.manual_seed(319)
    hidden = torch.randn(2, 11, 64, requires_grad=True)
    weight = torch.randn(19, 64, requires_grad=True)
    labels = torch.randint(0, 19, (2, 11))
    labels[:, 2:6] = -100
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    reference = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper(
        ref_hidden, labels, ref_weight, chunk_size=3
    )
    (reference * 0.37).backward()
    expected_tokens = (
        torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden, weight).reshape(-1, 19),
            labels.reshape(-1),
            reduction="none",
        )
        .reshape(2, 11)
        .detach()
    )

    original_ce = torch.nn.functional.cross_entropy
    original_linear = torch.nn.functional.linear
    projections, packed, unpacked, observed, errors = [], [], [], [], []

    def ce(*args, **kwargs):
        if fail_observer and kwargs.get("reduction") == "none":
            raise RuntimeError("observer-only failure")
        return original_ce(*args, **kwargs)

    def linear(*args, **kwargs):
        projections.append(1)
        return original_linear(*args, **kwargs)

    def pack(tensor):
        packed.append(tensor.shape)
        return tensor.detach().cpu().clone()

    def unpack(tensor):
        unpacked.append(tensor.shape)
        return tensor

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", ce)
    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    with capture_chunk_loss_per_token(observed.append, errors.append), saved_tensors_hooks(pack, unpack):
        actual = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper(
            hidden, labels, weight, chunk_size=3
        )
        (actual * 0.37).backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=0, atol=0)
    assert len(projections) == 4  # observer reuses the main projection
    assert packed == unpacked == [hidden.shape, weight.shape]
    if fail_observer:
        assert not observed
        assert len(errors) == 1 and "observer-only failure" in errors[0]
    else:
        assert not errors and len(observed) == 1
        torch.testing.assert_close(observed[0], expected_tokens)


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
        assert not computer.strict_observation_errors
        assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
        assert computer._result[0]["loss_sum"].item() == pytest.approx(loss.item() * 6, rel=1e-6)
    finally:
        computer.uninstall()
