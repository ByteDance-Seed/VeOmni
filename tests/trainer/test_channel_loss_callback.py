import importlib.util
import inspect
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


def _legacy_computer(*args, **kwargs):
    """Exercise the internal legacy aggregation helpers without a producer."""
    computer = ChannelLossComputer(*args, **kwargs)
    computer._require_source_provenance = False
    return computer


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
        computer = _legacy_computer(
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

    computer = _legacy_computer()
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
            "sample_count": 1,
            "input_token_count": 3,
        },
        {
            "source_id": 1,
            "loss_sum": pytest.approx(loss_1),
            "token_count": tokens_1,
            "sample_count": 1,
            "input_token_count": 2,
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

    logits_computer = _legacy_computer()
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

    hs_computer = _legacy_computer()
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


def test_channel_loss_computer_counts_attention_masked_input_tokens():
    computer = _legacy_computer()
    computer._source_ids = ["repoqa"]
    computer._position_ids = torch.tensor([[0, 1, 2, 3]])

    result = computer._compute_channel_loss(
        logits=torch.randn(1, 4, 8),
        labels=torch.tensor([[1, 2, 3, 4]]),
        vocab_size=8,
        hidden_states=None,
        weights=None,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        shift_labels=None,
        ignore_index=IGNORE_INDEX,
    )

    assert result[0]["sample_count"].item() == 1
    assert result[0]["input_token_count"].item() == 3
    assert result[0]["token_count"].item() == 3


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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    assert computer._result[0]["sample_count"].item() == 1
    assert computer._result[0]["input_token_count"].item() == 17


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

    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
        packed_global_cu_seqlens=[0, 4, 8],
        tail_padding_length=0,
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

    result = ChannelLossComputer._reduce_sp(
        local_segments=[
            {
                "batch_idx": 0,
                "sp_rank": 0,
                "local_order": 0,
                "is_start": True,
                "has_explicit_padding_mask": torch.tensor(False),
                "has_only_zero_positions": False,
                "loss_sum": torch.tensor(3.0),
                "token_count": torch.tensor(2),
            }
        ],
        source_ids=["doc"],
        device=torch.device("cpu"),
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
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    with pytest.raises(ChannelLossMetadataError, match="global geometry|do not describe the same headwise microbatch"):
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
    with pytest.raises(ChannelLossMetadataError, match="tail"):
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


def test_channel_loss_sp_rejects_inconsistent_global_cu_for_ulysses_slice():
    parallel_state = _test_parallel_state(sp_enabled=True, sp_size=2)

    kwargs = dict(
        per_token_loss=torch.arange(1, 5, dtype=torch.float32),
        labels_flat=torch.ones(4, dtype=torch.long),
        attention_mask_flat=torch.ones(4, dtype=torch.long),
        source_ids=["first", "second"],
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        packed_cu_seqlens=[0, 4, 8],
        packed_global_cu_seqlens=[0, 4, 8],
        tail_padding_length=0,
        ignore_index=IGNORE_INDEX,
        sp_enabled=True,
        parallel_state=parallel_state,
    )
    # A rank-zero width-four contiguous shard of [0,4,8] has no reset at
    # local position two. Treating the local reset as source two misattributes.
    assert ChannelLossComputer._aggregate_by_source(**kwargs, strict=False) == []
    with pytest.raises(ChannelLossMetadataError, match="global CU cannot prove"):
        ChannelLossComputer._aggregate_by_source(**kwargs, strict=True)


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
    with pytest.raises(ChannelLossMetadataError, match="endpoint"):
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
    computer = _legacy_computer(parallel_state=parallel_state)
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


@pytest.mark.parametrize(
    "field", ["source", "missing", "attempts", "layout", "error", "observation_source", "data_index"]
)
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
        "data_observation_index": 0,
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
    elif field == "data_index":
        remote["data_observation_index"] = None
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
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    computer = _legacy_computer(parallel_state=state)
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
    monkeypatch.setattr(computer, "_compute_channel_loss", lambda *, defer_sp=False, **kwargs: observation)
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

    result = _legacy_computer()._per_token_ce(
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
    assert ChannelLossConfig().release_cache is False
    with pytest.raises(ValueError, match="interval must be at least 1"):
        ChannelLossConfig(interval=0)


def test_channel_loss_releases_observer_cache_after_side_channel(monkeypatch):
    events = []
    computer = _legacy_computer()
    computer.release_cache = True

    def compute(**kwargs):
        events.append("compute")
        return []

    def release(device):
        assert device.type == "cpu"
        assert events == ["compute"]
        events.append("release")

    monkeypatch.setattr(computer, "_compute_channel_loss", compute)
    monkeypatch.setattr(channel_loss_module, "_release_observer_cache", release)
    computer.compute_side_channel(
        logits=torch.randn(1, 2, 4),
        labels=torch.tensor([[1, 2]]),
        vocab_size=4,
        hidden_states=None,
        weights=None,
    )

    assert events == ["compute", "release"]


def test_channel_loss_observer_cache_cleanup_orders_sync_before_empty(monkeypatch):
    events = []

    class _DeviceContext:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append(("enter", self.device))

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("exit", self.device))

    class _Accelerator:
        def device(self, device):
            return _DeviceContext(device)

    device = SimpleNamespace(type="npu")
    monkeypatch.setattr(channel_loss_module, "get_torch_device", lambda: _Accelerator())
    monkeypatch.setattr(channel_loss_module, "stream_synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(channel_loss_module, "empty_cache", lambda: events.append("empty_cache"))

    channel_loss_module._release_observer_cache(device)

    assert events == [
        ("enter", device),
        "synchronize",
        "empty_cache",
        ("exit", device),
    ]


def test_channel_loss_observer_cache_cleanup_is_noop_on_cpu(monkeypatch):
    monkeypatch.setattr(
        channel_loss_module,
        "get_torch_device",
        lambda: pytest.fail("CPU cleanup must not resolve an accelerator"),
    )

    channel_loss_module._release_observer_cache(torch.device("cpu"))


def test_channel_loss_cleanup_failure_does_not_abort_observation(monkeypatch):
    computer = _legacy_computer()
    computer.release_cache = True
    computer._source_ids = ["source-a"]
    observed = [
        {
            "source_id": "source-a",
            "loss_sum": torch.tensor(2.0),
            "token_count": torch.tensor(1),
            "sample_count": torch.tensor(1),
            "input_token_count": torch.tensor(2),
        }
    ]
    monkeypatch.setattr(computer, "_compute_channel_loss", lambda **kwargs: observed)
    monkeypatch.setattr(
        channel_loss_module,
        "_release_observer_cache",
        lambda device: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    computer.compute_side_channel(
        logits=torch.randn(1, 2, 4),
        labels=torch.tensor([[1, 2]]),
        vocab_size=4,
        hidden_states=None,
        weights=None,
    )

    assert computer._result == observed


def test_channel_loss_cleanup_failure_preserves_strict_metadata_error(monkeypatch):
    computer = _legacy_computer()
    computer.strict = True
    computer.release_cache = True

    def fail_compute(**kwargs):
        raise ChannelLossMetadataError("metadata failed")

    monkeypatch.setattr(computer, "_compute_channel_loss", fail_compute)
    monkeypatch.setattr(
        channel_loss_module,
        "_release_observer_cache",
        lambda device: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(ChannelLossMetadataError, match="metadata failed"):
        computer.compute_side_channel(
            logits=torch.randn(1, 2, 4),
            labels=torch.tensor([[1, 2]]),
            vocab_size=4,
            hidden_states=None,
            weights=None,
        )

    assert computer._result is None
    assert computer.strict_observation_errors
    assert "metadata failed" in computer.strict_observation_errors[0][1]


def test_channel_loss_multiple_objectives_count_primary_data_once(monkeypatch):
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    assert (sample_count.item(), input_token_count.item(), label_token_count.item()) == (1, 2, 1)


@pytest.mark.parametrize("auxiliary_first", [False, True])
@pytest.mark.parametrize("copy_primary_labels", [False, True])
def test_channel_loss_qwen35_mtp_counts_primary_batch_once(auxiliary_first, copy_primary_labels):
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import compute_mtp_loss

    class MTPModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 3))

        def loss_function(self, hidden_states, labels, weights, shift_labels=None, **kwargs):
            logits = F.linear(hidden_states, weights)
            targets = shift_labels if shift_labels is not None else F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX)
            return loss, logits, None

        def forward(self, hidden, labels, mtp_labels):
            def primary():
                return self.loss_function(
                    logits=None,
                    hidden_states=hidden,
                    labels=labels.clone() if copy_primary_labels else labels,
                    weights=self.weight,
                    vocab_size=8,
                )[0]

            def auxiliary():
                return compute_mtp_loss(self.loss_function, (hidden,), mtp_labels, self.weight, vocab_size=8)

            if auxiliary_first:
                aux, main = auxiliary(), primary()
            else:
                main, aux = primary(), auxiliary()
            return main + 0.2 * aux

    torch.manual_seed(719)
    model = MTPModel()
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    mtp_labels = torch.tensor([[[3, 4, IGNORE_INDEX, IGNORE_INDEX]]])
    expected = model(hidden, labels, mtp_labels)
    expected_grads = torch.autograd.grad(expected, (hidden, model.weight))
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer.begin_step([["source-a"]], [torch.arange(4)[None]], [None])
    computer.before_micro_step(primary_labels=labels, mtp_labels=mtp_labels)
    try:
        computer.install(model)
        with computer.capture():
            actual = model(hidden, labels, mtp_labels)
        computer.after_micro_step()
    finally:
        computer.uninstall()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (hidden, model.weight)), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert computer.strict_observation_errors == []
    assert tuple(value.item() for value in computer.step_data_totals["source-a"]) == (1, 4, 3)
    assert computer.step_totals["source-a"][1].item() == 5  # Primary CE plus two auxiliary targets.


@pytest.mark.parametrize(
    ("labels", "mtp_labels", "positions", "sources", "expected_counts"),
    [
        (
            [[1, 2, 3, 4, 5, 6]],
            [[[3, 4, 5, 6, IGNORE_INDEX, IGNORE_INDEX], [4, 5, 6, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]],
            [[0, 1, 2, 0, 1, 2]],
            ["a", "b"],
            [9, 3],
        ),
        ([[IGNORE_INDEX, 1, IGNORE_INDEX, IGNORE_INDEX]], [[[IGNORE_INDEX] * 4]], [[0, 1, 2, 3]], ["a"], [1]),
    ],
)
@pytest.mark.parametrize("with_provenance", [False, True])
def test_qwen35_mtp_observer_handles_multiple_depths_and_empty_targets(
    labels, mtp_labels, positions, sources, expected_counts, with_provenance
):
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import compute_mtp_loss

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(9, 3))

        def loss_function(self, hidden_states, labels, weights, shift_labels=None, **kwargs):
            logits = F.linear(hidden_states, weights)
            target = shift_labels if shift_labels is not None else F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
            return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=IGNORE_INDEX), logits, None

        def forward(self, hidden, labels, mtp_labels):
            main = self.loss_function(hidden_states=hidden, labels=labels, weights=self.weight)[0]
            auxiliary = compute_mtp_loss(
                self.loss_function, (hidden,) * mtp_labels.size(1), mtp_labels, self.weight, vocab_size=9
            )
            return main + auxiliary

    torch.manual_seed(830)
    labels, mtp_labels, positions = map(torch.tensor, (labels, mtp_labels, positions))
    model = Model()
    hidden = torch.randn(1, labels.size(-1), 3, requires_grad=True)
    expected = model(hidden, labels, mtp_labels)
    reference_grad = torch.autograd.grad(expected, (hidden, model.weight))
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer.strict = True
    points = positions[0].eq(0).nonzero(as_tuple=True)[0].tolist() + [positions.size(-1)]
    provenance = {
        "source_ids": sources,
        "source_cu": points,
        "input_mask": torch.ones(positions.numel(), dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer.begin_step(
        [sources], [positions], [None], per_mb_source_provenances=[provenance] if with_provenance else None
    )
    computer.before_micro_step(primary_labels=labels, mtp_labels=mtp_labels)
    try:
        computer.install(model)
        with computer.capture():
            actual = model(hidden, labels, mtp_labels)
        computer.after_micro_step()
    finally:
        computer.uninstall()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, reference in zip(torch.autograd.grad(actual, (hidden, model.weight)), reference_grad, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert [computer.step_totals[source][1].item() for source in sources] == expected_counts
    assert not computer.strict_observation_errors
    if with_provenance:
        assert [computer.step_data_totals[source][0].item() for source in sources] == [1] * len(sources)
        assert [computer.step_data_totals[source][1].item() for source in sources] == [
            end - start for start, end in zip(points, points[1:])
        ]


def test_channel_loss_rejects_ambiguous_primary_and_mtp_targets():
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer._primary_labels = torch.tensor([[1, 1, 1]])
    computer._require_primary_labels = True
    computer._mtp_labels = torch.tensor([[[1, 1, 1]]])
    copied_labels = computer._primary_labels.clone()
    with pytest.raises(ChannelLossMetadataError, match="primary and MTP targets are ambiguous"):
        computer._matched_mtp_targets(copied_labels, copied_labels, IGNORE_INDEX)


def test_mtp_provenance_checks_every_depth_for_supervised_padding():
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 2],
        "input_mask": torch.ones(2, dtype=torch.bool),
        "tail": 2,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._primary_labels = torch.tensor([[0, 1, IGNORE_INDEX, IGNORE_INDEX]])
    computer._require_primary_labels = True
    computer._mtp_labels = torch.tensor([[[1, 2, IGNORE_INDEX, IGNORE_INDEX], [2, 3, 4, IGNORE_INDEX]]])
    targets = computer._mtp_labels.reshape(2, 4)
    with pytest.raises(ChannelLossMetadataError, match="tail contains supervised labels"):
        computer._compute_channel_loss(
            logits=torch.randn(2, 4, 5),
            labels=targets,
            vocab_size=5,
            hidden_states=None,
            weights=None,
            attention_mask=None,
            shift_labels=targets,
            ignore_index=IGNORE_INDEX,
        )


def test_channel_loss_strict_rejects_missing_primary_microbatch_labels():
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer.begin_step([["source"]], [torch.tensor([[0, 1]])], [None])
    computer.before_micro_step(primary_labels=None)
    with pytest.raises(ChannelLossMetadataError, match="primary micro-batch labels are missing"):
        with computer.capture():
            computer.compute_side_channel(torch.randn(1, 2, 4), torch.tensor([[1, 2]]), 4, None, None)


def test_channel_loss_primary_data_is_reset_between_microbatches():
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer.begin_step([["same"], ["same"]], [torch.arange(4)[None]] * 2, [None, None])
    for labels in (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, IGNORE_INDEX, 3, IGNORE_INDEX]])):
        computer.before_micro_step(primary_labels=labels)
        for _ in range(2):
            with computer.capture():
                computer.compute_side_channel(torch.randn(1, 4, 8), labels, 8, None, None)
        computer.after_micro_step()
    assert computer.strict_observation_errors == []
    assert tuple(value.item() for value in computer.step_data_totals["same"]) == (2, 8, 4)


@pytest.mark.parametrize("sp_enabled", [False, True])
def test_dpo_channel_loss_masks_segment_boundaries_and_counts_preference_pairs(sp_enabled):
    from veomni.trainer.text_dpo_trainer import _build_dpo_labels_list

    raw = torch.tensor([[1, 2, 3, 4, 5]])
    labels = F.pad(raw[:, 1:], (0, 1), value=IGNORE_INDEX) if sp_enabled else raw
    before = labels.clone()
    computer = _legacy_computer(parallel_state=_test_parallel_state(sp_enabled=sp_enabled))
    computer.strict = True
    computer._dpo_segments = True
    computer._source_repeat = 2
    computer.begin_step([["source", "source"]], [torch.tensor([[0, 1, 2, 0, 1]])], [None])
    computer.before_micro_step(primary_labels=labels)
    with computer.capture():
        computer.compute_side_channel(torch.randn(1, 5, 8), labels, 8, None, None)
    computer.after_micro_step()
    objective_targets = sum(
        (part != IGNORE_INDEX).sum().item() for part in _build_dpo_labels_list(labels.flatten(), [3, 2], sp_enabled)
    )
    assert objective_targets == 3
    assert computer.step_totals["source"][1].item() == objective_targets
    assert tuple(item.item() for item in computer.step_data_totals["source"]) == (1, 5, 3)
    torch.testing.assert_close(labels, before, rtol=0, atol=0)


def test_dpo_channel_loss_rejects_unproven_headwise_cp_tail():
    computer = _legacy_computer(
        parallel_state=_test_parallel_state(sp_enabled=True, sp_size=2, sp_rank=0, sp_group=object())
    )
    computer._source_ids = ["source", "source"]
    computer._dpo_segments = True
    computer._position_ids = torch.tensor([[0, 1, 2]])
    computer._packed_cu_seqlens = [0, 1, 3]
    computer._packed_global_cu_seqlens = [0, 3, 6]
    with pytest.raises(ChannelLossMetadataError, match="cannot prove segment tails"):
        computer._dpo_tail_mask(torch.tensor([[2, 3, 4]]), {"enabled": True, "sp_size": 2, "sp_rank": 0})

    computer.parallel_state.cp_size = 2
    computer._packed_cu_seqlens = computer._packed_global_cu_seqlens
    with pytest.raises(ChannelLossMetadataError, match="context parallelism"):
        computer._dpo_tail_mask(torch.tensor([[2, 3, 4]]), {"enabled": True, "sp_size": 2, "sp_rank": 0})


def test_channel_loss_strict_success_plus_failure_discards_partial_evidence(monkeypatch):
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    computer = _legacy_computer()
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

    computer._data_result = computer._result
    computer.after_micro_step()

    assert set(computer.step_totals) == {"a", "b"}
    assert computer.step_totals["a"][0].item() == 6.0
    assert computer.step_totals["a"][1].item() == 3
    assert tuple(value.item() for value in computer.step_data_totals["a"]) == (2, 7, 3)


def test_channel_loss_begin_step_rejects_parallel_metadata_list_mismatch():
    computer = _legacy_computer(parallel_state=_test_parallel_state())

    with pytest.raises(ChannelLossMetadataError, match="per-microbatch metadata list lengths"):
        computer.begin_step(
            [["a"]],
            [torch.tensor([[0]])],
            [torch.ones(1, 1, dtype=torch.long)],
            per_mb_packed_global_physical_seq_lens=[],
        )


def test_base_step_begin_captures_channel_metadata_before_multisource_meter():
    calls = []
    step_end_calls = []

    class RecordingCallback:
        def __init__(self, name, consume_metadata=False):
            self.name = name
            self.consume_metadata = consume_metadata

        def on_step_begin(self, state, micro_batches=None, **kwargs):
            calls.append((self.name, "ds_idx" in micro_batches[0], kwargs["channel_loss_source_repeat"]))
            if self.consume_metadata:
                micro_batches[0].pop("ds_idx")
                micro_batches[0].pop("source_name")

        def on_step_end(self, state, **kwargs):
            step_end_calls.append(self.name)

    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.channel_loss_callback = RecordingCallback("channel")
    meter = RecordingCallback("meter", consume_metadata=True)
    tail = RecordingCallback("tail")
    # Production keeps the meter before channel loss for ``on_step_end`` so
    # the meter resets step metrics before channel loss publishes its values.
    # ``on_step_begin`` must still let channel loss snapshot source metadata
    # before the meter consumes it.
    trainer._callbacks = [meter, trainer.channel_loss_callback, tail]
    micro_batches = [{"ds_idx": torch.tensor([7]), "source_name": ["repoqa"]}]

    BaseTrainer.on_step_begin(trainer, micro_batches=micro_batches, channel_loss_source_repeat=2)

    assert calls == [("channel", True, 2), ("meter", True, 2), ("tail", False, 2)]

    BaseTrainer.on_step_end(trainer)

    assert step_end_calls == ["meter", "channel", "tail"]


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


@pytest.mark.parametrize("callback", [None, object()], ids=["missing", "unregistered"])
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


def test_base_forward_backward_refreshes_primary_labels_after_device_transfer(monkeypatch):
    _install_test_parallel_state(monkeypatch)
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=1)
    trainer.device = torch.device("meta")  # Second-device seam; no accelerator required.
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            channel_loss=ChannelLossConfig(enable=True, interval=1),
            enable_batch_invariant_mode=False,
            local_rank=0,
        )
    )
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.LOG_SAMPLE = False
    trainer.postforward = lambda outputs, batch: (outputs.loss, {"loss": outputs.loss.detach()}, {})
    trainer.preforward = lambda batch: BaseTrainer.preforward(trainer, batch)

    class DeviceTransferProbe(torch.nn.Module):
        parallel_state = "base"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, labels, **kwargs):
            computer = trainer.channel_loss_callback.computer
            assert labels.device.type == "meta"
            assert computer._primary_labels is labels
            computer.compute_side_channel(None, labels, 8, None, None)
            return SimpleNamespace(loss=self.weight * 2)

    trainer.model = DeviceTransferProbe()
    trainer.channel_loss_callback = ChannelLossCallback(trainer)
    trainer._callbacks = [trainer.channel_loss_callback]
    # Isolate lifecycle from CE, which cannot materialize a meta tensor.
    monkeypatch.setattr(
        trainer.channel_loss_callback.computer,
        "_compute_channel_loss",
        lambda **kwargs: [
            {
                "source_id": 3,
                "loss_sum": torch.tensor(2.0),
                "token_count": torch.tensor(2),
                "sample_count": torch.tensor(1),
                "input_token_count": torch.tensor(3),
            }
        ],
    )
    batch = {"labels": torch.tensor([[1, 2, 3]]), "position_ids": torch.arange(3)[None], "ds_idx": torch.tensor([3])}
    trainer.channel_loss_callback.on_step_begin(trainer.state, micro_batches=[batch])
    loss, _, _ = BaseTrainer.forward_backward_step(trainer, batch)
    computer = trainer.channel_loss_callback.computer
    assert loss.item() == 2
    assert computer.strict_observation_errors == []
    assert tuple(value.item() for value in computer.step_data_totals[3]) == (1, 3, 2)


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
        return {**micro_batch, "labels": micro_batch["labels"].to("meta")}

    base.preforward = preforward
    trainer = object.__new__(TextDPOTrainer)
    trainer.base = base
    trainer.policy_model = policy_model
    trainer.reference_model = reference_model
    forward_calls = []

    def concatenated_forward(model, micro_batch):
        if model is policy_model:
            assert base.channel_loss_callback.computer._primary_labels is micro_batch["labels"]
            assert micro_batch["labels"].device.type == "meta"
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
        "cu_seq_lens_q": torch.tensor([0, 2, 4, 6, 8]),
        "_veomni_channel_loss_source_provenance": channel_loss_module._ChannelLossProvenance(
            {
                "source_ids": [3, 4],
                "source_cu": [0, 4, 8],
                "input_mask": torch.ones(8, dtype=torch.bool),
                "tail": 0,
                "sp_size": 1,
                "sp_rank": 0,
            }
        ),
        "ds_idx": torch.tensor([3, 4]),
        "source_name": ["train/a", "train/b"],
    }

    try:
        base.channel_loss_callback.computer.install(policy_model)
        TextDPOTrainer.on_step_begin(trainer, micro_batches=[micro_batch])
        assert base.channel_loss_callback.computer._per_mb_source_ids == [[3, 4]]
        TextDPOTrainer.forward_backward_step(trainer, micro_batch)

        assert policy_model.loss_calls == 1
        assert reference_model.loss_calls == 1
        assert set(base.channel_loss_callback.computer.step_totals) == {3, 4}
        assert base.channel_loss_callback.computer.step_totals[3][1].item() == 2
        assert base.channel_loss_callback.computer.step_totals[4][1].item() == 2
        for source_id in (3, 4):
            samples, inputs, labels = base.channel_loss_callback.computer.step_data_totals[source_id]
            assert (samples.item(), inputs.item(), labels.item()) == (1, 4, 2)
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


def test_channel_loss_builds_four_data_metrics_from_compact_counts():
    cfg = ChannelLossConfig(enable=True)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)), environ_meter=None)
    callback = ChannelLossCallback(trainer)

    metrics = callback._build_data_metrics({7: (2, 11, 5)}, {7: "repoqa"})

    assert metrics == {
        "samples/source-i-7__repoqa": 2.0,
        "input_tokens/source-i-7__repoqa": 11.0,
        "label_tokens/source-i-7__repoqa": 5.0,
        "label_tokens_per_sample/source-i-7__repoqa": 2.5,
    }


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
    callback.computer.step_data_totals = {
        0: (torch.tensor(2), torch.tensor(10), torch.tensor(3)),
        1: (torch.tensor(1), torch.tensor(4), torch.tensor(1)),
    }
    callback._collect_step = True

    callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0)

    assert math.isclose(trainer.step_env_metrics["channel_loss/source-i-0__source_a"], 2.0)
    assert math.isclose(trainer.step_env_metrics["channel_loss/source-i-1__source-b"], 3.0)
    assert math.isclose(trainer.step_env_metrics["channel_loss_weighted/source-i-0__source_a"], 1.5)
    assert math.isclose(trainer.step_env_metrics["channel_tokens/source-i-1__source-b"], 1.0)
    assert trainer.step_env_metrics["samples/source-i-0__source_a"] == 2.0
    assert trainer.step_env_metrics["input_tokens/source-i-0__source_a"] == 10.0
    assert trainer.step_env_metrics["label_tokens/source-i-0__source_a"] == 3.0
    assert trainer.step_env_metrics["label_tokens_per_sample/source-i-0__source_a"] == 1.5
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
        callback.computer.step_data_totals = {0: (torch.tensor(1), torch.tensor(2), torch.tensor(1))}
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
        "group-data",
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
        "group-data",
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
        "group-data",
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

    computer = _legacy_computer(parallel_state=_test_parallel_state())
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


def test_strict_missing_primary_labels_prevents_train_step_optimizer_update(monkeypatch):
    import veomni.trainer.base as base_module

    state = _test_parallel_state()
    trainer = object.__new__(BaseTrainer)
    trainer.state = TrainerState(global_step=0)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(channel_loss=ChannelLossConfig(enable=True, interval=1, strict=True))
    )
    calls = []
    weight = torch.nn.Parameter(torch.tensor(2.0))
    trainer.model = SimpleNamespace(
        optimizer=SimpleNamespace(step=lambda: calls.append("optimizer"), zero_grad=lambda: calls.append("zero")),
        lr_scheduler=SimpleNamespace(step=lambda: calls.append("scheduler")),
    )
    with use_parallel_state(state):
        callback = ChannelLossCallback(trainer)
    monkeypatch.setattr(
        callback.computer,
        "_compute_channel_loss",
        lambda **kwargs: [{"source_id": 3, "loss_sum": torch.tensor(1.0), "token_count": torch.tensor(1)}],
    )
    trainer._reset_async_activation_offload_if_enabled = lambda *args: None
    trainer.on_step_begin = lambda micro_batches: callback.on_step_begin(trainer.state, micro_batches=micro_batches)
    trainer.sync_before_train_step = lambda: None
    trainer.model_reshard = lambda *args: None
    trainer._configure_hsdp_allreduce = lambda *args: None
    monkeypatch.setattr(base_module, "count_loss_token", lambda *args: 1)
    monkeypatch.setattr(base_module, "reduce_global_loss_token", lambda value: value)
    monkeypatch.setattr(base_module, "mark_compile_step_begin", lambda *args: None)

    def forward_backward_step(micro_batch):
        with callback.micro_step_context(trainer.state, micro_batch):
            with callback.model_forward_context():
                loss = weight * 2
                callback.computer.compute_side_channel(
                    logits=torch.randn(1, 2, 4),
                    labels=torch.tensor([[3, 4]]),
                    vocab_size=4,
                    hidden_states=None,
                    weights=None,
                )
            loss.backward()
        return loss, {}, {}

    trainer.forward_backward_step = forward_backward_step
    batch = {"labels": torch.tensor([[1, 2]]), "position_ids": torch.tensor([[0, 1]]), "ds_idx": torch.tensor([3])}
    with pytest.raises(ChannelLossMetadataError, match="before optimizer update"):
        BaseTrainer.train_step(trainer, iter([[batch]]))
    assert calls == []
    assert weight.item() == 2.0
    assert weight.grad is not None


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
            "primary_mismatch",
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

            computer = _legacy_computer(parallel_state=scenario_state)
            computer.strict = not (scenario == "strict_mismatch" and rank == 1)
            computer._source_ids = ["source-a"]
            computer._position_ids = torch.tensor([[0, 1]]) if rank == 0 else torch.tensor([[2, 3]])
            computer._packed_global_cu_seqlens = [0, 4]
            computer._tail_padding_length = 0
            if scenario == "primary_mismatch" and rank == 1:
                computer._primary_labels = torch.tensor([[3, 4]])
            original_serialize = channel_loss_module._serialize_channel_sp_observation
            original_digest = channel_loss_module._source_signature_digest

            if scenario == "source" and rank == 1:
                computer._source_ids = ["source-b"]
            elif scenario == "oversized_source_id" and rank == 1:
                computer._source_ids = [10**4300]
            elif scenario == "missing" and rank == 1:
                computer._source_ids = []
            elif scenario == "generic" and rank == 1:
                computer._compute_channel_loss = lambda *, defer_sp=False, **kwargs: (_ for _ in ()).throw(
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
        computer = _legacy_computer(parallel_state=state)
        computer.strict = True
        computer._source_ids = ["a", "b", "c"]
        computer._position_ids = torch.tensor([[0, 1, 2, 0]]) if rank == 0 else torch.tensor([[1, 2, 0, 1]])
        computer._packed_cu_seqlens = [0, 3, 4, 4] if rank == 0 else [0, 0, 2, 4]
        computer._packed_global_cu_seqlens = [0, 3, 6, 8]
        computer._tail_padding_length = 0
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


def _gloo_dpo_mtp_channel_worker(rank, world_size, init_file, result_queue):
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        parallel = _test_parallel_state(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )

        dpo = _legacy_computer(parallel_state=parallel)
        dpo.strict = True
        dpo._source_ids = ["source", "source"]
        dpo._position_ids = torch.tensor([[0, 1, 2]])
        dpo._packed_cu_seqlens = dpo._packed_global_cu_seqlens = [0, 3, 6]
        dpo._tail_padding_length = 0
        dpo._dpo_segments = True
        dpo._source_repeat = 2
        dpo_labels = torch.tensor([[2, 3, 4]]) if rank == 0 else torch.tensor([[5, 6, IGNORE_INDEX]])
        dpo._primary_labels = dpo_labels
        dpo._require_primary_labels = True
        with dpo.capture():
            dpo.compute_side_channel(torch.randn(1, 3, 8), dpo_labels, 8, None, None)
        dpo.after_micro_step()
        assert dpo.step_totals["source"][1].item() == 4
        assert tuple(item.item() for item in dpo.step_data_totals["source"]) == (1, 6, 4)

        computer = _legacy_computer(parallel_state=parallel)
        computer.strict = True
        computer._source_ids = ["a", "b", "c"]
        computer._position_ids = torch.tensor([[0, 1, 2, 0]]) if rank == 0 else torch.tensor([[1, 2, 0, 1]])
        computer._packed_cu_seqlens = [0, 3, 4, 4] if rank == 0 else [0, 0, 2, 4]
        computer._packed_global_cu_seqlens = [0, 3, 6, 8]
        computer._tail_padding_length = 0
        primary = torch.tensor([[0, 1, IGNORE_INDEX, 2]]) if rank == 0 else torch.tensor([[3, 4, 5, 6]])
        mtp = (
            torch.tensor([[[1, IGNORE_INDEX, 2, 3], [IGNORE_INDEX, 4, 5, IGNORE_INDEX]]])
            if rank == 0
            else torch.full((1, 2, 4), IGNORE_INDEX)
        )
        computer._primary_labels = primary
        computer._require_primary_labels = True
        computer._mtp_labels = mtp
        safe_mtp = mtp.reshape(2, 4).clone()
        has_valid = bool(safe_mtp.ne(IGNORE_INDEX).any())
        if not has_valid:
            safe_mtp.reshape(-1)[0] = 0
        main_logits = torch.randn(1, 4, 8, requires_grad=True)
        mtp_logits = torch.randn(2, 4, 8, requires_grad=True)
        ref_main = main_logits.detach().clone().requires_grad_()
        ref_mtp = mtp_logits.detach().clone().requires_grad_()
        main_loss = F.cross_entropy(main_logits.flatten(0, 1), primary.flatten(), ignore_index=IGNORE_INDEX)
        aux_loss = F.cross_entropy(mtp_logits.flatten(0, 1), safe_mtp.flatten(), ignore_index=IGNORE_INDEX) * int(
            has_valid
        )
        reference = F.cross_entropy(ref_main.flatten(0, 1), primary.flatten(), ignore_index=IGNORE_INDEX)
        reference = reference + F.cross_entropy(
            ref_mtp.flatten(0, 1), safe_mtp.flatten(), ignore_index=IGNORE_INDEX
        ) * int(has_valid)
        with computer.capture():
            computer.compute_side_channel(main_logits, primary, 8, None, None)
            computer.compute_side_channel(mtp_logits, safe_mtp, 8, None, None, shift_labels=safe_mtp)
        computer.after_micro_step()
        (main_loss + aux_loss).backward()
        reference.backward()
        torch.testing.assert_close(main_logits.grad, ref_main.grad, rtol=0, atol=0)
        torch.testing.assert_close(mtp_logits.grad, ref_mtp.grad, rtol=0, atol=0)
        assert [computer.step_totals[source][1].item() for source in ("a", "b", "c")] == [6, 4, 2]
        assert [computer.step_data_totals[source][0].item() for source in ("a", "b", "c")] == [1, 1, 1]
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_channel_loss_real_gloo_sp_preflight_avoids_divergent_payload_collectives(tmp_path):
    _run_gloo_workers(
        _gloo_sp_preflight_worker,
        world_size=2,
        init_file=tmp_path / "sp-preflight-init",
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_channel_loss_real_gloo_two_sp_stripes_propagate_strict_failure_worldwide(tmp_path):
    _run_gloo_workers(
        _gloo_two_stripe_worker,
        world_size=4,
        init_file=tmp_path / "sp-stripes-init",
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_channel_loss_real_gloo_packed_loss_and_training_gradient_parity(tmp_path):
    _run_gloo_workers(_gloo_packed_channel_parity_worker, world_size=2, init_file=tmp_path / "sp-packed-init")


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_channel_loss_real_gloo_dpo_and_mtp_depths(tmp_path):
    _run_gloo_workers(_gloo_dpo_mtp_channel_worker, world_size=2, init_file=tmp_path / "sp-dpo-mtp-init")


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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
    computer = _legacy_computer()
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
@pytest.mark.parametrize("enabled", [False, True])
def test_channel_loss_does_not_change_nonfirst_empty_sample_handling(monkeypatch, enabled):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    cfg = ChannelLossConfig(enable=enabled)
    keys = tuple(cfg.source_id_keys) if enabled else None
    with pytest.raises(IndexError):
        data_collator.MainCollator(channel_loss_source_id_keys=keys)(list(reversed(_channel_features())))


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
    callback.computer.before_micro_step(primary_labels=observed["labels"])
    callback.computer._original_loss_fn = lambda logits, labels, vocab_size: F.cross_entropy(
        logits.reshape(-1, vocab_size),
        F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    callback.computer._original_loss_fn.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
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
@pytest.mark.parametrize("strict", [False, True])
def test_channel_loss_empty_first_source_and_padding_counts_input_tokens(monkeypatch, strict):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)

    def features():
        examples = _channel_features()
        examples[1].update(
            input_ids=torch.tensor([0, 1, 2]),
            labels=torch.tensor([0, 1, 2]),
            attention_mask=torch.ones(3, dtype=torch.long),
            position_ids=torch.arange(3),
        )
        return examples

    cfg = ChannelLossConfig(enable=True, interval=1, strict=strict)
    baseline = data_collator.MainCollator(pad_to_length=4)(features())
    batch = data_collator.MainCollator(pad_to_length=4, channel_loss_source_id_keys=tuple(cfg.source_id_keys))(
        features()
    )
    assert batch[data_collator.CHANNEL_LOSS_PROVENANCE]["source_cu"] == [0, 0, 3]
    assert batch["tail_padding_length"].item() == 1
    torch.testing.assert_close(batch["labels"], baseline["labels"], rtol=0, atol=0)

    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg)))
    callback = ChannelLossCallback(trainer)
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch])
    callback.computer.before_micro_step(primary_labels=batch["labels"])
    callback.computer._original_loss_fn = lambda logits, labels, vocab_size: F.cross_entropy(
        logits.reshape(-1, vocab_size),
        F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    callback.computer._original_loss_fn.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
    logits = torch.randn(1, 4, 5, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    with callback.computer.capture():
        loss = callback.computer._wrapped_loss_fn(logits, batch["labels"], 5)
    reference_loss = callback.computer._original_loss_fn(reference, baseline["labels"], 5)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0, atol=0)
    assert [(item["source_id"], item["input_token_count"].item()) for item in callback.computer._data_result] == [
        (0, 0),
        (1, 3),
    ]


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
    callback.computer.before_micro_step(primary_labels=batch["labels"])
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
    base._get_registered_channel_loss_callback = lambda: base.channel_loss_callback
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
        "_veomni_channel_loss_source_provenance": channel_loss_module._ChannelLossProvenance(
            {
                "source_ids": [3, 4],
                "source_cu": [0, 4, 8],
                "input_mask": torch.ones(8, dtype=torch.bool),
                "tail": 0,
                "sp_size": 1,
                "sp_rank": 0,
            }
        ),
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
        CHANNEL_LOSS_PROVENANCE: channel_loss_module._ChannelLossProvenance(
            {
                "source_ids": [0],
                "source_cu": [0, 132],
                "input_mask": torch.ones(132, dtype=torch.bool),
                "tail": 0,
                "sp_size": 1,
                "sp_rank": 0,
            }
        ),
    }
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch], channel_loss_source_repeat=2)
    callback.computer.before_micro_step(primary_labels=batch["labels"])

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
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
    computer = _legacy_computer(parallel_state=_test_parallel_state(sp_enabled=True, sp_size=1))
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._capture_topology = computer._snapshot_topology()
    computer._observation_attempt_count = 2
    segment = {
        "batch_idx": 0,
        "sp_rank": 0,
        "local_order": 0,
        "is_start": True,
        "loss_sum": torch.tensor(1.0),
        "token_count": torch.tensor(1),
        "has_explicit_padding_mask": torch.tensor(False),
        "has_only_zero_positions": False,
    }
    computer._pending_observations = [
        {
            "segments": [segment],
            "source_ids": ["source"],
            "device": torch.device("cpu"),
            "data": False,
            "mode": "source",
        }
        for _ in range(2)
    ]
    results = iter([[{"source_id": "source", "loss_sum": torch.tensor(1.0), "token_count": torch.tensor(1)}], []])
    monkeypatch.setattr(computer, "_merge_sp_segments", lambda *args, **kwargs: next(results))
    context = pytest.raises(ChannelLossMetadataError, match="no source-aligned") if strict else nullcontext()
    with context:
        computer._finalize_capture()
    assert computer._result is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_strict_rejects_missing_non_sp_observation():
    computer = _legacy_computer(parallel_state=_test_parallel_state())
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.tensor([[0, 1]])
    computer._original_loss_fn = lambda labels: torch.tensor(1.0)
    with pytest.raises(ChannelLossMetadataError, match="no source-aligned"):
        with computer.capture():
            computer._wrapped_loss_fn(labels=torch.tensor([[1, 2]]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("nested", [False, True])
def test_channel_loss_discards_observation_when_forward_fails(nested):
    computer = _legacy_computer(parallel_state=_test_parallel_state())
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
        if scenario == "missing_provenance":
            batch.pop(data_collator.CHANNEL_LOSS_PROVENANCE)
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
        callback.on_step_begin(
            TrainerState(global_step=1), micro_batches=[batch], channel_loss_source_repeat=2 if dpo else 1
        )
        callback.computer.before_micro_step(primary_labels=batch["labels"])

        def training_loss(logits, labels, vocab_size, shift_labels=None):
            return F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX)

        training_loss.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
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
            "missing_provenance",
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
        assert [int(item["sample_count"].item()) for item in result] == [1] * len(positions)
        assert [int(item["input_token_count"].item()) for item in result] == [
            sum(i % 3 != 1 for i in range(len(pos))) for pos in positions
        ]
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
        ("missing_provenance", 2),
        ("mismatched_global_cu", 2),
        ("dpo_global_cu_mismatch", 2),
        ("dpo_attention_cu_mismatch", 2),
        ("dpo_observer_error", 2),
        ("two_stripes", 4),
        ("strict_two_stripes", 4),
    ],
)
@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
def test_real_gloo_channel_provenance(scenario, world_size):
    spawn_with_rendezvous(_gloo_channel_provenance_worker, args=(world_size, scenario), nprocs=world_size)


def _gloo_parent_packed_loss_worker(rank, world_size, scenario, endpoint):
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
        computer = _legacy_computer(parallel_state=state)
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
                logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
            )

        main_loss.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
        computer._original_loss_fn = main_loss
        if scenario in ("strict_missing", "bad_global_no_local_strict", "cp_only_global_strict"):
            with pytest.raises(
                ChannelLossMetadataError,
                match="SP ranks disagree|no causal-loss|global packed CU|CP requires|source metadata is missing",
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
    spawn_with_rendezvous(_gloo_parent_packed_loss_worker, args=(world_size, scenario), nprocs=world_size)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("key", ["_veomni_channel_loss_hook_changed", "_veomni_channel_loss_source_provenance"])
def test_channel_loss_does_not_consume_colliding_user_metadata(monkeypatch, key):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)

    def features():
        values = _channel_features()
        for feature in values:
            feature[key] = False
        return values

    cfg = ChannelLossConfig(enable=True, interval=1)
    baseline = data_collator.MainCollator()(features())
    batch = data_collator.MainCollator(channel_loss_source_id_keys=tuple(cfg.source_id_keys))(features())
    callback = ChannelLossCallback(SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(channel_loss=cfg))))
    callback.on_step_begin(TrainerState(global_step=1), micro_batches=[batch])
    callback.strip_model_inputs(batch)
    torch.testing.assert_close(batch[key], baseline[key], rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_base_preforward_strips_only_private_provenance_without_callback():
    from veomni.data.data_collator import CHANNEL_LOSS_PROVENANCE, _ChannelLossProvenance

    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer.LOG_SAMPLE = False
    private = {"input_ids": torch.tensor([[1, 2]]), CHANNEL_LOSS_PROVENANCE: _ChannelLossProvenance({})}
    assert CHANNEL_LOSS_PROVENANCE not in trainer.preforward(private)
    user = {"input_ids": torch.tensor([[1, 2]]), CHANNEL_LOSS_PROVENANCE: torch.tensor([7])}
    torch.testing.assert_close(trainer.preforward(user)[CHANNEL_LOSS_PROVENANCE], torch.tensor([7]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_provenance_survives_dataloader_worker_serialization():
    import pickle

    from veomni.data.data_collator import CHANNEL_LOSS_PROVENANCE, _ChannelLossProvenance

    batch = {CHANNEL_LOSS_PROVENANCE: _ChannelLossProvenance({"source_cu": [0, 3]})}
    restored = pickle.loads(pickle.dumps(batch))
    assert isinstance(restored[CHANNEL_LOSS_PROVENANCE], _ChannelLossProvenance)
    assert channel_loss_module._pop_channel_loss_provenance(restored)["source_cu"] == [0, 3]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("add_labels", [False, True])
def test_channel_loss_rejects_metadata_hook_changing_mtp_labels(monkeypatch, add_labels):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    features = _channel_features()
    if not add_labels:
        for feature in features:
            feature["mtp_labels"] = feature["labels"].clone()

    def mutate_mtp(batch, _sp_pad):
        if add_labels:
            batch["mtp_labels"] = torch.zeros_like(batch["labels"])
        batch["mtp_labels"][0, -1] = 7

    cfg = ChannelLossConfig(enable=True)
    batch = data_collator.MainCollator(
        channel_loss_source_id_keys=tuple(cfg.source_id_keys), metadata_collate_func=mutate_mtp
    )(features)
    assert batch[data_collator.CHANNEL_LOSS_PROVENANCE] == {"error": "unsupported source layout"}
    assert batch["mtp_labels"][0, -1].item() == 7


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_channel_loss_rejects_metadata_hook_adding_attention_mask(monkeypatch):
    import veomni.data.data_collator as data_collator

    state = _stub_parallel_state(monkeypatch)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: state)
    features = _channel_features()
    for feature in features:
        feature.pop("attention_mask")

    def add_mask(batch, _sp_pad):
        batch["attention_mask"] = torch.tensor([[1, 0, 1, 1]])
        batch["attention_mask_is_all_ones"] = False

    cfg = ChannelLossConfig(enable=True)
    batch = data_collator.MainCollator(
        channel_loss_source_id_keys=tuple(cfg.source_id_keys), metadata_collate_func=add_mask
    )(features)
    assert batch[data_collator.CHANNEL_LOSS_PROVENANCE] == {"error": "unsupported source layout"}
    assert batch["attention_mask"].sum().item() == 3


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_direct_channel_computer_without_source_proof_skips_without_changing_loss():
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["empty", "real"]
    computer._position_ids = torch.tensor([[0, 1, 0, 1]])
    computer._original_loss_fn = lambda logits, labels, vocab_size: F.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    labels = torch.tensor([[0, 1, 2, 3]])
    logits = torch.randn(1, 4, 5, requires_grad=True)
    baseline_logits = logits.detach().clone().requires_grad_()
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, labels, 5)
    baseline = computer._original_loss_fn(baseline_logits, labels, 5)
    loss.backward()
    baseline.backward()
    assert computer._result is None
    torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, baseline_logits.grad, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dpo_half_precision_log_probs_accumulate_in_fp32(dtype):
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._dpo_segments = True
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 14000],
        "input_mask": torch.ones(14000, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    labels = torch.zeros(1, 14000, dtype=torch.long)
    log_probs = torch.full((1, 14000), -8, dtype=dtype, requires_grad=True)
    baseline_logits = log_probs.detach().clone().requires_grad_()
    with computer.capture():
        objective = log_probs.float().square().mean()
        computer.observe_dpo_log_probs(log_probs, labels, [0, 7000, 14000])
    baseline = baseline_logits.float().square().mean()
    objective.backward()
    baseline.backward()
    torch.testing.assert_close(objective, baseline, rtol=0, atol=0)
    torch.testing.assert_close(log_probs.grad, baseline_logits.grad, rtol=0, atol=0)
    result = computer._result[0]
    assert result["token_count"].item() == 13998
    assert result["loss_sum"].dtype == torch.float32
    assert result["loss_sum"].item() == 111984


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
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 4],
        "input_mask": torch.ones(4, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }

    def model_loss(logits, labels, vocab_size, hidden_states, weights):
        shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
        return compiled(hidden_states, shifted, weights, chunk_size=3)

    model_loss.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
    computer._original_loss_fn = model_loss
    with torch.no_grad(), computer.capture():
        actual = computer._wrapped_loss_fn(None, labels, 7, hidden, weight)
    torch.testing.assert_close(actual, reference)
    expected = F.cross_entropy(F.linear(hidden, weight).reshape(-1, 7), shift.reshape(-1), reduction="none")
    torch.testing.assert_close(computer._result[0]["loss_sum"], expected.sum())
    assert computer._result[0]["token_count"].item() == 3


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_compiled_custom_loss_without_input_contract_skips_channel_observation():
    from veomni.ops import resolve_op

    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    compiled = torch.compile(op, backend="eager", fullgraph=True)
    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[0, 1, 2, 3]])
    shift = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]

    def custom_loss(logits, labels, vocab_size, hidden_states, weights):
        return compiled(hidden_states * 2, shift, weights, chunk_size=3)

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["source"]
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 4],
        "input_mask": torch.ones(4, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._original_loss_fn = custom_loss
    with torch.no_grad(), computer.capture():
        loss = computer._wrapped_loss_fn(None, labels, 7, hidden, weight)
    assert computer._result is None
    with torch.no_grad():
        torch.testing.assert_close(loss, op(hidden * 2, shift, weight, chunk_size=3))


@pytest.mark.parametrize("compiled_diagnostic", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_contracted_custom_loss_ignores_unrelated_chunk_capture(compiled_diagnostic):
    from veomni.ops import resolve_op

    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    diagnostic = torch.compile(op, backend="eager", fullgraph=True) if compiled_diagnostic else op
    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[0, 1, 2, 3]])
    shift = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    if compiled_diagnostic:
        with torch.no_grad():
            diagnostic(hidden * 2, shift, weight, chunk_size=3)

    def custom_loss(logits, labels, vocab_size, hidden_states, weights, shift_labels):
        actual = F.cross_entropy(
            F.linear(hidden_states, weights).reshape(-1, vocab_size), shift_labels.reshape(-1), reduction="sum"
        )
        unused = diagnostic(hidden_states * 2, shift_labels, weights, chunk_size=3)
        return actual + 0 * unused

    custom_loss.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._source_ids = ["a", "b"]
    computer._source_provenance = {
        "source_ids": ["a", "b"],
        "source_cu": [0, 2, 4],
        "input_mask": torch.ones(4, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._original_loss_fn = custom_loss
    with torch.no_grad(), computer.capture():
        loss = computer._wrapped_loss_fn(None, labels, 7, hidden, weight, shift)
    expected = F.cross_entropy(F.linear(hidden, weight).reshape(-1, 7), shift.reshape(-1), reduction="none")
    torch.testing.assert_close(loss, expected.sum())
    torch.testing.assert_close(computer._result[0]["loss_sum"], expected[:2].sum())
    torch.testing.assert_close(computer._result[1]["loss_sum"], expected[2:].sum())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_loss_signature_preserves_positional_aliases_without_mislabeling_weights():
    values = (torch.tensor(1), torch.tensor(2), 3)

    def generic(a, b, c):
        return a, b, c

    bound = channel_loss_module._bind_loss_call_args(generic, values, {})
    assert (bound["logits"], bound["labels"], bound["vocab_size"]) == values

    def named(logits, labels, vocab_size, hidden_states, weights):
        return logits, labels, vocab_size, hidden_states, weights

    bound = channel_loss_module._bind_loss_call_args(named, (*values, torch.tensor(4), torch.tensor(5)), {})
    assert "ignore_index" not in bound and "num_items_in_batch" not in bound
    assert bound["hidden_states"].item() == 4 and bound["weights"].item() == 5

    def named_with_varargs(logits, labels, vocab_size, hidden_states, weights, *args):
        return logits, labels, vocab_size, hidden_states, weights, args

    bound = channel_loss_module._bind_loss_call_args(
        named_with_varargs, (*values, torch.tensor(4), torch.tensor(5)), {}
    )
    assert "ignore_index" not in bound and "num_items_in_batch" not in bound


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
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    loss.backward()
    assert calls == ["side" if override == "compute_side_channel" else "compute"]
    torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("override", ["compute_side_channel", "_compute_channel_loss"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("variadic", [False, True])
def test_dpo_legacy_observer_without_expanded_log_probs_fails_closed(override, strict, variadic):
    calls = []

    def legacy_side(self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100):
        calls.append(1)
        return [{"source_id": "source", "loss_sum": torch.tensor(123.0), "token_count": torch.tensor(1)}]

    def legacy_compute(
        self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index
    ):
        calls.append(1)
        return [{"source_id": "source", "loss_sum": torch.tensor(123.0), "token_count": torch.tensor(1)}]

    def legacy_side_variadic(
        self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100, **kwargs
    ):
        calls.append(1)

    def legacy_compute_variadic(
        self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index, **kwargs
    ):
        calls.append(1)
        return []

    methods = {
        "compute_side_channel": legacy_side_variadic if variadic else legacy_side,
        "_compute_channel_loss": legacy_compute_variadic if variadic else legacy_compute,
    }
    custom_computer = type("LegacyComputer", (ChannelLossComputer,), {override: methods[override]})
    computer = custom_computer(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._dpo_segments = True
    computer._source_ids = ["source"]
    log_probs = torch.randn(1, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3]])

    def observe():
        with computer.capture():
            objective = log_probs.square().sum()
            computer.observe_dpo_log_probs(log_probs, labels, [0, 2, 4])
        return objective

    if strict:
        with pytest.raises(ChannelLossMetadataError, match="expanded per-token log-probs"):
            observe()
    else:
        loss = observe()
        loss.backward()
        torch.testing.assert_close(log_probs.grad, 2 * log_probs.detach(), rtol=0, atol=0)
    assert not calls
    assert computer._result is None


@pytest.mark.parametrize("strict", [False, True])
def test_legacy_observer_exception_is_handled_without_losing_main_loss(strict):
    class LegacyComputer(ChannelLossComputer):
        def compute_side_channel(
            self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100
        ):
            raise RuntimeError("observer only")

    computer = LegacyComputer(parallel_state=_test_parallel_state())
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)

    def observe():
        with computer.capture():
            return computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)

    if strict:
        with pytest.raises(ChannelLossMetadataError, match="observer only"):
            observe()
    else:
        loss = observe()
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
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
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
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
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size, logits_bias=None: logits.square().sum()
    bias = torch.randn(1, 2, 3)
    with computer.capture():
        loss = computer._wrapped_loss_fn(
            torch.ones(1, 2, 3, requires_grad=True), torch.tensor([[0, 1]]), 3, logits_bias=bias
        )
    loss.backward()
    assert seen == [(bias, None)]


@pytest.mark.parametrize("override", ["compute_side_channel", "_compute_channel_loss"])
def test_legacy_observer_without_bias_keeps_source_metrics(override):
    def side(self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100):
        return super(type(self), self).compute_side_channel(
            logits, labels, vocab_size, hidden_states, weights, shift_labels, ignore_index
        )

    def compute(self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index):
        return super(type(self), self)._compute_channel_loss(
            logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index
        )

    legacy = type(
        "LegacyComputer", (ChannelLossComputer,), {override: side if override == "compute_side_channel" else compute}
    )
    computer = legacy(parallel_state=_test_parallel_state())
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, labels, 4)
    loss.backward()
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    expected = F.cross_entropy(
        logits.detach().reshape(-1, 4), shifted.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
    )
    torch.testing.assert_close(computer._result[0]["loss_sum"], expected)
    assert computer._result[0]["token_count"].item() == 2


def _gloo_legacy_compute_without_defer_worker(rank, world_size, strict, variadic, endpoint):
    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        calls = []

        def legacy(
            self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index
        ):
            calls.append(1)
            return []

        def legacy_variadic(
            self,
            logits,
            labels,
            vocab_size,
            hidden_states,
            weights,
            attention_mask,
            shift_labels,
            ignore_index,
            **kwargs,
        ):
            calls.append(1)
            return []

        LegacyComputer = type(
            "LegacyComputer",
            (ChannelLossComputer,),
            {"_compute_channel_loss": legacy_variadic if variadic else legacy},
        )

        state = SimpleNamespace(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = LegacyComputer(parallel_state=state)
        computer._require_source_provenance = False  # Exercise the legacy custom-observer path.
        computer.strict = strict
        computer._source_ids = ["source"]
        computer._position_ids = torch.tensor([[rank * 2, rank * 2 + 1]])
        computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
        logits = torch.randn(1, 2, 4, requires_grad=True)
        labels = torch.tensor([[0, 1]])
        with pytest.raises(ChannelLossMetadataError, match="cannot defer SP") if strict else nullcontext():
            with computer.capture():
                loss = computer._wrapped_loss_fn(logits, labels, 4)
        if not strict:
            loss.backward()
            torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
        assert not calls and computer._result is None
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("variadic", [False, True])
def test_gloo_legacy_compute_without_defer_fails_closed(strict, variadic):
    spawn_with_rendezvous(_gloo_legacy_compute_without_defer_worker, args=(2, strict, variadic), nprocs=2)


def _gloo_modern_compute_accepts_defer_worker(rank, world_size, endpoint):
    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        calls = []

        class ModernComputer(ChannelLossComputer):
            def _compute_channel_loss(self, *args, defer_sp=False, **kwargs):
                calls.append(defer_sp)
                return super()._compute_channel_loss(*args, defer_sp=defer_sp, **kwargs)

        state = SimpleNamespace(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = ModernComputer(parallel_state=state)
        computer._require_source_provenance = False  # Exercise the legacy custom-observer path.
        computer.strict = True
        computer._source_ids = ["source"]
        computer._position_ids = torch.tensor([[rank * 2, rank * 2 + 1]])
        computer._source_provenance = {
            "source_ids": ["source"],
            "source_cu": [0, 4],
            "input_mask": torch.ones(4, dtype=torch.bool),
            "tail": 0,
            "sp_size": 2,
            "sp_rank": rank,
        }
        computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
        logits = torch.randn(1, 2, 4, requires_grad=True)
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 4)
        loss.backward()
        assert calls == [True]
        assert computer._result[0]["token_count"].item() == 4
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_gloo_modern_compute_receives_deferred_sp():
    spawn_with_rendezvous(_gloo_modern_compute_accepts_defer_worker, args=(2,), nprocs=2)


def _gloo_chunk_capture_failure_worker(rank, world_size, endpoint):
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        state = SimpleNamespace(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = ChannelLossComputer(parallel_state=state)
        computer._require_source_provenance = False  # Exercise the legacy custom-observer path.
        computer._source_ids = ["source"]
        computer._position_ids = torch.tensor([[rank * 2, rank * 2 + 1]])
        computer._source_provenance = {
            "source_ids": ["source"],
            "source_cu": [0, 4],
            "input_mask": torch.ones(4, dtype=torch.bool),
            "tail": 0,
            "sp_size": 2,
            "sp_rank": rank,
        }
        labels = torch.tensor([[0, 1]]) if rank == 0 else torch.tensor([[2, IGNORE_INDEX]])
        logits = torch.randn(1, 2, 4, requires_grad=True)

        def loss_fn(logits, labels, vocab_size):
            capture = chunk_loss._CAPTURE.get()
            capture["op_calls"] = 1
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
            assert computer._observation_attempt_count == 1
            if rank == 0:
                assert len(computer._capture_errors) == 1
                assert "observer failed" in computer._capture_errors[0][1]
                assert not computer._pending_observations
            else:
                assert not computer._capture_errors
                assert len(computer._pending_observations) == 1
        loss.backward()
        assert computer._result is None
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_gloo_one_rank_custom_chunk_capture_failure_skips_synchronously():
    spawn_with_rendezvous(_gloo_chunk_capture_failure_worker, args=(2,), nprocs=2)


def test_positional_only_bias_is_not_claimed_as_supported():
    def observer(logits_bias=None, /, **kwargs):
        pass

    bias = torch.ones(1)
    supported = channel_loss_module._supported_observer_kwargs(observer, {"logits_bias": bias, "defer_sp": True})
    assert "logits_bias" not in supported and "defer_sp" not in supported


@pytest.mark.parametrize("strict", [False, True])
def test_dpo_observer_detach_error_is_coordinated(strict):
    class BadLogProbs:
        def detach(self):
            raise RuntimeError("observer detach failed")

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._dpo_segments = True
    computer._source_ids = ["source"]
    logits = torch.randn(1, 4, requires_grad=True)
    with pytest.raises(ChannelLossMetadataError, match="observer detach failed") if strict else nullcontext():
        with computer.capture():
            loss = logits.square().sum()
            computer.observe_dpo_log_probs(BadLogProbs(), torch.tensor([[0, 1, 2, 3]]), [0, 2, 4])
    if not strict:
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    assert computer._result is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_chunk_capture_error_recomputes_metrics_and_preserves_previous_result(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._result = [{"source_id": "stale"}]

    original_ce = F.cross_entropy
    calls = []

    def fail_once(*args, **kwargs):
        if kwargs.get("reduction") == "none" and not calls:
            calls.append(True)
            raise RuntimeError("observer failed")
        return original_ce(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", fail_once)
    computer._original_loss_fn = partial(ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"))
    labels = torch.tensor([[0, 1, 2]])
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(4, 4, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_logits = F.linear(reference_hidden, reference_weight)
    target = torch.tensor([1, 2, IGNORE_INDEX])
    reference_sum = original_ce(reference_logits.flatten(0, 1), target, ignore_index=IGNORE_INDEX, reduction="sum")
    (reference_sum / 2).backward()
    with computer.capture():
        loss, _, _ = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight)
    loss.backward()
    assert computer._result[0]["source_id"] == "stale"
    assert computer._result[-1]["source_id"] == "source"
    assert computer._result[-1]["token_count"].item() == 2
    torch.testing.assert_close(computer._result[-1]["loss_sum"], reference_sum.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ambiguous_multiple_chunk_observations_are_not_recomputed(strict):
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }

    def loss_fn(logits, labels, vocab_size):
        chunk_loss._CAPTURE.get().update(op_calls=2, error="observer failed")
        return logits.square().sum()

    computer._original_loss_fn = loss_fn
    logits = torch.randn(1, 3, 4, requires_grad=True)
    with pytest.raises(ChannelLossMetadataError, match="multiple chunk-loss") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1, 2]]), 4)
    if not strict:
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    assert computer._result is None


@pytest.mark.parametrize("fail_call", [1, 2])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_real_chunk_capture_counts_multiple_calls_after_observer_failure(monkeypatch, fail_call):
    from veomni.ops import VeomniOp
    from veomni.trainer.callbacks.channel_loss_callback import capture_chunk_loss_per_token

    hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2])
    op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
    original_ce = F.cross_entropy
    attempts = []

    def fail_observer(*args, **kwargs):
        if kwargs.get("reduction") == "none":
            attempts.append(1)
            if len(attempts) == fail_call:
                raise RuntimeError("injected observer failure")
        return original_ce(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", fail_observer)
    observed, errors = [], []
    with capture_chunk_loss_per_token(observed.append, errors.append):
        first = op(hidden, target, weight, num_items_in_batch=None, ignore_index=IGNORE_INDEX)
        second = op(hidden * 2, target, weight, num_items_in_batch=None, ignore_index=IGNORE_INDEX)
    (first + second).backward()
    assert errors == ["multiple chunk-loss observations in one causal-loss call"]
    assert not observed
    assert hidden.grad is not None and weight.grad is not None


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("inject_error", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_custom_transformed_chunk_loss_does_not_report_outer_ce(monkeypatch, strict, inject_error):
    from veomni.ops import VeomniOp

    computer = ChannelLossComputer(parallel_state=_test_parallel_state())
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")

    def loss_fn(hidden_states, labels, weights):
        return op(hidden_states.reshape(-1, 4) * 2, labels.flatten(), weights, num_items_in_batch=None)

    original_ce = F.cross_entropy

    def fail_observer(*args, **kwargs):
        if kwargs.get("reduction") == "none":
            raise RuntimeError("injected observer failure")
        return original_ce(*args, **kwargs)

    if inject_error:
        monkeypatch.setattr(F, "cross_entropy", fail_observer)
    computer._original_loss_fn = loss_fn
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    with pytest.raises(ChannelLossMetadataError, match="custom causal loss") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(hidden_states=hidden, labels=labels, weights=weight)
    if not strict:
        loss.backward()
        assert hidden.grad is not None and weight.grad is not None
    assert computer._result is None


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("inject_error", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_official_loss_with_custom_chunk_op_is_not_a_proven_fallback(monkeypatch, strict, inject_error):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    real_op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")

    class CustomOp:
        impl = "chunk_loss"

        def __call__(self, hidden, labels, weight, **kwargs):
            return real_op(hidden * 2, labels, weight, **kwargs)

    original_ce = F.cross_entropy

    def fail_observer(*args, **kwargs):
        if kwargs.get("reduction") == "none":
            raise RuntimeError("injected observer failure")
        return original_ce(*args, **kwargs)

    if inject_error:
        monkeypatch.setattr(F, "cross_entropy", fail_observer)
    computer._original_loss_fn = partial(ForCausalLMLoss, op=CustomOp())
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    with pytest.raises(ChannelLossMetadataError, match="custom causal loss") if strict else nullcontext():
        with computer.capture():
            loss, _, _ = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight)
    if not strict:
        loss.backward()
        assert hidden.grad is not None and weight.grad is not None
    assert computer._result is None


@pytest.mark.parametrize("prebound", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_return_log_probs_is_not_treated_as_causal_ce(monkeypatch, prebound):
    from veomni.models.loss_utils import ForCausalLMLoss

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._original_loss_fn = partial(ForCausalLMLoss, return_log_probs=True) if prebound else ForCausalLMLoss
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    with computer.capture():
        kwargs = {} if prebound else {"return_log_probs": True}
        loss, _, auxiliary = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight, **kwargs)
    assert loss is None and auxiliary.log_probs is not None
    auxiliary.log_probs.sum().backward()
    assert computer._result is None
    assert hidden.grad is not None and weight.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_prebound_logits_bias_is_used_for_chunk_fallback(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    bias = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    targets = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1)
    original_ce = F.cross_entropy
    expected = original_ce((F.linear(hidden, weight) + bias).flatten(0, 1), targets, reduction="none")
    failed = []

    def fail_once(*args, **kwargs):
        if kwargs.get("reduction") == "none" and not failed:
            failed.append(True)
            raise RuntimeError("injected capture failure")
        return original_ce(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", fail_once)
    computer._original_loss_fn = partial(
        ForCausalLMLoss,
        op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss"),
        logits_bias=bias,
    )
    with computer.capture():
        loss, _, _ = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight)
    loss.backward()
    torch.testing.assert_close(computer._result[0]["loss_sum"], expected.sum(), rtol=1e-6, atol=1e-6)
    assert bias.grad is not None and hidden.grad is not None and weight.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_standard_liger_handle_keeps_channel_loss_observation(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import eager, liger_kernel
    from veomni.ops.registry import OpEntry

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer.strict = True
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    # Exercise the framework handle contract on CPU; actual Liger numerical
    # behavior must be validated on the backend that provides the kernel.
    handle = object.__new__(VeomniOp)
    handle.op, handle.variant, handle.impl = "cross_entropy_loss", "standard", "liger_kernel"
    monkeypatch.setattr(liger_kernel, "forward", eager.forward)
    monkeypatch.setattr(liger_kernel, "backward", eager.backward)
    handle._entry = OpEntry(
        op="cross_entropy_loss",
        variant="standard",
        impl="liger_kernel",
        description="CPU registry test",
        forward=liger_kernel.forward,
        backward=liger_kernel.backward,
    )
    computer._original_loss_fn = partial(ForCausalLMLoss, op=handle)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    with computer.capture():
        loss, _, _ = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight)
    loss.backward()
    assert computer._result[0]["token_count"].item() == 2
    assert hidden.grad is not None and weight.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_device_override_of_standard_ce_is_not_treated_as_builtin(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    handle = object.__new__(VeomniOp)
    handle.op, handle.variant, handle.impl = "cross_entropy_loss", "standard", "eager"

    def custom_forward(*args, **kwargs):
        raise AssertionError("identity marker only")

    handle._entry = SimpleNamespace(
        forward=custom_forward,
        backward=None,
        wrapper=lambda hidden, target, weight, **kwargs: F.cross_entropy(F.linear(hidden * 2, weight), target),
    )
    computer._original_loss_fn = partial(ForCausalLMLoss, op=handle)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    with computer.capture():
        loss, _, _ = computer._wrapped_loss_fn(labels=labels, hidden_states=hidden, weights=weight)
    loss.backward()
    assert computer._result is None
    assert hidden.grad is not None and weight.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_mutated_standard_ce_wrapper_is_not_trusted(monkeypatch):
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    handle = VeomniOp("cross_entropy_loss", "standard", "eager")
    monkeypatch.setattr(
        handle.entry,
        "wrapper",
        lambda hidden, target, weight, **kwargs: F.cross_entropy(F.linear(hidden * 2, weight), target),
    )
    computer._original_loss_fn = partial(ForCausalLMLoss, op=handle)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    with computer.capture():
        loss, _, _ = computer._wrapped_loss_fn(labels=torch.tensor([[0, 1, 2]]), hidden_states=hidden, weights=weight)
    loss.backward()
    assert computer._result is None
    assert hidden.grad is not None and weight.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("anonymous_args", [(), (7, 0)])
def test_custom_partial_contract_rejects_ambiguous_anonymous_args(monkeypatch, anonymous_args):
    state = _install_test_parallel_state(monkeypatch)
    computer = ChannelLossComputer(parallel_state=state)
    computer.strict = False
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }

    def custom_ce(logits, labels, vocab_size, *extra):
        del vocab_size, extra
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), shifted.reshape(-1), ignore_index=-100)

    fn = partial(custom_ce)
    fn.__veomni_channel_loss_contract__ = "outer_causal_ce_v1"
    computer._original_loss_fn = fn
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[2, 0, 1]])
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, labels, 4, *anonymous_args)
    loss.backward()
    if anonymous_args:
        assert computer._result is None
    else:
        assert computer._result[0]["token_count"].item() == 2
        torch.testing.assert_close(computer._result[0]["loss_sum"] / 2, loss.detach())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_custom_loss_uninspectable_signature_keeps_training_in_non_strict_mode(monkeypatch):
    class Loss:
        __veomni_channel_loss_contract__ = "outer_causal_ce_v1"

        @property
        def __signature__(self):
            raise RuntimeError("signature unavailable")

        def __call__(self, logits, labels, vocab_size):
            shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
            return F.cross_entropy(logits.reshape(-1, vocab_size), shifted.reshape(-1), ignore_index=-100)

    computer = ChannelLossComputer(parallel_state=_install_test_parallel_state(monkeypatch))
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._original_loss_fn = Loss()
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[2, 0, 1]])
    with computer.capture():
        loss = computer._wrapped_loss_fn(logits, labels, 4)
    loss.backward()
    assert computer._result is None
    assert logits.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("mode", ["later_signature_error", "noncanonical_ignore"])
def test_custom_loss_ambiguous_observer_keeps_training_in_non_strict_mode(monkeypatch, mode):
    class Loss:
        __veomni_channel_loss_contract__ = "outer_causal_ce_v1"

        def __init__(self):
            self.inspections = 0

        @property
        def __signature__(self):
            self.inspections += 1
            if mode == "later_signature_error":
                if self.inspections > 1:
                    raise RuntimeError("signature unavailable")
                return inspect.signature(canonical)
            return inspect.signature(self.__call__)

        def __call__(self, logits, labels, vocab_size, *, ignored_label=-100):
            shifted = F.pad(labels, (0, 1), value=ignored_label)[..., 1:]
            return F.cross_entropy(logits.reshape(-1, vocab_size), shifted.reshape(-1), ignore_index=ignored_label)

    def canonical(logits, labels, vocab_size):
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
        return F.cross_entropy(logits.reshape(-1, vocab_size), shifted.reshape(-1), ignore_index=-100)

    obj = Loss()
    computer = ChannelLossComputer(parallel_state=_install_test_parallel_state(monkeypatch))
    computer._source_ids = ["source"]
    computer._position_ids = torch.arange(3).unsqueeze(0)
    computer._source_provenance = {
        "source_ids": ["source"],
        "source_cu": [0, 3],
        "input_mask": torch.ones(3, dtype=torch.bool),
        "tail": 0,
        "sp_size": 1,
        "sp_rank": 0,
    }
    computer._original_loss_fn = obj
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[2, 0, 1]])
    with computer.capture():
        loss = computer._wrapped_loss_fn(
            logits, labels, 4, **({"ignored_label": 0} if mode == "noncanonical_ignore" else {})
        )
    loss.backward()
    assert computer._result is None
    assert logits.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_loss_call_binding_does_not_treat_weight_as_ignore_index():
    def loss_fn(logits, labels, vocab_size, hidden_states, weights, *args):
        pass

    values = (torch.zeros(2), torch.ones(2), 3, torch.ones(2, 4), torch.ones(3, 4))
    bound = channel_loss_module._bind_loss_call_args(loss_fn, values, {})
    assert bound["weights"] is values[-1]
    assert "ignore_index" not in bound


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_legacy_observer_exception_preserves_main_loss_and_respects_strict_mode(strict):
    class LegacyComputer(ChannelLossComputer):
        def compute_side_channel(
            self, logits, labels, vocab_size, hidden_states, weights, shift_labels=None, ignore_index=-100
        ):
            raise RuntimeError("observer only")

    computer = LegacyComputer(parallel_state=_test_parallel_state())
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with pytest.raises(ChannelLossMetadataError, match="observer only") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    if not strict:
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    assert computer._result is None


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_legacy_compute_exception_respects_strict_mode(strict):
    class LegacyComputer(ChannelLossComputer):
        def _compute_channel_loss(
            self, logits, labels, vocab_size, hidden_states, weights, attention_mask, shift_labels, ignore_index
        ):
            raise RuntimeError("legacy compute failed")

    computer = LegacyComputer(parallel_state=_test_parallel_state())
    computer._require_source_provenance = False  # Legacy observer contract, tested separately from provenance.
    computer.strict = strict
    computer._source_ids = ["source"]
    computer._original_loss_fn = lambda logits, labels, vocab_size: logits.square().sum()
    logits = torch.ones(1, 2, 3, requires_grad=True)
    with pytest.raises(ChannelLossMetadataError, match="legacy compute failed") if strict else nullcontext():
        with computer.capture():
            loss = computer._wrapped_loss_fn(logits, torch.tensor([[0, 1]]), 3)
    if not strict:
        loss.backward()
        torch.testing.assert_close(logits.grad, 2 * logits.detach(), rtol=0, atol=0)
    assert computer._result is None


def _gloo_chunk_capture_fallback_worker(rank, world_size, fail_rank, endpoint):
    from datetime import timedelta

    import veomni.models.loss_utils.cross_entropy_loss as loss_module
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        state = SimpleNamespace(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = ChannelLossComputer(parallel_state=state)
        computer._source_ids = ["a", "b"]
        computer._position_ids = torch.tensor([[0, 1, 0], [1, 2, 3]])[rank : rank + 1]
        computer._source_provenance = {
            "source_ids": ["a", "b"],
            "source_cu": [0, 2, 6],
            "input_mask": torch.ones(6, dtype=torch.bool),
            "tail": 0,
            "sp_size": world_size,
            "sp_rank": rank,
        }
        torch.manual_seed(133)
        all_hidden = torch.randn(6, 4)
        all_targets = torch.tensor([1, 2, 3, 0, 1, IGNORE_INDEX])
        weight = torch.randn(4, 4).requires_grad_()
        hidden = all_hidden[rank * 3 : (rank + 1) * 3].unsqueeze(0).clone().requires_grad_()
        targets = all_targets[rank * 3 : (rank + 1) * 3].unsqueeze(0)
        original_ce = F.cross_entropy
        token_losses = original_ce(F.linear(all_hidden, weight.detach()), all_targets, reduction="none")
        ref_hidden = hidden.detach().clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()
        reference = original_ce(F.linear(ref_hidden, ref_weight).flatten(0, 1), targets.flatten())
        reference.backward()
        calls = []

        def fail_once(*args, **kwargs):
            if rank == fail_rank and kwargs.get("reduction") == "none" and not calls:
                calls.append(True)
                raise RuntimeError("observer failed")
            return original_ce(*args, **kwargs)

        F.cross_entropy = fail_once
        loss_module.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
        computer._original_loss_fn = partial(
            ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
        )
        with computer.capture():
            loss, _, _ = computer._wrapped_loss_fn(
                labels=targets, shift_labels=targets, hidden_states=hidden, weights=weight
            )
        loss.backward()
        assert [item["token_count"].item() for item in computer._result] == [2, 3]
        torch.testing.assert_close(computer._result[0]["loss_sum"], token_losses[:2].sum(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(computer._result[1]["loss_sum"], token_losses[2:].sum(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-6, atol=1e-6)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("fail_rank", [0, 1])
def test_gloo_chunk_capture_fallback_keeps_ranks_in_same_collective(fail_rank):
    spawn_with_rendezvous(_gloo_chunk_capture_fallback_worker, args=(2, fail_rank), nprocs=2)


def _gloo_real_sp_chunk_fallback_worker(rank, world_size, endpoint):
    from datetime import timedelta

    import veomni.models.loss_utils.cross_entropy_loss as loss_module
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    torch.distributed.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        state = SimpleNamespace(
            sp_enabled=True, sp_group=torch.distributed.group.WORLD, sp_size=world_size, sp_rank=rank
        )
        computer = ChannelLossComputer(parallel_state=state)
        computer._source_ids = ["a", "b"]
        computer._position_ids = torch.tensor([[0, 1, 0], [1, 2, 3]])[rank : rank + 1]
        computer._source_provenance = {
            "source_ids": ["a", "b"],
            "source_cu": [0, 2, 6],
            "input_mask": torch.ones(6, dtype=torch.bool),
            "tail": 0,
            "sp_size": world_size,
            "sp_rank": rank,
        }
        torch.manual_seed(133)
        all_hidden = torch.randn(6, 4)
        targets = torch.tensor([1, 2, 3, 0, 1, IGNORE_INDEX])
        weight = torch.randn(4, 4).requires_grad_()
        ref_hidden = all_hidden.clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()
        ref_loss = F.cross_entropy(F.linear(ref_hidden, ref_weight), targets)
        ref_loss.backward()
        hidden = all_hidden[rank * 3 : (rank + 1) * 3].unsqueeze(0).clone().requires_grad_()
        local_targets = targets[rank * 3 : (rank + 1) * 3].unsqueeze(0)
        original_ce = F.cross_entropy
        failed = []

        def fail_once(*args, **kwargs):
            if rank == 1 and kwargs.get("reduction") == "none" and not failed:
                failed.append(True)
                raise RuntimeError("observer failed")
            return original_ce(*args, **kwargs)

        F.cross_entropy = fail_once
        loss_module.get_parallel_state = lambda: state
        computer._original_loss_fn = partial(
            ForCausalLMLoss, op=VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
        )
        with computer.capture():
            loss, _, _ = computer._wrapped_loss_fn(
                labels=local_targets,
                shift_labels=local_targets,
                hidden_states=hidden,
                weights=weight,
                loss_reduction_group=torch.distributed.group.WORLD,
            )
        loss.backward()
        assert [item["token_count"].item() for item in computer._result] == [2, 3]
        torch.testing.assert_close(loss.detach(), ref_loss.detach(), rtol=1e-6, atol=1e-6)
        # SP backward scales each local contribution by world_size because
        # data-parallel parameter gradients are subsequently averaged.
        torch.testing.assert_close(hidden.grad.squeeze(0) / world_size, ref_hidden.grad[rank * 3 : (rank + 1) * 3])
        torch.distributed.all_reduce(weight.grad)
        torch.testing.assert_close(weight.grad / world_size, ref_weight.grad, rtol=1e-6, atol=1e-6)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_gloo_real_sp_reduction_channel_fallback_preserves_global_gradient():
    spawn_with_rendezvous(_gloo_real_sp_chunk_fallback_worker, args=(2,), nprocs=2)
