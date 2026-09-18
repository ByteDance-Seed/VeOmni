import gc
import types
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, noop_context_fn

from veomni.arguments import GradientCheckpointingConfig, MixedPrecisionConfig
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.utils import recompute_utils


class _CheckpointingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


@pytest.mark.parametrize("early_stop", [True, False])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_build_parallelize_model_forwards_checkpoint_early_stop(monkeypatch, early_stop, use_reentrant):
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda model, **kwargs: model)
    model = _CheckpointingModel()

    result = build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        early_stop=early_stop,
        enable_reentrant=use_reentrant,
    )

    assert result is model
    expected = {
        "use_reentrant": use_reentrant,
        "context_fn": noop_context_fn,
    }
    if not use_reentrant:
        expected["early_stop"] = early_stop
    assert model.gradient_checkpointing_kwargs == expected


def test_gradient_checkpointing_config_enables_early_stop_by_default():
    assert GradientCheckpointingConfig().early_stop is True


# ---------------------------------------------------------------------------
# Layer selection and SAC, driven by the framework instead of the model
# ---------------------------------------------------------------------------

BLOCK_TOTAL = 20


class _Recorder:
    """Stands in for ``torch.utils.checkpoint.checkpoint``, recording its kwargs."""

    #: Consumed by checkpoint itself, never forwarded to the wrapped callable.
    _CONSUMED_KWARGS = ("use_reentrant", "context_fn", "early_stop", "preserve_rng_state", "determinism_check")

    def __init__(self):
        self.calls = []

    def __call__(self, func, *args, **kwargs):
        self.calls.append(kwargs)
        forwarded = {key: value for key, value in kwargs.items() if key not in self._CONSUMED_KWARGS}
        return func(*args, **forwarded)


class _ToyLayer(nn.Module):
    """A HF ``GradientCheckpointingLayer``-style block: it owns the entry point."""

    gradient_checkpointing = False

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1


class _ToyModel(nn.Module):
    _no_split_modules = ["_ToyLayer"]

    def __init__(self, depth=6):
        super().__init__()
        self.layers = nn.ModuleList(_ToyLayer() for _ in range(depth))


class _FoldedModel(nn.Module):
    """A hand-written block loop: the container owns the entry point (flux style)."""

    _no_split_modules = ["_ToyLayer"]

    def __init__(self, depth=3, single_depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(_ToyLayer() for _ in range(depth))
        self.single_blocks = nn.ModuleList(_ToyLayer() for _ in range(single_depth))
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def forward(self, x):
        for block in self.blocks:
            x = self._gradient_checkpointing_func(block, x)
        for block in self.single_blocks:
            x = self._gradient_checkpointing_func(block.__call__, x)
        return x


def _expected_decisions(recompute_n, selective_n, total=BLOCK_TOTAL):
    """Oracle: recompute range from the end, SAC from the front of that range."""
    recompute_n = -1 if recompute_n < 0 or recompute_n > total else recompute_n
    recompute_start = 0 if recompute_n < 0 else total - recompute_n
    decisions = []
    for index in range(total):
        if index < recompute_start:
            decisions.append(recompute_utils.Decision.DIRECT)
        elif 0 < selective_n and index - recompute_start < selective_n:
            decisions.append(recompute_utils.Decision.SAC)
        else:
            decisions.append(recompute_utils.Decision.FULL)
    return decisions


@pytest.mark.parametrize("recompute_n", [-1, 0, 1, 5, 10, 19, 20, 25])
@pytest.mark.parametrize("selective_n", [0, 1, 5, 19, 20, 25])
def test_layer_decisions_match_previous_semantics(recompute_n, selective_n):
    policy = recompute_utils.RecomputePolicy(
        recompute_last_n_layers=recompute_n, selective_n_layers=selective_n, context_fn=noop_context_fn
    )
    decisions = [recompute_utils.plan_block(policy, index, BLOCK_TOTAL).decision for index in range(BLOCK_TOTAL)]
    assert decisions == _expected_decisions(recompute_n, selective_n)


@pytest.mark.parametrize("recompute_n", [-1, 0, 5, 20])
@pytest.mark.parametrize("selective_n", [0, 5, 20])
def test_without_context_fn_nothing_is_sac(recompute_n, selective_n):
    """SAC unavailable (disabled, reentrant, or offload) falls back to full recomputation."""
    policy = recompute_utils.RecomputePolicy(recompute_last_n_layers=recompute_n, selective_n_layers=selective_n)
    decisions = [recompute_utils.plan_block(policy, index, BLOCK_TOTAL).decision for index in range(BLOCK_TOTAL)]
    assert recompute_utils.Decision.SAC not in decisions
    assert decisions == _expected_decisions(recompute_n, 0)


def test_checkpoint_kwargs_shapes():
    sac = recompute_utils.RecomputePolicy(
        recompute_last_n_layers=-1, selective_n_layers=10, context_fn=noop_context_fn
    )
    assert recompute_utils.plan_block(sac, 0, BLOCK_TOTAL).checkpoint_kwargs == {
        "use_reentrant": False,
        "context_fn": noop_context_fn,
        "early_stop": True,
    }
    assert recompute_utils.plan_block(sac, 15, BLOCK_TOTAL).checkpoint_kwargs == {
        "use_reentrant": False,
        "early_stop": True,
    }

    reentrant = recompute_utils.RecomputePolicy(
        recompute_last_n_layers=-1, selective_n_layers=10, context_fn=noop_context_fn, use_reentrant=True
    )
    # torch rejects context_fn and early_stop on the reentrant path.
    assert recompute_utils.plan_block(reentrant, 0, BLOCK_TOTAL).checkpoint_kwargs == {"use_reentrant": True}

    direct = recompute_utils.RecomputePolicy(recompute_last_n_layers=5)
    assert recompute_utils.plan_block(direct, 0, BLOCK_TOTAL).checkpoint_kwargs == {}


def test_out_of_range_index_raises_even_with_optimizations():
    with pytest.raises(ValueError):
        recompute_utils.plan_block(recompute_utils.RecomputePolicy(), BLOCK_TOTAL, BLOCK_TOTAL)


def test_layer_entry_point_gets_per_layer_decisions(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _ToyModel(depth=6)
    for layer in model.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = recorder  # what HF sets

    policy = recompute_utils.RecomputePolicy(
        recompute_last_n_layers=-1, selective_n_layers=2, context_fn=noop_context_fn
    )
    report = recompute_utils.apply_recompute_policy(model, policy)
    assert report.bound and report.bound_blocks == 6 and report.patched_blocks == 6

    for layer in model.layers:  # what GradientCheckpointingLayer.__call__ does
        layer._gradient_checkpointing_func(partial(layer, x=torch.zeros(1)))

    assert len(recorder.calls) == 6
    assert [call.get("context_fn") is noop_context_fn for call in recorder.calls] == [True] * 2 + [False] * 4


def test_blocks_outside_the_filter_skip_checkpointing(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _ToyModel(depth=4)
    for layer in model.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = recorder

    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1))
    for layer in model.layers:
        layer._gradient_checkpointing_func(partial(layer, x=torch.zeros(1)))

    assert len(recorder.calls) == 1  # only the last block is checkpointed
    assert [layer.calls for layer in model.layers] == [1, 1, 1, 1]  # every block still runs


def test_inactive_policy_touches_nothing():
    model = _ToyModel(depth=4)
    sentinel = object()
    for layer in model.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = sentinel

    report = recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy())

    assert report.bound is False
    assert all(layer._gradient_checkpointing_func is sentinel for layer in model.layers)


def test_container_entry_point_resolves_the_block(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=3, single_depth=4)
    model._gradient_checkpointing_func = recorder

    report = recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=2, selective_n_layers=1, context_fn=noop_context_fn),
    )
    assert report.stack.total == 7  # blocks + single_blocks run as one sequence
    assert report.stack.describe() == "blocks+single_blocks (7 blocks, folded blocks=3, single_blocks=4)"

    model(torch.zeros(1))

    assert len(recorder.calls) == 2  # the last two blocks, first five are direct
    assert recorder.calls[0]["context_fn"] is noop_context_fn  # SAC is the first of the range


def test_unbound_block_falls_back_to_full_recomputation(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=2, single_depth=2)
    model._gradient_checkpointing_func = recorder
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=-1, selective_n_layers=4, context_fn=noop_context_fn),
    )

    model._gradient_checkpointing_func(_ToyLayer(), torch.zeros(1))

    assert len(recorder.calls) == 1
    assert "context_fn" not in recorder.calls[0]
    assert recorder.calls[0]["use_reentrant"] is False


def test_checkpoint_forward_uses_the_bound_plan(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=3, single_depth=0)
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=-1, selective_n_layers=1, context_fn=noop_context_fn),
    )

    for block in model.blocks:
        recompute_utils.checkpoint_forward(block, True, False, torch.zeros(1))

    assert len(recorder.calls) == 3
    assert recorder.calls[0]["context_fn"] is noop_context_fn
    assert [block.calls for block in model.blocks] == [1, 1, 1]


def test_checkpoint_forward_without_a_binding_still_checkpoints(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)

    recompute_utils.checkpoint_forward(_ToyLayer(), True, False, torch.zeros(1))

    assert len(recorder.calls) == 1
    assert "context_fn" not in recorder.calls[0]


def test_stacks_under_another_parent_are_reported_not_filtered():
    model = _ToyModel(depth=3)
    model.text = nn.Module()
    model.text.layers = nn.ModuleList(_ToyLayer() for _ in range(2))
    model.vision = nn.Module()
    model.vision.blocks = nn.ModuleList(_ToyLayer() for _ in range(2))

    report = recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=2))

    assert report.stack.describe() == "layers (3 blocks)"
    assert [stack.describe() for stack in report.excluded] == ["text.layers (2 blocks)", "vision.blocks (2 blocks)"]
    assert report.bound_blocks == 3  # only the main stack is filtered


def test_declared_block_class_outranks_block_count():
    """A vision tower with more blocks than the decoder must not become the main stack."""

    class _VisionLayer(_ToyLayer):
        pass

    class _TwoTowerModel(nn.Module):
        _no_split_modules = ["_ToyLayer", "_VisionLayer"]

        def __init__(self, text_depth=2, vision_depth=5):
            super().__init__()
            self.text = nn.Module()
            self.text.layers = nn.ModuleList(_ToyLayer() for _ in range(text_depth))
            self.vision = nn.Module()
            self.vision.blocks = nn.ModuleList(_VisionLayer() for _ in range(vision_depth))

    model = _TwoTowerModel()

    report = recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1))

    assert report.stack.describe() == "text.layers (2 blocks)"
    assert [stack.describe() for stack in report.excluded] == ["vision.blocks (5 blocks)"]
    assert report.bound_blocks == 2


def test_model_without_blocks_is_left_alone():
    model = _CheckpointingModel()
    model.gradient_checkpointing_kwargs = {"use_reentrant": False}

    report = recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=-1, selective_n_layers=4, context_fn=noop_context_fn),
    )

    assert report.bound is False
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_build_policy_drops_sac_when_it_cannot_apply():
    enabled = dict(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=5)
    assert recompute_utils.build_policy(**enabled).context_fn is not None
    assert recompute_utils.build_policy(**{**enabled, "offload_active": True}).context_fn is None
    assert recompute_utils.build_policy(**{**enabled, "enable_reentrant": True}).context_fn is None
    assert recompute_utils.build_policy(**{**enabled, "enabled": False}).context_fn is None


def test_build_policy_is_inactive_by_default():
    policy = recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True)
    assert policy.active is False


# ---------------------------------------------------------------------------
# Bindings: lifetime, idempotence, and what installing must leave alone
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_warnings():
    """Warnings are once-per-process; each test starts with a clean slate."""
    recompute_utils._warned.clear()
    yield
    recompute_utils._warned.clear()


def test_bindings_do_not_keep_models_alive():
    recompute_utils._block_bindings.clear()
    assert len(recompute_utils._block_bindings) == 0

    model = _ToyModel(depth=4)
    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=2))
    assert len(recompute_utils._block_bindings) == 4

    del model
    gc.collect()

    assert len(recompute_utils._block_bindings) == 0


def test_install_is_idempotent_and_reinstallable():
    model = _ToyModel(depth=3)

    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1))
    first = [recompute_utils._block_bindings[layer].plan for layer in model.layers]
    assert [plan.decision for plan in first] == [recompute_utils.Decision.DIRECT] * 2 + [recompute_utils.Decision.FULL]

    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1))
    assert [recompute_utils._block_bindings[layer].plan for layer in model.layers] == first

    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=3),
    )
    assert [recompute_utils._block_bindings[layer].plan.decision for layer in model.layers] == [
        recompute_utils.Decision.SAC
    ] * 3


def test_install_leaves_gradient_checkpointing_kwargs_alone(monkeypatch):
    model = _ToyModel(depth=2)
    model.gradient_checkpointing_kwargs = {"use_reentrant": False, "context_fn": noop_context_fn, "early_stop": True}
    expected = dict(model.gradient_checkpointing_kwargs)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))

    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=2),
    )

    assert model.gradient_checkpointing_kwargs == expected
    # No layer owns an entry point here, so nothing may be written onto them.
    assert all("_gradient_checkpointing_func" not in vars(layer) for layer in model.layers)


# ---------------------------------------------------------------------------
# Resolving the block a container-level checkpoint call wraps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapped",
    [
        pytest.param(lambda block: block, id="module"),
        pytest.param(lambda block: block.__call__, id="bound-call"),
        pytest.param(lambda block: partial(block), id="partial-module"),
        pytest.param(lambda block: partial(block.__call__), id="partial-bound-call"),
        pytest.param(lambda block: partial(partial(block.__call__)), id="nested-partial"),
    ],
)
def test_resolve_block_finds_the_wrapped_module(wrapped):
    block = _ToyLayer()
    assert recompute_utils._resolve_block(wrapped(block), (torch.zeros(1),)) is block


def test_resolve_block_gives_up_on_plain_callables():
    assert recompute_utils._resolve_block(lambda *_: None, (torch.zeros(1),)) is None
    assert recompute_utils._resolve_block(print, ()) is None


def test_container_call_on_an_unknown_block_still_checkpoints(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=2, single_depth=0)
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=2),
    )

    model._gradient_checkpointing_func(lambda x: x + 1, torch.zeros(1))

    assert len(recorder.calls) == 1
    assert recorder.calls[0] == {"use_reentrant": False, "early_stop": True}


# ---------------------------------------------------------------------------
# Execution priority: direct call > offload > SAC > full recomputation
# ---------------------------------------------------------------------------


def _bound_model(depth=2, **policy_kwargs):
    model = _FoldedModel(depth=depth, single_depth=0)
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, **policy_kwargs),
    )
    return model


def test_offload_takes_priority_over_sac(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _bound_model(selective_n_layers=2)

    recompute_utils.checkpoint_forward(model.blocks[0], True, True, torch.zeros(1))

    assert len(recorder.calls) == 1
    assert recorder.calls[0] == {"use_reentrant": False}  # save_on_cpu never carries a context_fn


def test_direct_beats_offload(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=3, single_depth=0)
    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=2))

    recompute_utils.checkpoint_forward(model.blocks[0], True, True, torch.zeros(1))

    assert recorder.calls == []
    assert model.blocks[0].calls == 1


def test_checkpointing_disabled_calls_the_block(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _bound_model(selective_n_layers=2)

    recompute_utils.checkpoint_forward(model.blocks[0], False, False, torch.zeros(1))

    assert recorder.calls == []
    assert model.blocks[0].calls == 1


def test_direct_block_keeps_the_module_hooks():
    model = _FoldedModel(depth=2, single_depth=0)
    recompute_utils.apply_recompute_policy(model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1))
    seen = []
    model.blocks[0].register_forward_hook(lambda *_: seen.append("block0"))

    recompute_utils.checkpoint_forward(model.blocks[0], True, False, torch.zeros(1))

    assert seen == ["block0"]


# ---------------------------------------------------------------------------
# Decisions: boundaries and log rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recompute_n", "selective_n", "active"),
    [(-1, 0, False), (-1, 3, True), (5, 0, True), (0, 0, True)],
)
def test_policy_active_flag(recompute_n, selective_n, active):
    policy = recompute_utils.RecomputePolicy(recompute_last_n_layers=recompute_n, selective_n_layers=selective_n)
    assert policy.active is active


def test_plan_block_saturates_when_selective_exceeds_the_range():
    policy = recompute_utils.RecomputePolicy(
        recompute_last_n_layers=3, selective_n_layers=99, context_fn=noop_context_fn
    )

    decisions = [recompute_utils.plan_block(policy, index, 10).decision for index in range(10)]

    assert decisions == [recompute_utils.Decision.DIRECT] * 7 + [recompute_utils.Decision.SAC] * 3


def test_plan_block_clamps_a_count_beyond_the_stack():
    policy = recompute_utils.RecomputePolicy(recompute_last_n_layers=99)

    assert [recompute_utils.plan_block(policy, index, 4).decision for index in range(4)] == [
        recompute_utils.Decision.FULL
    ] * 4


def test_plan_block_handles_a_single_block_stack():
    policy = recompute_utils.RecomputePolicy(recompute_last_n_layers=1)

    assert recompute_utils.plan_block(policy, 0, 1).decision is recompute_utils.Decision.FULL


def test_describe_decisions_groups_consecutive_blocks():
    decisions = [
        recompute_utils.Decision.FULL,
        recompute_utils.Decision.FULL,
        recompute_utils.Decision.SAC,
        recompute_utils.Decision.DIRECT,
    ]

    assert recompute_utils._describe_decisions(decisions) == (
        "blocks 0-1 full recompute, blocks 2-2 SAC, blocks 3-3 no recompute"
    )


def test_warn_once_emits_each_concern_once(monkeypatch):
    messages = []
    monkeypatch.setattr(recompute_utils.logger, "warning", lambda *args, **kwargs: messages.append(args[2]))

    recompute_utils._warn_once("concern", "first")
    recompute_utils._warn_once("concern", "second")
    recompute_utils._warn_once("other", "third")

    assert messages == ["first", "third"]


def test_unbound_block_warns_only_while_a_policy_is_in_force(monkeypatch):
    recompute_utils._block_bindings.clear()
    messages = []
    monkeypatch.setattr(recompute_utils.logger, "warning", lambda *args, **kwargs: messages.append(args[2]))
    unbound = _ToyLayer()

    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))
    assert messages == []  # nothing is bound anywhere: the documented default, no noise

    bound = _bound_model(selective_n_layers=1)  # keep it alive: the bindings are weak
    assert recompute_utils._block_bindings[bound.blocks[0]].plan.decision is recompute_utils.Decision.SAC
    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))
    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))

    assert len(messages) == 1


# ---------------------------------------------------------------------------
# Layer discovery: what counts as a stack
# ---------------------------------------------------------------------------


def test_a_single_block_container_is_not_a_stack():
    report = recompute_utils.apply_recompute_policy(
        _ToyModel(depth=1), recompute_utils.RecomputePolicy(recompute_last_n_layers=1)
    )

    assert report.bound is False


def test_sibling_containers_fold_into_one_stack():
    class _TwoSiblingContainers(nn.Module):
        _no_split_modules = ["_ToyLayer"]

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(_ToyLayer() for _ in range(3))
            self.extra = nn.ModuleList(_ToyLayer() for _ in range(2))

    report = recompute_utils.apply_recompute_policy(
        _TwoSiblingContainers(), recompute_utils.RecomputePolicy(recompute_last_n_layers=2)
    )

    assert report.stack.describe() == "blocks+extra (5 blocks, folded blocks=3, extra=2)"
    assert report.excluded == ()
    assert report.bound_blocks == 5


def test_block_classes_fall_back_to_the_checkpointing_marker():
    class _Undeclared(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(_ToyLayer() for _ in range(3))

    report = recompute_utils.apply_recompute_policy(
        _Undeclared(), recompute_utils.RecomputePolicy(recompute_last_n_layers=1)
    )

    assert report.stack.describe() == "layers (3 blocks)"


def test_the_deeper_sibling_container_wins():
    class _UnevenSiblings(nn.Module):
        _no_split_modules = ["_ToyLayer"]

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(_ToyLayer() for _ in range(3))
            self.extra = nn.ModuleList(_ToyLayer() for _ in range(2))

    report = recompute_utils.apply_recompute_policy(
        _UnevenSiblings(), recompute_utils.RecomputePolicy(recompute_last_n_layers=1)
    )

    assert report.stack.total == 5


# ---------------------------------------------------------------------------
# SAC operator selection
# ---------------------------------------------------------------------------


class _FakeOp:
    """Stands in for an OpOverload; only its schema name is read."""

    def __init__(self, name):
        self._schema = types.SimpleNamespace(name=name)


def test_token_matching_skips_the_decomposed_efficient_attention_helpers():
    matches = lambda name: recompute_utils._token_matches_qualname(  # noqa: E731
        name, recompute_utils.DEFAULT_SELECTIVE_TOKENS
    )

    assert matches("npu::npu_fusion_attention")
    assert matches("aten::_scaled_dot_product_flash_attention")
    assert not matches("aten::_efficient_attention_forward")
    assert not matches("aten::addmm")


def test_exact_policy_saves_only_the_listed_ops():
    listed, unlisted = object(), object()
    policy = recompute_utils._make_selective_policy([listed], prefix_mode=False)

    assert policy(None, listed) is CheckpointPolicy.MUST_SAVE
    assert policy(None, unlisted) is CheckpointPolicy.PREFER_RECOMPUTE


def test_prefix_policy_matches_by_operator_name():
    policy = recompute_utils._make_selective_policy((), prefix_mode=True)

    assert policy(None, _FakeOp("npu::npu_fusion_attention")) is CheckpointPolicy.MUST_SAVE
    assert policy(None, _FakeOp("aten::mm")) is CheckpointPolicy.PREFER_RECOMPUTE


def test_resolve_exact_ops_reports_unresolvable_extras_without_raising():
    ops, failed = recompute_utils.resolve_exact_ops(["not.a.real.op"])

    assert failed == ["not.a.real.op"]
    assert len(ops) == len(set(ops))  # de-duplicated


def test_build_context_fn_falls_back_to_the_name_policy(monkeypatch):
    monkeypatch.setattr(recompute_utils, "resolve_exact_ops", lambda extras: ([], list(extras or ())))
    messages = []
    monkeypatch.setattr(recompute_utils.logger, "warning", lambda *args, **kwargs: messages.append(args[2]))

    context_fn = recompute_utils._build_context_fn(["bogus.op"])

    assert context_fn is not None
    assert len(messages) == 1
    assert "bogus.op" in messages[0]


# ---------------------------------------------------------------------------
# SAC end to end: the attention output is saved, everything else recomputed
# ---------------------------------------------------------------------------


class _AttentionBlock(nn.Module):
    """A block whose expensive part is a real, SAC-visible attention operator."""

    gradient_checkpointing = False

    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        shape = (*q.shape[:-1], self.heads, q.shape[-1] // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # F.scaled_dot_product_attention dispatches to aten::_scaled_dot_product_attention,
        # one of the operators SAC resolves to MUST_SAVE.
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return x + self.proj(attn.transpose(1, 2).reshape(*x.shape))


class _AttentionModel(nn.Module):
    _no_split_modules = ["_AttentionBlock"]

    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList(_AttentionBlock() for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = recompute_utils.checkpoint_forward(layer, True, False, x)
        return x


class _DispatchRecorder(TorchDispatchMode):
    """Records the aten operators dispatched while it is active."""

    def __init__(self):
        super().__init__()
        self.forward_ops = []
        self.backward_ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        target = self.backward_ops if "backward" in name else self.forward_ops
        target.append(name)
        return func(*args, **(kwargs or {}))


def _is_attention(op_name):
    return "attention" in op_name or "scaled_dot_product" in op_name


def _attention_dispatch_count(policy):
    """Run one step under ``policy`` and observe which operators run twice.

    The observable has to be the aten dispatch, not Python calls: during the
    backward replay the block's ``forward`` runs again either way — so a
    ``scaled_dot_product_attention`` wrapper keeps counting. What a MUST_SAVE
    policy changes is that the cached operator is *not executed again*, while
    PREFER_RECOMPUTE really re-executes it.
    """
    torch.manual_seed(0)
    model = _AttentionModel()
    recompute_utils.apply_recompute_policy(model, policy)
    recorder = _DispatchRecorder()
    x = torch.randn(2, 6, 8, requires_grad=True)

    with recorder:
        out = model(x)
        forward_ops = list(recorder.forward_ops)
        out.square().mean().backward()

    replayed = recorder.forward_ops[len(forward_ops) :]

    return types.SimpleNamespace(
        attention_forward=sum(_is_attention(name) for name in forward_ops),
        attention_replayed=sum(_is_attention(name) for name in replayed),
        ops_replayed=len(replayed),
    )


def _gradients(policy):
    torch.manual_seed(0)
    model = _AttentionModel()
    recompute_utils.apply_recompute_policy(model, policy)
    x = torch.randn(2, 6, 8, requires_grad=True)

    out = model(x)
    out.square().mean().backward()

    return out.detach(), x.grad.detach(), [param.grad.detach().clone() for param in model.parameters()]


def test_sac_saves_attention_outputs_and_recomputes_the_rest():
    direct = _attention_dispatch_count(recompute_utils.RecomputePolicy(recompute_last_n_layers=0))
    full = _attention_dispatch_count(recompute_utils.RecomputePolicy(recompute_last_n_layers=-1))
    sac = _attention_dispatch_count(
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=3)
    )

    assert direct.attention_forward == 3  # three blocks, forward only
    assert direct.attention_replayed == 0  # nothing is recomputed

    assert full.attention_replayed == 3  # every block replays its attention
    assert sac.attention_forward == direct.attention_forward
    assert sac.attention_replayed == 0  # the saved attention output is reused as is
    assert sac.ops_replayed > 0  # but the rest of each block really is recomputed


def test_sac_results_match_the_direct_reference():
    reference = _gradients(recompute_utils.RecomputePolicy(recompute_last_n_layers=0))

    for policy in (
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=3),
        recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True),
    ):
        out, input_grad, param_grads = _gradients(policy)

        torch.testing.assert_close(out, reference[0])
        torch.testing.assert_close(input_grad, reference[1])
        for actual, expected in zip(param_grads, reference[2]):
            torch.testing.assert_close(actual, expected)
