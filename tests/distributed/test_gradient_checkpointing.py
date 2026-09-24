import gc
import types
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, noop_context_fn

from veomni.arguments import GradientCheckpointingConfig, MixedPrecisionConfig
from veomni.distributed.checkpoint import CheckpointFunction
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


def test_gradient_checkpointing_config_takes_a_bare_operator_name():
    assert GradientCheckpointingConfig(selective_ops="aten.foo.default").selective_ops == ["aten.foo.default"]
    assert GradientCheckpointingConfig(selective_ops=None).selective_ops == []


def test_gradient_checkpointing_config_rejects_unusable_values():
    with pytest.raises(ValueError, match="selective_ops must be a list"):
        GradientCheckpointingConfig(selective_ops=5)
    with pytest.raises(ValueError, match="recompute_last_n_layers must be an integer"):
        GradientCheckpointingConfig(recompute_last_n_layers="10")


# ---------------------------------------------------------------------------
# Layer selection and SAC, driven by the framework instead of the model
# ---------------------------------------------------------------------------

BLOCK_TOTAL = 20


class _Recorder:
    """Stands in for ``torch.utils.checkpoint.checkpoint``, recording callable and kwargs."""

    #: Consumed by checkpoint itself, never forwarded to the wrapped callable.
    _CONSUMED_KWARGS = ("use_reentrant", "context_fn", "early_stop", "preserve_rng_state", "determinism_check")

    def __init__(self):
        self.calls = []
        self.funcs = []

    def __call__(self, func, *args, **kwargs):
        self.calls.append(kwargs)
        self.funcs.append(func)
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


class _SelfContainedModel(nn.Module):
    """MiniMax-H3 style: the container installs its own default checkpoint function."""

    _no_split_modules = ["_ToyLayer"]

    def __init__(self, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList(_ToyLayer() for _ in range(depth))
        self._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    def forward(self, x):
        for block in self.blocks:
            x = self._gradient_checkpointing_func(block, x)
        return x


class _KeywordToyLayer(nn.Module):
    """A block whose arguments arrive by keyword, the way MiniMax-H3 passes its own."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x, *, shift):
        return x * self.weight + shift


class _KeywordModel(nn.Module):
    """A hand-written loop that passes its block arguments by keyword, as H3 does."""

    _no_split_modules = ["_KeywordToyLayer"]

    def __init__(self, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList(_KeywordToyLayer() for _ in range(depth))
        self._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    def forward(self, x):
        for block in self.blocks:
            x = self._gradient_checkpointing_func(block, x, shift=1.0)
        return x


class _WeightedToyLayer(_ToyLayer):
    """A toy block that owns a parameter, so a stack can be frozen or trainable."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class _TextBlock(_WeightedToyLayer):
    pass


class _VisionBlock(_WeightedToyLayer):
    pass


class _TwoStackModel(nn.Module):
    """Two block stacks under different parents, the way a multimodal model has them."""

    def __init__(self, text_depth=6, vision_depth=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_TextBlock() for _ in range(text_depth))
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList(_VisionBlock() for _ in range(vision_depth))


def test_target_class_names_sorts_the_models_own_set():
    model = _TwoStackModel()
    model._no_split_modules = {"_VisionBlock", "_TextBlock"}

    assert recompute_utils._target_class_names(model, []) == ("_TextBlock", "_VisionBlock")
    assert recompute_utils._target_class_names(model, ["_VisionBlock"]) == ("_VisionBlock", "_TextBlock")


def test_main_stack_is_the_same_whatever_order_the_classes_are_declared_in():
    model = _TwoStackModel()
    model._no_split_modules = {"_TextBlock", "_VisionBlock"}
    first, _ = recompute_utils.discover_block_stack(model)
    model._no_split_modules = {"_VisionBlock", "_TextBlock"}
    second, _ = recompute_utils.discover_block_stack(model)

    assert first.fqn == second.fqn == "model.layers"


def test_main_stack_prefers_the_trainable_stack():
    model = _TwoStackModel(text_depth=2, vision_depth=6)
    model.visual.requires_grad_(False)
    model._no_split_modules = {"_VisionBlock", "_TextBlock"}

    stack, excluded = recompute_utils.discover_block_stack(model)

    assert stack.fqn == "model.layers"
    assert [left_out.fqn for left_out in excluded] == ["visual.blocks"]


def test_configured_module_names_the_main_stack():
    model = _TwoStackModel(text_depth=2, vision_depth=6)
    model._no_split_modules = {"_TextBlock", "_VisionBlock"}

    stack, _ = recompute_utils.discover_block_stack(model, ["_VisionBlock"])

    assert stack.fqn == "visual.blocks"


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


def test_a_containers_own_checkpoint_function_checkpoints_without_a_policy(monkeypatch):
    """The default an H3-style container brings must work with no policy at all."""
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _SelfContainedModel(depth=2)

    out = model(torch.zeros(4))

    assert len(recorder.calls) == 2  # one per block, through the container's own function
    assert recorder.calls[0]["use_reentrant"] is False
    assert "context_fn" not in recorder.calls[0]
    assert torch.equal(out, torch.full((4,), 2.0))


def test_a_containers_own_checkpoint_function_is_replaced_by_the_policy(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _SelfContainedModel(depth=4)

    report = recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=2, selective_n_layers=1, context_fn=noop_context_fn),
    )

    assert report.patched_blocks == 0 and report.patched_containers == 1  # the container, not the blocks
    assert report.covered

    model(torch.zeros(1))

    assert len(recorder.calls) == 2  # the last two blocks, first two are direct
    assert recorder.calls[0]["context_fn"] is noop_context_fn  # SAC is the first of the range


def test_the_container_entry_point_forwards_keyword_arguments(monkeypatch):
    """A block can be called by keyword through the container the policy replaced."""
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)

    model = _KeywordModel()
    report = recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=1, selective_n_layers=1, context_fn=noop_context_fn),
    )
    assert report.patched_containers == 1 and report.patched_blocks == 0

    x = torch.zeros(2, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert torch.equal(out.detach(), torch.full((2,), 3.0))
    assert torch.equal(x.grad, torch.ones(2))
    assert len(recorder.calls) == 1 and recorder.calls[0]["context_fn"] is noop_context_fn  # only the last block


def test_the_container_entry_point_hands_torch_the_block_itself(monkeypatch):
    """Reentrant checkpointing reads ``run_function.__self__``, so the block must arrive unwrapped."""
    recorder = _Recorder()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)
    model = _FoldedModel(depth=2, single_depth=2)
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=2, selective_n_layers=1, context_fn=noop_context_fn),
    )

    model(torch.zeros(1))

    # The two trailing blocks are checkpointed, as module and as bound method.
    assert [getattr(func, "__self__", func) for func in recorder.funcs] == [
        model.single_blocks[0],
        model.single_blocks[1],
    ]


def test_reentrant_checkpointing_survives_veomnis_own_checkpoint_function(monkeypatch):
    """The CheckpointFunction enable_reentrant swaps in must accept what the policy hands torch."""
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", CheckpointFunction)
    model = _FoldedModel(depth=2, single_depth=2)
    recompute_utils.apply_recompute_policy(
        model, recompute_utils.RecomputePolicy(recompute_last_n_layers=2, use_reentrant=True)
    )

    x = torch.zeros(2, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert torch.equal(out.detach(), torch.full((2,), 4.0))  # four blocks, each +1
    assert torch.equal(x.grad, torch.ones(2))


def test_keyword_arguments_still_reach_the_block_through_real_checkpointing():
    """Nothing stands between the loop and torch, so keywords keep working without a closure."""
    model = _KeywordModel()
    recompute_utils.apply_recompute_policy(
        model,
        recompute_utils.RecomputePolicy(recompute_last_n_layers=1, selective_n_layers=1, context_fn=noop_context_fn),
    )

    x = torch.zeros(2, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert torch.equal(out.detach(), torch.full((2,), 3.0))  # three blocks, each ``x + shift``
    assert torch.equal(x.grad, torch.ones(2))


def test_reentrant_refuses_keyword_arguments_through_the_container():
    """Reentrant checkpointing saves positional tensors only, so say so instead of crashing in torch."""
    model = _KeywordModel(depth=2)
    recompute_utils.apply_recompute_policy(
        model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1, use_reentrant=True)
    )

    with pytest.raises(ValueError) as failure:
        model(torch.zeros(2))

    message = str(failure.value)
    assert "model.accelerator.gradient_checkpointing.enable_reentrant=True" in message
    assert "_KeywordToyLayer" in message and "shift" in message


def test_reentrant_refuses_keyword_arguments_through_checkpoint_forward():
    model = _KeywordModel(depth=2)
    recompute_utils.apply_recompute_policy(
        model, recompute_utils.RecomputePolicy(recompute_last_n_layers=1, use_reentrant=True)
    )

    with pytest.raises(ValueError) as failure:
        recompute_utils.checkpoint_forward(model.blocks[-1], True, False, torch.zeros(2), shift=1.0)

    assert "enable_reentrant=True" in str(failure.value)


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
    # No policy is in reach from this entry point, so the remaining options stay
    # at torch's defaults instead of echoing a config value.
    assert "early_stop" not in recorder.calls[0]


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


def test_block_count_names_the_main_stack_when_nothing_is_configured():
    """The class declaration order must not be read: HF hands it over as a set.

    Two declared classes under two parents: the deeper stack wins and the other
    one is reported. A model that needs the other one names it in
    ``basic_modules`` — see :func:`test_configured_module_names_the_main_stack`.
    """

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

    assert report.stack.describe() == "vision.blocks (5 blocks)"
    assert [stack.describe() for stack in report.excluded] == ["text.layers (2 blocks)"]
    assert report.bound_blocks == 5


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
    """Warnings are once-per-process (framework ``warning_once``); each test starts clean."""
    recompute_utils.logger.warning_once.cache_clear()
    yield
    recompute_utils.logger.warning_once.cache_clear()


@pytest.fixture
def captured_warnings(monkeypatch):
    """Messages the module logs as warnings; ``warning_once`` is the framework's, so it dedupes."""
    messages = []
    monkeypatch.setattr(recompute_utils.logger, "warning", lambda message, *args, **kwargs: messages.append(message))
    return messages


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


def test_unbound_block_warns_only_while_a_policy_is_in_force(captured_warnings):
    recompute_utils._block_bindings.clear()
    unbound = _ToyLayer()

    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))
    assert captured_warnings == []  # nothing is bound anywhere: the documented default, no noise

    bound = _bound_model(selective_n_layers=1)  # keep it alive: the bindings are weak
    assert recompute_utils._block_bindings[bound.blocks[0]].plan.decision is recompute_utils.Decision.SAC
    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))
    recompute_utils.checkpoint_forward(unbound, True, False, torch.zeros(1))

    assert len(captured_warnings) == 1


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


def _is_saved(op_name):
    """What the default policy decides for one operator name."""
    policy = recompute_utils._make_selective_policy(())
    return policy(None, _FakeOp(op_name)) is CheckpointPolicy.MUST_SAVE


#: One representative operator per attention family VeOmni can select. The
#: default SAC set must keep the attention output of every one of them.
ATTENTION_OPS = (
    # aten SDPA family, its composite entry point, and torch flex
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_attention_math",
    "aten::scaled_dot_product_attention",
    "aten::_native_multi_head_attention",
    "aten::_triton_scaled_dot_attention",
    "higher_order::flex_attention",
    # FlashAttention 2/3/4
    "flash_attn::_flash_attn_forward",
    "flash_attn::_flash_attn_varlen_forward",  # FA2 wheel: the packed path dispatches through this op
    "flash_attn_3::_flash_attn_forward",
    "flash_attn_4::_flash_attn_forward",
    # torch_npu
    "npu::npu_fusion_attention",
    "npu::npu_prompt_flash_attention",
    "npu::npu_incre_flash_attention",
    "npu::npu_sparse_flash_attention",
    "npu::npu_kv_quant_sparse_flash_attention",
    "npu::npu_fused_attention_score_fwd",
    "npu::npu_multi_head_attention",
    "npu::npu_block_sparse_attention",
    "npu::npu_fused_floyd_attention",
    "npu::npu_quant_fusion_attention",
    "npu::npu_nsa_select_attention",
    "npu::npu_nsa_compress_attention",
    "npu::npu_attn_softmax_",
    "npu::npu_attention_update",
    "npu::npu_advance_step_flashattn",
    # sparse MLA / indexer (cuDNN FE and torch_npu)
    "flash_mla::flash_mla_sparse_fwd",
    "DSA::indexer_forward_wrapper",
    "npu::npu_lightning_indexer",
    "npu::npu_quant_lightning_indexer",
    # third-party kernels
    "xformers::efficient_attention_forward_cutlass",
    "sageattention::sageattn",
)

#: Operators the policy must NOT save: decomposed implementations that
#: materialize the attention matrix, fused kernels whose output (qkv,
#: compressed KV) is far larger than an attention output, and names no
#: attention token matches at all.
NOT_SAVED_OPS = (
    "aten::_efficient_attention_forward",
    "npu::npu_fused_attention_layernorm_qkv_fwd",
    "npu::npu_fused_attention_qkv_grad",
    "npu::npu_mla_prolog_v3",
    "npu::npu_nsa_compress",  # no attention token in the name: never matched, not vetoed
    "aten::addmm",
    "aten::matmul",
    "aten::softmax",
    "npu::npu_rms_norm",
)


@pytest.mark.parametrize("op_name", ATTENTION_OPS)
def test_default_set_saves_every_attention_family(op_name):
    assert _is_saved(op_name), op_name


@pytest.mark.parametrize("op_name", NOT_SAVED_OPS)
def test_default_set_recomputes_non_attention_and_oversized_kernels(op_name):
    assert not _is_saved(op_name), op_name


def test_namespace_probe_skips_vetoed_operators(monkeypatch):
    """A vetoed op is not even resolved into the exact set."""
    namespace = types.SimpleNamespace(
        npu_fused_attention_score=types.SimpleNamespace(default="score"),
        npu_fused_attention_layernorm_qkv_fwd=types.SimpleNamespace(default="qkv"),
    )
    monkeypatch.setattr(recompute_utils, "_registered_namespaces", lambda: ["npu"])
    monkeypatch.setattr(torch.ops, "npu", namespace, raising=False)

    ops, failed = recompute_utils.resolve_exact_ops()

    assert "score" in ops
    assert "qkv" not in ops  # the vetoed fused kernel stays out of the exact set
    assert failed == []


def test_exact_policy_saves_only_the_listed_ops():
    listed, unlisted = _FakeOp("custom::listed"), _FakeOp("custom::unlisted")
    policy = recompute_utils._make_selective_policy([listed])

    assert policy(None, listed) is CheckpointPolicy.MUST_SAVE
    assert policy(None, unlisted) is CheckpointPolicy.PREFER_RECOMPUTE


def test_policy_matches_by_operator_name_alongside_the_exact_ops():
    """A resolved exact op must not switch the name fallback off (FlashAttention 3)."""
    policy = recompute_utils._make_selective_policy([_FakeOp("custom::listed")])

    assert policy(None, _FakeOp("flash_attn_3::_flash_attn_forward")) is CheckpointPolicy.MUST_SAVE
    assert policy(None, _FakeOp("npu::npu_fusion_attention")) is CheckpointPolicy.MUST_SAVE
    assert policy(None, _FakeOp("aten::mm")) is CheckpointPolicy.PREFER_RECOMPUTE
    assert policy(None, _FakeOp("aten::_efficient_attention_forward")) is CheckpointPolicy.PREFER_RECOMPUTE


def test_selective_namespaces_add_the_fixed_candidates_and_matching_registered_ones():
    selected = recompute_utils._selective_namespaces(["aten", "flash_attn_3", "flash_attn_4", "_private_attn"])

    assert "flash_attn_3" in selected  # discovered, not only listed in the candidates
    assert "flash_attn_4" in selected  # future namespace, discovered by token
    assert "npu" in selected  # fixed candidate, absent from the registered names
    assert "flash_attn" in selected  # fixed candidate
    assert "aten" not in selected  # registered but not attention
    assert "_private_attn" not in selected  # private namespaces are not probed


def test_registered_namespaces_reads_the_dispatcher():
    namespaces = recompute_utils._registered_namespaces()

    assert "aten" in namespaces
    assert namespaces == sorted(set(namespaces))


def test_resolve_exact_ops_discovers_a_late_registered_namespace(monkeypatch):
    """flash_attn_3 registers its ops without ever being touched through torch.ops."""
    namespace = types.SimpleNamespace(_flash_attn_forward=types.SimpleNamespace(default="resolved"))
    monkeypatch.setattr(recompute_utils, "_registered_namespaces", lambda: ["flash_attn_3"])
    monkeypatch.setattr(torch.ops, "flash_attn_3", namespace, raising=False)

    ops, failed = recompute_utils.resolve_exact_ops()

    assert "resolved" in ops
    assert failed == []


def test_resolve_exact_ops_reports_unresolvable_extras_without_raising():
    ops, failed = recompute_utils.resolve_exact_ops(["not.a.real.op"])

    assert failed == ["not.a.real.op"]
    assert len(ops) == len(set(ops))  # de-duplicated


def test_build_context_fn_warns_about_unresolvable_extras(monkeypatch, captured_warnings):
    monkeypatch.setattr(recompute_utils, "resolve_exact_ops", lambda extras: ([], list(extras or ())))

    context_fn = recompute_utils._build_context_fn(["bogus.op"])

    assert len(captured_warnings) == 1
    assert "bogus.op" in captured_warnings[0]
    # The policy still saves name-matching operators, so one bad extra is not fatal.
    policy = context_fn.args[0]
    assert policy(None, _FakeOp("flash_attn_3::_flash_attn_forward")) is CheckpointPolicy.MUST_SAVE


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


# ---------------------------------------------------------------------------
# SAC over a backend kernel registered as a torch.library custom op (FlashAttention 3)
# ---------------------------------------------------------------------------

_FA3_CALLS: list[str] = []

#: Test-owned namespace standing in for FA3's ``flash_attn_3::_flash_attn_forward``.
#: Registering the real name would collide in any process that imports the real
#: library afterwards (the ``gpu`` extra installs ``flash-attn-3``), and an operator
#: registration cannot be undone. The stand-in is discovered exactly like the real
#: one: the namespace matches the ``flash_attn`` token, so it is probed, and the
#: operator name matches the same token.
_FA3_NS = "flash_attn_3_test"
_FA3_OP_NAME = "_flash_attn_forward"


def _register_fake_flash_attn_3():
    """Register ``flash_attn_3_test::_flash_attn_forward`` the way FA3 registers its own.

    Upstream FA3 wraps its kernel in ``torch.library.custom_op``, which is what
    makes it visible to the SAC dispatch mode at all. The body is a stand-in —
    only the call count is observed — so this runs on any device.
    """

    @torch.library.custom_op(f"{_FA3_NS}::{_FA3_OP_NAME}", mutates_args=())
    def _flash_attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _FA3_CALLS.append("fwd")
        return q + k + v

    @_flash_attn_forward.register_fake
    def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(q)

    torch.library.register_autograd(
        f"{_FA3_NS}::{_FA3_OP_NAME}",
        lambda ctx, grad_out: (grad_out, grad_out, grad_out),
    )
    return getattr(getattr(torch.ops, _FA3_NS), _FA3_OP_NAME).default


try:
    _FA3_OP = _register_fake_flash_attn_3()
except (RuntimeError, ValueError):  # pragma: no cover - the namespace is test-owned
    _FA3_OP = None


class _FlashAttn3Block(nn.Module):
    """Calls the custom op the way a model backend does: through torch.ops.

    The kernel output feeds a later layer inside the block, so a replay has to
    reproduce it — otherwise ``early_stop`` could end the replay before reaching
    the operator and the count would prove nothing.
    """

    gradient_checkpointing = False

    def __init__(self, dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.norm(x)
        attn = getattr(getattr(torch.ops, _FA3_NS), _FA3_OP_NAME)(q, q, q)
        return x + self.proj(attn)


class _FlashAttn3Model(nn.Module):
    _no_split_modules = ["_FlashAttn3Block"]

    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList(_FlashAttn3Block() for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = recompute_utils.checkpoint_forward(layer, True, False, x)
        return x


def _replays_under(policy, model, calls):
    """Run one step and return how many fwd calls the backward replay added."""
    torch.manual_seed(0)
    recompute_utils.apply_recompute_policy(model, policy)
    x = torch.randn(2, 6, 8, requires_grad=True)

    calls.clear()
    out = model(x)
    forward_calls = len(calls)
    out.square().mean().backward()
    return forward_calls, len(calls) - forward_calls


SAC_POLICY = recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True, selective_n_layers=3)
FULL_POLICY = recompute_utils.build_policy(enabled=True, enable_reentrant=False, early_stop=True)


@pytest.mark.skipif(_FA3_OP is None, reason="the test namespace is already registered")
def test_backend_kernel_registered_as_custom_op_is_saved_then_recomputed():
    """A custom-op attention kernel is resolved exactly and kept out of the replay."""
    assert _FA3_OP in recompute_utils.resolve_exact_ops()[0]

    sac_forward, sac_replayed = _replays_under(SAC_POLICY, _FlashAttn3Model(), _FA3_CALLS)
    full_forward, full_replayed = _replays_under(FULL_POLICY, _FlashAttn3Model(), _FA3_CALLS)

    assert sac_forward == full_forward == 3  # one call per block in the forward
    assert sac_replayed == 0  # MUST_SAVE: the kernel is not re-executed in the replay
    assert full_replayed == 3  # whole-block recompute: every block re-runs it


class _FlashAttn2Function(torch.autograd.Function):
    """Shape of the *upstream* FA2 path: a python autograd.Function over a raw kernel.

    Upstream FA2 reaches its kernel through a pybind extension, so no dispatcher
    operator carries the attention name — SAC has nothing to attach MUST_SAVE to.
    The pinned VeOmni FA2 wheel is a fork that adds `flash_attn::_flash_attn_*`
    custom ops, so *that* build is visible; see the op-name case above.
    """

    @staticmethod
    def forward(ctx, q, k, v):
        _FA2_CALLS.append("fwd")
        scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        ctx.save_for_backward(q, k, v, probs)
        return torch.matmul(probs, v)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, probs = ctx.saved_tensors
        grad_probs = torch.matmul(grad_out, v.transpose(-1, -2))
        grad_scores = probs * (grad_probs - (grad_probs * probs).sum(-1, keepdim=True))
        grad_q = torch.matmul(grad_scores, k) * (q.shape[-1] ** -0.5)
        grad_k = torch.matmul(grad_scores.transpose(-1, -2), q) * (q.shape[-1] ** -0.5)
        grad_v = torch.matmul(probs.transpose(-1, -2), grad_out)
        return grad_q, grad_k, grad_v


_FA2_CALLS: list[str] = []


class _FlashAttn2Block(nn.Module):
    gradient_checkpointing = False

    def __init__(self, dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.norm(x)
        attn = _FlashAttn2Function.apply(q, q, q)
        return x + self.proj(attn)


class _FlashAttn2Model(nn.Module):
    _no_split_modules = ["_FlashAttn2Block"]

    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList(_FlashAttn2Block() for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = recompute_utils.checkpoint_forward(layer, True, False, x)
        return x


def test_upstream_pybind_kernel_without_a_dispatcher_name_gets_no_sac_benefit():
    """Upstream FA2 keeps re-executing under SAC: invisible to the dispatch mode."""
    _, sac_replayed = _replays_under(SAC_POLICY, _FlashAttn2Model(), _FA2_CALLS)
    _, full_replayed = _replays_under(FULL_POLICY, _FlashAttn2Model(), _FA2_CALLS)

    assert sac_replayed == full_replayed == 3
