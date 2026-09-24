"""VeOmni runtime over SeedOmni graphs: executor, OmniModelRuntime, graph profiling."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch.nn as nn

from veomni.arguments.omni_arguments_types import OmniGraphProfileArguments
from veomni.models.seed_omni.accelerated import OmniModelRuntime
from veomni.models.seed_omni.accelerated.utils import iter_named_omni_modules
from veomni.models.seed_omni.accelerated.utils.executor import (
    TrainNodeRunner,
    execute_generation_node,
    execute_train_node,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.inference_module_mixin import InferenceModuleMixin
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig
from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule
from veomni.models.seed_omni.utils import graph_profiler
from veomni.models.seed_omni.utils.graph_profiler import GraphProfiler
from veomni.trainer.callbacks.omni_callbacks import GraphProfileCallback
from veomni.trainer.omni.omni_trainer import OmniTrainer


def _fan_in_edges() -> list[dict]:
    """Two encoders feeding one backbone."""
    return [
        {"from": "module_A", "to": "module_C"},
        {"from": "module_B", "to": "module_C"},
        {"from": "module_C", "to": "end"},
    ]


class _StubConfig(OmniModuleConfig):
    model_type = "graph_runtime_stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _FakeOmniModule(PretrainedOmniModule, TrainingModuleMixin, BaseMixin, InferenceModuleMixin):
    """Minimal OmniModule: callable (→ forward) + pre/post hooks, traces each call.

    ``__call__`` delegates to ``self.forward`` so the non-``forward`` alias trick
    (``raw.forward = encode``) works exactly as on a real ``nn.Module``.
    """

    config_class = _StubConfig

    def __init__(self, name: str):
        super().__init__(_StubConfig())
        self.name = name

    def pre_forward(self, method, **kwargs):
        return kwargs

    def post_forward(self, method, **outputs):
        return outputs

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def _trace(self, kwargs, label: str) -> dict:
        trace = list(kwargs.get("trace", []))
        trace.append(f"{self.name}.{label}")
        return {"trace": trace}

    def forward(self, **kwargs):
        return self._trace(kwargs, "forward")

    def encode(self, **kwargs):
        return self._trace(kwargs, "encode")

    def generate(self, **kwargs):
        return self._trace(kwargs, "generate")

    def generate_via_forward(self, **kwargs):
        out = self.forward(**kwargs)
        out["trace"].append(f"{self.name}.generate_via_forward")
        return out


def _fake_modules(g: TrainingGraph) -> dict:
    return {name: _FakeOmniModule(name) for name in {g.module_of(n) for n in g.execution_order}}


def _minimal_generation_graphs(module: str = "module_C") -> dict:
    return {
        "infer_gen": {
            "initial": "run",
            "states": {
                "run": {
                    "body": [{"from": module, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    }


def _omni_model(edges: list[dict], modules: dict) -> OmniModel:
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in modules},
        training_graphs={"default": edges},
        generation_graphs=_minimal_generation_graphs(),
    )
    return OmniModel(config, modules)


class _DdpStyleWrapper(nn.Module):
    """Minimal DDP-shaped wrapper: hooks live on ``.module``; records each call."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.module = inner
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


def test_plan_loop_flows_carrier_in_topological_order():
    """Driving iter_nodes() + execute_train_node mirrors OmniModelRuntime.forward."""
    g = TrainingGraph(_fan_in_edges())
    modules = _fake_modules(g)
    batch = {"trace": []}
    profiler = GraphProfiler()
    g.reset()
    for node in g.iter_nodes():
        execute_train_node(modules[node.module], node, batch, profiler=profiler)
    records = profiler.save_records()
    assert batch["trace"][-1] == "module_C.forward"
    assert set(batch["trace"]) == {"module_A.forward", "module_B.forward", "module_C.forward"}
    assert [t for t in records if t.startswith("forward:")] == [f"forward:{n}" for n in g.execution_order]


def test_omni_model_runtime_forward_matches_manual_executor():
    edges = _fan_in_edges()
    g = TrainingGraph(edges)
    modules = _fake_modules(g)
    runtime = OmniModelRuntime(_omni_model(edges, modules))

    batch_runtime: dict = {"trace": []}
    batch_manual: dict = {"trace": []}
    runtime.forward(batch_runtime, profiler=GraphProfiler())
    g.reset()
    for node in g.iter_nodes():
        execute_train_node(modules[node.module], node, batch_manual, profiler=GraphProfiler())

    assert batch_runtime == batch_manual


def test_omni_model_runtime_forward_enters_composed_model_call(monkeypatch: pytest.MonkeyPatch):
    """FSDP leftover unshard requires ``OmniModel.__call__``, not an out-of-band graph loop."""
    edges = _fan_in_edges()
    model = _omni_model(edges, _fake_modules(TrainingGraph(edges)))
    runtime = OmniModelRuntime(model)

    called: list[int] = []
    orig_call = nn.Module.__call__

    def _tracking_call(self, *args, **kwargs):
        called.append(id(self))
        return orig_call(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "__call__", _tracking_call)
    runtime.forward({"trace": []})

    assert called and called[0] == id(model)


def test_graph_profiler_can_append_request_peak_memory(monkeypatch):
    class _FakeDevice:
        def __init__(self):
            self.reset_calls = 0

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def max_memory_allocated(self):
            return 2 * 1024**3

        def max_memory_reserved(self):
            return 3 * 1024**3

    device = _FakeDevice()
    monkeypatch.setattr(graph_profiler, "get_torch_device", lambda: device)

    profiler = GraphProfiler(enable_memory=True)
    with profiler.node("forward:module_C.forward"):
        pass

    assert device.reset_calls == 1
    assert profiler.save_records() == ["forward:module_C.forward | peak_allocated_gb=2.000 | peak_reserved_gb=3.000"]


def _make_graph_profile_callback(output_dir, *, global_rank=0, **profile_kwargs):
    """A GraphProfileCallback wired to a stub OmniTrainer (no full trainer init)."""
    profile = OmniGraphProfileArguments(**profile_kwargs)
    edges = _fan_in_edges()
    model = _omni_model(edges, _fake_modules(TrainingGraph(edges)))
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            global_rank=global_rank,
            graph_profile=profile,
            checkpoint=SimpleNamespace(output_dir=str(output_dir)),
        ),
    )
    trainer.model = OmniModelRuntime(model, module_runtimes={})
    return GraphProfileCallback(trainer), trainer


def test_graph_profile_callback_saves_training_graph_profile(tmp_path):
    output_dir = tmp_path / "output"
    callback, trainer = _make_graph_profile_callback(
        output_dir, enable_wall_time=True, train_start_step=2, train_end_step=3
    )
    state = SimpleNamespace(global_step=3)

    callback.on_step_begin(state)
    profiler = trainer.model.step_profiler
    assert profiler is not None
    with profiler.node("forward:module_C.forward"):
        pass

    callback.on_step_end(state)
    trace_path = output_dir / "graph_trace" / "step_000003_rank_0.txt"
    assert trace_path.exists()
    assert "forward:module_C.forward | wall_ms=" in trace_path.read_text()
    assert trainer.model.step_profiler is None


def test_graph_profile_callback_is_gated_outside_step_window(tmp_path):
    output_dir = tmp_path / "output"
    callback, trainer = _make_graph_profile_callback(
        output_dir, enable_wall_time=True, train_start_step=2, train_end_step=3
    )

    callback.on_step_begin(SimpleNamespace(global_step=5))
    assert trainer.model.step_profiler is None
    callback.on_step_end(SimpleNamespace(global_step=5))
    assert not (output_dir / "graph_trace").exists()


def test_execute_train_node_dispatches_non_forward_method_via_wrapper():
    """A dotted ``module.encode`` node must run the module's ``encode`` (alias trick)."""
    g = TrainingGraph([{"from": "module_B.encode", "to": "end"}])
    modules = _fake_modules(g)
    node = next(g.iter_nodes())
    batch = execute_train_node(modules[node.module], node, {"trace": []})
    assert batch["trace"] == ["module_B.encode"]
    assert modules["module_B"].forward.__name__ == "forward"


def test_execute_train_node_unwraps_ddp_style_wrapper():
    """A wrapper without ``pre_forward`` is unwrapped via ``.module`` (DDP)."""

    class _DDPWrap:
        def __init__(self, inner):
            self.module = inner

        def __call__(self, **kwargs):
            return self.module.forward(**kwargs)

    g = TrainingGraph([{"from": "module_C", "to": "end"}])
    node = next(g.iter_nodes())
    batch = execute_train_node(_DDPWrap(_FakeOmniModule("module_C")), node, {"trace": []})
    assert batch["trace"] == ["module_C.forward"]


class _TrackingWrapper:
    def __init__(self, inner):
        self.module = inner
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.module.forward(**kwargs)


def _one_step_generation_graph(endpoint: str) -> GenerationGraph:
    return GenerationGraph(
        {
            "initial": "run",
            "states": {
                "run": {
                    "body": [{"from": endpoint, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    )


def _run_one_body(g: GenerationGraph, modules: dict, ctx: dict) -> dict:
    """Drive one FSM body iteration: graph selects nodes, executor runs them."""
    for node in g.iter_nodes(ctx):
        execute_generation_node(modules, node, ctx, state_name=g.current_state_name)
    return ctx


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("module_C", ["module_C.generate"]),
        ("module_C.encode", ["module_C.encode"]),
        ("module_C.generate_via_forward", ["module_C.forward", "module_C.generate_via_forward"]),
    ],
)
def test_generation_dispatches_through_the_wrapper_call(endpoint, expected):
    inner = _FakeOmniModule("module_C")
    wrapped = _TrackingWrapper(inner)

    ctx = _run_one_body(_one_step_generation_graph(endpoint), {"module_C": wrapped}, {"trace": []})

    assert wrapped.calls == 1
    assert ctx["trace"] == expected
    assert inner.forward.__name__ == "forward"


def test_execute_train_node_applies_module_scope():
    scoped: list[str] = []

    @contextmanager
    def scope_fn(name: str):
        scoped.append(name)
        yield

    g = TrainingGraph([{"from": "module_C", "to": "end"}])
    node = next(g.iter_nodes())
    execute_train_node(_fake_modules(g)[node.module], node, {"trace": []}, scope_fn=scope_fn)
    assert scoped == ["module_C"]


class _LossModule(_FakeOmniModule):
    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        out["_loss"] = 1.5
        return out


def test_execute_train_node_merges_loss_into_batch():
    g = TrainingGraph([{"from": "module_C", "to": "end"}])
    node = next(g.iter_nodes())
    batch = execute_train_node(_LossModule("module_C"), node, {"trace": []})
    assert batch["_loss"] == 1.5


def test_train_node_runner_records_transitions_and_losses():
    """The runner, not ``OmniModel.forward``, owns the profiler markers."""
    g = TrainingGraph(_fan_in_edges())
    modules = {name: _LossModule(name) for name in {g.module_of(n) for n in g.execution_order}}
    profiler = GraphProfiler()
    runner = TrainNodeRunner(profiler=profiler)

    batch: dict = {"trace": []}
    g.reset()
    for node in g.iter_nodes():
        runner(modules[node.module], node, batch)
        batch.pop("_loss", None)

    records = profiler.save_records()
    # No transition before the first node; one per node boundary after that.
    assert [t for t in records if t.startswith("transition:")] == [
        f"transition: -> {n}" for n in g.execution_order[1:]
    ]
    assert [t for t in records if t.startswith("loss:")] == [f"loss:{n}" for n in g.execution_order]


def test_iter_named_omni_modules_unwraps_ddp_style_wrapper():
    g = TrainingGraph(_fan_in_edges())
    raw_modules = _fake_modules(g)
    wrapped_modules = {name: _DdpStyleWrapper(mod) for name, mod in raw_modules.items()}

    resolved = dict(iter_named_omni_modules(list(raw_modules), wrapped_modules))
    assert set(resolved) == set(raw_modules)
    for name, raw in resolved.items():
        assert raw is raw_modules[name]
