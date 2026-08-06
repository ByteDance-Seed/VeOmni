"""Unit tests for the SeedOmni V2 graph layer (flat edge-list training subset)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch.nn as nn

from veomni.arguments import OmniGraphProfileArguments
from veomni.models.seed_omni import EdgeDef, NodeDef
from veomni.models.seed_omni.accelerator import OmniModelRuntime
from veomni.models.seed_omni.accelerator.executor import execute_generation_node, execute_train_node
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.base import END
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.mixins.module_mixin import ModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.utils import graph_profiler
from veomni.models.seed_omni.utils.graph_profiler import GraphProfiler
from veomni.trainer.callbacks.omni_callbacks import GraphProfileCallback
from veomni.trainer.omni.omni_trainer import OmniTrainer


# ── NodeDef parsing ───────────────────────────────────────────────────────────


def test_from_endpoint_default_method():
    n = NodeDef.from_endpoint("ar_llm", default_method="forward")
    assert n.module == "ar_llm" and n.method == "forward"
    assert n.name == "ar_llm.forward"


def test_from_endpoint_dotted_form():
    n = NodeDef.from_endpoint("vq_decoder.encode", default_method="forward")
    assert n.module == "vq_decoder" and n.method == "encode"
    assert n.name == "vq_decoder.encode"


def test_from_endpoint_generate_default():
    n = NodeDef.from_endpoint("ar_llm", default_method="generate")
    assert n.module == "ar_llm" and n.method == "generate"


def test_from_endpoint_rejects_reserved_end():
    with pytest.raises(ValueError, match=f"'{END}' is the virtual sink"):
        NodeDef.from_endpoint(END, default_method="forward")


def test_from_endpoint_rejects_empty():
    with pytest.raises(ValueError, match="non-empty 'module"):
        NodeDef.from_endpoint("   ", default_method="forward")


# ── EdgeDef parsing ───────────────────────────────────────────────────────────


def test_parse_edge():
    e = EdgeDef.parse({"from": "vision_encoder", "to": "run_ar"}, default_method="forward")
    assert e.from_ == "vision_encoder.forward" and e.to == "run_ar.forward"
    assert e.from_node.module == "vision_encoder" and e.to_node.module == "run_ar"
    assert not e.is_sink()


def test_parse_edge_to_end_is_sink():
    e = EdgeDef.parse({"from": "tok_decode", "to": "end"}, default_method="forward")
    assert e.is_sink() and e.to == END and e.to_node is None
    assert e.from_ == "tok_decode.forward"


def test_parse_edge_rejects_from_end():
    with pytest.raises(ValueError, match="`from: end` is forbidden"):
        EdgeDef.parse({"from": "end", "to": "run_ar"}, default_method="forward")


def test_parse_edge_rejects_node_fields():
    with pytest.raises(ValueError, match="must not contain node fields"):
        EdgeDef.parse({"from": "a", "to": "b", "module": "x"}, default_method="forward")


def test_parse_edge_rejects_missing_endpoints():
    with pytest.raises(ValueError, match="must declare both"):
        EdgeDef.parse({"from": "a"}, default_method="forward")


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _janus_joint_edges() -> list[dict]:
    """Janus joint training edges: vq_decoder appears under TWO methods.

    Adds an explicit ``to: end`` sink for the leaf so every node is visible to
    the active subset purely from the edge list.
    """
    return [
        {"from": "vision_encoder", "to": "run_ar"},
        {"from": "vq_decoder.encode", "to": "run_ar"},
        {"from": "run_ar", "to": "vq_decoder.gen_loss"},
        {"from": "vq_decoder.gen_loss", "to": "end"},
    ]


def _understanding_only_edges() -> list[dict]:
    """Two encoders → ar_llm, simple DAG with end-sink."""
    return [
        {"from": "vision_encoder", "to": "run_ar"},
        {"from": "vq_decoder", "to": "run_ar"},
        {"from": "run_ar", "to": "end"},
    ]


# ── Validation ────────────────────────────────────────────────────────────────


def test_missing_edges_raises():
    with pytest.raises(ValueError, match="non-empty `training_graph`"):
        TrainingGraph([])


def test_duplicate_edge_raises():
    with pytest.raises(ValueError, match="Duplicate edge"):
        TrainingGraph(
            [
                {"from": "vision_encoder", "to": "run_ar"},
                {"from": "vision_encoder", "to": "run_ar"},
            ]
        )


def test_single_node_with_only_end_edge():
    """``[{from: ar_llm, to: end}]`` derives exactly one real node."""
    g = TrainingGraph([{"from": "ar_llm", "to": "end"}])
    assert g.execution_order == ["ar_llm.forward"]
    assert g.sources == ["ar_llm.forward"] and g.sinks == ["ar_llm.forward"]


# ── Topological order ─────────────────────────────────────────────────────────


def test_understanding_only_topological_order():
    g = TrainingGraph(_understanding_only_edges())
    assert g.execution_order[-1] == "run_ar.forward"
    assert set(g.execution_order[:-1]) == {"vision_encoder.forward", "vq_decoder.forward"}


def test_janus_joint_topological_order():
    """vq_decoder appears as TWO nodes; topo must place them on either side of run_ar."""
    g = TrainingGraph(_janus_joint_edges())
    order = g.execution_order
    assert order.index("vq_decoder.gen_loss") > order.index("run_ar.forward")
    assert order.index("run_ar.forward") > order.index("vq_decoder.encode")
    assert order.index("run_ar.forward") > order.index("vision_encoder.forward")


def test_cycle_in_active_set_raises():
    with pytest.raises(ValueError, match="Circular dependency"):
        TrainingGraph(
            [
                {"from": "vq_decoder", "to": "ar_llm"},
                {"from": "ar_llm", "to": "vq_decoder"},
            ]
        )


# ── Sources / sinks ───────────────────────────────────────────────────────────


def test_sources_and_sinks_understanding_only():
    g = TrainingGraph(_understanding_only_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.forward"}
    # run_ar's only outgoing edge targets `end`, so it's a sink.
    assert g.sinks == ["run_ar.forward"]


def test_sources_and_sinks_janus_joint():
    g = TrainingGraph(_janus_joint_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.encode"}
    # vq_decoder.gen_loss is the only sink (its only outgoing edge goes to `end`).
    assert g.sinks == ["vq_decoder.gen_loss"]


# ── module / method accessors ────────────────────────────────────────────────


def test_module_and_method_lookup():
    g = TrainingGraph(_janus_joint_edges())
    assert g.module_of("vq_decoder.encode") == "vq_decoder"
    assert g.method_of("vq_decoder.encode") == "encode"
    assert g.module_of("vq_decoder.gen_loss") == "vq_decoder"
    assert g.method_of("vq_decoder.gen_loss") == "gen_loss"
    assert g.method_of("run_ar.forward") == "forward"


def test_module_lookup_raises_for_unknown():
    g = TrainingGraph(_janus_joint_edges())
    with pytest.raises(KeyError):
        g.module_of("not_a_node")


# ── Execution lifecycle (cursor + step + maybe_transition) ────────────────────


class _FakeOmniModule(nn.Module, ModuleMixin):
    """Minimal stand-in for an OmniModule: callable (→ forward) + pre/post hooks.

    ``__call__`` delegates to ``self.forward`` so the non-``forward`` alias trick
    (``raw.forward = encode``) works exactly as on a real ``nn.Module``.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def pre_forward(self, method, **kwargs):
        return kwargs

    def post_forward(self, method, **outputs):
        return outputs

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.forward")
        return {"conversation_list": cl}

    def encode(self, **kwargs):
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.encode")
        return {"conversation_list": cl}

    def generate(self, **kwargs):
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.generate")
        return {"conversation_list": cl}

    def generate_via_forward(self, **kwargs):
        out = self.forward(**kwargs)
        out["conversation_list"].append(f"{self.name}.generate_via_forward")
        return out


def _fake_modules(g: TrainingGraph) -> dict:
    return {name: _FakeOmniModule(name) for name in {g.module_of(n) for n in g.execution_order}}


def test_cursor_lifecycle():
    g = TrainingGraph(_understanding_only_edges())
    assert not g.is_done()
    assert g.current_node_name == g.execution_order[0]
    # Walk the cursor manually.
    seen = []
    while not g.is_done():
        seen.append(g.current_node_name)
        g.maybe_transition()
    assert seen == g.execution_order
    assert g.is_done()
    with pytest.raises(RuntimeError, match="cursor past the last node"):
        _ = g.current_node_name
    g.reset()
    assert not g.is_done() and g.current_node_name == g.execution_order[0]


def test_plan_loop_flows_carrier_in_topological_order():
    """Driving iter_nodes() + execute_train_node mirrors OmniModelRuntime.forward."""
    g = TrainingGraph(_understanding_only_edges())
    modules = _fake_modules(g)
    batch = {"conversation_list": []}
    profiler = GraphProfiler()
    g.reset()
    for node in g.iter_nodes():
        execute_train_node(modules, node, batch, profiler=profiler)
    # run_ar runs last; both encoders precede it.
    trace = profiler.save_records()
    assert batch["conversation_list"][-1] == "run_ar.forward"
    assert set(batch["conversation_list"]) == {"vision_encoder.forward", "vq_decoder.forward", "run_ar.forward"}
    assert [t for t in trace if t.startswith("forward:")] == [f"forward:{n}" for n in g.execution_order]


def _minimal_generation_graph(module: str = "run_ar") -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(module: str = "run_ar") -> dict:
    return {"infer_gen": _minimal_generation_graph(module)}


def test_omni_model_runtime_forward_matches_manual_executor():
    """OmniModelRuntime.forward must match the manual executor loop."""
    edges = _understanding_only_edges()
    g = TrainingGraph(edges)
    modules = _fake_modules(g)
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in {g.module_of(n) for n in g.execution_order}},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, modules)
    runtime = OmniModelRuntime(model)

    batch_runtime: dict = {"conversation_list": []}
    batch_manual: dict = {"conversation_list": []}
    profiler = GraphProfiler()

    runtime.forward(batch_runtime, profiler=profiler)
    g.reset()
    for node in g.iter_nodes():
        execute_train_node(modules, node, batch_manual, profiler=GraphProfiler())

    assert batch_runtime == batch_manual


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
    with profiler.node("forward:run_ar.forward"):
        pass

    assert device.reset_calls == 1
    assert profiler.save_records() == ["forward:run_ar.forward | peak_allocated_gb=2.000 | peak_reserved_gb=3.000"]


def _make_graph_profile_callback(output_dir, *, global_rank=0, **profile_kwargs):
    """A GraphProfileCallback wired to a stub OmniTrainer (no full trainer init)."""
    profile = OmniGraphProfileArguments(**profile_kwargs)
    edges = _understanding_only_edges()
    g = TrainingGraph(edges)
    modules = _fake_modules(g)
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in {g.module_of(n) for n in g.execution_order}},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, modules)
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(
            global_rank=global_rank,
            graph_profile=profile,
            checkpoint=SimpleNamespace(output_dir=str(output_dir)),
        ),
    )
    trainer.model = OmniModelRuntime(model, module_runtimes={}, module_parallel_state_names=())
    return GraphProfileCallback(trainer), trainer


def test_graph_profile_callback_saves_training_graph_profile(tmp_path):
    output_dir = tmp_path / "output"
    callback, trainer = _make_graph_profile_callback(
        output_dir, enable_wall_time=True, train_start_step=2, train_end_step=3
    )
    state = SimpleNamespace(global_step=3)

    # Step begin builds + binds the profiler for the step's forwards to consume.
    callback.on_step_begin(state)
    profiler = trainer.model.step_profiler
    assert profiler is not None
    with profiler.node("forward:run_ar.forward"):
        pass

    # Step end writes the per-step trace file and clears the slot.
    callback.on_step_end(state)
    trace_path = output_dir / "graph_trace" / "step_000003_rank_0.txt"
    assert trace_path.exists()
    assert "forward:run_ar.forward | wall_ms=" in trace_path.read_text()
    assert trainer.model.step_profiler is None


def test_graph_profile_callback_is_gated_outside_step_window(tmp_path):
    output_dir = tmp_path / "output"
    callback, trainer = _make_graph_profile_callback(
        output_dir, enable_wall_time=True, train_start_step=2, train_end_step=3
    )

    # global_step outside [train_start_step, train_end_step] → callback skips init,
    # and flush is a no-op (no trace file written).
    callback.on_step_begin(SimpleNamespace(global_step=5))
    assert trainer.model.step_profiler is None
    callback.on_step_end(SimpleNamespace(global_step=5))
    assert not (output_dir / "graph_trace").exists()


def test_execute_train_node_dispatches_non_forward_method_via_wrapper():
    """A dotted ``module.encode`` node must run the module's ``encode`` (alias trick)."""
    g = TrainingGraph([{"from": "vq_decoder.encode", "to": "end"}])
    modules = _fake_modules(g)
    node = next(g.iter_nodes())
    batch = execute_train_node(modules, node, {"conversation_list": []})
    assert batch["conversation_list"] == ["vq_decoder.encode"]
    # forward restored after the aliased call.
    assert modules["vq_decoder"].forward.__name__ == "forward"


def test_execute_train_node_unwraps_ddp_style_wrapper():
    """A wrapper without ``pre_forward`` is unwrapped via ``.module`` (DDP)."""

    class _DDPWrap:
        def __init__(self, inner):
            self.module = inner

        def __call__(self, **kwargs):
            return self.module.forward(**kwargs)

    g = TrainingGraph([{"from": "run_ar", "to": "end"}])
    inner = _FakeOmniModule("run_ar")
    node = next(g.iter_nodes())
    batch = execute_train_node({"run_ar": _DDPWrap(inner)}, node, {"conversation_list": []})
    assert batch["conversation_list"] == ["run_ar.forward"]


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


def test_generation_dispatches_bare_generate_via_wrapper_call():
    g = _one_step_generation_graph("run_ar")
    inner = _FakeOmniModule("run_ar")
    wrapped = _TrackingWrapper(inner)

    ctx = _run_one_body(g, {"run_ar": wrapped}, {"conversation_list": []})

    assert wrapped.calls == 1
    assert ctx["conversation_list"] == ["run_ar.generate"]
    assert inner.forward.__name__ == "forward"


def test_generation_dispatches_dotted_method_via_wrapper_call():
    g = _one_step_generation_graph("run_ar.encode")
    inner = _FakeOmniModule("run_ar")
    wrapped = _TrackingWrapper(inner)

    ctx = _run_one_body(g, {"run_ar": wrapped}, {"conversation_list": []})

    assert wrapped.calls == 1
    assert ctx["conversation_list"] == ["run_ar.encode"]
    assert inner.forward.__name__ == "forward"


def test_generation_endpoint_can_call_original_forward_without_recursing():
    g = _one_step_generation_graph("run_ar.generate_via_forward")
    inner = _FakeOmniModule("run_ar")
    wrapped = _TrackingWrapper(inner)

    ctx = _run_one_body(g, {"run_ar": wrapped}, {"conversation_list": []})

    assert wrapped.calls == 1
    assert ctx["conversation_list"] == ["run_ar.forward", "run_ar.generate_via_forward"]
    assert inner.forward.__name__ == "forward"


def test_execute_train_node_applies_module_scope():
    from contextlib import contextmanager

    scoped: list[str] = []

    @contextmanager
    def scope_fn(name: str):
        scoped.append(name)
        yield

    g = TrainingGraph([{"from": "run_ar", "to": "end"}])
    node = next(g.iter_nodes())
    execute_train_node(_fake_modules(g), node, {"conversation_list": []}, scope_fn=scope_fn)
    assert scoped == ["run_ar"]


def test_execute_train_node_merges_loss_into_batch():
    class _LossModule(_FakeOmniModule):
        def forward(self, **kwargs):
            out = super().forward(**kwargs)
            out["_loss"] = 1.5
            return out

    g = TrainingGraph([{"from": "run_ar", "to": "end"}])
    node = next(g.iter_nodes())
    batch = execute_train_node({"run_ar": _LossModule("run_ar")}, node, {"conversation_list": []})
    assert batch["_loss"] == 1.5


def test_execute_train_node_raises_for_missing_module():
    g = TrainingGraph([{"from": "run_ar", "to": "end"}])
    node = next(g.iter_nodes())
    with pytest.raises(KeyError, match="missing from modules dict"):
        execute_train_node({}, node, {"conversation_list": []})


# ── Mermaid visualisation ────────────────────────────────────────────────────


def test_to_mermaid_janus_joint_contains_node_labels_and_end_sink():
    g = TrainingGraph(_janus_joint_edges())
    out = g.to_mermaid(title="Janus Joint Training")

    # Frontmatter, ELK renderer hint, then LR flowchart.
    assert out.startswith("---\ntitle: Janus Joint Training\n---\n")
    assert "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%" in out
    assert "flowchart LR" in out

    # Node ids sanitise dots → underscores; labels keep the canonical name.
    assert re.search(r'\bvision_encoder_forward\["<i>vision_encoder\.forward</i>"\]', out)
    assert re.search(r'\bvq_decoder_encode\["<i>vq_decoder\.encode</i>"\]', out)
    assert re.search(r'\brun_ar_forward\["<i>run_ar\.forward</i>"\]', out)
    assert re.search(r'\bvq_decoder_gen_loss\["<i>vq_decoder\.gen_loss</i>"\]', out)

    assert "vision_encoder_forward -->" in out and "run_ar_forward" in out
    assert "vq_decoder_encode -->" in out
    assert "run_ar_forward -->" in out and "vq_decoder_gen_loss" in out

    # `end` rendered as the dashed terminal.
    assert "end_sink" in out and "vq_decoder_gen_loss --> end_sink" in out

    assert ":::source" in out and ":::sink" in out

    # Per-rank invisible subgraphs (col0 = sources, col1 = middle, col2 = sinks).
    assert "subgraph col0" in out and "subgraph col1" in out and "subgraph col2" in out
    assert "style col0 fill:transparent,stroke:none" in out

    assert "data -.-> vision_encoder_forward" in out
    assert "data -.-> vq_decoder_encode" in out

    # Single-loss protocol — no `losses` collector node.
    assert "losses" not in out


def test_to_mermaid_always_draws_data_pseudo_node():
    g = TrainingGraph(_janus_joint_edges())
    out = g.to_mermaid()
    assert "data[(data)]" in out
    assert "data -.-> vision_encoder_forward" in out
    assert "losses" not in out
    assert "end_sink" in out


def test_generation_graph_mermaid_stacks_state_body_nodes():
    g = GenerationGraph(
        {
            "initial": "prompt",
            "states": {
                "prompt": {
                    "body": [{"from": "encoder.encode", "to": "decoder.decode"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    )

    out = g.to_mermaid(title="Compact FSM")

    assert "flowchart LR" in out
    assert "subgraph state_prompt [prompt]\n        direction TB" in out
    assert "prompt__encoder_encode --> prompt__decoder_decode" in out


class _DdpStyleWrapper(nn.Module):
    """Minimal DDP-shaped wrapper: hooks live on ``.module``."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.module = inner

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_named_omni_modules_unwraps_ddp_style_wrapper():
    edges = _understanding_only_edges()
    g = TrainingGraph(edges)
    raw_modules = _fake_modules(g)
    wrapped_modules = {name: _DdpStyleWrapper(mod) for name, mod in raw_modules.items()}
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in raw_modules},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, wrapped_modules)

    resolved = dict(model.named_omni_modules())
    assert set(resolved) == set(raw_modules)
    for name, raw in resolved.items():
        assert raw is raw_modules[name]
        assert not isinstance(raw, _DdpStyleWrapper)


def test_named_omni_modules_skips_non_modulemixin():
    class _PlainModule(nn.Module):
        def forward(self, x):
            return x

    config = OmniConfig(
        modules={"plain": {"subfolder": "plain"}},
        training_graph=[{"from": "plain", "to": "end"}],
        generation_graphs=_minimal_generation_graphs("plain"),
    )
    model = OmniModel(config, {"plain": _PlainModule()})
    assert list(model.named_omni_modules()) == []
