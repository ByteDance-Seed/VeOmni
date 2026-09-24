"""``OmniModelRuntime`` runs graph nodes the way the eager :class:`OmniModel` does.

Training goes through ``OmniModel.forward`` with the runtime's node runner;
inference loops the FSM itself. Either path must see the same hooks and keep the
same artefacts as the eager model, so a module does not behave differently
once it is served under VeOmni.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime
from veomni.models.seed_omni.accelerated.utils import executor
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.inference_module_mixin import InferenceModuleMixin, post_generate, pre_generate
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig
from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule


class _StubConfig(OmniModuleConfig):
    model_type = "stub_omni_runtime_module"


class _Module(PretrainedOmniModule, TrainingModuleMixin, InferenceModuleMixin, BaseMixin):
    """Emits one artefact per ``generate`` and traces the generation hooks."""

    config_class = _StubConfig

    def __init__(self, name: str):
        super().__init__(_StubConfig())
        self.name = name
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, **kwargs):
        return {f"{self.name}_out": self.weight * 2}

    @pre_generate("generate")
    def generate_pre(self, **kwargs):
        return {**kwargs, "trace": [*kwargs.get("trace", []), f"{self.name}:pre"]}

    def generate(self, generation_kwargs=None, **kwargs):
        del generation_kwargs
        return {
            "trace": [*kwargs.get("trace", []), f"{self.name}:generate"],
            "generated": {"type": "text", "value": self.name},
        }

    @post_generate("generate")
    def generate_post(self, **outputs):
        return {**outputs, "trace": [*outputs.get("trace", []), f"{self.name}:post"]}


_BODY = [{"from": "module_a", "to": "module_b"}, {"from": "module_b", "to": "end"}]


def _model() -> OmniModel:
    config = OmniConfig(
        _module_entries={"module_a": {"model_path": "module_a"}, "module_b": {"model_path": "module_b"}},
        training_graphs={"default": _BODY},
        generation_graphs={
            "infer_gen": {
                "initial": "run",
                "states": {
                    "run": {
                        "body": _BODY,
                        "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                    }
                },
            }
        },
    )
    return OmniModel(config, {"module_a": _Module("module_a"), "module_b": _Module("module_b")})


def test_training_forward_runs_every_node_through_the_runtime_runner(monkeypatch):
    """``OmniModel.forward`` must use the runner it is handed, not its own eager node call."""
    seen = []
    run_node = executor.execute_train_node

    def spy(wrapped, node, batch, **kwargs):
        seen.append(node.module)
        return run_node(wrapped, node, batch, **kwargs)

    monkeypatch.setattr(executor, "execute_train_node", spy)
    runtime = OmniModelRuntime(_model())

    runtime.forward({})

    assert seen == ["module_a", "module_b"]


def test_runtime_generation_matches_eager_generation():
    eager = _model()
    eager.reset()
    eager_ctx: dict = {}
    eager_generated = eager.generate(eager_ctx)

    runtime = OmniModelRuntime(_model())
    runtime.reset()
    runtime_ctx: dict = {}
    runtime_generated = runtime.generate(runtime_ctx)

    assert (
        runtime_ctx["trace"]
        == eager_ctx["trace"]
        == [
            "module_a:pre",
            "module_a:generate",
            "module_a:post",
            "module_b:pre",
            "module_b:generate",
            "module_b:post",
        ]
    )
    assert [item["value"] for item in runtime_generated] == [item["value"] for item in eager_generated]
    assert [item["value"] for item in runtime_generated] == ["module_a", "module_b"]


class _NativeModule(PretrainedOmniModule):
    """A module with no graph mixins, as an unregistered ``model_type`` resolves to."""

    config_class = _StubConfig

    def __init__(self):
        super().__init__(_StubConfig())
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.resets = 0

    def forward(self, **kwargs):
        return {"native_out": self.weight * 3}

    def reset_global_inference_state(self) -> None:
        self.resets += 1


def test_runtime_runs_and_resets_a_module_without_graph_mixins():
    """The eager model treats ``pre_forward`` / ``post_forward`` as optional and
    resets every participant; served under VeOmni the module must behave the same."""
    model = _model()
    native = _NativeModule()
    model.module_b = native
    runtime = OmniModelRuntime(model)
    batch: dict = {}

    runtime.forward(batch)
    runtime.reset()

    assert torch.equal(batch["native_out"], torch.tensor([3.0]))
    assert dict(runtime.named_omni_modules())["module_b"] is native
    assert native.resets == 1


class _DdpStyleWrapper(torch.nn.Module):
    """DDP-shaped: the module lives on ``.module``; counts the calls it receives."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.module = inner
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


class _Exportable(torch.nn.Module):
    """Records the directories its ``save_pretrained`` was asked to write."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.saved_to = []

    def save_pretrained(self, save_directory, **kwargs):
        self.saved_to.append(save_directory)


def test_weight_export_reaches_a_ddp_wrapped_module(tmp_path):
    """DDP does not forward ``save_pretrained``, so exporting a DDP module raised."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from veomni.models.seed_omni.accelerated.utils.modules import save_module_subdirectory

    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'rendezvous'}", world_size=1, rank=0)
    try:
        inner = _Exportable()
        save_module_subdirectory("module_a", DistributedDataParallel(inner), str(tmp_path), save_module_weights=True)
    finally:
        dist.destroy_process_group()

    assert inner.saved_to == [str(tmp_path / "module_a")]


def test_runtime_calls_a_wrapped_module_through_its_runtime():
    """OmniModel holds the bare module; DDP syncs gradients only if its own forward runs."""
    model = _model()
    wrapped = _DdpStyleWrapper(model.modules_dict["module_a"])
    module_runtime = SimpleNamespace(model=wrapped, _scoped=nullcontext)
    runtime = OmniModelRuntime(model, module_runtimes={"module_a": module_runtime})

    runtime.forward({})
    assert wrapped.calls == 1

    runtime.reset()
    ctx: dict = {}
    runtime.generate(ctx)
    assert wrapped.calls == 2
    assert ctx["trace"][:3] == ["module_a:pre", "module_a:generate", "module_a:post"]
    assert dict(runtime.named_omni_modules())["module_a"] is model.modules_dict["module_a"]
