# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``VeOmniModelRuntime`` drives a model with no trainer anywhere in sight.

The point of the class is that the model-bound half of a job stands on its
own, so the integration tests here go build -> parallelize -> optimizer ->
clip through the runtime alone. They wrap with DDP rather than FSDP2: a
one-rank mesh reports ``fsdp_enabled=False`` and the parallelize path then
rejects meta init, so FSDP2 needs the multi-rank suites. The seam tests below
need no distribution at all.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from veomni.arguments import (
    AcceleratorConfig,
    FSDPConfig,
    ModelArguments,
    ModelRuntimeArguments,
)
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    use_parallel_state,
)
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer.base import BaseTrainer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

from ..tools.training_utils import make_eager_ops_config, unbuilt_runtime


TOY_CONFIG = "./tests/toy_config/qwen3_toy"


def train_args(load_path=None):
    """The job-wide slice a runtime reads: where checkpoints go, and where to resume."""
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            load_path=load_path,
            save_path=None,
            output_dir=None,
            manager="dcp",
            save_async=False,
            dcp_save_to_lowest_rank=False,
        )
    )


@pytest.fixture
def single_rank_group(tmp_path):
    """One rank on the local accelerator, enough to drive the parallelize path."""
    if not torch.accelerator.is_available():
        pytest.skip("needs an accelerator for the FSDP2 meta-init path")

    get_torch_device().set_device(f"{get_device_type()}:0")
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{tmp_path / 'rendezvous'}",
        world_size=1,
        rank=0,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


def make_runtime(name="base", **kwargs):
    # DDP on the local accelerator rather than FSDP2 on meta: a one-rank mesh
    # reports ``fsdp_enabled=False``, and the parallelize path rejects meta
    # init without FSDP. DDP exercises the same runtime sequence at one rank.
    args = ModelArguments(
        config_path=TOY_CONFIG,
        ops_implementation=make_eager_ops_config(),
        accelerator=AcceleratorConfig(
            init_device=get_device_type(),
            fsdp_config=FSDPConfig(fsdp_mode="ddp"),
        ),
    )
    return VeOmniModelRuntime(args, name, train=train_args(**kwargs))


def test_a_runtime_builds_and_optimizes_a_model_without_a_trainer(single_rank_group):
    # Constructing it is the whole build: mesh, model, parallelize, optimizer.
    runtime = make_runtime()

    assert runtime.model is not None
    assert runtime.model_config is runtime.model.config

    runtime.build_lr_scheduler(total_steps=10)

    assert runtime.optimizer is not None
    assert runtime.lr_scheduler is not None
    # The optimizer must actually cover the model it was built from.
    optimized = {id(p) for group in runtime.optimizer.param_groups for p in group["params"]}
    assert optimized and optimized <= {id(p) for p in runtime.model.parameters()}


def test_the_runtime_reads_its_mesh_from_the_registry_under_its_own_name(single_rank_group):
    runtime = make_runtime(name="vision_tower")

    # A lookup rather than a stored handle, so the registry stays authoritative.
    assert runtime.parallel_state is not None
    assert runtime.parallel_state.dp_size == 1

    with use_parallel_state(runtime.model_name):
        assert get_parallel_state() is runtime.parallel_state


def test_building_a_model_leaves_the_ambient_state_as_it_found_it(single_rank_group):
    # Everything a trainer builds after the model — dataloader, lr scheduler,
    # callbacks — reads the ambient state rather than opening a scope of its own,
    # which is only safe because the runtime's build scope hands it back. The
    # job state is FSDP2 and the runtime's is DDP so the two are distinguishable
    # at one rank (``dp_mode`` is part of the topology cache key).
    job_state = _init_parallel_state(dp_mode="fsdp2", name="base")
    assert get_parallel_state() is job_state

    runtime = make_runtime(name="vision_tower")

    assert runtime.parallel_state is not job_state, "the runtime built over a mesh of its own"
    assert get_parallel_state() is job_state


def test_clip_grad_norm_falls_back_to_the_models_own_max_grad_norm(single_rank_group):
    runtime = make_runtime()
    runtime.args.optimizer.max_grad_norm = 1.0

    for param in runtime.model.parameters():
        param.grad = torch.ones_like(param)

    norm = runtime.clip_grad_norm()

    assert float(norm) > 1.0, "the pre-clip norm is what gets reported"
    clipped = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in runtime.model.parameters()])
    )
    assert float(clipped) <= 1.0 + 1e-3, "gradients must actually be scaled down"


def test_a_lora_free_config_leaves_the_model_untouched(single_rank_group):
    runtime = make_runtime()

    before = runtime.model
    runtime.setup_lora()

    assert runtime.model is before


class TestHowATrainerHoldsItsModel:
    """The trainer composes a runtime; it does not inherit one."""

    def test_a_trainer_hands_the_runtime_its_own_argument_shape(self, monkeypatch):
        # The trainer unpacks its job config: the runtime gets the model's own
        # arguments, the name the one ParallelState of a single-model job
        # registers under, and the job-wide half it still needs. Built through
        # ``__new__`` exactly as the composed trainers build it; the constructor
        # is stubbed down to what it records, because this is about what the
        # trainer hands over, not about the model that gets built out of it.
        args = SimpleNamespace(
            model=SimpleNamespace(name="model args"),
            data=SimpleNamespace(chat_template="chatml"),
            train=SimpleNamespace(checkpoint=SimpleNamespace(load_path="/ckpt")),
        )
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.args = args

        def record_only(runtime, args, model_name="base", *, train=None, chat_template_name=None):
            runtime.args = args
            runtime.model_name = model_name
            runtime.train = train
            runtime.chat_template_name = chat_template_name

        monkeypatch.setattr(VeOmniModelRuntime, "__init__", record_only)

        runtime = trainer.build_model_runtime()

        assert isinstance(runtime, VeOmniModelRuntime)
        assert runtime.args is args.model, "the runtime is handed its own slice, not the job"
        assert runtime.model_name == "base"
        assert runtime.train is args.train, "and the job-wide half it still needs"
        assert runtime.chat_template_name == "chatml", (
            "including which chat template to build, since only the runtime holds the preprocessor to build it from"
        )

    def test_a_trainer_is_not_itself_a_model_runtime(self):
        # The regression this guards: BaseTrainer used to *inherit* the runtime,
        # which made every model method look like a trainer method.
        assert not issubclass(BaseTrainer, VeOmniModelRuntime)

    def test_optimizer_and_scheduler_live_on_the_runtime_not_the_trainer(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"))

        trainer.model.optimizer = "optimizer"
        trainer.model.lr_scheduler = "scheduler"

        assert trainer.model.optimizer == "optimizer"
        assert trainer.model.lr_scheduler == "scheduler"
        assert not hasattr(BaseTrainer, "optimizer")
        assert not hasattr(BaseTrainer, "lr_scheduler")
        assert not hasattr(BaseTrainer, "model_config")

    def test_a_standalone_runtime_defaults_to_the_single_model_name(self):
        args = ModelRuntimeArguments(model_path="somewhere")
        runtime = unbuilt_runtime(args)

        assert runtime.args is args
        assert runtime.model_name == "base"

    def test_a_named_runtime_carries_its_own_name(self):
        # Sibling models in one job register their meshes under distinct names.
        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"), name="audio")

        assert runtime.model_name == "audio"

    def test_a_fresh_run_still_materializes_hf_weights(self):
        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"), train=train_args())

        assert runtime.skip_hf_weight_load is False

    def test_a_full_resume_skips_the_second_memory_peak(self):
        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"), train=train_args(load_path="/ckpt"))

        assert runtime.skip_hf_weight_load is True

    def test_a_lora_resume_still_needs_the_hf_base(self):
        runtime = unbuilt_runtime(
            ModelRuntimeArguments(model_path="somewhere", lora_config={"rank": 8}),
            train=train_args(load_path="/ckpt"),
        )

        assert runtime.skip_hf_weight_load is False


class TestWhereTheLoaderReadsTheConfigFrom:
    """Every unit answers ``config_path``, so the loader never asks what it holds."""

    def test_a_module_falls_back_to_its_weights_folder(self):
        # A module inside a composed checkpoint is addressed by its own subfolder,
        # so it never configures a config path separately.
        assert ModelRuntimeArguments(model_path="somewhere").config_path == "somewhere"

    def test_a_whole_model_honours_a_separate_config_path(self):
        # Toy-config runs rely on this: architecture from a local json, weights
        # (and tokenizer) from somewhere else entirely.
        assert ModelArguments(model_path="weights", config_path="cfg").config_path == "cfg"

    def test_a_whole_model_falls_back_to_its_weights_path(self):
        assert ModelArguments(model_path="weights").config_path == "weights"

    def test_the_tokenizer_follows_the_config_the_base_settled(self):
        assert ModelArguments(model_path="weights", config_path="cfg").tokenizer_path == "cfg"


class TestHowAJobOverridesItsPreprocessor:
    """``processor_config`` is to the preprocessor what ``model_config`` is to the architecture."""

    @staticmethod
    def _record_loader(monkeypatch, seen):
        def fake_build_processor(path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = kwargs
            return object()

        monkeypatch.setattr("veomni.models.auto.build_processor", fake_build_processor)

    def test_a_run_can_resize_what_the_repository_ships(self, monkeypatch):
        # A pixel budget belongs to the run, not to the model class: the same
        # checkpoint is trained at different resolutions by different jobs.
        seen = {}
        self._record_loader(monkeypatch, seen)
        size = {"shortest_edge": 3136, "longest_edge": 602112}
        runtime = unbuilt_runtime(ModelArguments(model_path="somewhere", processor_config={"size": size}))

        runtime.build_model_assets()

        assert seen == {"path": "somewhere", "kwargs": {"size": size}}

    def test_a_run_that_overrides_nothing_leaves_the_repository_alone(self, monkeypatch):
        # No kwargs at all, so the preprocessor the repository ships is
        # authoritative — nothing silently narrows it behind the job's back.
        seen = {}
        self._record_loader(monkeypatch, seen)
        runtime = unbuilt_runtime(ModelArguments(model_path="somewhere"))

        runtime.build_model_assets()

        assert seen["kwargs"] == {}


class TestWhatTheRuntimeAsksTheModel:
    """Build policy that follows from how a model is built travels with the model."""

    def test_a_model_that_wraps_itself_is_used_verbatim(self):
        # The escape hatch for models the generic GPU-materializing loader has no
        # hook for — a MoE backbone streaming EP-sharded experts to CPU, say.
        wrapped = nn.Linear(1, 1)
        seen = {}

        class SelfWrapping(nn.Module):
            def build_parallelize_model(self, *, weights_path, args):
                seen["weights_path"] = weights_path
                seen["args"] = args
                return wrapped

        args = ModelRuntimeArguments(model_path="somewhere")
        runtime = unbuilt_runtime(args, train=train_args())
        runtime.model = SelfWrapping()

        runtime.build_parallelized_model()

        assert runtime.model is wrapped
        assert seen == {"weights_path": "somewhere", "args": args}

    def test_an_ordinary_model_leaves_the_generic_path_alone(self, monkeypatch):
        # No hook on the model means the generic loader still runs; the hook is
        # an override, not a gate the ordinary path has to pass through.
        generically_wrapped = nn.Linear(1, 1)
        monkeypatch.setattr(
            "veomni.distributed.torch_parallelize.build_parallelize_model",
            lambda model, **kwargs: generically_wrapped,
        )
        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"), train=train_args())
        runtime.model = nn.Linear(1, 1)

        runtime.build_parallelized_model()

        assert runtime.model is generically_wrapped

    def test_a_model_gets_to_freeze_itself(self):
        # Which of its own parts a model keeps frozen is the model's knowledge,
        # not the runtime's — an omni module reading its own ``config.freeze``.
        class SelfFreezing(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.frozen = False

            def freeze_model(self):
                self.frozen = True

        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere"))
        runtime.model = SelfFreezing()

        runtime.freeze_model()

        assert runtime.model.frozen is True

    def test_a_model_that_wraps_its_own_lora_is_used_verbatim(self):
        wrapped = nn.Linear(1, 1)

        class SelfAdapting(nn.Module):
            def setup_lora(self, lora_config):
                return wrapped

        runtime = unbuilt_runtime(ModelRuntimeArguments(model_path="somewhere", lora_config={"rank": 8}))
        runtime.model = SelfAdapting()

        runtime.setup_lora()

        assert runtime.model is wrapped

    def test_the_hooks_are_named_for_what_a_model_does_not_for_the_asking(self):
        # A model declares ``setup_lora`` / ``freeze_model`` /
        # ``build_parallelize_model``; the ``customized_`` prefix belongs to the
        # runtime's question, so a model carrying the prefixed name is ignored
        # and the generic adapter is built over it instead.
        from veomni.lora import VeOmniLoraModel

        class PrefixedByMistake(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def customized_setup_lora(self, lora_config):
                raise AssertionError("the prefixed name is the runtime's, not the model's")

        args = ModelRuntimeArguments(
            model_path="somewhere",
            lora_config={"rank": 8, "alpha": 16, "lora_modules": ["proj"]},
        )
        runtime = unbuilt_runtime(args)
        runtime.model = PrefixedByMistake()

        runtime.setup_lora()

        assert isinstance(runtime.model, VeOmniLoraModel)
