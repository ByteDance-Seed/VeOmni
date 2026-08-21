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

from veomni.arguments import (
    AcceleratorConfig,
    FSDPConfig,
    ModelArguments,
    ModelRuntimeArguments,
)
from veomni.distributed.parallel_state import clear_parallel_state, get_parallel_state
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer.base import BaseTrainer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

from ..tools.training_utils import make_eager_ops_config


TOY_CONFIG = "./tests/toy_config/qwen3_toy"


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
    return VeOmniModelRuntime(args, name=name, **kwargs)


def test_a_runtime_builds_and_optimizes_a_model_without_a_trainer(single_rank_group):
    runtime = make_runtime()
    runtime.register_parallel_state()

    runtime.build_model()
    assert runtime.model is not None
    assert runtime.model_config is runtime.model.config

    runtime.build_parallelized_model()
    runtime.build_optimizer()
    runtime.build_lr_scheduler(total_steps=10)

    assert runtime.optimizer is not None
    assert runtime.lr_scheduler is not None
    # The optimizer must actually cover the model it was built from.
    optimized = {id(p) for group in runtime.optimizer.param_groups for p in group["params"]}
    assert optimized and optimized <= {id(p) for p in runtime.model.parameters()}


def test_the_runtime_reads_its_mesh_from_the_registry_under_its_own_name(single_rank_group):
    runtime = make_runtime(name="vision_tower")
    runtime.register_parallel_state()

    # A lookup rather than a stored handle, so the registry stays authoritative.
    assert runtime.parallel_state is not None
    assert runtime.parallel_state.dp_size == 1

    with runtime.scoped():
        assert get_parallel_state() is runtime.parallel_state


def test_clip_grad_norm_falls_back_to_the_models_own_max_grad_norm(single_rank_group):
    runtime = make_runtime()
    runtime.model_args.optimizer.max_grad_norm = 1.0
    runtime.register_parallel_state()
    runtime.build_model()
    runtime.build_parallelized_model()

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
    runtime.register_parallel_state()
    runtime.build_model()

    before = runtime.model
    runtime.setup_lora()

    assert runtime.model is before


class TestTheThreeSeams:
    """The seams are all a subclass must supply to reuse the whole build."""

    def test_a_trainer_routes_every_seam_at_its_own_argument_shape(self):
        # BaseTrainer nests the runtime args under ``model`` and never calls
        # ``VeOmniModelRuntime.__init__``, so the seams have to read off
        # ``self.args``. Built through ``__new__`` exactly as the composed
        # trainers build it, which is what would break if an override were lost.
        args = SimpleNamespace(
            model=SimpleNamespace(name="model args"),
            train=SimpleNamespace(checkpoint=SimpleNamespace(load_path="/ckpt")),
        )
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.args = args

        assert trainer.model_args is args.model
        assert trainer.runtime_name == "base"
        assert trainer.checkpoint_load_path == "/ckpt"

    def test_a_standalone_runtime_defaults_every_seam(self):
        args = ModelRuntimeArguments(model_path="somewhere")
        runtime = VeOmniModelRuntime(args)

        assert runtime.model_args is args
        assert runtime.runtime_name == "base"
        assert runtime.checkpoint_load_path is None

    def test_a_named_runtime_carries_its_name_and_resume_path(self):
        args = ModelRuntimeArguments(model_path="somewhere")
        runtime = VeOmniModelRuntime(args, name="audio", checkpoint_load_path="/ckpt")

        assert runtime.runtime_name == "audio"
        assert runtime.checkpoint_load_path == "/ckpt"

    def test_a_fresh_run_still_materializes_hf_weights(self):
        runtime = VeOmniModelRuntime(ModelRuntimeArguments(model_path="somewhere"))

        assert runtime.skip_hf_weight_load is False

    def test_a_full_resume_skips_the_second_memory_peak(self):
        runtime = VeOmniModelRuntime(
            ModelRuntimeArguments(model_path="somewhere"),
            checkpoint_load_path="/ckpt",
        )

        assert runtime.skip_hf_weight_load is True

    def test_a_lora_resume_still_needs_the_hf_base(self):
        runtime = VeOmniModelRuntime(
            ModelRuntimeArguments(model_path="somewhere", lora_config={"rank": 8}),
            checkpoint_load_path="/ckpt",
        )

        assert runtime.skip_hf_weight_load is False


class TestWhereTheLoaderReadsTheConfigFrom:
    """A module is addressed by its own subfolder; a whole model has a config path."""

    def test_a_module_uses_its_weights_folder(self):
        assert ModelRuntimeArguments(model_path="somewhere").foundation_config_path == "somewhere"

    def test_a_whole_model_honours_a_separate_config_path(self):
        assert ModelArguments(model_path="weights", config_path="cfg").foundation_config_path == "cfg"

    def test_a_whole_model_falls_back_to_its_weights_path(self):
        assert ModelArguments(model_path="weights").foundation_config_path == "weights"
