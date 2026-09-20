# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""The trainer composes a runtime; it does not inherit one."""

from types import SimpleNamespace

from tests.tools.training_utils import unbuilt_runtime
from veomni.arguments import ModelArguments
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer.base import BaseTrainer


def _train_args(load_path=None):
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


class TestHowATrainerHoldsItsModel:
    def test_a_trainer_hands_the_runtime_its_own_argument_shape(self, monkeypatch):
        # The trainer unpacks its job config: the runtime gets the model's own
        # arguments, the name the one ParallelState of a single-model job
        # registers under, and the job-wide half it still needs. Built through
        # ``__new__`` exactly as the composed trainers build it; the constructor
        # is stubbed down to what it records, because this is about what the
        # trainer hands over, not about the model that gets built out of it.
        args = SimpleNamespace(
            model=SimpleNamespace(name="model args", chat_template="chatml"),
            data=SimpleNamespace(),
            train=SimpleNamespace(checkpoint=SimpleNamespace(load_path="/ckpt")),
        )
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.args = args

        def record_only(runtime, args, model_name="base", *, train=None):
            runtime.args = args
            runtime.model_name = model_name
            runtime.train_args = train

        monkeypatch.setattr(VeOmniModelRuntime, "__init__", record_only)

        runtime = trainer._build_model_runtime()

        assert isinstance(runtime, VeOmniModelRuntime)
        assert runtime.args is args.model, "the runtime is handed its own slice, not the job"
        assert runtime.model_name == "base"
        assert trainer._build_model_runtime("policy").model_name == "policy"
        assert runtime.train_args is args.train, "and the job-wide half it still needs"
        assert runtime.args.chat_template == "chatml", (
            "which chat template to build lives on the model slice, since only the runtime holds the preprocessor"
        )

    def test_a_trainer_is_not_itself_a_model_runtime(self):
        # BaseTrainer composes a runtime; it is not one.
        assert not issubclass(BaseTrainer, VeOmniModelRuntime)

    def test_optimizer_and_scheduler_live_on_the_runtime_not_the_trainer(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = unbuilt_runtime(ModelArguments(model_path="somewhere"))

        trainer.model.optimizer = "optimizer"
        trainer.model.lr_scheduler = "scheduler"

        assert trainer.model.optimizer == "optimizer"
        assert trainer.model.lr_scheduler == "scheduler"
        assert not hasattr(BaseTrainer, "optimizer")
        assert not hasattr(BaseTrainer, "lr_scheduler")
        assert not hasattr(BaseTrainer, "model_config")

    def test_a_standalone_runtime_defaults_to_the_single_model_name(self):
        args = ModelArguments(model_path="somewhere")
        runtime = unbuilt_runtime(args)

        assert runtime.args is args
        assert runtime.model_name == "base"

    def test_a_named_runtime_carries_its_own_name(self):
        # Sibling models in one job register their meshes under distinct names.
        runtime = unbuilt_runtime(ModelArguments(model_path="somewhere"), name="audio")

        assert runtime.model_name == "audio"

    def test_a_fresh_run_still_materializes_hf_weights(self):
        runtime = unbuilt_runtime(ModelArguments(model_path="somewhere"), train=_train_args())

        assert runtime.skip_hf_weight_load is False

    def test_a_full_resume_skips_the_second_memory_peak(self):
        runtime = unbuilt_runtime(ModelArguments(model_path="somewhere"), train=_train_args(load_path="/ckpt"))

        assert runtime.skip_hf_weight_load is True

    def test_a_lora_resume_still_needs_the_hf_base(self):
        runtime = unbuilt_runtime(
            ModelArguments(model_path="somewhere", lora_config={"rank": 8}),
            train=_train_args(load_path="/ckpt"),
        )

        assert runtime.skip_hf_weight_load is False
