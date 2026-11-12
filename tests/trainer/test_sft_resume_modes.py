# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer import text_trainer, vlm_trainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("module,trainer_name", [(text_trainer, "TextTrainer"), (vlm_trainer, "VLMTrainer")])
@pytest.mark.parametrize(
    "data_type,expected", [("plaintext", True), ("conversation", True), ("classification", False)]
)
def test_exact_sft_flag_is_set_before_callbacks(monkeypatch, module, trainer_name, data_type, expected):
    class Base:
        def _setup(self, args):
            return "cpu"

        def _build_model_runtime(self):
            return SimpleNamespace(parallel_state=None)

        def _build_dataset(self):
            pass

        def _build_collate_fn(self):
            pass

        def _build_dataloader(self):
            pass

        def _build_lr_scheduler(self):
            pass

        def _build_training_context(self, model):
            pass

        def _init_callbacks(self):
            self.flag_seen_by_callbacks = self._sft_dynamic_exact_resume

    trainer_class = getattr(module, trainer_name)
    monkeypatch.setattr(module, "BaseTrainer", Base)
    monkeypatch.setattr(module, "use_parallel_state", lambda _: nullcontext())
    monkeypatch.setattr(trainer_class, "_build_data_transform", lambda self: None)
    if trainer_name == "VLMTrainer":
        monkeypatch.setattr(trainer_class, "_build_model_runtime", lambda self: SimpleNamespace(parallel_state=None))
        monkeypatch.setattr(trainer_class, "_build_collate_fn", lambda self: None)
    trainer = trainer_class(SimpleNamespace(data=SimpleNamespace(data_type=data_type)))
    assert trainer.base.flag_seen_by_callbacks is expected
