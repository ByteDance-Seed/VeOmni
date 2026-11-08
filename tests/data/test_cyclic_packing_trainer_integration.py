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

"""The special loader is opt-in and requires the checkpoint identity chain."""

import inspect
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from datasets import Dataset

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import DataloaderConfig, TrainingArguments
from veomni.data.dataset import MappingDataset
from veomni.trainer import base as trainer_base


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _trainer():
    trainer = trainer_base.BaseTrainer.__new__(trainer_base.BaseTrainer)
    config = DataloaderConfig(
        type="cyclic_packed",
        num_workers=0,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
    )
    checkpoint = SimpleNamespace(save_async=False, load_path=None)
    data = SimpleNamespace(
        dataloader=config,
        data_type="conversation",
        datasets_type="mapping",
        enable_multisource=False,
        max_seq_len=262144,
        dyn_bsz_buffer_size=16,
    )
    train = SimpleNamespace(
        cyclic_input_contract="sha256:" + "a" * 64,
        cyclic_input_deterministic=True,
        checkpoint=checkpoint,
        micro_batch_size=1,
        global_batch_size=64,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        bsz_warmup_ratio=0,
        num_train_epochs=3,
    )
    trainer.args = SimpleNamespace(data=data, train=train, train_steps=5)
    trainer.model = SimpleNamespace(
        optimizer=SimpleNamespace(_is_multi_optimizer=True),
        checkpoint=SimpleNamespace(_last_saved_optimizer_generation=None),
    )
    trainer.train_dataset = MappingDataset(Dataset.from_dict({"sample": [1, 2]}))
    trainer.collate_fn = object()
    return trainer


def test_sft_explicitly_opts_in_and_dpo_default_conversation_is_rejected(monkeypatch):
    from veomni.data import cyclic_packing

    monkeypatch.setattr(
        trainer_base, "get_parallel_state", lambda: SimpleNamespace(dp_size=1, tp_size=1, sp_size=2, sp_group="sp")
    )
    monkeypatch.setattr(cyclic_packing, "_StatelessMainCollator", lambda collator: collator)
    captured = {}
    monkeypatch.setattr(cyclic_packing, "CyclicPackedLoader", lambda *args, **kwargs: captured.update(kwargs))

    # DPO also defaults to data_type=conversation; it must not opt in just
    # because that field happens to match the SFT configuration.
    with pytest.raises(ValueError, match="map-style SFT"):
        _trainer()._build_dataloader()
    trainer = _trainer()
    trainer._build_dataloader(allow_cyclic=True)
    assert captured["micro_batches_per_step"] == 64
    assert captured["max_seq_len"] == 262144
    assert captured["train_steps"] == 5 and captured["train_epochs"] == 3
    assert captured["source_fingerprint"] == trainer.train_dataset._data._fingerprint


def test_unproven_optimizer_or_async_save_is_rejected(monkeypatch):
    monkeypatch.setattr(
        trainer_base, "get_parallel_state", lambda: SimpleNamespace(dp_size=1, tp_size=1, sp_size=2, sp_group="sp")
    )
    trainer = _trainer()
    trainer.model.optimizer._is_multi_optimizer = False
    with pytest.raises(ValueError, match="generation-aware MultiOptimizer"):
        trainer._build_dataloader(allow_cyclic=True)
    trainer = _trainer()
    trainer.args.train.checkpoint.save_async = True
    with pytest.raises(ValueError, match="map-style SFT"):
        trainer._build_dataloader(allow_cyclic=True)


def test_legacy_subclass_dataloader_override_remains_compatible(monkeypatch):
    monkeypatch.setattr(trainer_base, "use_parallel_state", lambda _state: nullcontext())

    class LegacyTrainer(trainer_base.BaseTrainer):
        @staticmethod
        def _setup(_args):
            return "cpu"

        def _build_model_runtime(self):
            return SimpleNamespace(parallel_state=None)

        def _build_data_transform(self):
            pass

        def _build_dataset(self):
            pass

        def _build_collate_fn(self):
            pass

        def _build_dataloader(self):
            self.loaded = True

        def _build_lr_scheduler(self):
            pass

        def _build_training_context(self, _model):
            pass

        def _init_callbacks(self):
            pass

    assert LegacyTrainer(SimpleNamespace()).loaded


def test_default_dataloader_kwargs_remain_compatible_with_native_builder():
    from veomni.data.data_loader import build_native_dataloader

    # Omni removes these two control fields before passing the remainder on.
    kwargs = asdict(DataloaderConfig())
    kwargs.pop("type")
    kwargs.pop("use_background_prefetcher")
    assert set(kwargs) <= set(inspect.signature(build_native_dataloader).parameters)
    assert next(iter(inspect.signature(TrainingArguments).parameters)) == "dyn_bsz"
    assert (
        inspect.signature(TrainingArguments).parameters["cyclic_input_contract"].kind is inspect.Parameter.KEYWORD_ONLY
    )
