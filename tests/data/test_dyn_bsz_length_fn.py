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
"""A trainer-supplied ``dyn_bsz_length_fn`` has to reach whichever runtime spends the budget.

``dyn_bsz_runtime: main`` batches in ``TextBatchingStrategy``, ``worker`` in
``DynamicBatchingSizeDataset``. A measure threaded to one and not the other would
leave half the configurations costing samples the old way; both produce batches
that look right, so the only symptom would be step times that do not match the budget.
"""

import types
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from utils import DummyDataset, process_dummy_example

import veomni.trainer.base as base_trainer
from veomni.arguments.arguments_types import DataloaderConfig
from veomni.data import build_dataloader, build_dataset
from veomni.trainer.base import BaseTrainer


def double_attention_mask_cost(item):
    return 2 * int(item["attention_mask"].sum())


@pytest.fixture(scope="module")
def dummy_dataset():
    dummy = DummyDataset(size=40, num_shard=1, dataset_name="ci_dyn_bsz_length_fn")
    yield dummy
    dummy.clean_cache()


def _micro_batches(dataset_path, runtime, monkeypatch, length_fn=None, count_mode="total"):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    single = types.SimpleNamespace(dp_size=1, dp_rank=0, sp_enabled=False, sp_size=1, sp_rank=0)
    for module in (m_dl, m_ds, m_col):
        monkeypatch.setattr(module, "get_parallel_state", lambda: single)

    micro_batch_size, global_batch_size, max_seq_len = 2, 8, 100
    return build_dataloader(
        "native",
        dataset=build_dataset(
            dataset_name="iterable",
            train_path=dataset_path,
            transform=partial(process_dummy_example, max_seq_len=max_seq_len),
            seed=0,
        ),
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        dataloader_batch_size=1 if runtime == "main" else global_batch_size // micro_batch_size,
        max_seq_len=max_seq_len,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime=runtime,
        dyn_bsz_count_mode=count_mode,
        dyn_bsz_buffer_size=1,
        dyn_bsz_length_fn=length_fn,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )


def _samples_taken(dataloader):
    return sum(len(micro_batch["id"]) for micro_batch in next(iter(dataloader)))


@pytest.mark.parametrize("runtime", ["main", "worker"])
def test_a_measure_charging_double_halves_what_a_step_holds(runtime, dummy_dataset, monkeypatch):
    """Bounded on both sides: unchanged means the measure reached the bookkeeping but
    not the selection, a quarter means it was applied twice."""
    default = _samples_taken(_micro_batches(dummy_dataset.save_path, runtime, monkeypatch))
    doubled = _samples_taken(
        _micro_batches(dummy_dataset.save_path, runtime, monkeypatch, length_fn=double_attention_mask_cost)
    )
    assert 0.35 < doubled / default < 0.65


def test_a_measure_cannot_be_combined_with_effective_count_mode(dummy_dataset, monkeypatch):
    with pytest.raises(ValueError, match="dyn_bsz_count_mode"):
        _micro_batches(
            dummy_dataset.save_path,
            "main",
            monkeypatch,
            length_fn=double_attention_mask_cost,
            count_mode="effective",
        )


def _kwargs_passed_by_trainer(length_fn):
    stub = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(
                dataloader=DataloaderConfig(num_workers=0),
                max_seq_len=128,
                dyn_bsz_buffer_size=1,
            ),
            train=SimpleNamespace(
                micro_batch_size=1,
                global_batch_size=1,
                dataloader_batch_size=1,
                bsz_warmup_ratio=0.0,
                bsz_warmup_init_mbtoken=8,
                dyn_bsz=True,
                dyn_bsz_runtime="main",
                dyn_bsz_count_mode="total",
                dyn_bsz_physical_overflow_ratio=1.5,
                seed=0,
                checkpoint=SimpleNamespace(save_steps=1),
            ),
            train_steps=1,
        ),
        train_dataset=None,
        collate_fn=None,
        dyn_bsz_length_fn=length_fn,
    )
    seen = {}
    with patch.object(base_trainer, "build_dataloader", side_effect=lambda **kwargs: seen.update(kwargs)):
        BaseTrainer._build_dataloader(stub)
    return seen


def test_the_trainer_forwards_its_measure_to_the_dataloader():
    assert _kwargs_passed_by_trainer(double_attention_mask_cost)["dyn_bsz_length_fn"] is double_attention_mask_cost


def test_a_trainer_that_sets_no_measure_gets_the_default():
    assert BaseTrainer.dyn_bsz_length_fn is None
    assert _kwargs_passed_by_trainer(None)["dyn_bsz_length_fn"] is None
