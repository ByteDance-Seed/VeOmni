import argparse
import copy
import dataclasses
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import field
from functools import partial
from typing import Any, Dict, List, Literal, cast
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
from tools import resolve_ops_overrides
from tools.launch_utils import find_free_port
from torch.utils.data import IterableDataset
from transformers import PretrainedConfig
from utils import (
    DummyDataset,
    FakeModel,
    StepAwareResumeCheckpointerCallback,
    compare_global_batch,
    compare_metrics,
    mock_empty_cache,
    setup_test_distributed,
)

from veomni.arguments import VeOmniArguments, parse_args
from veomni.data import build_dataloader
from veomni.data.data_collator import MainCollator
from veomni.data.dataset import (
    DynamicBatchingSizeDataset,
    InterleavedIterableDataset,
    WeightedMultiSourceDataset,
    build_dataset,
)
from veomni.data.source_metadata import (
    attach_source_metadata,
    make_source_metadata,
    normalize_packed_source_metadata,
    normalize_source_metadata,
)
from veomni.distributed.parallel_state import get_parallel_state
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.utils import helper
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import get_device_type
from veomni.utils.dist_utils import all_reduce
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)


def _torch_shm_manager_executable() -> bool:
    torch_dir = os.path.dirname(torch.__file__)
    shm_manager = os.path.join(torch_dir, "bin", "torch_shm_manager")
    return os.path.exists(shm_manager) and os.access(shm_manager, os.X_OK)


class MockIterableDataset(IterableDataset):
    def __init__(self, data, name="mock"):
        self.data = list(data)
        self.name = name
        self._state = {"consumed": 0}

    def __iter__(self):
        for item in self.data:
            self._state["consumed"] += 1
            yield item

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


def _convert_list_to_tensor_fn(sample: Dict[str, Any], max_seq_len: int, **kwargs) -> Dict[str, Any]:
    """Convert list fields in the sample to truncated tensors."""
    converted = {}
    for k, v in sample.items():
        if isinstance(v, list):
            converted[k] = torch.tensor(v[:max_seq_len], dtype=torch.long)
        else:
            converted[k] = v
    return converted


class TrainerTest(BaseTrainer):
    gt_data_list: List[List[Dict[str, Any]]] = []
    pred_data_list: List[List[Dict[str, Any]]] = []
    golden_env_metrics: Dict[str, Any] = {}
    resume_dcp_path: str
    tmp_yaml_path: str

    save_epoch, save_step = 1, None
    is_resume_train: bool = False
    multisource_names = ["dataset_a", "dataset_b"]
    multisource_weights = [0.5, 0.5]

    def _setup(self):
        self.device, _ = setup_test_distributed(self.args)

        self.multisource_datasets = [DummyDataset(size=100, dataset_name=name) for name in self.multisource_names]
        self.multisource_paths = [dataset.save_path for dataset in self.multisource_datasets]

        multisource_config = dict(
            sources=self.multisource_paths,
            names=self.multisource_names,
            schedule=[dict(schedule_type="const", weights=self.multisource_weights)],
            level="token",
            stopping_strategy="all_exhausted",
            upstream_sharded=False,
        )
        self.tmp_train_path = os.path.join(get_cache_dir("./tmp_train_path.yaml"), "tmp_train_path.yaml")
        if dist.get_rank() == 0:
            with open(self.tmp_train_path, "w") as f:
                yaml.safe_dump(multisource_config, f)
        if dist.is_initialized():
            dist.barrier()

        self.args.data.train_path = self.tmp_train_path
        self.args.data.enable_multisource = True
        self.args.data.dataset_name = "veomni_weighted_multisource"

        self.args.train.num_train_epochs = 3

        # we have to add a shuffle field to the args because it does not have one,
        # and we need it to control the behavior of HF datasets,
        # because it is shuffled it will store samples in a buffer,
        # and such a buffer will be en  which will be discarded during resuming,
        # thus causing the resumed training to see different samples from the original training
        shuffle_field = field(default=True)
        shuffle_field.name = "shuffle"
        shuffle_field.type = bool
        shuffle_field._field_type = dataclasses._FIELD
        self.args.data.__dataclass_fields__["shuffle"] = shuffle_field
        self.args.data.shuffle = False

    def _freeze_model_module(self):
        pass

    def _build_model(self):
        self.model = FakeModel().to(get_device_type())
        self.model_config = PretrainedConfig()

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        self.data_transform = partial(_convert_list_to_tensor_fn, max_seq_len=self.args.data.max_seq_len)

    def _build_dataset(self):
        super()._build_dataset()

        dist.barrier()

        state = cast(WeightedMultiSourceDataset, self.train_dataset).state_dict()
        assert state["version"] == 1
        assert state["topology"]["stopping_strategy"] == "all_exhausted"
        assert state["topology"]["level"] == "token"
        assert state["topology"]["source_names"] == self.multisource_names
        source_ids = state["topology"]["source_ids"]
        assert len(source_ids) == len(self.multisource_names)
        assert len(set(source_ids)) == len(source_ids)
        assert sorted(state["runtime"]["avg_len_sum"].keys()) == sorted(source_ids)
        assert sorted(state["runtime"]["avg_len_count"].keys()) == sorted(source_ids)
        assert sorted(state["runtime"]["dataset_states"].keys()) == sorted(source_ids)

        self.args.compute_train_steps(dataset_length=None)
        self.train_steps = self.args.train_steps
        self.save_step = self.train_steps - 2

    def _build_dataloader(self):
        args = self.args
        global_batch_size = cast(int, args.train.global_batch_size)
        self.train_dataloader = build_dataloader(
            dataloader_type="native",
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_runtime=args.train.dyn_bsz_runtime,
            dyn_bsz_count_mode=args.train.dyn_bsz_count_mode,
            dyn_bsz_physical_overflow_ratio=args.train.dyn_bsz_physical_overflow_ratio,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_buffer_size=1,
            dyn_bsz_dataset_save_by_idx=False,
            num_workers=1,
            drop_last=args.data.dataloader.drop_last,
            # Force pin_memory=False: on NPU the pin_memory background thread
            # races with HCCL teardown (triggered inside destroy_distributed)
            # and aborts the process with SIGABRT. The test uses DummyDataset
            # so pin_memory provides no benefit anyway.
            pin_memory=False,
            prefetch_factor=args.data.dataloader.prefetch_factor,
        )

    def _build_parallelized_model(self):
        self.model.train()

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.train.optimizer.lr)

    def _build_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def _build_training_context(self):
        self.model_fwd_context = nullcontext()
        self.model_bwd_context = nullcontext()

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallbackTest(self)
        self.checkpointer_callback = StepAwareResumeCheckpointerCallback(self)
        self.check_callback = CheckCallback(self)
        self.state = TrainerState()

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.state.curr_step = self.start_step - 1
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        # we need to put the check callback before environ meter callback because the later one will remove 'ds_idx' and 'source_name' from it
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.check_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        self.state.global_step += 1
        self.state.curr_step += 1
        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        self.on_step_end(loss=0.0, loss_dict={}, grad_norm=0.0)

    def resume_train(self):
        self.is_resume_train = True
        super().train()

    def destroy_distributed(self):
        if self.is_resume_train:
            super().destroy_distributed()


class EnvironMeterCallbackTest(Callback):
    trainer: TrainerTest

    def __init__(self, trainer: TrainerTest) -> None:
        super().__init__(trainer)
        args = self.trainer.args
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=trainer.tmp_train_path,
            gc_steps=args.train.gc_steps,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        try:
            step_env_metrics = self.trainer.environ_meter.step(delta_time, global_step=state.global_step)
        except AttributeError as e:
            logger.warning(f"[rank{self.trainer.args.train.global_rank}] Skipping metrics: {e}")
            step_env_metrics = {}

        step_train_metrics = {"total_loss": loss}
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=get_parallel_state().fsdp_group)
            for k, v in step_train_metrics.items()
        }
        step_train_metrics["training/lr"] = max(self.trainer.lr_scheduler.get_last_lr())

        step_env_metrics.update(step_train_metrics)
        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        if state.global_step == 1:
            helper.print_example(example=micro_batches[0], rank=self.trainer.args.train.local_rank)
            for micro_batch in micro_batches:
                assert "ds_idx" in micro_batch
                assert "source_name" in micro_batch
                source_name = micro_batch["source_name"]
                if isinstance(source_name, list):
                    assert all(name in self.trainer.multisource_names for name in source_name)
                else:
                    assert source_name in self.trainer.multisource_names
                ds_idx = micro_batch["ds_idx"]
                if isinstance(ds_idx, torch.Tensor):
                    assert torch.all((ds_idx >= 0) & (ds_idx < len(self.trainer.multisource_names)))
                elif isinstance(ds_idx, list):
                    assert all(0 <= int(idx) < len(self.trainer.multisource_names) for idx in ds_idx)
                else:
                    assert 0 <= int(ds_idx) < len(self.trainer.multisource_names)
                assert micro_batch["attention_mask"].shape[-1] == micro_batch["input_ids"].shape[-1]
                assert micro_batch["labels"].shape[-1] == micro_batch["input_ids"].shape[-1]
                assert torch.all(micro_batch["attention_mask"] == 1)
                assert torch.all(
                    (micro_batch["labels"] == IGNORE_INDEX) | (micro_batch["labels"] == micro_batch["input_ids"])
                )

        if state.epoch == self.trainer.save_epoch and state.curr_step > self.trainer.save_step:
            if not self.trainer.is_resume_train:
                self.trainer.gt_data_list.append(micro_batches)
            else:
                self.trainer.pred_data_list.append(micro_batches)

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.is_resume_train:
            assert len(self.trainer.gt_data_list) == len(self.trainer.pred_data_list), (
                f"Batch count mismatch: gt={len(self.trainer.gt_data_list)}, pred={len(self.trainer.pred_data_list)}"
            )

            """
            gt_data_list_output = [
                [(list(set(micro_batch["input_ids"].tolist()[0])), micro_batch.get("ds_idx", None))
                for micro_batch in micro_batches
                ]
                for micro_batches in self.trainer.gt_data_list
            ]
            pred_data_list_output = [
                [(list(set(micro_batch["input_ids"].tolist()[0])), micro_batch.get("ds_idx", None))
                for micro_batch in micro_batches
                ]
                for micro_batches in self.trainer.pred_data_list
            ]
            logger.error(f"[rank{self.trainer.args.train.global_rank}] gt_data_list_output: {gt_data_list_output}")
            logger.error(f"[rank{self.trainer.args.train.global_rank}] pred_data_list_output: {pred_data_list_output}")
            """
            compare_global_batch(self.trainer.gt_data_list, self.trainer.pred_data_list)

            metrics = self.trainer.step_env_metrics
            metrics_resume = self.trainer.golden_env_metrics
            compare_metrics(metrics, metrics_resume)

            logger.info_rank0(
                "dataset_a: "
                f"{metrics.get('multi_source/consumed_chunk_num/dataset_a', 0)} "
                f"dataset_b: {metrics.get('multi_source/consumed_chunk_num/dataset_b', 0)}"
            )

            if dist.is_initialized():
                dist.barrier()

            if (not dist.is_initialized() or dist.get_rank() == 0) and os.path.exists(self.trainer.tmp_train_path):
                os.remove(self.trainer.tmp_train_path)
        else:
            self.trainer.golden_env_metrics = copy.deepcopy(self.trainer.step_env_metrics)


def _main_distributed_test():
    """Entry point for the distributed test launched by ``torchrun``."""
    _parser = argparse.ArgumentParser()
    _, remaining_argv = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    # Patch empty_cache to avoid AttributeError on CPU.
    with patch("veomni.utils.device.empty_cache", mock_empty_cache):
        args = parse_args(VeOmniArguments)
        trainer = TrainerTest(args)
        trainer.train()
        assert trainer.args.train.checkpoint.load_path is not None
        trainer.resume_train()


def _make_simple_dataset(
    datasets,
    weights,
    level="sample",
    stopping_strategy: Literal["first_exhausted", "all_exhausted", "never_exhausted"] = "first_exhausted",
    source_names=None,
    source_ids=None,
):
    return WeightedMultiSourceDataset(
        datasets=datasets,
        weights=weights,
        seed=123,
        level=level,
        sample_token_len_fn=None,
        source_names=source_names,
        source_ids=source_ids,
        upstream_sharded=False,
        stopping_strategy=stopping_strategy,
    )


def test_weighted_multisource_builder_emits_configured_stable_source_id(tmp_path, monkeypatch):
    config_path = tmp_path / "multisource.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": ["source.jsonl"],
                "names": ["display-name"],
                "source_ids": [17],
                "schedule": [{"schedule_type": "const", "weights": [1.0]}],
                "upstream_sharded": False,
            }
        )
    )

    def fake_build_iterable_dataset(*, transform, **_kwargs):
        transformed = transform({"input_ids": [1, 2]})
        return MockIterableDataset([transformed])

    monkeypatch.setattr("veomni.data.dataset.build_iterable_dataset", fake_build_iterable_dataset)
    dataset = build_dataset(
        "veomni_weighted_multisource",
        train_path=str(config_path),
        transform=lambda sample: {**sample, "source_id": "discarded-by-source-wrapper"},
        shuffle=False,
    )

    sample = next(iter(dataset))

    assert sample["source_id"] == 17
    assert sample["source_name"] == "display-name"
    assert sample["_veomni_source_metadata"] == {
        "schema_version": 1,
        "source_id": 17,
        "source_name": "display-name",
    }


def test_weighted_multisource_rejects_boolean_source_id():
    with pytest.raises(ValueError, match="source_ids must contain only int or str"):
        _make_simple_dataset(
            datasets=[MockIterableDataset([{"input_ids": [1]}])],
            weights=[1.0],
            source_ids=[True],
        )


def test_attach_source_metadata_assigns_missing_logical_part_index():
    chunks = [
        {"input_ids": [1]},
        {"input_ids": [2], "part_index": 7},
    ]

    attach_source_metadata(chunks, source_id="stable-id", source_name="display-name")

    assert [chunk["_veomni_source_metadata"]["part_index"] for chunk in chunks] == [0, 7]


def test_attach_source_metadata_removes_stale_name_when_canonical_name_is_missing():
    sample = {
        "input_ids": [1],
        "channel_name": "stale-channel",
        "source_name": "stale-source",
        "dataset_name": "stale-dataset",
        "data_name": "stale-data",
    }

    attach_source_metadata(sample, source_id="stable-id", source_name=None)

    assert not {"channel_name", "source_name", "dataset_name", "data_name"} & sample.keys()
    assert "source_name" not in sample["_veomni_source_metadata"]


def test_make_source_metadata_rejects_non_string_source_name():
    with pytest.raises(ValueError, match="source_name must be a str or None"):
        make_source_metadata("stable-id", True)


def test_interleaved_iterable_preserves_source_metadata_after_transform():
    dataset = InterleavedIterableDataset(
        data=[
            {
                "input_ids": [1, 2],
                "ds_idx": 0,
                "source_id": "stable-id",
                "source_name": "display-name",
            }
        ],
        transform=lambda _sample, **_kwargs: {"input_ids": [3, 4]},
    )

    sample = next(iter(dataset))

    assert sample["source_id"] == "stable-id"
    assert sample["source_name"] == "display-name"
    assert sample["_veomni_source_metadata"] == {
        "schema_version": 1,
        "source_id": "stable-id",
        "source_name": "display-name",
    }


def test_interleaved_iterable_checkpoint_validates_typed_source_id_topology():
    class StatefulRows:
        def __init__(self):
            self.cursor = 0

        def __iter__(self):
            return iter(())

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    rows = StatefulRows()
    rows.cursor = 3
    dataset = InterleavedIterableDataset(rows, source_ids=[7, "7"])

    state = dataset.state_dict()

    assert state == {
        "version": 1,
        "topology": {"source_ids": [7, "7"]},
        "dataset": {"cursor": 3},
    }

    restored_rows = StatefulRows()
    restored = InterleavedIterableDataset(restored_rows, source_ids=[7, "7"])
    restored.load_state_dict(copy.deepcopy(state))
    assert restored_rows.cursor == 3

    reordered = InterleavedIterableDataset(StatefulRows(), source_ids=["7", 7])
    with pytest.raises(ValueError, match="source_ids topology"):
        reordered.load_state_dict(copy.deepcopy(state))

    # Version-less states from the old wrapper remain loadable, but cannot
    # provide topology validation retroactively.
    restored.load_state_dict({"dataset": {"cursor": 4}})
    assert restored_rows.cursor == 4


def test_interleave_builder_preserves_typed_source_ids_after_transform(tmp_path, monkeypatch):
    config_path = tmp_path / "interleave.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": ["source-a.jsonl", "source-b.jsonl"],
                "names": ["a", "b"],
                "source_ids": [101, "101"],
                "schedule": [{"schedule_type": "const", "weights": [0.5, 0.5]}],
            }
        )
    )

    class MappableRows:
        def __init__(self, rows):
            self.rows = rows

        def map(self, transform):
            return MappableRows([transform(row) for row in self.rows])

        def __iter__(self):
            return iter(self.rows)

    def fake_build_iterable_dataset(train_path, **_kwargs):
        return type("DatasetWrapper", (), {"_data": MappableRows([{"input_ids": [len(train_path)]}])})()

    monkeypatch.setattr("veomni.data.dataset.build_iterable_dataset", fake_build_iterable_dataset)
    monkeypatch.setattr(
        "veomni.data.dataset.interleave_datasets",
        lambda *, datasets, **_kwargs: [row for dataset in datasets for row in dataset],
    )
    monkeypatch.setattr("veomni.data.dataset.split_dataset_by_node", lambda dataset, *_args: dataset)

    dataset = build_dataset(
        "interleave",
        train_path=str(config_path),
        datasets_type="iterable",
        transform=lambda sample, **_kwargs: {"input_ids": sample["input_ids"]},
    )

    samples = list(dataset)

    assert [sample["source_id"] for sample in samples] == [101, "101"]
    assert [sample["_veomni_source_metadata"]["source_id"] for sample in samples] == [101, "101"]


def test_interleave_mapping_builder_attaches_metadata_to_transformed_parts(tmp_path, monkeypatch):
    config_path = tmp_path / "interleave-mapping.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": ["source-a.jsonl", "source-b.jsonl"],
                "names": ["a", "b"],
                "source_ids": [202, "202"],
                "schedule": [{"schedule_type": "const", "weights": [0.5, 0.5]}],
            }
        )
    )

    class ColumnRows:
        def __init__(self, rows):
            self.rows = rows

        def add_column(self, name, values):
            return ColumnRows([{**row, name: value} for row, value in zip(self.rows, values)])

        def __getitem__(self, index):
            return self.rows[index]

        def __len__(self):
            return len(self.rows)

    def fake_build_mapping_dataset(train_path, **_kwargs):
        return type("DatasetWrapper", (), {"_data": ColumnRows([{"input_ids": [len(train_path)]}])})()

    monkeypatch.setattr("veomni.data.dataset.build_mapping_dataset", fake_build_mapping_dataset)
    monkeypatch.setattr(
        "veomni.data.dataset.interleave_datasets",
        lambda *, datasets, **_kwargs: ColumnRows([row for dataset in datasets for row in dataset.rows]),
    )

    dataset = build_dataset(
        "interleave",
        train_path=str(config_path),
        datasets_type="mapping",
        transform=lambda sample, **_kwargs: [
            {"input_ids": sample["input_ids"]},
            {"input_ids": sample["input_ids"]},
        ],
    )

    first_parts = dataset[0]
    second_parts = dataset[1]

    assert [part["_veomni_source_metadata"]["source_id"] for part in first_parts] == [202, 202]
    assert [part["_veomni_source_metadata"]["part_index"] for part in first_parts] == [0, 1]
    assert [part["_veomni_source_metadata"]["source_id"] for part in second_parts] == ["202", "202"]


def test_real_hf_streaming_interleave_reaches_canonical_main_collator(tmp_path):
    source_a = tmp_path / "source-a.jsonl"
    source_b = tmp_path / "source-b.jsonl"
    source_a.write_text("\n".join(f'{{"tokens": [{token}, {token + 1}]}}' for token in (11, 21, 31, 41)) + "\n")
    source_b.write_text('{"tokens": [101, 102]}\n')
    config_path = tmp_path / "streaming-interleave.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": [str(source_a), str(source_b)],
                "names": ["a", "b"],
                "source_ids": [303, "303"],
                "schedule": [{"schedule_type": "const", "weights": [1.0, 0.0]}],
            }
        )
    )

    def transform(sample, **_kwargs):
        input_ids = torch.tensor(sample["tokens"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }

    dataset = build_dataset(
        "interleave",
        train_path=str(config_path),
        datasets_type="iterable",
        transform=transform,
        shuffle=False,
    )
    features = list(dataset)

    assert [feature["input_ids"].tolist() for feature in features[:4]] == [
        [11, 12],
        [21, 22],
        [31, 32],
        [41, 42],
    ]
    features = features[:4]
    assert [feature["_veomni_source_metadata"]["source_id"] for feature in features] == [303] * 4

    packed = MainCollator()(features)

    assert packed["_veomni_packed_source_metadata"]["valid_token_count"] == 8
    assert [segment["source_id"] for segment in packed["_veomni_packed_source_metadata"]["segments"]] == [303] * 4
    assert [segment["sample_index"] for segment in packed["_veomni_packed_source_metadata"]["segments"]] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    "operation",
    [
        lambda: make_source_metadata(True, "name"),
        lambda: attach_source_metadata({}, source_id=1.5, source_name="name"),
    ],
)
def test_public_source_metadata_helpers_validate_source_id(operation):
    with pytest.raises(ValueError, match="source_ids must contain only int or str"):
        operation()


def test_public_source_metadata_schema_helpers_fail_closed():
    with pytest.raises(ValueError, match="schema_version must be 1"):
        normalize_source_metadata({"schema_version": 2, "source_id": "a"})

    with pytest.raises(ValueError, match="valid_token_count must equal"):
        normalize_packed_source_metadata(
            {
                "schema_version": 1,
                "coordinate_space": "packed_pre_sp",
                "valid_token_count": 3,
                "segments": [
                    {
                        "source_id": "a",
                        "segment_index": 0,
                        "sample_index": 0,
                        "subsegment_index": 0,
                        "token_start": 0,
                        "token_length": 2,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "sample_coordinates",
    [
        [(1, 0), (1, 1)],
        [(0, 0), (0, 2)],
    ],
)
def test_packed_source_metadata_rejects_invalid_sample_subsegment_order(sample_coordinates):
    segments = [
        {
            "source_id": "a",
            "segment_index": segment_index,
            "sample_index": sample_index,
            "subsegment_index": subsegment_index,
            "token_start": segment_index,
            "token_length": 1,
        }
        for segment_index, (sample_index, subsegment_index) in enumerate(sample_coordinates)
    ]

    with pytest.raises(ValueError, match="sample_index/subsegment_index"):
        normalize_packed_source_metadata(
            {
                "schema_version": 1,
                "coordinate_space": "packed_pre_sp",
                "valid_token_count": 2,
                "segments": segments,
            }
        )


def test_packed_source_metadata_rejects_partially_missing_source_names():
    with pytest.raises(ValueError, match="source_name must be present on all segments or none"):
        normalize_packed_source_metadata(
            {
                "schema_version": 1,
                "coordinate_space": "packed_pre_sp",
                "valid_token_count": 2,
                "segments": [
                    {
                        "source_id": "a",
                        "source_name": "source-a",
                        "segment_index": 0,
                        "sample_index": 0,
                        "subsegment_index": 0,
                        "token_start": 0,
                        "token_length": 1,
                    },
                    {
                        "source_id": "b",
                        "segment_index": 1,
                        "sample_index": 1,
                        "subsegment_index": 0,
                        "token_start": 1,
                        "token_length": 1,
                    },
                ],
            }
        )


def test_packed_source_metadata_rejects_conflicting_names_for_same_typed_source_id():
    with pytest.raises(ValueError, match="same source_id must use one source_name"):
        normalize_packed_source_metadata(
            {
                "schema_version": 1,
                "coordinate_space": "packed_pre_sp",
                "valid_token_count": 2,
                "segments": [
                    {
                        "source_id": 7,
                        "source_name": "source-a",
                        "segment_index": 0,
                        "sample_index": 0,
                        "subsegment_index": 0,
                        "token_start": 0,
                        "token_length": 1,
                    },
                    {
                        "source_id": 7,
                        "source_name": "source-b",
                        "segment_index": 1,
                        "sample_index": 1,
                        "subsegment_index": 0,
                        "token_start": 1,
                        "token_length": 1,
                    },
                ],
            }
        )


def test_state_dict_structure():
    ds1 = MockIterableDataset([{"input_ids": [1, 2]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [3, 4, 5]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        level="token",
        stopping_strategy="all_exhausted",
        source_names=["a", "b"],
        source_ids=["id_a", "id_b"],
    )
    state = dataset.state_dict()
    assert state["version"] == 1
    assert state["topology"]["source_ids"] == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_sum"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_count"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["dataset_states"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["exhausted"].keys()) == ["id_a", "id_b"]


def _legacy_weighted_state(
    *,
    source_names,
    dataset_states,
    avg_len_sum,
    avg_len_count,
    exhausted,
    global_sample_idx=7,
):
    """Build the version-0 shape emitted while the public builder ignored source_ids."""
    return {
        "version": 0,
        "topology": {
            # The old builder typo made these aliases equal to source_names.
            "source_ids": list(source_names),
            "source_names": list(source_names),
            "weights": [1.0 / len(source_names)] * len(source_names),
            "level": "token",
            "stopping_strategy": "all_exhausted",
        },
        "runtime": {
            "random_state": np.random.RandomState(321).get_state(),
            "avg_len_sum": dict(avg_len_sum),
            "avg_len_count": dict(avg_len_count),
            "exhausted": dict(exhausted),
            "global_sample_idx": global_sample_idx,
            "dataset_states": dict(dataset_states),
        },
    }


def test_legacy_name_keyed_checkpoint_migrates_to_typed_stable_ids_and_roundtrips():
    old_state = _legacy_weighted_state(
        source_names=["integer-id-source", "string-id-source"],
        dataset_states={
            "integer-id-source": {"consumed": 3},
            "string-id-source": {"consumed": 5},
        },
        avg_len_sum={"integer-id-source": 11.0, "string-id-source": 22.0},
        avg_len_count={"integer-id-source": 2, "string-id-source": 4},
        exhausted={"integer-id-source": True, "string-id-source": False},
    )
    restored_sources = [
        MockIterableDataset([{"input_ids": [1]}], name="integer-id-source"),
        MockIterableDataset([{"input_ids": [2]}], name="string-id-source"),
    ]
    restored = _make_simple_dataset(
        datasets=restored_sources,
        weights=[0.5, 0.5],
        level="token",
        stopping_strategy="all_exhausted",
        source_names=["integer-id-source", "string-id-source"],
        source_ids=[1, "1"],
    )

    restored.load_state_dict(copy.deepcopy(old_state))

    assert restored._avg_len_sum == [11.0, 22.0]
    assert restored._avg_len_count == [2, 4]
    assert restored._exhausted == [True, False]
    assert restored._global_sample_idx == 7
    assert [source.state_dict() for source in restored_sources] == [{"consumed": 3}, {"consumed": 5}]

    migrated_state = restored.state_dict()
    assert migrated_state["version"] == 1
    assert migrated_state["topology"]["source_ids"] == [1, "1"]
    assert migrated_state["runtime"]["avg_len_sum"] == {1: 11.0, "1": 22.0}
    assert migrated_state["runtime"]["avg_len_count"] == {1: 2, "1": 4}
    assert migrated_state["runtime"]["exhausted"] == {1: True, "1": False}
    assert migrated_state["runtime"]["dataset_states"] == {
        1: {"consumed": 3},
        "1": {"consumed": 5},
    }

    cold_sources = [
        MockIterableDataset([{"input_ids": [1]}], name="integer-id-source"),
        MockIterableDataset([{"input_ids": [2]}], name="string-id-source"),
    ]
    cold = _make_simple_dataset(
        datasets=cold_sources,
        weights=[0.5, 0.5],
        level="token",
        stopping_strategy="all_exhausted",
        source_names=["integer-id-source", "string-id-source"],
        source_ids=[1, "1"],
    )
    cold.load_state_dict(copy.deepcopy(migrated_state), reconcile_policy="strict")
    assert cold.state_dict()["runtime"]["dataset_states"] == migrated_state["runtime"]["dataset_states"]


def test_legacy_checkpoint_rejects_ambiguous_source_name_mapping():
    old_state = _legacy_weighted_state(
        source_names=["duplicate", "duplicate"],
        dataset_states={"duplicate": {"consumed": 1}},
        avg_len_sum={"duplicate": 1.0},
        avg_len_count={"duplicate": 1},
        exhausted={"duplicate": False},
    )
    restored = _make_simple_dataset(
        datasets=[
            MockIterableDataset([{"input_ids": [1]}]),
            MockIterableDataset([{"input_ids": [2]}]),
        ],
        weights=[0.5, 0.5],
        source_names=["duplicate", "other"],
        source_ids=["stable-a", "stable-b"],
    )

    with pytest.raises(ValueError, match="ambiguous legacy source_names"):
        restored.load_state_dict(old_state)


def test_legacy_checkpoint_rejects_alias_collision_with_current_stable_id():
    old_state = _legacy_weighted_state(
        source_names=["legacy-a", "legacy-b"],
        dataset_states={"legacy-a": {"consumed": 1}, "legacy-b": {"consumed": 2}},
        avg_len_sum={"legacy-a": 1.0, "legacy-b": 2.0},
        avg_len_count={"legacy-a": 1, "legacy-b": 1},
        exhausted={"legacy-a": False, "legacy-b": False},
    )
    restored = _make_simple_dataset(
        datasets=[
            MockIterableDataset([{"input_ids": [1]}]),
            MockIterableDataset([{"input_ids": [2]}]),
        ],
        weights=[0.5, 0.5],
        source_names=["legacy-a", "legacy-b"],
        # The legacy alias "legacy-b" would select source 0 through the new
        # stable-ID table but source 1 through the version-0 name mapping.
        source_ids=["legacy-b", "stable-b"],
    )

    with pytest.raises(ValueError, match="legacy source alias.*conflicts with current source_id"):
        restored.load_state_dict(old_state)


class _ResumeIndexDataset(IterableDataset):
    def __init__(self):
        self.rows = [
            {"input_ids": [10], "attention_mask": [1]},
            {"input_ids": [11], "attention_mask": [1]},
            {"input_ids": [12], "attention_mask": [1]},
        ]
        self.output_index_for_resume = False
        self.cursor = 0
        self.loaded_cursor = None

    def __iter__(self):
        while self.cursor < len(self.rows):
            row_idx = self.cursor
            self.cursor += 1
            row = copy.deepcopy(self.rows[row_idx])
            if self.output_index_for_resume:
                yield row, row_idx
            else:
                yield row

    def get_item(self, row_idx):
        # Buffer reconstruction must happen only after the upstream cursor was
        # restored, otherwise exact cold resume can duplicate/skip rows.
        if self.loaded_cursor is None:
            raise RuntimeError("get_item called before upstream state restore")
        return copy.deepcopy(self.rows[row_idx])

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.loaded_cursor = self.cursor


def _make_dynamic_weighted_resume_dataset():
    source = _ResumeIndexDataset()
    weighted = WeightedMultiSourceDataset(
        datasets=[source],
        weights=[1.0],
        source_names=["legacy-name"],
        source_ids=["stable-id"],
        upstream_sharded=True,
        output_index_for_resume=True,
    )
    dynamic = DynamicBatchingSizeDataset(
        dataset=weighted,
        micro_batch_seq_length=1,
        ready_for_micro_batch_threshold=1,
        dynamic_batching_collate_fn=lambda samples: samples,
        save_by_idx=True,
        get_length_fn=lambda sample: len(sample["input_ids"]),
    )
    return source, weighted, dynamic


def test_dynamic_save_by_idx_cold_resume_migrates_legacy_source_alias_before_buffer_rebuild():
    legacy_upstream_state = _legacy_weighted_state(
        source_names=["legacy-name"],
        dataset_states={"legacy-name": {"cursor": 2}},
        avg_len_sum={"legacy-name": 0.0},
        avg_len_count={"legacy-name": 0},
        exhausted={"legacy-name": False},
        global_sample_idx=2,
    )
    old_dynamic_state = {
        "save_by_idx": True,
        "buffer_token_count": 1,
        "buffer_physical_token_count": 1,
        "buffer": [(("legacy-name", 1), 0)],
        "dynamic_batch_upstream_dataset_state": legacy_upstream_state,
    }
    source, _weighted, dynamic = _make_dynamic_weighted_resume_dataset()

    dynamic.load_state_dict(copy.deepcopy(old_dynamic_state))

    assert source.loaded_cursor == 2
    assert dynamic._buffer[0][0]["input_ids"] == [11]
    assert dynamic._buffer[0][0]["source_id"] == "stable-id"
    migrated = dynamic.state_dict()
    assert migrated["buffer"] == [(("stable-id", 1), 0)]
    assert migrated["dynamic_batch_upstream_dataset_state"]["version"] == 1
    assert migrated["dynamic_batch_upstream_dataset_state"]["topology"]["source_ids"] == ["stable-id"]

    cold_source, _cold_weighted, cold_dynamic = _make_dynamic_weighted_resume_dataset()
    cold_dynamic.load_state_dict(copy.deepcopy(migrated))
    resumed_rows = [sample["input_ids"][0] for micro_batch in cold_dynamic for sample in micro_batch]
    assert cold_source.loaded_cursor == 2
    assert resumed_rows == [11, 12]


def test_dynamic_v0_buffer_rejects_removed_alias_reused_as_current_stable_id():
    class StatelessResumeIndexDataset(_ResumeIndexDataset):
        def get_item(self, row_idx):
            return copy.deepcopy(self.rows[row_idx])

    legacy_upstream_state = _legacy_weighted_state(
        source_names=["removed-source"],
        dataset_states={"removed-source": {"cursor": 2}},
        avg_len_sum={"removed-source": 0.0},
        avg_len_count={"removed-source": 0},
        exhausted={"removed-source": False},
        global_sample_idx=2,
    )
    old_dynamic_state = {
        "save_by_idx": True,
        "buffer_token_count": 1,
        "buffer_physical_token_count": 1,
        "buffer": [(("removed-source", 1), 0)],
        "dynamic_batch_upstream_dataset_state": legacy_upstream_state,
    }
    new_source = StatelessResumeIndexDataset()
    weighted = WeightedMultiSourceDataset(
        datasets=[new_source],
        weights=[1.0],
        source_names=["new-source"],
        # Reusing the removed v0 name as a new stable ID must not redirect
        # the old buffer entry to this unrelated source.
        source_ids=["removed-source"],
        upstream_sharded=True,
        output_index_for_resume=True,
    )
    dynamic = DynamicBatchingSizeDataset(
        dataset=weighted,
        micro_batch_seq_length=1,
        ready_for_micro_batch_threshold=1,
        dynamic_batching_collate_fn=lambda samples: samples,
        save_by_idx=True,
        get_length_fn=lambda sample: len(sample["input_ids"]),
    )

    with pytest.raises(ValueError, match="removed legacy source alias"):
        dynamic.load_state_dict(copy.deepcopy(old_dynamic_state))


def test_exhausted_state_save_restore_and_elastic():
    """Test exhausted state save/restore with elastic source add/remove scenarios."""
    # Scenario 1: Basic save and restore
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}, {"input_ids": [3]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        stopping_strategy="all_exhausted",
        source_ids=["id_a", "id_b"],
    )
    dataset._exhausted = [True, False]
    state = dataset.state_dict()
    assert state["runtime"]["exhausted"] == {"id_a": True, "id_b": False}

    # Restore to same structure
    ds1_new = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2_new = MockIterableDataset([{"input_ids": [2]}, {"input_ids": [3]}], name="b")
    dataset_new = _make_simple_dataset(
        datasets=[ds1_new, ds2_new],
        weights=[0.5, 0.5],
        stopping_strategy="all_exhausted",
        source_ids=["id_a", "id_b"],
    )
    dataset_new.load_state_dict(state)
    assert dataset_new._exhausted == [True, False]

    # Scenario 2: Add a new source - new source should default to False
    ds1_new2 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2_new2 = MockIterableDataset([{"input_ids": [2]}, {"input_ids": [3]}], name="b")
    ds3_new = MockIterableDataset([{"input_ids": [4]}], name="c")
    dataset_with_new = _make_simple_dataset(
        datasets=[ds1_new2, ds2_new2, ds3_new],
        weights=[0.3, 0.3, 0.4],
        stopping_strategy="all_exhausted",
        source_ids=["id_a", "id_b", "id_c"],
    )
    dataset_with_new.load_state_dict(state, reconcile_policy="allow_add")
    assert dataset_with_new._exhausted == [True, False, False]

    # Scenario 3: Remove a source - only remaining sources' states preserved
    ds1_new3 = MockIterableDataset([{"input_ids": [1]}], name="a")
    dataset_removed = _make_simple_dataset(
        datasets=[ds1_new3],
        weights=[1.0],
        stopping_strategy="all_exhausted",
        source_ids=["id_a"],
    )
    dataset_removed.load_state_dict(state, reconcile_policy="allow_add_remove")
    assert dataset_removed._exhausted == [True]


def test_exhausted_state_backward_compatible():
    """Test that loading old checkpoint without exhausted field defaults to all False."""
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        stopping_strategy="all_exhausted",
        source_ids=["id_a", "id_b"],
    )

    # Simulate old checkpoint without exhausted field
    old_state = {
        "topology": {"source_ids": ["id_a", "id_b"]},
        "runtime": {
            "random_state": np.random.RandomState(42).get_state(),
            "avg_len_sum": {"id_a": 1.0, "id_b": 2.0},
            "avg_len_count": {"id_a": 1, "id_b": 2},
            "dataset_states": {"id_a": {"consumed": 1}, "id_b": {"consumed": 2}},
        },
    }

    dataset.load_state_dict(old_state)
    assert dataset._exhausted == [False, False]


def test_elastic_load_add_source():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
    )
    next(iter(dataset))
    state = dataset.state_dict()
    ds3 = MockIterableDataset([{"input_ids": [3]}], name="c")
    dataset_new = _make_simple_dataset(
        datasets=[ds1, ds2, ds3],
        weights=[0.3, 0.3, 0.4],
        source_ids=["id_a", "id_b", "id_c"],
    )
    dataset_new.load_state_dict(state, reconcile_policy="allow_add")
    assert ds1.state_dict()["consumed"] >= 1


def test_elastic_load_remove_source():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
    )
    next(iter(dataset))
    state = dataset.state_dict()
    dataset_new = _make_simple_dataset(
        datasets=[ds1],
        weights=[1.0],
        source_ids=["id_a"],
    )
    dataset_new.load_state_dict(state, reconcile_policy="allow_add_remove")
    assert ds1.state_dict()["consumed"] >= 1


def test_elastic_load_strict_policy():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
    )
    state = dataset.state_dict()
    dataset_new = _make_simple_dataset(
        datasets=[ds1],
        weights=[1.0],
        source_ids=["id_a"],
    )
    with pytest.raises(ValueError):
        dataset_new.load_state_dict(state, reconcile_policy="strict")


def test_stopping_strategy_first_exhausted():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
        stopping_strategy="first_exhausted",
    )
    dataset._iters = [iter(ds1), iter(ds2)]
    dataset._exhausted = [False, False]
    first = dataset._next_sample(0)
    assert first["input_ids"] == [1]
    with pytest.raises(StopIteration):
        dataset._next_sample(0)


def test_stopping_strategy_all_exhausted():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
        stopping_strategy="all_exhausted",
    )
    dataset._iters = [iter(ds1), iter(ds2)]
    dataset._exhausted = [False, False]
    first = dataset._next_sample(0)
    second = dataset._next_sample(0)
    assert first["input_ids"] == [1]
    assert second["input_ids"] == [1]


def test_stopping_strategy_never_exhausted():
    ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
        stopping_strategy="never_exhausted",
    )
    dataset._iters = [iter(ds1), iter(ds2)]
    dataset._exhausted = [False, False]
    first = dataset._next_sample(0)
    second = dataset._next_sample(0)
    assert first["input_ids"] == [1]
    assert second["input_ids"] == [1]


def test_determinism_with_seed():
    data_a = [{"input_ids": [i]} for i in range(10)]
    data_b = [{"input_ids": [i]} for i in range(10, 20)]
    ds1_a = MockIterableDataset(data_a, name="a")
    ds2_a = MockIterableDataset(data_b, name="b")
    ds1_b = MockIterableDataset(data_a, name="a")
    ds2_b = MockIterableDataset(data_b, name="b")
    dataset1 = _make_simple_dataset(
        datasets=[ds1_a, ds2_a],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
        stopping_strategy="all_exhausted",
    )
    dataset2 = _make_simple_dataset(
        datasets=[ds1_b, ds2_b],
        weights=[0.5, 0.5],
        source_ids=["id_a", "id_b"],
        stopping_strategy="all_exhausted",
    )
    dataset1.set_epoch(0)
    dataset2.set_epoch(0)
    it1 = iter(dataset1)
    it2 = iter(dataset2)
    for _ in range(10):
        sample1 = cast(dict, next(it1))
        sample2 = cast(dict, next(it2))
        assert sample1["ds_idx"] == sample2["ds_idx"]


def test_level_token_weighting():
    ds1 = MockIterableDataset([{"input_ids": [1, 2, 3, 4]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [5]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[1.0, 1.0],
        level="token",
        source_ids=["id_a", "id_b"],
    )
    dataset._avg_len_sum = [4.0, 1.0]
    dataset._avg_len_count = [1, 1]
    weights = dataset._runtime_weights()
    assert weights[0] == 0.2
    assert weights[1] == 0.8


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}]), MockIterableDataset([{"input_ids": [2]}])],
                "weights": [1.0],
            },
            "weights length must match datasets length",
        ),
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}]), MockIterableDataset([{"input_ids": [2]}])],
                "weights": [0.5, 0.5],
                "source_names": ["only_one"],
            },
            "source_names length must match datasets length",
        ),
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}]), MockIterableDataset([{"input_ids": [2]}])],
                "weights": [0.5, 0.5],
                "source_ids": ["id_a"],
            },
            "source_ids length must match datasets length",
        ),
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}]), MockIterableDataset([{"input_ids": [2]}])],
                "weights": [0.5, 0.5],
                "source_ids": ["same_id", "same_id"],
            },
            "source_ids must be unique",
        ),
        (
            {"datasets": [MockIterableDataset([{"input_ids": [1]}])], "weights": [1.0], "level": "invalid"},
            "level must be 'sample' or 'token'",
        ),
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}])],
                "weights": [1.0],
                "stopping_strategy": cast(Literal["first_exhausted", "all_exhausted", "never_exhausted"], "invalid"),
            },
            "stopping_strategy must be",
        ),
    ],
)
def test_init_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WeightedMultiSourceDataset(**kwargs, seed=42)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"attention_mask": torch.tensor([1, 1, 0])}, 2.0),
        ({"attention_mask": [1, 1, 1, 0]}, 3.0),
        ({"input_ids": torch.tensor([1, 2, 3])}, 3.0),
        ({"input_ids": [1, 2, 3, 4]}, 4.0),
        ([{"input_ids": [1, 2]}, {"input_ids": [3, 4, 5]}], 5.0),
        ({"other_field": "value"}, 1.0),
        (None, 0.0),
    ],
)
def test_default_sample_token_len(sample, expected):
    ds1 = MockIterableDataset([{"input_ids": [1, 2, 3]}], name="a")
    dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
    assert dataset._default_sample_token_len(sample) == expected


class TestLoadStateDictBoundary:
    def test_missing_topology(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
        with pytest.raises(ValueError, match="state_dict missing required keys"):
            dataset.load_state_dict({"runtime": {}})

    def test_missing_runtime(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
        with pytest.raises(ValueError, match="state_dict missing required keys"):
            dataset.load_state_dict({"topology": {}})

    def test_missing_source_ids_in_topology(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
        state = {
            "topology": {"weights": [1.0], "level": "sample"},
            "runtime": {
                "random_state": np.random.RandomState(42).get_state(),
                "avg_len_sum": {},
                "avg_len_count": {},
                "dataset_states": {},
            },
        }
        with pytest.raises(ValueError, match="state_dict missing topology.source_ids"):
            dataset.load_state_dict(state)

    def test_avg_len_not_dict(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
        state = {
            "topology": {"source_ids": ["id_a"]},
            "runtime": {
                "random_state": np.random.RandomState(42).get_state(),
                "avg_len_sum": [1.0],
                "avg_len_count": [1],
                "dataset_states": {},
            },
        }
        with pytest.raises(ValueError, match="must be dicts keyed by source_id"):
            dataset.load_state_dict(state)

    def test_dataset_states_not_dict(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        dataset = _make_simple_dataset(datasets=[ds1], weights=[1.0], source_ids=["id_a"])
        state = {
            "topology": {"source_ids": ["id_a"]},
            "runtime": {
                "random_state": np.random.RandomState(42).get_state(),
                "avg_len_sum": {"id_a": 1.0},
                "avg_len_count": {"id_a": 1},
                "dataset_states": [],
            },
        }
        with pytest.raises(ValueError, match="must be a dict keyed by source_id"):
            dataset.load_state_dict(state)

    def test_warn_only_policy(self):
        ds1 = MockIterableDataset([{"input_ids": [1]}], name="a")
        ds2 = MockIterableDataset([{"input_ids": [2]}], name="b")
        dataset = _make_simple_dataset(
            datasets=[ds1, ds2],
            weights=[0.5, 0.5],
            source_ids=["id_a", "id_b"],
        )
        dataset._avg_len_sum = [2.0, 5.0]
        dataset._avg_len_count = [1, 2]
        dataset._global_sample_idx = 7
        dataset._random_state = np.random.RandomState(999)
        state = dataset.state_dict()
        dataset_new = _make_simple_dataset(
            datasets=[ds1],
            weights=[1.0],
            source_ids=["id_a"],
        )
        dataset_new.load_state_dict(state, reconcile_policy="warn_only")
        assert dataset_new._avg_len_sum == [2.0]
        assert dataset_new._avg_len_count == [1]
        assert dataset_new._global_sample_idx == 7
        rng = np.random.RandomState()
        rng.set_state(state["runtime"]["random_state"])
        assert dataset_new._random_state.randint(0, 2**31 - 1) == rng.randint(0, 2**31 - 1)


def build_command():
    port = find_free_port()
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master_port={port}",
        "tests/data/test_multisource_dataset.py",
        "--model.config_path=test",
        "--data.train_path=None",
        "--data.train_size=1000",
        "--data.max_seq_len=32",
        "--data.datasets_type=iterable",
        "--train.global_batch_size=8",
        "--train.micro_batch_size=2",
        "--train.accelerator.fsdp_config.fsdp_mode=ddp",
        "--train.checkpoint.manager=dcp",
        "--train.checkpoint.output_dir=.tests/cache",
        "--train.dyn_bsz=true",
        "--train.dyn_bsz_runtime=worker",
        "--train.bsz_warmup_ratio=0",
        "--train.max_steps=6",
        # Hardware-aware ops_implementation overrides; see test_datasets.py.
        *resolve_ops_overrides(None),
    ]
    return command


def test_multisource_dataset_chain():
    if sys.platform == "darwin":
        pytest.skip(f"torch_shm_manager not supported on macOS: executable={_torch_shm_manager_executable()}")
    command = build_command()
    result = subprocess.run(command, check=True, env=os.environ.copy())
    assert result.returncode == 0


if __name__ == "__main__":
    _main_distributed_test()
