import argparse
import copy
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, cast
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
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
from veomni.data.dataset import WeightedMultiSourceDataset
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
        self.device = setup_test_distributed(self.args)

        self.multisource_datasets = [DummyDataset(size=100, dataset_name=name) for name in self.multisource_names]
        self.multisource_paths = [dataset.save_path for dataset in self.multisource_datasets]

        multisource_config = dict(
            sources=self.multisource_paths,
            names=self.multisource_names,
            schedule=[dict(schedule_type="const", weights=self.multisource_weights)],
        )
        self.tmp_train_path = os.path.join(get_cache_dir("./tmp_train_path.yaml"), "tmp_train_path.yaml")
        if dist.get_rank() == 0:
            with open(self.tmp_train_path, "w") as f:
                yaml.safe_dump(multisource_config, f)

        self.args.data.train_path = self.tmp_train_path
        self.args.data.enable_multisource = True
        self.args.data.level = "token"
        self.args.data.split_by_node = False
        self.args.data.stopping_strategy = "all_exhausted"
        self.args.data.dataset_name = "veomni_weighted_multisource"

        self.args.train.num_train_epochs = 3

    def _freeze_model_module(self):
        pass

    def _build_model(self):
        self.model = FakeModel().to(get_device_type())
        self.model_config = PretrainedConfig()

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        self.data_transform = None

    def _build_dataset(self):
        super()._build_dataset()

        dist.barrier()

        state = cast(WeightedMultiSourceDataset, self.train_dataset).state_dict()
        assert state["version"] == 0
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
            dyn_bsz_run_in=args.train.dyn_bsz_run_in,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_buffer_size=1,
            dyn_bsz_dataset_save_by_idx=True,
            num_workers=1,
            drop_last=args.data.dataloader.drop_last,
            pin_memory=args.data.dataloader.pin_memory,
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
            data_path=trainer.tmp_yaml_path,
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

            if (not dist.is_initialized() or dist.get_rank() == 0) and os.path.exists(self.trainer.tmp_yaml_path):
                os.remove(self.trainer.tmp_yaml_path)
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
    assert state["version"] == 0
    assert state["topology"]["source_ids"] == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_sum"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_count"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["dataset_states"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["exhausted"].keys()) == ["id_a", "id_b"]


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
        "--train.dyn_bsz_run_in=worker",
        "--train.bsz_warmup_ratio=0",
        "--train.max_steps=6",
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
