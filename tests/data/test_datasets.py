import copy
import os
import random
import subprocess
from functools import partial
from typing import Any, Dict, List

import pytest
import yaml
from transformers import PretrainedConfig
from utils import DummyDataset, FakeModel, compare_global_batch, compare_items, compare_metrics, process_dummy_example

from veomni.arguments import parse_args
from veomni.distributed.parallel_state import get_parallel_state
from veomni.trainer.base import BaseTrainer, VeOmniArguments
from veomni.trainer.callbacks import (
    Callback,
    CheckpointerCallback,
    EnvironMeterCallback,
    TrainerState,
)
from veomni.utils import helper
from veomni.utils.device import get_device_type
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)
os.environ["NCCL_DEBUG"] = "OFF"


class TrainerTest(BaseTrainer):
    gt_data_list: List[Dict[str, Any]] = []
    pred_data_list: List[Dict[str, Any]] = []
    golden_env_metrics: helper.EnvironMeter
    resume_dcp_path: str

    is_resume: bool = False
    start_save_data: bool = False

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.checkpointer_callback = CheckpointerCallbackTest(self)
        self.check_callback = CheckCallback(self)
        self.state = TrainerState()

    def _build_model(self):
        # only build fake model
        self.model = FakeModel().to(get_device_type())
        self.model_config = PretrainedConfig()

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        args: VeOmniArguments = self.args
        self.data_transform = partial(
            process_dummy_example,
            max_seq_len=args.data.max_seq_len,
        )

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.check_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        self.state.global_step += 1
        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        self.on_step_end(loss=0.0, loss_dict={}, grad_norm=0.0)

    def resume_fit(self):
        self.is_resume = True
        self.start_save_data = True
        super().fit()

    def destroy_distributed(self):
        if self.is_resume:  # do not destroy distributed when gt train
            super().destroy_distributed()


class CheckpointerCallbackTest(CheckpointerCallback):
    trainer: TrainerTest

    def on_step_end(self, state: TrainerState, **kwargs):
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if state.epoch == 1 and not self.trainer.is_resume:
            self._save_checkpoint(state)
            self.trainer.resume_dcp_path = os.path.join(
                self.trainer.args.train.save_checkpoint_path, f"global_step_{state.global_step}"
            )
            self.trainer.args.train.load_checkpoint_path = self.trainer.resume_dcp_path
            self.trainer.start_save_data = True

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.is_resume:
            self._load_checkpoint()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_step_begin(self, state: TrainerState, micro_batches: List[List[Dict[str, Any]]] = None, **kwargs) -> None:
        if state.global_step == 1 and get_parallel_state().sp_enabled:
            assert (
                micro_batches[0]["input_ids"].shape[-1] * get_parallel_state().sp_size
                == micro_batches[0]["attention_mask"].shape[-1]
            )
            assert compare_items(
                micro_batches[0]["attention_mask"],
                rank=get_parallel_state().sp_rank,
                group_size=get_parallel_state().sp_size,
                group=get_parallel_state().sp_group,
            )
            assert compare_items(
                micro_batches[0]["cu_seq_lens_q"],
                rank=get_parallel_state().sp_rank,
                group_size=get_parallel_state().sp_size,
                group=get_parallel_state().sp_group,
            )
        if self.trainer.start_save_data:
            if not self.trainer.is_resume:
                self.trainer.gt_data_list.append(micro_batches)
            else:
                self.trainer.pred_data_list.append(micro_batches)

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.is_resume:
            compare_global_batch(self.trainer.gt_data_list, self.trainer.pred_data_list)
            compare_metrics(self.trainer.step_env_metrics, self.trainer.golden_env_metrics)

            if self.trainer.args.data.enable_multisource:
                dataset_a_consumed_chunk_num = self.trainer.step_env_metrics[
                    "multi_source/consumed_chunk_num/dataset_a"
                ]
                dataset_b_consumed_chunk_num = self.trainer.step_env_metrics[
                    "multi_source/consumed_chunk_num/dataset_b"
                ]
                # assert abs(dataset_a_consumed_chunk_num / dataset_b_consumed_chunk_num - 0.2 / 0.8) < 0.1
                logger.info(
                    f"dataset_a_consumed_chunk_num: {dataset_a_consumed_chunk_num}, "
                    f"dataset_b_consumed_chunk_num: {dataset_b_consumed_chunk_num}"
                )

                if not self.trainer.args.train.dyn_bsz:
                    assert (
                        dataset_a_consumed_chunk_num + dataset_b_consumed_chunk_num
                        == self.trainer.args.train.global_batch_size
                        * self.trainer.train_steps
                        * self.trainer.args.train.num_train_epochs
                    )
            else:
                consumed_chunk_num = self.trainer.step_env_metrics["consumed_chunk_num"]
                if not self.trainer.args.train.dyn_bsz:
                    assert (
                        consumed_chunk_num
                        == self.trainer.args.train.global_batch_size
                        * self.trainer.train_steps
                        * self.trainer.args.train.num_train_epochs
                    )
        else:
            self.trainer.golden_env_metrics = copy.deepcopy(self.trainer.step_env_metrics)


def main():
    args: VeOmniArguments = parse_args(VeOmniArguments)
    trainer = TrainerTest(args)
    trainer.fit()
    trainer.resume_fit()


if __name__ == "__main__":
    main()


def build_command(dataset_type: str, dyn_bsz: bool, data_path: str):
    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        f"--master_port={port}",
        "tests/data/test_datasets.py",
        "--model.config_path=test",
        f"--data.train_path={data_path}",
        "--data.train_size=1000",
        "--data.train_sample=4",  # iterable & not dyn_bsz
        "--data.max_seq_len=16",
        "--train.global_batch_size=16",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=ddp",
        f"--data.datasets_type={dataset_type}",
        f"--train.dyn_bsz={dyn_bsz}",
        "--train.use_wandb=True",
        "--train.ulysses_parallel_size=2",
        "--train.bsz_warmup_ratio=0",
        "--data.num_workers=1",
        "--train.num_train_epochs=5",
        "--train.output_dir=.tests/cache",
    ]
    return command


@pytest.fixture(scope="session")
def dummy_multisource_dataset_ci():
    # build dummy data
    multisource_names = ["dataset_a", "dataset_b"]
    multisource_weights = [0.2, 0.8]
    multisource_datasets = [DummyDataset(size=100, dataset_name=name) for name in multisource_names]
    multisource_path = [dataset.save_path for dataset in multisource_datasets]

    multisource_config = dict(
        sources=multisource_path,
        names=multisource_names,
        schedule=[
            dict(
                schedule_type="const",
                weights=multisource_weights,
            )
        ],
    )

    tmp_yaml_path = os.path.join(get_cache_dir("./tmp.yaml"), "tmp.yaml")

    with open(tmp_yaml_path, "w") as f:
        yaml.safe_dump(multisource_config, f)

    yield tmp_yaml_path

    del multisource_datasets
    os.remove(tmp_yaml_path)


@pytest.fixture(scope="session")
def dummy_native_dataset_ci():
    dummy_dataset = DummyDataset(size=20)
    train_path = dummy_dataset.save_path

    yield train_path
    del dummy_dataset


TEST_DATASETS = ["mapping", "iterable"]
DYN_BSZ = [True, False]


@pytest.mark.parametrize("dataset_type", TEST_DATASETS)
@pytest.mark.parametrize("dyn_bsz", DYN_BSZ)
def test_multisource_dataset(dataset_type: str, dyn_bsz: bool, dummy_multisource_dataset_ci):
    data_path = dummy_multisource_dataset_ci
    command = build_command(dataset_type, dyn_bsz, data_path=data_path)
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


@pytest.mark.parametrize("dataset_type", TEST_DATASETS)
@pytest.mark.parametrize("dyn_bsz", DYN_BSZ)
def test_native_dataset(dataset_type: str, dyn_bsz: bool, dummy_native_dataset_ci):
    data_path = dummy_native_dataset_ci
    command = build_command(dataset_type, dyn_bsz, data_path=data_path)
    result = subprocess.run(command, check=True)
    assert result.returncode == 0
