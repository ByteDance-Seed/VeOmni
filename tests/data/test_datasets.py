import copy
import os
import random
import subprocess
import sys
from functools import partial
from itertools import islice
from typing import Any, Dict, List


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import yaml
from tools import resolve_ops_overrides
from torch.utils.data import DistributedSampler, IterableDataset
from transformers import PretrainedConfig
from utils import (
    DummyDataset,
    FakeModel,
    ShardedMappingDataset,
    compare_global_batch,
    compare_items,
    compare_metrics,
    process_dummy_example,
)

from veomni.arguments import parse_args
from veomni.data import dataset as dataset_module
from veomni.data.data_collator import MainCollator
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import (
    DynamicBatchingSizeDataset,
    InterleavedMappingDataset,
    MappingDataset,
    ShardedIterableDataset,
    _MapStyleSamplerWrapper,
    build_dataset,
    get_data_files,
)
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
        self.state.curr_step = 0

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
        self.state.curr_step += 1
        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        self.on_step_end(loss=0.0, loss_dict={}, grad_norm=0.0)

    def resume_train(self):
        self.is_resume = True
        self.start_save_data = True
        super().train()

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
                self.trainer.args.train.checkpoint.save_path, f"global_step_{state.global_step}"
            )
            self.trainer.args.train.checkpoint.load_path = self.trainer.resume_dcp_path
            self.trainer.start_save_data = True

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.is_resume:
            self._load_checkpoint()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        """
        from itertools import groupby
        micro_batches_output = [[k  for k, _ in groupby(micro_batch["input_ids"].tolist()[0])]  for micro_batch in micro_batches]
        logger.error(f"[rank{get_parallel_state().global_rank}][epoch{state.epoch}][curr_step{state.curr_step}][step {state.global_step}] micro_batches_output: {micro_batches_output}")
        """
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
    trainer.train()
    assert trainer.args.train.checkpoint.load_path is not None
    trainer.resume_train()


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
        "--train.accelerator.fsdp_config.fsdp_mode=ddp",
        f"--data.datasets_type={dataset_type}",
        f"--train.dyn_bsz={dyn_bsz}",
        "--train.wandb.enable=True",
        "--train.accelerator.ulysses_size=2",
        "--train.bsz_warmup_ratio=0",
        "--data.dataloader.num_workers=1",
        # Force pin_memory=False: on NPU the pin_memory background thread
        # races with HCCL teardown (triggered inside destroy_distributed) and
        # aborts the process with SIGABRT. The tests use DummyDataset so
        # pin_memory provides no benefit anyway. Mirrors the fix in #639 for
        # tests/data/test_multisource_dataset.py.
        "--data.dataloader.pin_memory=False",
        "--train.num_train_epochs=5",
        "--train.checkpoint.output_dir=.tests/cache",
        # Hardware-aware ops_implementation overrides. ``resolve_ops_overrides``
        # emits NPU-supported per-op backends on NPU and ``[]`` on GPU (the
        # dataclass defaults are already GPU-optimal). FakeModel is not
        # registered, so ``model_name=None`` skips per-model eager fallbacks.
        *resolve_ops_overrides(None),
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


# When tested under mapping datasets, each rank will see the same data from the upstream dataset but will only visit a chunk of it by StatefulDistributedSampler defined in build_native_dataloader. When the dataset is drained, the dataloader will re-create the iterator of the dataset. The random shuffle of the data by the sampler will be updated for each epoch by the StatefulDistributedSampler controlled by its seed, which is the same behavior as PyTorch's DistributedSampler
# When tested under iterable datasets, each rank will get the same data from the upstream dataset, and the dataset is shuffled and splited into the small chunk which each rank visit. When it is drained, the dataloader will just re-create an iter of the dataset. The random shuffle of the data will be updated for each epoch by set_epoch method of the dataset controled by the seed passed in.
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


_WRAPPER_DATASET_SIZE = 50
_WRAPPER_MICRO_BATCH_SEQ_LENGTH = 64
_WRAPPER_READY_THRESHOLD = 4


def _wrapper_get_length(item):
    return int(item["attention_mask"].sum())


def _wrapper_take_first(micro_batches):
    # batch_size=1 -> one micro batch per item; module-level so it stays picklable for num_workers > 0.
    return micro_batches[0]


def _assert_batch_stream_equal(got_batches, want_batches):
    assert len(got_batches) == len(want_batches)
    for got, want in zip(got_batches, want_batches):
        assert got.keys() == want.keys()
        for key in got:
            if torch.is_tensor(got[key]):
                assert torch.equal(got[key], want[key]), f"mismatch in {key}"


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_replicas", [1, 2, 4])
@pytest.mark.parametrize("epoch", [0, 3])
def test_map_style_sampler_wrapper_rank_parity(shuffle, num_replicas, epoch):
    """Per-rank index assignment must be bit-identical to ``DistributedSampler``."""
    dataset = ShardedMappingDataset(size=_WRAPPER_DATASET_SIZE)
    seed = 7
    for rank in range(num_replicas):
        wrapper = _MapStyleSamplerWrapper(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        wrapper.set_epoch(epoch)
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        sampler.set_epoch(epoch)
        assert wrapper._rank_indices().tolist() == list(sampler)
        assert len(wrapper) == sampler.num_samples


def _build_wrapper_pipeline(num_workers, save_by_idx):
    """mapping dataset -> _MapStyleSamplerWrapper -> DynamicBatchingSizeDataset -> DistributedDataloader."""
    dataset = ShardedMappingDataset(size=_WRAPPER_DATASET_SIZE)
    wrapper = _MapStyleSamplerWrapper(dataset, num_replicas=2, rank=0, shuffle=True, seed=11)
    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=wrapper,
        micro_batch_seq_length=_WRAPPER_MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=_WRAPPER_READY_THRESHOLD,
        dynamic_batching_collate_fn=MainCollator(),
        get_length_fn=_wrapper_get_length,
        save_by_idx=save_by_idx,
    )
    return DistributedDataloader(dynamic_ds, batch_size=1, num_workers=num_workers, collate_fn=_wrapper_take_first)


def test_map_style_sampler_wrapper_resume():
    """Checkpoint resume preserves the full worker-side dynamic-batching pipeline."""
    for num_workers in (0, 2):
        for save_by_idx in (False, True):
            # Use an uninterrupted epoch as the reference stream for all resume checks.
            golden_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            golden_dataloader.set_epoch(2)
            golden = list(golden_dataloader)
            assert len(golden) > 4, "test setup must produce enough micro batches"
            epoch_boundary_state = golden_dataloader.state_dict()

            # Case 1: A mid-epoch save/resume must reproduce the uninterrupted stream exactly.
            dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            dataloader.set_epoch(2)
            it = iter(dataloader)
            head = [next(it) for _ in range(3)]
            dataloader_state = dataloader.state_dict()
            del it, dataloader

            resumed_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            resumed_dataloader.load_state_dict(dataloader_state)
            resumed_dataloader.set_epoch(2)
            batches = head + list(resumed_dataloader)
            _assert_batch_stream_equal(batches, golden)

            # Case 2: Re-saving before consuming a restored batch must preserve the pending resume state.
            resaved_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            resaved_dataloader.load_state_dict(dataloader_state)
            resaved_dataloader.set_epoch(2)
            resaved_state = resaved_dataloader.state_dict()
            del resaved_dataloader

            resumed_from_resaved = _build_wrapper_pipeline(num_workers, save_by_idx)
            resumed_from_resaved.load_state_dict(resaved_state)
            resumed_from_resaved.set_epoch(2)
            resaved_batches = head + list(resumed_from_resaved)
            _assert_batch_stream_equal(resaved_batches, golden)

            # Case 3: An epoch-boundary checkpoint with an empty buffer must start the next epoch cleanly.
            next_epoch_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            next_epoch_dataloader.load_state_dict(epoch_boundary_state)
            iter(
                next_epoch_dataloader
            )  # check _load_checkpoint() of checkpoint_callback.py for more details on why we need to call iter() when the checkpoint is saved at the epoch boundary
            next_epoch_dataloader.set_epoch(3)
            next_epoch_batches = list(next_epoch_dataloader)

            full_next_epoch_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            full_next_epoch_dataloader.set_epoch(3)
            full_next_epoch_batches = list(full_next_epoch_dataloader)
            _assert_batch_stream_equal(next_epoch_batches, full_next_epoch_batches)

            # Case 4: An epoch-boundary checkpoint with a non-empty buffer must start the next epoch cleanly.
            buffered_boundary_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            buffered_boundary_dataloader.load_state_dict(dataloader_state)
            iter(buffered_boundary_dataloader)
            buffered_boundary_dataloader.set_epoch(3)
            buffered_next_epoch_batches = list(buffered_boundary_dataloader)
            _assert_batch_stream_equal(buffered_next_epoch_batches, full_next_epoch_batches)

            # Case 5: Starting a new epoch after materializing an unconsumed restored iterator must reset pending state.
            materialized_dataloader = _build_wrapper_pipeline(num_workers, save_by_idx)
            materialized_dataloader.load_state_dict(dataloader_state)
            materialized_dataloader.set_epoch(2)
            restored_it = iter(materialized_dataloader)
            del restored_it
            materialized_dataloader.set_epoch(3)
            materialized_next_epoch_batches = list(materialized_dataloader)
            _assert_batch_stream_equal(materialized_next_epoch_batches, full_next_epoch_batches)


def test_map_style_sampler_wrapper_rejects_incompatible_resume_state():
    """Restoring a different sampler configuration must fail instead of changing the sample stream."""
    source = _MapStyleSamplerWrapper(
        ShardedMappingDataset(size=_WRAPPER_DATASET_SIZE),
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=5,
    )
    state = source.state_dict()

    incompatible = _MapStyleSamplerWrapper(
        ShardedMappingDataset(size=_WRAPPER_DATASET_SIZE - 1),
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=6,
    )
    with pytest.raises(
        RuntimeError,
        match=r"num_replicas.*seed.*dataset_size.*shuffle",
    ):
        incompatible.load_state_dict(state)

    with pytest.raises(RuntimeError, match="missing sampler fingerprint fields"):
        source.load_state_dict({"epoch": 0, "yielded": 0})


def _read_cycle(dataset, cycle):
    start = cycle * len(dataset)
    return [dataset[index] for index in range(start, start + len(dataset))]


def test_mapping_dataset_overlength_indices_are_deterministic():
    first = MappingDataset(list(range(8)), seed=17)
    second = MappingDataset(list(range(8)), seed=17)

    expected = _read_cycle(first, 1)
    assert expected != list(range(8))
    assert sorted(expected) == list(range(8))
    assert _read_cycle(second, 1) == expected
    assert [second[index] for index in (12, 8, 15, 9)] == [expected[index] for index in (4, 0, 7, 1)]


def test_mapping_dataset_preserves_python_negative_index_semantics():
    dataset = MappingDataset(list(range(8)), seed=17)
    assert dataset[-1] == 7
    with pytest.raises(IndexError, match="out of range"):
        dataset[-9]


def test_mapping_dataset_cycle_is_reproducible_after_restart():
    uninterrupted = MappingDataset(list(range(8)), seed=9)
    expected = _read_cycle(uninterrupted, 2)

    resumed = MappingDataset(list(range(8)), seed=9)
    assert [resumed[index] for index in range(19, 24)] == expected[3:]


def test_mapping_dataset_mapping_is_independent_of_cycle_access_order():
    dataset = MappingDataset(list(range(8)), seed=9)
    expected = {index: dataset[index] for index in range(8, 32)}

    for index in (25, 9, 17, 31, 8, 24, 16):
        assert dataset[index] == expected[index]


def test_empty_mapping_dataset_raises_index_error():
    dataset = MappingDataset([], seed=9)
    with pytest.raises(IndexError, match="empty dataset"):
        dataset[0]
    with pytest.raises(IndexError, match="empty dataset"):
        dataset[-1]


def test_mapping_dataset_seed_controls_overlength_order_without_global_rng_side_effect():
    first = MappingDataset(list(range(8)), seed=3)
    second = MappingDataset(list(range(8)), seed=4)
    assert _read_cycle(first, 1) != _read_cycle(second, 1)

    random.seed(123)
    expected = random.random()
    random.seed(123)
    first[len(first)]
    assert random.random() == expected


def test_build_mapping_dataset_forwards_seed(monkeypatch):
    monkeypatch.setattr(dataset_module, "get_data_files", lambda _: (["unused.jsonl"], "json"))
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: list(range(8)))

    first = build_dataset(dataset_name="mapping", train_path="unused.jsonl", seed=3)
    second = build_dataset(dataset_name="mapping", train_path="unused.jsonl", seed=4)

    assert _read_cycle(first, 1) != _read_cycle(second, 1)


def test_overlength_read_does_not_change_first_cycle_order():
    dataset = MappingDataset(list(range(8)), seed=3)
    dataset[len(dataset)]
    assert _read_cycle(dataset, 0) == list(range(8))


def test_interleaved_mapping_dataset_uses_same_deterministic_mapping():
    data = [{"value": index, "ds_idx": index % 2} for index in range(8)]

    first = InterleavedMappingDataset(data, seed=7)
    second = InterleavedMappingDataset(data, seed=7)

    assert _read_cycle(first, 1) == _read_cycle(second, 1)


class _ListStream(IterableDataset):
    def __init__(self, values):
        self.values = list(values)
        self.epoch = 0

    def __iter__(self):
        yield from self.values

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def test_get_data_files_skips_non_table_artifacts(tmp_path):
    nested = tmp_path / "rank0"
    nested.mkdir()
    parquet_path = nested / "shard.parquet"
    parquet_path.write_bytes(b"PAR1")
    (tmp_path / "veomni_cli.yaml").write_text("train: {}\n")
    (tmp_path / "first.png").write_bytes(b"\x89PNG")

    files, loader = get_data_files(str(tmp_path))
    assert loader == "parquet"
    assert files == [str(parquet_path)]


def test_get_data_files_rejects_mixed_table_types(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Mixed data file types"):
        get_data_files(str(tmp_path))


def test_get_data_files_empty_directory_errors(tmp_path):
    (tmp_path / "readme.md").write_text("not data")
    with pytest.raises(FileNotFoundError, match="No supported data files"):
        get_data_files(str(tmp_path))


def test_iterable_drop_last_equalizes_ranks():
    source = list(range(15))
    ranks = [
        list(ShardedIterableDataset(_ListStream(source), dp_rank=rank, dp_size=8, repeat=False)) for rank in range(8)
    ]
    assert [len(items) for items in ranks] == [1] * 8
    assert [items[0] for items in ranks] == list(range(8))


def test_iterable_one_pass_does_not_replay():
    stream = ShardedIterableDataset(_ListStream([0, 1]), repeat=False)
    assert list(stream) == [0, 1]


def test_iterable_repeat_replays_until_consumed():
    stream = ShardedIterableDataset(_ListStream([0, 1]), repeat=True)
    assert list(islice(stream, 5)) == [0, 1, 0, 1, 0]


def test_iterable_repeat_drops_incomplete_round_before_replay():
    source = [0, 1, 2]
    ranks = [
        list(islice(ShardedIterableDataset(_ListStream(source), dp_rank=rank, dp_size=2, repeat=True), 4))
        for rank in range(2)
    ]
    assert ranks[0] == [0, 0, 0, 0]
    assert ranks[1] == [1, 1, 1, 1]


def test_iterable_repeat_exits_when_source_shorter_than_dp():
    ranks = [
        list(islice(ShardedIterableDataset(_ListStream([0]), dp_rank=rank, dp_size=2, repeat=True), 4))
        for rank in range(2)
    ]
    assert ranks == [[], []]


def test_iterable_repeat_advances_inner_epoch():
    inner = _ListStream(["a"])
    stream = ShardedIterableDataset(inner, repeat=True, seed=10)
    stream.set_epoch(3)
    assert list(islice(stream, 2)) == ["a", "a"]
    assert inner.epoch == 14
