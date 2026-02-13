import os
import random
import subprocess
import sys
import time
from typing import Literal, cast

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import IterableDataset
from transformers import PretrainedConfig
from utils import DummyDataset, FakeModel, compare_global_batch, compare_metrics

from veomni.arguments import VeOmniArguments, parse_args
from veomni.checkpoint import build_checkpointer
from veomni.data import build_dataloader, build_dataset
from veomni.data.simple_multisource_dataset import SimpleMultiSourceIterableDataset
from veomni.distributed.parallel_state import init_parallel_state
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
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


def process_dummy_example_no_truncate(example, source_name=None):
    tokenized_example = {}
    for k, v in example.items():
        if k in ("ds_idx", "source_name"):
            continue
        tokenized_example[k] = torch.tensor(v, dtype=torch.long)
    return [tokenized_example]


def run_data_test():
    args = parse_args(VeOmniArguments)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device_type = get_device_type()
    if device_type != "cpu":
        get_torch_device().set_device(f"{device_type}:{args.train.local_rank}")
    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    multisource_names = ["dataset_a", "dataset_b"]
    multisource_weights = [0.5, 0.5]
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

    tmp_yaml_path = os.path.join(get_cache_dir("./tmp_simple_ms.yaml"), "tmp_simple_ms.yaml")

    if dist.get_rank() == 0:
        with open(tmp_yaml_path, "w") as f:
            yaml.safe_dump(multisource_config, f)
    logger.info_rank0(f"[{rank}] multisource_config saved in {tmp_yaml_path}")
    dist.barrier()

    args.data.enable_multisource = True
    train_dataset = build_dataset(
        dataset_name="simple_multisource",
        train_path=tmp_yaml_path,
        datasets_type="iterable",
        transform=process_dummy_example_no_truncate,
        seed=args.train.seed,
        level="token",
        stopping_strategy="all_exhausted",
        max_seq_len=args.data.max_seq_len,
        overlong_strategy="truncate",
    )
    state = cast(SimpleMultiSourceIterableDataset, train_dataset).state_dict()
    assert state["version"] == 2
    assert state["topology"]["stopping_strategy"] == "all_exhausted"
    assert state["topology"]["max_seq_len"] == args.data.max_seq_len
    assert state["topology"]["overlong_strategy"] == "truncate"
    assert state["topology"]["level"] == "token"
    assert state["topology"]["source_names"] == multisource_names
    source_ids = state["topology"]["source_ids"]
    assert len(source_ids) == len(multisource_names)
    assert len(set(source_ids)) == len(source_ids)
    assert sorted(state["runtime"]["avg_len_sum"].keys()) == sorted(source_ids)
    assert sorted(state["runtime"]["avg_len_count"].keys()) == sorted(source_ids)
    assert sorted(state["runtime"]["dataset_states"].keys()) == sorted(source_ids)

    dataset_length = None
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    global_batch_size = cast(int, args.train.global_batch_size)
    dataloader = build_dataloader(
        dataloader_type="native",
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        bsz_warmup_ratio=0.0,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        num_workers=1,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
        dyn_bsz=args.train.dyn_bsz,
        dyn_bsz_buffer_size=1,
    )

    config = PretrainedConfig()
    environ_meter = helper.EnvironMeter(
        config=config,
        global_batch_size=global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=dataloader,
        data_path=tmp_yaml_path,
    )

    gt_global_batch_list = []
    epoch_num = 3
    train_steps = args.train.train_steps
    start_epoch, start_step, global_step = 0, 0, 0
    save_step = int(args.train.train_steps * 2)

    fake_model = FakeModel().to(get_device_type())
    for epoch in range(start_epoch, epoch_num):
        dataloader.set_epoch(epoch)
        data_iterator = iter(dataloader)
        start_time = time.time()
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)
                for micro_batch in micro_batches:
                    assert "ds_idx" in micro_batch
                    assert "source_name" in micro_batch
                    source_name = micro_batch["source_name"]
                    if isinstance(source_name, list):
                        assert all(name in multisource_names for name in source_name)
                    else:
                        assert source_name in multisource_names
                    ds_idx = micro_batch["ds_idx"]
                    if isinstance(ds_idx, torch.Tensor):
                        assert torch.all((ds_idx >= 0) & (ds_idx < len(multisource_names)))
                    elif isinstance(ds_idx, list):
                        assert all(0 <= int(idx) < len(multisource_names) for idx in ds_idx)
                    else:
                        assert 0 <= int(ds_idx) < len(multisource_names)
                    assert micro_batch["input_ids"].shape[-1] <= args.data.max_seq_len
                    assert micro_batch["attention_mask"].shape[-1] == micro_batch["input_ids"].shape[-1]
                    assert micro_batch["labels"].shape[-1] == micro_batch["input_ids"].shape[-1]
                    assert torch.all(micro_batch["attention_mask"] == 1)
                    assert torch.all(micro_batch["labels"] == micro_batch["input_ids"])

            if global_step > save_step:
                gt_global_batch_list.append(micro_batches)

            for micro_step, micro_batch in enumerate(micro_batches):
                if global_step == 1:
                    logger.info(f"[rank{rank}] micro step: {micro_step}, {type(micro_batch)}")

                environ_meter.add(micro_batch)

            delta_time = time.time() - start_time
            metrics = environ_meter.step(delta_time, global_step=global_step)
            if global_step == save_step:
                state = {
                    "model": fake_model,
                    "extra_state": {
                        "global_step": global_step,
                        "train_dataloader": dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                    },
                }
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
    state = {"model": fake_model, "extra_state": {}}
    Checkpointer.load(save_checkpoint_path, state)
    dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
    environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
    global_step = state["extra_state"]["global_step"]
    start_epoch = global_step // train_steps
    start_step = global_step % train_steps

    if start_step == 0:
        iter(dataloader)

    pred_global_batch_list = []

    for epoch in range(start_epoch, epoch_num):
        dataloader.set_epoch(epoch)
        data_iter = iter(dataloader)
        for _ in range(start_step, train_steps):
            global_step += 1
            global_batch = next(data_iter)

            if global_step > save_step:
                pred_global_batch_list.append(global_batch)

            start_time = time.time()
            for micro_batch in global_batch:
                environ_meter.add(micro_batch)
            delta_time = time.time() - start_time
            metrics_resume = environ_meter.step(delta_time, global_step=global_step)
        start_step = 0

    compare_global_batch(gt_global_batch_list, pred_global_batch_list)
    compare_metrics(metrics, metrics_resume)

    logger.info_rank0(
        f"dataset_a: {metrics.get('multi_source/consumed_chunk_num/dataset_a', 0)} dataset_b: {metrics.get('multi_source/consumed_chunk_num/dataset_b', 0)}"
    )

    if dist.is_initialized():
        dist.barrier()

    del multisource_datasets

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.remove(tmp_yaml_path)

    if world_size > 1:
        dist.destroy_process_group()


def _make_simple_dataset(
    datasets,
    weights,
    level="sample",
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    max_seq_len=None,
    overlong_strategy: Literal["drop", "truncate"] = "drop",
    source_names=None,
    source_ids=None,
):
    return SimpleMultiSourceIterableDataset(
        datasets=datasets,
        weights=weights,
        seed=123,
        level=level,
        transforms=None,
        sample_token_len_fn=None,
        source_names=source_names,
        source_ids=source_ids,
        sharded=False,
        stopping_strategy=stopping_strategy,
        max_seq_len=max_seq_len,
        overlong_strategy=overlong_strategy,
    )


def test_source_id_stability():
    from veomni.data.simple_multisource_dataset import _build_source_id

    a1 = _build_source_id("path/a", "name")
    a2 = _build_source_id("path/a", "name")
    b1 = _build_source_id("path/a", "name2")
    b2 = _build_source_id("path/b", "name")
    assert a1 == a2
    assert a1 != b1
    assert a1 != b2


def test_state_dict_structure():
    ds1 = MockIterableDataset([{"input_ids": [1, 2]}], name="a")
    ds2 = MockIterableDataset([{"input_ids": [3, 4, 5]}], name="b")
    dataset = _make_simple_dataset(
        datasets=[ds1, ds2],
        weights=[0.5, 0.5],
        level="token",
        stopping_strategy="all_exhausted",
        max_seq_len=8,
        overlong_strategy="truncate",
        source_names=["a", "b"],
        source_ids=["id_a", "id_b"],
    )
    state = dataset.state_dict()
    assert state["version"] == 2
    assert state["topology"]["source_ids"] == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_sum"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["avg_len_count"].keys()) == ["id_a", "id_b"]
    assert sorted(state["runtime"]["dataset_states"].keys()) == ["id_a", "id_b"]


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


def test_overlong_strategy_drop():
    ds1 = MockIterableDataset([{"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}], name="a")
    dataset = _make_simple_dataset(
        datasets=[ds1],
        weights=[1.0],
        source_ids=["id_a"],
        max_seq_len=3,
        overlong_strategy="drop",
    )
    sample, token_len = dataset._maybe_apply_max_seq_len({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]})
    assert sample is None
    assert token_len == 0.0


def test_overlong_strategy_truncate():
    ds1 = MockIterableDataset([{"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}], name="a")
    dataset = _make_simple_dataset(
        datasets=[ds1],
        weights=[1.0],
        source_ids=["id_a"],
        max_seq_len=3,
        overlong_strategy="truncate",
    )
    sample, token_len = dataset._maybe_apply_max_seq_len({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]})
    assert sample["input_ids"] == [1, 2, 3]
    assert sample["attention_mask"] == [1, 1, 1]
    assert token_len == 3.0


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
        assert next(it1)["ds_idx"] == next(it2)["ds_idx"]


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
    assert weights[1] > weights[0]


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
                "stopping_strategy": cast(Literal["first_exhausted", "all_exhausted"], "invalid"),
            },
            "stopping_strategy must be",
        ),
        (
            {
                "datasets": [MockIterableDataset([{"input_ids": [1]}])],
                "weights": [1.0],
                "overlong_strategy": cast(Literal["drop", "truncate"], "invalid"),
            },
            "overlong_strategy must be",
        ),
    ],
)
def test_init_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimpleMultiSourceIterableDataset(**kwargs, seed=42)


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
    port = 12345 + random.randint(0, 100)
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master_port={port}",
        "tests/data/test_simple_multisource_dataset.py",
        "--data.enable_multisource=True",
        "--model.config_path=test",
        "--data.train_path=None",
        "--data.train_size=1000",
        "--data.max_seq_len=8",
        "--data.datasets_type=iterable",
        "--train.global_batch_size=8",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=ddp",
        "--train.ckpt_manager=dcp",
        "--train.ulysses_parallel_size=1",
        "--train.bsz_warmup_ratio=0",
        "--train.output_dir=.tests/cache",
        "--train.rmpad=False",
        "--train.dyn_bsz=False",
        "--train.max_steps=6",
    ]
    return command


def test_simple_multisource_dataset_chain():
    if sys.platform == "darwin":
        pytest.skip(f"torch_shm_manager not supported on macOS: executable={_torch_shm_manager_executable()}")
    command = build_command()
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    run_data_test()
