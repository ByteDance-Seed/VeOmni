import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from functools import partial

import torch.distributed as dist
import yaml
from transformers import PretrainedConfig
from utils import DummyDataset, FakeModel, compare_global_batch, compare_items, compare_metrics, process_dummy_example

from veomni.checkpoint import build_checkpointer
from veomni.data import (
    build_dataloader,
    build_dataset,
)
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def run_data_test():
    args = parse_args(Arguments)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend(), world_size=world_size, rank=rank)

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

    transform = partial(
        process_dummy_example,
        max_seq_len=args.data.max_seq_len,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
    )

    # build dummy data
    multisource_names = ["dataset_a", "dataset_b"]
    multisource_weights = [0.5, 0.5]
    multisource_datasets = [DummyDataset(size=1000, dataset_name=name) for name in multisource_names]
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

    if dist.get_rank() == 0:
        with open(tmp_yaml_path, "w") as f:
            yaml.safe_dump(multisource_config, f)
    logger.info_rank0(f"[{rank}] multisource_config saved in {tmp_yaml_path}")
    dist.barrier()

    args.data.enable_multisource = True
    logger.info_rank0("Start building interleave dataset")
    train_dataset = build_dataset(
        dataset_name="interleave",
        train_path=tmp_yaml_path,
        datasets_type=args.data.datasets_type,
        transform=transform,
        seed=args.train.seed,
    )

    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    dataloader = build_dataloader(
        dataloader_type="native",
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
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
        dyn_bsz_buffer_size=1,
    )

    config = PretrainedConfig()
    environ_meter = helper.EnvironMeter(
        config=config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=dataloader,
        data_path=tmp_yaml_path,
    )

    gt_global_batch_list = []
    epoch_num = 5
    train_steps = args.train.train_steps
    start_epoch, start_step, global_step = 0, 0, 0
    save_step = int(args.train.train_steps * 2)  # due to dataset.buffer, cannot resume from mid_step

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

                if get_parallel_state().sp_enabled:
                    assert (
                        micro_batches[0]["input_ids"].shape[-1] * get_parallel_state().sp_size
                        == micro_batches[0]["attention_mask"].shape[-1]
                    )
                    compare_items(
                        micro_batches[0]["attention_mask"],
                        rank=get_parallel_state().sp_rank,
                        group_size=get_parallel_state().sp_size,
                        group=get_parallel_state().sp_group,
                    )
                    if args.train.rmpad_with_pos_ids:
                        compare_items(
                            micro_batches[0]["position_ids"],
                            rank=get_parallel_state().sp_rank,
                            group_size=get_parallel_state().sp_size,
                            group=get_parallel_state().sp_group,
                        )
                    if args.train.rmpad:
                        compare_items(
                            micro_batches[0]["cu_seqlens"],
                            rank=get_parallel_state().sp_rank,
                            group_size=get_parallel_state().sp_size,
                            group=get_parallel_state().sp_group,
                        )

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
    # resume
    state = {"model": fake_model, "extra_state": {}}  # cannot be None
    Checkpointer.load(save_checkpoint_path, state)
    dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
    environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
    global_step = state["extra_state"]["global_step"]
    start_epoch = global_step // train_steps
    start_step = global_step % train_steps

    if start_step == 0:  # resume at the end of epoch
        iter(dataloader)  # clear resume state and prefetch data

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


def build_command(dataset_type, dataloader_type):
    port = 12345 + random.randint(0, 100)

    if dataloader_type == "rmpad":
        rmpad = True
        rmpad_with_pos_ids = False
    elif dataloader_type == "rmpad_with_pos_ids":
        rmpad = False
        rmpad_with_pos_ids = True
    else:
        rmpad = False
        rmpad_with_pos_ids = False
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        f"--master_port={port}",
        "tests/data/test_multisource_datasets.py",
        "--data.enable_multisource=True",
        "--model.config_path=test",
        "--data.train_path=None",
        "--data.train_size=1000",
        "--data.max_seq_len=16",
        "--train.global_batch_size=16",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=ddp",
        "--train.ckpt_manager=dcp",
        f"--data.datasets_type={dataset_type}",
        "--train.ulysses_parallel_size=2",
        "--train.bsz_warmup_ratio=0",
        "--train.output_dir=.tests/cache",
        f"--train.rmpad={rmpad}",
        f"--train.rmpad_with_pos_ids={rmpad_with_pos_ids}",
    ]
    return command


def test_multisource_data_rmpad():
    command = build_command(dataset_type="mapping", dataloader_type="rmpad")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0

    command = build_command(dataset_type="iterable", dataloader_type="rmpad")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def test_multisource_data_rmpad_with_pos_ids():
    command = build_command(dataset_type="mapping", dataloader_type="rmpad_with_pos_ids")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0

    command = build_command(dataset_type="iterable", dataloader_type="rmpad_with_pos_ids")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def test_multisource_data_padding():
    command = build_command(dataset_type="mapping", dataloader_type="padding")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    run_data_test()
