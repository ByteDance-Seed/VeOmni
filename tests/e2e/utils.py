import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset
from rich.console import Console
from rich.table import Table

from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.utils.helper import get_cache_dir


def parse_training_log(log_content) -> pd.DataFrame:
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+):"
        r".*?(?P<step>\d+)/(?P<total_steps>\d+)"
        r".*?total_loss:\s+(?P<loss>[\d\.]+)"
        r".*?grad_norm:\s+(?P<grad_norm>[\d\.]+)"
        r".*?lr:\s+(?P<lr>[\d\.eE+-]+)"
    )

    data = []

    for match in pattern.finditer(log_content):
        row = match.groupdict()
        parsed_row = {
            "epoch": int(row["epoch"]),
            "total_epochs": int(row["total_epochs"]),
            "step": int(row["step"]),
            "total_steps": int(row["total_steps"]),
            "loss": float(row["loss"]),
            "grad_norm": float(row["grad_norm"]),
            "lr": float(row["lr"]),
        }
        parsed_row["global_step"] = (parsed_row["epoch"] - 1) * parsed_row["total_steps"] + parsed_row["step"]
        data.append(parsed_row)

    return pd.DataFrame(data)


def check_metric(base_series, compare_series, name, rtol=1e-5, atol=1e-5):
    a = base_series.to_numpy()
    b = compare_series.to_numpy()

    if len(a) != len(b):
        raise AssertionError(f"[{name}] Length mismatch: base({len(a)}) vs compare({len(b)})")

    is_close = np.isclose(a, b, rtol=rtol, atol=atol)

    if not np.all(is_close):
        first_mismatch = np.where(~is_close)[0][0]
        max_diff = np.max(np.abs(a - b))

        err_msg = (
            f"\n❌ [{name}] Comparison failed!\n"
            f"Max Absolute Error: {max_diff:.2e}\n"
            f"First mismatch at index: {first_mismatch}\n"
            f"Base value: {a[first_mismatch]:.8f}\n"
            f"Compare value: {b[first_mismatch]:.8f}\n"
            f"Tolerances: rtol={rtol}, atol={atol}"
        )
        raise AssertionError(err_msg)


def compare_log(base_log_df: pd.DataFrame, compare_log_df: pd.DataFrame):
    check_metric(base_log_df["loss"], compare_log_df["loss"], name="loss")
    check_metric(base_log_df["grad_norm"], compare_log_df["grad_norm"], name="grad_norm")


class DummyDataset:
    def __init__(self, num_samples=16, seq_len=8192, dataset_type: str = "text") -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_shard = 2

        self.save_path = get_cache_dir(f"./{dataset_type}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.dataset = build_dummy_dataset(dataset_type, self.num_samples, self.seq_len)
            self.build_dummy_dataset()

        if dist.is_initialized():
            dist.barrier()

    def generate_data(self):
        num_samples = len(self.dataset)
        for index in range(num_samples):
            item = self.dataset[index][0]
            yield item

    def build_dummy_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        batch_len = math.ceil(self.num_samples / self.num_shard)
        print(f"Total length: {self.num_samples}, batch length: {batch_len}")

        index = 0
        for i in range(0, self.num_samples, batch_len):
            print(f"Generating {index}th parquet file")
            ds = Dataset.from_generator(
                self.generate_data,
                keep_in_memory=True,
                num_proc=1,
            )
            ds.to_parquet(os.path.join(self.save_path, f"{index}.parquet"))
            index += 1

    def clean_cache(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if os.path.exists(self.save_path):
                os.system(f"rm -rf {self.save_path}")

    def __del__(self):
        self.clean_cache()


@dataclass(frozen=True)
class ModelMode:
    sp_size: int
    ep_size: int

    def __str__(self):
        return f"_[sp-{self.sp_size}]_[ep-{self.ep_size}]"


_SP_SIZE = [1, 2]
_EP_SIZE = [1, 2]


def _base_model_modes():
    modes = []
    for sp_size in _SP_SIZE:
        modes.append(ModelMode(sp_size, 1))
    return modes


def _moe_model_modes():
    modes = []
    for sp_size in _SP_SIZE:
        for ep_size in _EP_SIZE:
            modes.append(ModelMode(sp_size, ep_size))
    return modes


def prepare_exec_cmd(
    test_tasks: list[str],
    model_name: str,
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    is_moe: bool,
) -> str:
    model_modes: ModelMode = _base_model_modes() if not is_moe else _moe_model_modes()

    command_list = []
    for task in test_tasks:
        for mode in model_modes:
            port = 12345 + random.randint(0, 100)
            command = [
                "torchrun",
                "--nnodes=1",
                f"--nproc_per_node={mode.sp_size * 4}",
                f"--master_port={port}",
                f"tests/e2e/{task}.py",
                f"--model.config_path={config_path}",
                f"--data.train_path={train_path}",
                "--data.dyn_bsz_buffer_size=1",
                "--train.global_batch_size=16",
                "--train.micro_batch_size=1",
                "--train.data_parallel_mode=fsdp2",
                "--model.attn_implementation=flash_attention_2",
                "--model.moe_implementation=fused",
                "--train.init_device=meta",
                f"--train.ulysses_parallel_size={mode.sp_size}",
                f"--train.expert_parallel_size={mode.ep_size}",
                "--train.bsz_warmup_ratio=0",
                "--train.num_train_epochs=1",
                "--train.save_epochs=0",
                "--train.save_steps=0",
                "--train.save_hf_weights=False",
                "--train.enable_full_determinism=True",
                "--train.enable_batch_invariant_mode=True",
                "--train.max_steps=5",
                f"--train.output_dir={os.path.join(output_dir, f'{model_name}_{task}_{mode}')}",
                f"--model.model_path={model_path}",
            ]
            task_name = f"{model_name}_{task}_{mode}"
            command_list.append((task_name, command))

    return command_list


def print_all_values(output_dict, value_key: str, model_type: str = ""):
    console = Console()
    table = Table(title=f"Alignment Result: [bold magenta]{model_type} {value_key}[/bold magenta]")

    table.add_column("Task", style="cyan", justify="left")

    table.add_column(value_key.upper(), style="bold green", justify="right")

    for task_name, output in output_dict.items():
        row_cells = []
        row_cells.append(task_name)

        val_list = output.get(value_key)
        row_cells.append(", ".join([f"{v:.8f}" for v in val_list]))

        table.add_row(*row_cells)

    console.print(table)


def compare_multi_items(model_name: str, outputs_dict: Dict, rtol=0.01, atol=0.01):
    base_task = next(iter(outputs_dict))
    base_output = outputs_dict[base_task]

    for task, output in outputs_dict.items():
        if task == base_task:
            continue
        for key in output.keys():
            try:
                torch.testing.assert_close(
                    output[key],
                    base_output[key],
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError:
                print_all_values(outputs_dict, key, model_name)
                raise AssertionError(f"{key} not match")
