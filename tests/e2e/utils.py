import os
import random
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..tools import DummyDataset as DummyDataset
from ..tools import compare_metrics, print_comparison_table


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


# DummyDataset is imported from tests.tools and re-exported for backward compatibility.


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
    max_sp_size: int | None = None,
) -> str:
    """Build torchrun commands for every (task, parallel-mode) combination.

    Args:
        test_tasks: Script basenames under tests/e2e/ to run (e.g. ["train_text_test"]).
        model_name: Short name used for directory naming and log output.
        config_path: Path to the model's toy config directory or config.json.
        model_path: Path to materialized model weights.
        train_path: Path to the dummy training dataset directory.
        output_dir: Root directory for per-run output (logs, checkpoints).
        is_moe: If True, also iterates over ep_size values (expert parallelism).
        max_sp_size: If set, filters out modes with sp_size > this value.
            Use 1 to skip sp=2 when the model does not support sequence parallelism yet.

    Returns:
        List of (task_name, command) tuples, where command is a list of strings
        suitable for subprocess.run.
    """
    model_modes: ModelMode = _base_model_modes() if not is_moe else _moe_model_modes()
    if max_sp_size is not None:
        model_modes = [m for m in model_modes if m.sp_size <= max_sp_size]

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
                "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
                "--model.ops_implementation.attn_implementation=flash_attention_2",
                "--model.ops_implementation.moe_implementation=fused",
                "--train.init_device=meta",
                f"--train.accelerator.ulysses_size={mode.sp_size}",
                f"--train.accelerator.ep_size={mode.ep_size}",
                "--train.bsz_warmup_ratio=0",
                "--train.num_train_epochs=1",
                "--train.checkpoint.save_epochs=0",
                "--train.checkpoint.save_steps=0",
                "--train.checkpoint.save_hf_weights=False",
                "--train.enable_full_determinism=True",
                "--train.enable_batch_invariant_mode=True",
                "--train.max_steps=2",
                f"--train.checkpoint.output_dir={os.path.join(output_dir, f'{model_name}_{task}_{mode}')}",
                f"--model.model_path={model_path}",
            ]
            task_name = f"{model_name}_{task}_{mode}"
            command_list.append((task_name, command))

    return command_list


def print_all_values(output_dict, value_key: str, model_type: str = ""):
    """Thin wrapper around tests.tools.print_comparison_table for backward compatibility."""
    print_comparison_table(output_dict, value_key, title=model_type)


def compare_multi_items(model_name: str, outputs_dict: Dict, rtol=0.01, atol=0.01):
    """Thin wrapper around tests.tools.compare_metrics for backward compatibility."""
    try:
        compare_metrics(outputs_dict, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"[{model_name}] {e}") from e
