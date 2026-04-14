import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..tools import DummyDataset as DummyDataset
from ..tools import ParallelConfig


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


@dataclass(frozen=True)
class ParallelMode:
    sp_size: int
    ep_size: int

    def __str__(self):
        return f"_[sp-{self.sp_size}]_[ep-{self.ep_size}]"


_SP_SIZE = [1, 2]
_EP_SIZE = [1, 2]


def _base_model_modes():
    modes = []
    for sp_size in _SP_SIZE:
        modes.append(ParallelMode(sp_size, 1))
    return modes


def _moe_model_modes():
    modes = []
    for sp_size in _SP_SIZE:
        for ep_size in _EP_SIZE:
            modes.append(ParallelMode(sp_size, ep_size))
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
) -> list[tuple[str, dict]]:
    """Prepare torchrun command kwargs for every (task, parallel-mode) combination.

    Port allocation is deferred to execution time to avoid TOCTOU races —
    the caller must pass each dict to ``build_torchrun_cmd(**kwargs)`` right
    before ``subprocess.run()``.

    Args:
        test_tasks: Script basenames under tests/train_scripts/ to run (e.g. ["train_text_test"]).
        model_name: Short name used for directory naming and log output.
        config_path: Path to the model's toy config directory or config.json.
        model_path: Path to materialized model weights.
        train_path: Path to the dummy training dataset directory.
        output_dir: Root directory for per-run output (logs, checkpoints).
        is_moe: If True, also iterates over ep_size values (expert parallelism).
        max_sp_size: If set, filters out modes with sp_size > this value.
            Use 1 to skip sp=2 when the model does not support sequence parallelism yet.

    Returns:
        List of (task_name, cmd_kwargs) tuples, where cmd_kwargs is a dict of
        keyword arguments for :func:`build_torchrun_cmd`.
    """
    model_modes: list[ParallelMode] = _base_model_modes() if not is_moe else _moe_model_modes()
    if max_sp_size is not None:
        model_modes = [m for m in model_modes if m.sp_size <= max_sp_size]

    command_list = []
    for task in test_tasks:
        for mode in model_modes:
            task_name = f"{model_name}_{task}_{mode}"
            cmd_kwargs = dict(
                script=f"tests/train_scripts/{task}.py",
                config_path=config_path,
                model_path=model_path,
                train_path=train_path,
                output_dir=os.path.join(output_dir, task_name),
                parallel_config=ParallelConfig(sp_size=mode.sp_size, ep_size=mode.ep_size, fsdp_mode="fsdp2"),
                nproc=mode.sp_size * 4,
            )
            command_list.append((task_name, cmd_kwargs))

    return command_list
