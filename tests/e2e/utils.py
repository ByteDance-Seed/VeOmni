import re

import numpy as np
import pandas as pd


def parse_training_log(log_content) -> pd.DataFrame:
    r"""
    regex
    1. Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)  ->  Epoch 1/3
    2. .*?\|\s+(?P<step>\d+)/(?P<total_steps>\d+)    ->  1/46 (ignore bar)
    3. .*?loss:\s+(?P<loss>[\d\.]+)                  ->  loss: 1.34
    4. .*?grad_norm:\s+(?P<grad_norm>[\d\.]+)        ->  grad_norm: 7.52
    5. .*?lr:\s+(?P<lr>[\d\.eE+-]+)                  ->  lr: 1.00e-04
    """
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)"
        r".*?\|\s+(?P<step>\d+)/(?P<total_steps>\d+)"
        r".*?loss:\s+(?P<loss>[\d\.]+)"
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
