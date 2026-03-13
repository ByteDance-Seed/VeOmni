"""Shared distributed training test framework.

Provides a generic training loop for L2/L3 distributed tests that compares
training metrics (loss, gradient norms) across different parallelism configurations.
This module encapsulates the pattern of:
1. Materialize toy model weights
2. Run torchrun with different parallel configs
3. Collect per-step metrics from JSON logs
4. Compare metrics across configs within tolerance
"""

import json
import os
import subprocess
from typing import Dict, List, Optional

from .utils import (
    ParallelConfig,
    build_torchrun_cmd,
    compare_metrics,
    print_comparison_table,
)


def materialize_weights(config_path: str, output_path: str, save_original_format: bool = True) -> None:
    """Build a model from toy config and save random weights to disk.

    This avoids downloading real model weights for CI tests.
    """
    from veomni.models.auto import build_foundation_model
    from veomni.utils.device import get_device_type

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path, save_original_format=save_original_format)


def run_training_config(
    script: str,
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    parallel_config: ParallelConfig,
    task_name: str,
    nproc: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict:
    """Run a single training configuration and return metrics from log.

    Returns:
        Dict of {metric_name: list_of_values} loaded from the JSON log.
    """
    run_output_dir = os.path.join(output_dir, task_name)
    cmd = build_torchrun_cmd(
        script=script,
        config_path=config_path,
        model_path=model_path,
        train_path=train_path,
        output_dir=run_output_dir,
        parallel_config=parallel_config,
        nproc=nproc,
        extra_args=extra_args,
    )

    print(f"\n{'=' * 60}")
    print(f"Running: {task_name}")
    print(f"Config: {parallel_config}")
    print(f"{'=' * 60}")

    subprocess.run(cmd, check=True)

    log_path = os.path.join(run_output_dir, "log_dict.json")
    with open(log_path) as f:
        return json.load(f)


def run_and_compare(
    script: str,
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    configs: List[ParallelConfig],
    model_name: str,
    rtol: float = 0.1,
    atol: float = 0.1,
    nproc_override: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Run training with multiple parallel configs and compare metrics.

    This is the main entry point for L2/L3 distributed tests. It:
    1. Runs training for each ParallelConfig
    2. Collects per-step loss/grad_norm from JSON logs
    3. Compares all configs against the first (baseline) within tolerance

    Args:
        script: Path to the training script (e.g., tests/e2e/train_text_test.py).
        config_path: Path to the model's toy config.
        model_path: Path to materialized model weights.
        train_path: Path to the dummy training dataset.
        output_dir: Root directory for per-run output.
        configs: List of ParallelConfig to compare.
        model_name: Short model name for logging.
        rtol: Relative tolerance for metric comparison.
        atol: Absolute tolerance for metric comparison.
        nproc_override: If set, override world_size for all configs.
        extra_args: Additional command-line arguments.
    """
    results = {}
    log_keys = []

    for config in configs:
        task_name = f"{model_name}_{config}"
        nproc = nproc_override or config.world_size

        output = run_training_config(
            script=script,
            config_path=config_path,
            model_path=model_path,
            train_path=train_path,
            output_dir=output_dir,
            parallel_config=config,
            task_name=task_name,
            nproc=nproc,
            extra_args=extra_args,
        )

        if not log_keys:
            log_keys = list(output.keys())
        else:
            assert set(output.keys()) == set(log_keys), (
                f"Metric keys mismatch for {task_name}: expected {log_keys}, got {list(output.keys())}"
            )

        results[task_name] = output

    # Print all metrics for visual inspection
    for key in log_keys:
        print_comparison_table(results, key, title=model_name)

    # Compare all runs against baseline
    compare_metrics(results, rtol=rtol, atol=atol, keys=log_keys)
