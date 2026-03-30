"""Distributed test utilities for VeOmni.

Provides helpers for setting up distributed training environments,
building models with toy configs, running torchrun training, and
comparing training metrics across different parallelism configurations.
"""

import gc
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from veomni.testing import compare_metrics, find_free_port, print_comparison_table


# Re-export for consumers that previously imported from utils.py / _training_core.py
__all__ = [
    "ParallelConfig",
    "build_torchrun_cmd",
    "compare_metrics",
    "find_free_port",
    "materialize_weights",
    "print_comparison_table",
    "release_device_memory",
    "run_training_config",
]


def release_device_memory():
    """Synchronize GPU, run garbage collection, and empty CUDA cache."""
    from veomni.utils.device import empty_cache, synchronize

    synchronize()
    gc.collect()
    empty_cache()


@dataclass(frozen=True)
class ParallelConfig:
    """Describes a parallelism configuration for distributed tests."""

    sp_size: int = 1
    ep_size: int = 1
    fsdp_mode: str = "fsdp2"

    @property
    def world_size(self) -> int:
        return max(self.sp_size, self.ep_size) * 2

    def __str__(self) -> str:
        return f"fsdp_{self.fsdp_mode}_sp{self.sp_size}_ep{self.ep_size}"


def build_torchrun_cmd(
    script: str,
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    parallel_config: ParallelConfig,
    extra_args: Optional[List[str]] = None,
    nproc: Optional[int] = None,
    init_device: str = "meta",
) -> List[str]:
    """Build a torchrun command for distributed test execution.

    Args:
        init_device: Device for model initialization. Use "meta" for FSDP
            (multi-GPU), device type for single-GPU (no FSDP wrapping).
    """
    port = find_free_port()
    n = nproc or parallel_config.world_size

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={n}",
        f"--master_port={port}",
        script,
        f"--model.config_path={config_path}",
        f"--data.train_path={train_path}",
        "--data.dyn_bsz_buffer_size=1",
        "--train.global_batch_size=16",
        "--train.micro_batch_size=1",
        f"--train.accelerator.fsdp_config.fsdp_mode={parallel_config.fsdp_mode}",
        "--model.ops_implementation.attn_implementation=flash_attention_2",
        "--model.ops_implementation.moe_implementation=fused",
        f"--train.init_device={init_device}",
        f"--train.accelerator.ulysses_size={parallel_config.sp_size}",
        f"--train.accelerator.ep_size={parallel_config.ep_size}",
        "--train.bsz_warmup_ratio=0",
        "--train.num_train_epochs=1",
        "--train.checkpoint.save_epochs=0",
        "--train.checkpoint.save_steps=0",
        "--train.checkpoint.save_hf_weights=False",
        "--train.enable_full_determinism=True",
        "--train.enable_batch_invariant_mode=True",
        "--train.max_steps=2",
        f"--train.checkpoint.output_dir={output_dir}",
        f"--model.model_path={model_path}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Training helpers (previously in _training_core.py)
# ---------------------------------------------------------------------------


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
    init_device: str = "meta",
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
        init_device=init_device,
    )

    print(f"\n{'=' * 60}")
    print(f"Running: {task_name}")
    print(f"Config: {parallel_config}")
    print(f"{'=' * 60}")

    subprocess.run(cmd, check=True)

    log_path = os.path.join(run_output_dir, "log_dict.json")
    with open(log_path) as f:
        return json.load(f)
