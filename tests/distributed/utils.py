"""Distributed test utilities for VeOmni.

Provides helpers for setting up distributed training environments,
building models with toy configs, and comparing training metrics
across different parallelism configurations.
"""

import gc
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from rich.console import Console
from rich.table import Table


TOY_CONFIG_DIR = Path(__file__).parent.parent / "toy_config"


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_toy_config_path(model_name: str) -> str:
    """Return path to toy config for a model."""
    dir_path = TOY_CONFIG_DIR / f"{model_name}_toy"
    if dir_path.is_dir():
        config_json = dir_path / "config.json"
        if config_json.exists():
            return str(config_json)
        return str(dir_path)
    raise FileNotFoundError(f"No toy config found for {model_name} at {dir_path}")


def release_device_memory():
    """Synchronize GPU, run garbage collection, and empty CUDA cache."""
    from veomni.utils.device import empty_cache, synchronize

    synchronize()
    gc.collect()
    empty_cache()


def compare_metrics(
    outputs: Dict[str, Dict[str, Any]],
    *,
    rtol: float = 0.01,
    atol: float = 0.01,
    keys: Optional[Sequence[str]] = None,
) -> None:
    """Compare metrics across multiple runs.

    Args:
        outputs: {run_name: {metric_name: value}} mapping.
        rtol: Relative tolerance for torch.testing.assert_close.
        atol: Absolute tolerance for torch.testing.assert_close.
        keys: If provided, only compare these metric keys. Otherwise compare all.

    Raises:
        AssertionError: If any metric differs beyond tolerance.
    """
    base_name = next(iter(outputs))
    base = outputs[base_name]
    check_keys = keys or list(base.keys())

    for run_name, run_output in outputs.items():
        if run_name == base_name:
            continue
        for key in check_keys:
            base_val = base[key]
            run_val = run_output[key]
            if not isinstance(base_val, torch.Tensor):
                base_val = torch.tensor(base_val)
            if not isinstance(run_val, torch.Tensor):
                run_val = torch.tensor(run_val)
            try:
                torch.testing.assert_close(run_val, base_val, rtol=rtol, atol=atol)
            except AssertionError:
                print_comparison_table(outputs, key)
                raise AssertionError(f"Metric '{key}' mismatch: {base_name} vs {run_name}")


def print_comparison_table(
    outputs: Dict[str, Any],
    metric_key: str,
    title: str = "",
) -> None:
    """Pretty-print a comparison table for a single metric across runs."""
    console = Console()
    table = Table(title=f"Comparison: {title} {metric_key}")
    table.add_column("Run", style="cyan")
    table.add_column(metric_key.upper(), style="bold green", justify="right")

    for name, output in outputs.items():
        val = output.get(metric_key, "N/A")
        if isinstance(val, (list, tuple)):
            val_str = ", ".join(f"{v:.8f}" for v in val)
        elif hasattr(val, "item"):
            val_str = f"{val.item():.8f}"
        elif isinstance(val, float):
            val_str = f"{val:.8f}"
        else:
            val_str = str(val)
        table.add_row(str(name), val_str)

    console.print(table)


@dataclass(frozen=True)
class ParallelConfig:
    """Describes a parallelism configuration for distributed tests."""

    sp_size: int = 1
    ep_size: int = 1
    fsdp_mode: str = "fsdp2"

    @property
    def world_size(self) -> int:
        return max(self.sp_size * 2, 2)

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
) -> List[str]:
    """Build a torchrun command for distributed test execution."""
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
        "--train.init_device=meta",
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
