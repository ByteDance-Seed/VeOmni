"""Shared training utilities for distributed and e2e tests.

Provides helpers for building torchrun commands, materializing model weights,
running training configurations, and comparing results.
"""

import gc
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from veomni.utils.import_utils import is_torch_npu_available

from .launch_utils import find_free_port


# NPU ops_implementation overrides per model. The public
# ``OpsImplementationConfig`` defaults are GPU-optimal (Liger / Triton) and
# raise on NPU at config validation time, so every NPU test must override
# every per-op field. ``_NPU_OPS_DEFAULTS`` is the baseline; entries in
# ``_NPU_PER_MODEL_OVERRIDES`` (DeepSeek-V3, Qwen-VL family) pin specific
# fields to ``eager`` where the model has no NPU kernel.
_NPU_OPS_DEFAULTS: Dict[str, str] = {
    "attn_implementation": "flash_attention_2",
    "moe_implementation": "fused_npu",
    "cross_entropy_loss_implementation": "chunk_loss",
    "rms_norm_implementation": "npu",
    "rotary_pos_emb_implementation": "npu",
    "swiglu_mlp_implementation": "eager",  # no NPU backend
    # NPU ships ``triton-ascend`` (not mainline ``triton``); the validator
    # gates ``triton`` on ``is_package_available("triton")`` so the fused
    # load-balancing-loss kernel would raise on the NPU runner. Pin to eager.
    "load_balancing_loss_implementation": "eager",
}

_NPU_PER_MODEL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "deepseek_v3": {
        # batch-invariant RMSNorm + deterministic RoPE are GPU-only Triton
        "rms_norm_implementation": "eager",
        "rotary_pos_emb_implementation": "eager",
    },
    # Multimodal RoPE has no NPU backend in the Qwen-VL family.
    "qwen2vl": {"rotary_pos_emb_implementation": "eager"},
    "qwen25vl": {"rotary_pos_emb_implementation": "eager"},
    "qwen25_omni": {"rotary_pos_emb_implementation": "eager"},
}


def resolve_ops_overrides(model_name: Optional[str]) -> List[str]:
    """Return ``--model.ops_implementation.X=Y`` flags for the active hardware.

    On GPU returns ``[]`` — the dataclass defaults are already optimal.
    On NPU returns the NPU-supported backend per op, with per-model eager
    fallbacks for ops without an NPU kernel for that specific model.
    """
    if not is_torch_npu_available():
        return []
    merged = dict(_NPU_OPS_DEFAULTS)
    if model_name is not None:
        merged.update(_NPU_PER_MODEL_OVERRIDES.get(model_name, {}))
    return [f"--model.ops_implementation.{k}={v}" for k, v in merged.items()]


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
    parallel_config: Optional[ParallelConfig] = None,
    extra_args: Optional[List[str]] = None,
    nproc: Optional[int] = None,
    init_device: str = "meta",
    model_name: Optional[str] = None,
) -> List[str]:
    """Build a torchrun command for distributed test execution.

    Args:
        parallel_config: Parallelism configuration. When None, no parallel
            args (fsdp_mode, ulysses_size, ep_size) are passed -- suitable
            for plain single-GPU training.
        init_device: Device for model initialization. Use "meta" for FSDP
            (multi-GPU), device type for single-GPU (no FSDP wrapping).
        model_name: Short model identifier (e.g. ``"qwen3_moe"``,
            ``"deepseek_v3"``). Forwarded to ``resolve_ops_overrides`` so
            NPU runs pick the right per-model eager fallbacks. ``None`` is
            fine for tests not tied to a specific model.
    """
    port = find_free_port()
    if nproc is not None:
        n = nproc
    elif parallel_config is not None:
        n = parallel_config.world_size
    else:
        n = 1

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
        f"--train.init_device={init_device}",
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
        *resolve_ops_overrides(model_name),
    ]

    if parallel_config is not None:
        cmd.extend(
            [
                f"--train.accelerator.fsdp_config.fsdp_mode={parallel_config.fsdp_mode}",
                f"--train.accelerator.ulysses_size={parallel_config.sp_size}",
                f"--train.accelerator.ep_size={parallel_config.ep_size}",
            ]
        )

    if extra_args:
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Training helpers
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
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path, save_original_format=save_original_format)


def run_training_config(
    script: str,
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    task_name: str,
    parallel_config: Optional[ParallelConfig] = None,
    nproc: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
    init_device: str = "meta",
    model_name: Optional[str] = None,
) -> Dict:
    """Run a single training configuration and return metrics from log.

    Args:
        model_name: Short model identifier (forwarded to ``build_torchrun_cmd``
            for NPU-aware ops_implementation overrides).

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
        model_name=model_name,
    )

    print(f"\n{'=' * 60}")
    print(f"Running: {task_name}")
    print(f"Config: {parallel_config}")
    print(f"{'=' * 60}")

    subprocess.run(cmd, check=True)

    log_path = os.path.join(run_output_dir, "log_dict.json")
    with open(log_path) as f:
        return json.load(f)
