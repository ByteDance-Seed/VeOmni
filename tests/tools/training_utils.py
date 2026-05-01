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

import pytest

from veomni.utils.import_utils import is_torch_npu_available

from .launch_utils import find_free_port


# ── ops_implementation overrides for tests ──────────────────────────────────
#
# OpsImplementationConfig defaults are GPU-optimal (Liger / Triton). On NPU
# those defaults raise at config validation time (liger_kernel is GPU-only),
# so every NPU test must explicitly opt every op down to an NPU-supported
# value (or to ``eager`` for ops with no NPU backend at all).
#
# ``resolve_ops_overrides(model_name)`` is the single source of truth for
# this mapping. It returns the list of ``--model.ops_implementation.X=Y``
# CLI flags to inject into ``torchrun`` commands so every test on every
# accelerator passes the validator with the fastest backend that actually
# runs on the host.

# Default NPU values per op. Models without an NPU kernel for a given op
# override these in ``_NPU_PER_MODEL_OVERRIDES`` below.
_NPU_OPS_DEFAULTS: Dict[str, str] = {
    "attn_implementation": "flash_attention_2",
    "moe_implementation": "fused_npu",
    "cross_entropy_loss_implementation": "chunk_loss",
    "rms_norm_implementation": "npu",
    "rotary_pos_emb_implementation": "npu",
    # SwiGLU has no NPU backend in the registry — eager is the only option.
    "swiglu_mlp_implementation": "eager",
    # NPU ships ``triton-ascend`` (not mainline ``triton``); the existing
    # ``tests/models/utils.py`` infra already gates the fused load-balancing-loss
    # kernel behind ``is_package_available("triton")`` for the same reason. The
    # validator uses the same check, so the ``"triton"`` value would raise on
    # the NPU CI runner. Default to eager for safety.
    "load_balancing_loss_implementation": "eager",
}

# Per-model overrides: ops that exist in the registry but have no NPU backend
# for this specific model. These fall back to ``eager``. Anything not listed
# here picks up ``_NPU_OPS_DEFAULTS``.
_NPU_PER_MODEL_OVERRIDES: Dict[str, Dict[str, str]] = {
    # DeepSeek-V3 only registers GPU-only Triton kernels for RMSNorm
    # (batch-invariant) and RoPE (deterministic). Eager fallback on NPU.
    "deepseek_v3": {
        "rms_norm_implementation": "eager",
        "rotary_pos_emb_implementation": "eager",
    },
    # Multimodal RoPE has no NPU backend in any of the Qwen-VL family
    # (Qwen2-VL / Qwen2.5-VL / Qwen2.5-Omni share the same multimodal RoPE
    # symbol via qwen2_vl/device_patch.py). Eager fallback on NPU.
    "qwen2vl": {"rotary_pos_emb_implementation": "eager"},
    "qwen25vl": {"rotary_pos_emb_implementation": "eager"},
    "qwen25_omni": {"rotary_pos_emb_implementation": "eager"},
}

# GPU baseline. The OpsImplementationConfig defaults are already GPU-optimal,
# so these flags are effectively redundant — we still pass them explicitly so
# the CLI invocation matches what production users see in ``configs/*.yaml``
# and a future default change cannot silently shift CI semantics.
#
# NOTE: the unset fields (cross_entropy / rms_norm / rotary / swiglu /
# load_balancing) inherit the public OpsImplementationConfig defaults
# (``liger_kernel`` / ``triton``). Those defaults silently require
# ``liger-kernel`` and ``triton`` to be installed on the GPU CI image (both
# ship in the ``gpu`` extra of pyproject.toml). If a future GPU image drops
# either package, every test using this baseline will fail at config
# validation time with a clear error message — but the link back here may
# not be obvious. Add explicit ``--model.ops_implementation.X=eager`` flags
# below if the CI image diverges from the ``gpu`` extra.
_GPU_OPS_DEFAULTS: Dict[str, str] = {
    "attn_implementation": "flash_attention_2",
    "moe_implementation": "fused_triton",
}

# Models that cannot run on NPU at all — e.g. a forward path requires a kernel
# that has no NPU backend AND no eager fallback. Tests should mark the model
# parametrization with ``npu_skip_marker(model_name)`` so the test is skipped
# (rather than failing in validation) when ``is_torch_npu_available()``.
#
# Empty today; populated when models with strict eager-incompatible kernels
# land (e.g. Qwen3.5 GatedDeltaNet's ``chunk_gated_delta_rule`` OpSlot, which
# raises in eager mode for varlen training).
_NPU_SKIP_MODELS: set = set()


def resolve_ops_overrides(model_name: Optional[str]) -> List[str]:
    """Return the list of ``--model.ops_implementation.X=Y`` flags to apply
    for *model_name* on the active hardware.

    On GPU the OpsImplementationConfig defaults are already optimal; we still
    emit the attention + MoE flags explicitly to match production YAML.

    On NPU we override every configurable op to an NPU-supported value (or
    eager when no NPU backend exists for that op or that model). The returned
    flags pass the OpsImplementationConfig._validate_implementations gate.

    Pass ``model_name=None`` for tests that don't tie to a specific model
    (e.g. raw distributed tests); the result is the GPU/NPU baseline without
    any model-specific eager fallbacks.
    """
    if is_torch_npu_available():
        merged: Dict[str, str] = dict(_NPU_OPS_DEFAULTS)
        if model_name is not None:
            merged.update(_NPU_PER_MODEL_OVERRIDES.get(model_name, {}))
        return [f"--model.ops_implementation.{k}={v}" for k, v in merged.items()]
    return [f"--model.ops_implementation.{k}={v}" for k, v in _GPU_OPS_DEFAULTS.items()]


def npu_skip_marker(model_name: str):
    """``pytest.mark.skipif`` that skips a model's test parametrization on NPU
    when no NPU+eager combination of ops backends can run the model.

    Use as ``marks=[npu_skip_marker(model_name)]`` (or list-extended onto an
    existing marks list) on ``pytest.param(...)`` entries. The skip activates
    only when ``is_torch_npu_available()``; on GPU the marker is a no-op.
    """
    return pytest.mark.skipif(
        is_torch_npu_available() and model_name in _NPU_SKIP_MODELS,
        reason=(
            f"{model_name} has no NPU-compatible kernel for at least one "
            "required op (and no eager fallback) — skipping on NPU."
        ),
    )


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
            ``"deepseek_v3"``). Used by ``resolve_ops_overrides`` to pick
            NPU-compatible per-op backends when running on NPU. Pass ``None``
            for tests that don't tie to a specific model — the GPU/NPU
            baseline is still emitted but no model-specific eager fallbacks
            are applied.
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
