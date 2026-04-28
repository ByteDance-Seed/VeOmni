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

from veomni.utils.import_utils import is_liger_kernel_available, is_package_available, is_torch_npu_available

from .launch_utils import find_free_port


# Pick host-appropriate ops_implementation values. NPU users must opt in
# explicitly for every GPU-only kernel; without these, the e2e tests would
# trip ``OpsImplementationConfig._validate_device_compatibility`` (added in
# the kernel-defaults refactor). Hosts without ``liger-kernel`` installed
# (e.g. NPU CI) need ``eager`` for the Liger fields too.
_IS_NPU = is_torch_npu_available()
_HAS_LIGER = is_liger_kernel_available()
# ``triton`` (CUDA) and ``triton-ascend`` (NPU) both expose the same import
# name. Treat ``triton`` as available iff we can import it, regardless of
# device — the load-balancing-loss kernel works on both stacks but the
# standard ``--extra npu`` install does NOT ship triton-ascend.
_HAS_TRITON = is_package_available("triton")
_FUSED_MOE_IMPL = "fused_npu" if _IS_NPU else "fused_triton"
# Attention: keep ``flash_attention_2`` on both GPU and NPU. The Ulysses-SP
# rewrite in ``OpsImplementationConfig.__post_init__`` only kicks in for
# ``flash_attention_*`` names → ``veomni_flash_attention_*_with_sp`` (which
# does the cross-rank Q/K/V gather/scatter). ``sdpa`` is not in the rewrite
# map, so picking it on NPU with ``ulysses_size>1`` enables SP in
# ``parallel_state`` but leaves attention running locally per rank — the
# sp1/sp2 e2e alignment test (``test_text_parallel_align``) catches that
# regression. NPU + FA2 is supported by the OSS NPU CI image and by
# downstream third-party FA-on-Ascend providers.
_ATTN_IMPL = "flash_attention_2"
# RMSNorm / RoPE: NPU has its own fused kernel; GPU uses Liger if available.
_RMS_NORM_IMPL = "npu" if _IS_NPU else ("liger_kernel" if _HAS_LIGER else "eager")
_ROTARY_IMPL = "npu" if _IS_NPU else ("liger_kernel" if _HAS_LIGER else "eager")
# SwiGLU has no NPU fused kernel; CE on NPU goes through chunk_loss.
_SWIGLU_IMPL = "eager" if _IS_NPU else ("liger_kernel" if _HAS_LIGER else "eager")
_CE_IMPL = "npu" if _IS_NPU else ("liger_kernel" if _HAS_LIGER else "eager")
# Load-balancing-loss: triton-ascend is not in the standard ``--extra npu``
# install, so we can't unconditionally pick triton on NPU. Fall back to
# eager whenever the triton import would fail.
_LB_LOSS_IMPL = "triton" if _HAS_TRITON else "eager"


def host_appropriate_ops_cli_args(
    *,
    separator: str = "=",
    attn_implementation: Optional[str] = None,
    eager_only: bool = False,
) -> List[str]:
    """Return ``--model.ops_implementation.*`` CLI args that work on this host.

    The new GPU-reasonable defaults on ``OpsImplementationConfig`` raise on
    NPU for any field whose default is GPU-only (Liger / fused_triton /
    flash_attention_2). Tests that subprocess into a training script via
    torchrun therefore need to spell out the per-host values explicitly so
    ``OpsImplementationConfig.__post_init__`` does not blow up inside the
    child process.

    Args:
        separator: ``"="`` for ``--key=value`` form (used by build_torchrun_cmd /
            most train scripts), or ``" "`` for the ``--key value`` form used
            by ``tests/checkpoints/utils.py``.
        attn_implementation: Override for the attention impl. When ``None``
            (default) the helper picks a host-appropriate value (``sdpa`` on
            NPU, ``flash_attention_2`` on GPU). Pass an explicit value to
            keep the existing-test behaviour where attention mode is
            chosen by the test.
        eager_only: If True, every kernel field is set to ``"eager"`` —
            universal and has zero runtime dependencies. Suitable for tests
            that don't actually exercise model kernels (data-pipeline tests,
            arg-parsing tests). Equivalent to passing
            ``OpsImplementationConfig.eager_defaults()`` via CLI.
    """
    if eager_only:
        attn = attn_implementation if attn_implementation is not None else "eager"
        return [
            f"--model.ops_implementation.attn_implementation{separator}{attn}",
            f"--model.ops_implementation.moe_implementation{separator}eager",
            f"--model.ops_implementation.rms_norm_implementation{separator}eager",
            f"--model.ops_implementation.rotary_pos_emb_implementation{separator}eager",
            f"--model.ops_implementation.swiglu_mlp_implementation{separator}eager",
            f"--model.ops_implementation.cross_entropy_loss_implementation{separator}eager",
            f"--model.ops_implementation.load_balancing_loss_implementation{separator}eager",
        ]

    attn = attn_implementation if attn_implementation is not None else _ATTN_IMPL
    return [
        f"--model.ops_implementation.attn_implementation{separator}{attn}",
        f"--model.ops_implementation.moe_implementation{separator}{_FUSED_MOE_IMPL}",
        f"--model.ops_implementation.rms_norm_implementation{separator}{_RMS_NORM_IMPL}",
        f"--model.ops_implementation.rotary_pos_emb_implementation{separator}{_ROTARY_IMPL}",
        f"--model.ops_implementation.swiglu_mlp_implementation{separator}{_SWIGLU_IMPL}",
        f"--model.ops_implementation.cross_entropy_loss_implementation{separator}{_CE_IMPL}",
        f"--model.ops_implementation.load_balancing_loss_implementation{separator}{_LB_LOSS_IMPL}",
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
    parallel_config: Optional[ParallelConfig] = None,
    extra_args: Optional[List[str]] = None,
    nproc: Optional[int] = None,
    init_device: str = "meta",
) -> List[str]:
    """Build a torchrun command for distributed test execution.

    Args:
        parallel_config: Parallelism configuration. When None, no parallel
            args (fsdp_mode, ulysses_size, ep_size) are passed -- suitable
            for plain single-GPU training.
        init_device: Device for model initialization. Use "meta" for FSDP
            (multi-GPU), device type for single-GPU (no FSDP wrapping).
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
        *host_appropriate_ops_cli_args(),
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
    task_name: str,
    parallel_config: Optional[ParallelConfig] = None,
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
