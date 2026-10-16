"""
Pytest wrapper for SeedOmni multi-module runtime acceptance tests (#1064).

Runs ``tests/train_scripts/train_omni_fsdp_test.py`` via ``torchrun`` and validates:
  1. Two-model FSDP2 / DDP pipeline forward-backward-step smoke.
  2. ``sp1`` vs ``sp4`` per-module ``trace/<module>/consume_tokens`` exact equality.
  3. Per-module and global grad-clip numerical parity against a single-model
     reference under DDP(+SP), FSDP2, and FSDP2+EP.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess

import pytest

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_torch_device

from ..tools.launch_utils import find_free_port


_SCRIPT = "tests/train_scripts/train_omni_fsdp_test.py"
_DEVICE_COUNT = get_torch_device().device_count() if (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) else 0


def _run_omni_fsdp_test(nproc: int, output_dir: str, extra_args: list[str] | None = None) -> dict:
    port = find_free_port()
    env = {**os.environ, "OMNI_FSDP_OUTPUT_DIR": output_dir}
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        _SCRIPT,
        *(extra_args or []),
    ]
    result = subprocess.run(cmd, env=env, check=True, capture_output=False)
    assert result.returncode == 0
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("nproc", [1, 2])
def test_omni_fsdp_pipeline(nproc, tmp_path):
    """Two separately FSDP2-wrapped models should train correctly end-to-end."""
    if nproc > _DEVICE_COUNT:
        pytest.skip(f"nproc={nproc} but only {_DEVICE_COUNT} accelerator(s) available")

    output_dir = str(tmp_path / f"fsdp_nproc{nproc}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        metrics = _run_omni_fsdp_test(nproc=nproc, output_dir=output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    assert metrics["grad_norm_encoder"], "worker recorded no steps"

    for i, (gne, gnl, opt_ok) in enumerate(
        zip(metrics["grad_norm_encoder"], metrics["grad_norm_llm"], metrics["optimizer_state_ok"])
    ):
        assert math.isfinite(gne) and gne > 0, f"step {i}: encoder grad_norm={gne}"
        assert math.isfinite(gnl) and gnl > 0, f"step {i}: llm grad_norm={gnl}"
        assert opt_ok, f"step {i}: optimizer state empty"


@pytest.mark.skipif(_DEVICE_COUNT < 4, reason="sp1 vs sp4 comparison requires at least 4 accelerators")
def test_omni_sp1_vs_sp4_consume_tokens(tmp_path):
    """sp1 (dp=4) and sp4 (dp=1) must report identical per-module consume_tokens to the digit."""
    sp1_dir = str(tmp_path / "sp1")
    sp4_dir = str(tmp_path / "sp4")

    sp1 = _run_omni_fsdp_test(
        nproc=4,
        output_dir=sp1_dir,
        extra_args=["--mode=sp_consume_tokens", "--ulysses-size=1"],
    )
    sp4 = _run_omni_fsdp_test(
        nproc=4,
        output_dir=sp4_dir,
        extra_args=["--mode=sp_consume_tokens", "--ulysses-size=4"],
    )

    assert len(sp1["steps"]) == len(sp4["steps"]) == 2
    for step_sp1, step_sp4 in zip(sp1["steps"], sp4["steps"], strict=True):
        assert step_sp1["consumed_chunk_num"] == step_sp4["consumed_chunk_num"]
        assert step_sp1["consume_tokens"] == step_sp4["consume_tokens"]
        assert step_sp1["trace/vision_encoder/consume_tokens(M)"] == step_sp4["trace/vision_encoder/consume_tokens(M)"]
        assert step_sp1["trace/llm_backbone/consume_tokens(M)"] == step_sp4["trace/llm_backbone/consume_tokens(M)"]
        assert step_sp1["trace/vision_encoder/avg_seq_len"] == step_sp4["trace/vision_encoder/avg_seq_len"]
        assert step_sp1["trace/llm_backbone/avg_seq_len"] == step_sp4["trace/llm_backbone/avg_seq_len"]


@pytest.mark.parametrize(
    ("nproc", "dp_mode", "ulysses_size", "ep_size"),
    [
        pytest.param(4, "ddp", 4, 1, id="ddp_sp4"),
        pytest.param(2, "fsdp2", 1, 1, id="fsdp2"),
        pytest.param(4, "fsdp2", 1, 2, id="fsdp2_ep2"),
    ],
)
def test_omni_grad_clip_matches_single_model_reference(nproc, dp_mode, ulysses_size, ep_size, tmp_path):
    """Per-module and global Omni grad clip must match the single-model reference under DDP+SP, FSDP2, and FSDP2+EP."""
    if nproc > _DEVICE_COUNT:
        pytest.skip(f"nproc={nproc} but only {_DEVICE_COUNT} accelerator(s) available")

    out_dir = str(tmp_path / f"clip_{dp_mode}_sp{ulysses_size}_ep{ep_size}")
    metrics = _run_omni_fsdp_test(
        nproc=nproc,
        output_dir=out_dir,
        extra_args=[
            "--mode=grad_clip",
            f"--dp-mode={dp_mode}",
            f"--ulysses-size={ulysses_size}",
            f"--ep-size={ep_size}",
        ],
    )
    assert metrics["enc_norm"] > 0 and math.isfinite(metrics["enc_norm"])
    assert metrics["dec_norm"] > 0 and math.isfinite(metrics["dec_norm"])
    assert metrics["global_norm"] > 0 and math.isfinite(metrics["global_norm"])
