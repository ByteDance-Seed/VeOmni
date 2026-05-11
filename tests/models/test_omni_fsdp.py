"""
Pytest wrapper for the two-model FSDP2 pipeline test.

Runs train_omni_fsdp_test.py via torchrun and validates that:
  - Both the encoder (model A) and LLM (model B) receive non-zero finite gradients.
  - Optimizer state is populated.
  - The forward–backward pipeline completes without errors.

Requires: at least 1 GPU (works on 1 GPU too; FSDP2 sharding only on 2+).
"""

import json
import os
import shutil
import subprocess

import pytest

from ..tools.launch_utils import find_free_port


_SCRIPT = "tests/train_scripts/train_omni_fsdp_test.py"


def _run_omni_fsdp_test(nproc: int, output_dir: str) -> dict:
    port = find_free_port()
    env = {**os.environ, "OMNI_FSDP_OUTPUT_DIR": output_dir}
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        _SCRIPT,
    ]
    result = subprocess.run(cmd, env=env, check=True, capture_output=False)
    assert result.returncode == 0
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def _tmp_dir(tmp_path):
    yield tmp_path


@pytest.mark.parametrize("nproc", [1, 2])
def test_omni_fsdp_pipeline(nproc, tmp_path):
    """Two separately FSDP2-wrapped models should train correctly end-to-end."""
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if nproc > gpu_count:
        pytest.skip(f"nproc={nproc} but only {gpu_count} GPU(s) available")

    output_dir = str(tmp_path / f"fsdp_nproc{nproc}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        metrics = _run_omni_fsdp_test(nproc=nproc, output_dir=output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    import math

    for i, (gne, gnl, opt_ok) in enumerate(
        zip(metrics["grad_norm_encoder"], metrics["grad_norm_llm"], metrics["optimizer_state_ok"])
    ):
        assert math.isfinite(gne) and gne > 0, f"step {i}: encoder grad_norm={gne}"
        assert math.isfinite(gnl) and gnl > 0, f"step {i}: llm grad_norm={gnl}"
        assert opt_ok, f"step {i}: optimizer state empty"
