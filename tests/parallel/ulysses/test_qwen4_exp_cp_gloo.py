# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Collected CPU/Gloo gates for both generated Qwen4-Exp CP model variants."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_generated_cp_packed_forward_and_backward(backend, cp_size):
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        QWEN4_CP_TEST_MODULE=backend,
        QWEN4_CP_TEST_CP_SIZE=str(cp_size),
        QWEN4_CP_TEST_PACKED_EDGES="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    # U=2 deliberately does not divide the full combined SP size into QSA
    # heads. Packed boundaries exercise starts/ends around CP shard edges.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={2 * cp_size}",
            str(Path(__file__).with_name("test_qwen4_exp_cp.py")),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for gate in (
        "QSA packed/GQA",
        "PLE token/conv halos",
        "GDN packed recurrence",
        "packed text model with PLE/GDN/QSA/MoE/GR2",
        "packed text model with non-reentrant gradient checkpointing",
    ):
        assert f"PASS {gate}:" in result.stdout, result.stdout + result.stderr


def test_collecting_standalone_script_preserves_moe_package():
    before = sys.modules.get("veomni.distributed.moe")
    source = Path(__file__).with_name("test_qwen4_exp_cp.py")
    spec = importlib.util.spec_from_file_location("qwen4_cp_collection_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sys.modules.get("veomni.distributed.moe") is before
    assert "model_code" not in vars(module)
