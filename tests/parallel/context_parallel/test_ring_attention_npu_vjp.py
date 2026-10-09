"""Collected two-NPU gate for packed/dense Ring attention forward and VJP."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_ring_attention_native_npu_vjp():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        pytest.skip("Ring attention VJP requires two Ascend NPUs")
    root = Path(__file__).resolve().parents[3]
    script = Path(__file__).with_name("run_ring_attention_npu_vjp_gate.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
            "--seq-len=128",
            "--q-heads=4",
            "--kv-heads=2",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "AI4SE_RING_NPU_VJP_GATE_OK" in result.stdout, output
