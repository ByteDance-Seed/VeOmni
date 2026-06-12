"""Opt-in BAGEL multi-rank FSDP2 trainability smoke."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tests.tools.launch_utils import find_free_port


_ENV_PREFIX = "VEOMNI_V2_TEST_BAGEL_"
_SCRIPT = "tests/seed_omni/bagel/train_bagel_fsdp2_worker.py"


def _env_name(suffix: str) -> str:
    return f"{_ENV_PREFIX}{suffix}"


def _env_value(suffix: str) -> str | None:
    return os.environ.get(_env_name(suffix))


def _env_flag(suffix: str) -> bool:
    value = _env_value(suffix)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _env_flag("ENABLE_PARITY_CHECK"),
    reason=f"Set {_env_name('ENABLE_PARITY_CHECK')}=1 to run BAGEL opt-in FSDP2 smoke.",
)


def _bagel_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "Bagel" / "bagel_7b_mot" / "base.yaml"


def test_bagel_two_rank_fsdp2_fixture_smoke(tmp_path: Path) -> None:
    fixture_path = _env_value("GRADIENT_CE_MSE_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('GRADIENT_CE_MSE_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL two-rank FSDP2 smoke."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip(f"BAGEL two-rank FSDP2 smoke requires 2 CUDA GPUs, got {torch.cuda.device_count()}.")

    output_dir = tmp_path / "bagel_fsdp2_smoke"
    port = find_free_port()
    env = {
        **os.environ,
        "BAGEL_FSDP_SMOKE_FIXTURE": fixture_path,
        "BAGEL_FSDP_SMOKE_OUTPUT_DIR": str(output_dir),
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "WARN",
    }
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master_port={port}",
        _SCRIPT,
        str(_bagel_config_path()),
        "--model.model_path",
        model_root,
        "--train.global_batch_size",
        "2",
        "--train.micro_batch_size",
        "1",
        "--train.max_steps",
        "1",
        "--train.wandb.enable",
        "false",
        "--train.checkpoint.output_dir",
        str(output_dir),
        "--train.checkpoint.save_steps",
        "0",
        "--train.checkpoint.save_hf_weights",
        "false",
        "--train.checkpoint.hf_save_steps",
        "0",
        "--train.gradient_checkpointing.enable",
        "false",
        "--data.dataloader.num_workers",
        "0",
        "--data.dataloader.drop_last",
        "false",
    ]

    try:
        result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=900)
        if result.returncode != 0:
            stdout_tail = result.stdout[-4000:]
            stderr_tail = result.stderr[-4000:]
            raise AssertionError(
                f"BAGEL FSDP2 smoke failed with exit code {result.returncode}\n"
                f"stdout tail:\n{stdout_tail}\n"
                f"stderr tail:\n{stderr_tail}"
            )
        report_path = output_dir / "results.json"
        assert report_path.exists(), f"Missing BAGEL FSDP2 smoke report at {report_path}"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["all_pass"], report
        assert report["dp_mode"] == "fsdp2", report
        assert len(report["ranks"]) == 2, report
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
