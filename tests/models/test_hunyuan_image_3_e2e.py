# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Track E5: Hunyuan Image 3 T2I end-to-end training smoke.

Drives the real stack — toy parquet -> mapping dataset -> transform ->
MainCollator + metadata hook -> VLMTrainer -> flow forward/backward -> optimizer
— through tasks-style training via ``tests/train_scripts/train_hunyuan_image_3_test.py``,
and asserts finite per-step losses. posterior_cache latent source (no VAE weights).

Requires flash-attention varlen (SM80+ CUDA). The FSDP2 case needs 2 GPUs.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


_REPO_ROOT = Path(__file__).parents[2]
_TOY_CONFIG = _REPO_ROOT / "configs" / "multimodal" / "hunyuan_image_3" / "hunyuan_image_3_toy.yaml"
_TRAIN_SCRIPT = _REPO_ROOT / "tests" / "train_scripts" / "train_hunyuan_image_3_test.py"
_LATENT_CHANNELS = 4
_GRID = (2, 2)

_requires_flash_gpu = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 80,
    reason="Hunyuan Image 3 end-to-end training requires an NVIDIA CUDA SM80+ GPU.",
)


def _write_toy_parquet(path: Path, num_rows: int = 8) -> None:
    from datasets import Dataset

    generator = torch.Generator().manual_seed(0)
    rows = []
    for index in range(num_rows):
        mean = torch.randn(_LATENT_CHANNELS, _GRID[0], _GRID[1], generator=generator)
        rows.append(
            {
                "id": f"sample_{index}",
                "prompt": f"a toy prompt number {index}",
                "latent_mean": mean.tolist(),
                "latent_logvar": torch.zeros(_LATENT_CHANNELS, _GRID[0], _GRID[1]).tolist(),
            }
        )
    Dataset.from_list(rows).to_parquet(str(path))


def _run_training(tmp_path: Path, *, nproc: int, extra_args: list[str], port: int) -> dict:
    data_path = tmp_path / "toy_t2i.parquet"
    output_dir = tmp_path / "out"
    _write_toy_parquet(data_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        str(_TRAIN_SCRIPT),
        str(_TOY_CONFIG),
        f"--data.train_path={data_path}",
        f"--train.checkpoint.output_dir={output_dir}",
        *extra_args,
    ]
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    with open(output_dir / "log_dict.json") as handle:
        return json.load(handle)


def _assert_finite_training(log: dict, *, min_steps: int) -> None:
    losses = log["image_decoder_loss"]
    grad_norms = log["grad_norm"]
    assert len(losses) >= min_steps, f"expected >= {min_steps} steps, got {len(losses)}"
    assert all(math.isfinite(v) and v > 0 for v in losses), losses
    assert all(math.isfinite(v) for v in grad_norms), grad_norms


@_requires_flash_gpu
def test_end_to_end_single_gpu_posterior_cache(tmp_path):
    """Single-GPU (ddp/cuda, bf16) e2e: >=5 finite flow-loss steps."""
    log = _run_training(tmp_path, nproc=1, extra_args=[], port=29541)
    _assert_finite_training(log, min_steps=5)


@_requires_flash_gpu
def test_end_to_end_fsdp2_posterior_cache(tmp_path):
    """2-GPU FSDP2 meta-init + mixed precision e2e: >=4 finite flow-loss steps."""
    if torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 e2e needs 2 GPUs.")
    log = _run_training(
        tmp_path,
        nproc=2,
        extra_args=[
            "--train.global_batch_size=2",
            "--train.init_device=meta",
            "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
            "--train.accelerator.fsdp_config.mixed_precision.enable=true",
        ],
        port=29542,
    )
    _assert_finite_training(log, min_steps=4)
