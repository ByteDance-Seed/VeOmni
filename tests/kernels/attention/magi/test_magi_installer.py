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
# See the License for the specific language governing limitations
# under the License.

"""Magi SM90 installer CLI contract. The script is not leftover ops attention."""

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALLER = REPO_ROOT / "scripts" / "kernel" / "install_magi_sm90.sh"


def _run_installer(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_config(output: str) -> dict[str, str]:
    return dict(line.split("=", maxsplit=1) for line in output.splitlines())


def test_magi_sm90_installer_prints_verified_default() -> None:
    result = _run_installer("--print-config")

    assert result.returncode == 0, result.stderr
    assert _parse_config(result.stdout) == {
        "dtypes": "BF16+FP16",
        "head_dim_buckets": "128",
        "nfunc": "1,3,5",
        "max_jobs": "2",
        "nvcc_threads": "4",
        "split_compile": "32",
        "local_version": "sm90.fp16bf16.hdim128.nfunc1x3x5.split32",
    }


def test_magi_sm90_installer_canonicalizes_cli_overrides() -> None:
    result = _run_installer(
        "--dim",
        "256,64,128",
        "--nfunc",
        "5,1",
        "--dtype",
        "fp16,bf16",
        "--max-jobs",
        "3",
        "--nvcc-threads",
        "2",
        "--split-compile",
        "16",
        "--print-config",
    )

    assert result.returncode == 0, result.stderr
    assert _parse_config(result.stdout) == {
        "dtypes": "BF16+FP16",
        "head_dim_buckets": "64,128,256",
        "nfunc": "1,5",
        "max_jobs": "3",
        "nvcc_threads": "2",
        "split_compile": "16",
        "local_version": "sm90.fp16bf16.hdim64x128x256.nfunc1x5.split16",
    }


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (("--dtype", "fp16"), "cannot disable BF16"),
        (("--dim", "80"), "--dim supports only 64,96,128,192,256"),
    ],
)
def test_magi_sm90_installer_rejects_invalid_configuration(
    arguments: tuple[str, ...],
    expected_message: str,
) -> None:
    result = _run_installer(*arguments, "--print-config")

    assert result.returncode == 2
    assert expected_message in result.stderr
