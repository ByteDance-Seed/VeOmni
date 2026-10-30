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

"""Run the existing MXFP4 checks on one Modal container with two A100 GPUs."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/opt/veomni"
tracked = set()
if modal.is_local():
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", "veomni", "tests", "scripts", "pyproject.toml", "uv.lock"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("build-essential", "git", "ffmpeg", "libibverbs1")
    # Modal 1.6.3's uv_sync uses a TOML parser that rejects this valid pyproject.
    .dockerfile_commands("COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/")
    .add_local_file(ROOT / "pyproject.toml", "/opt/veomni-env/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/opt/veomni-env/uv.lock", copy=True)
    .run_commands(
        "uv sync --project /opt/veomni-env --python 3.12 --frozen --extra gpu --group test "
        "--no-install-workspace --compile-bytecode"
    )
    .dockerfile_commands("ENV PATH=/opt/veomni-env/.venv/bin:$PATH")
    .env({"PYTHONPATH": REMOTE, "PYTHONUNBUFFERED": "1", "MODELING_BACKEND": "veomni"})
    .workdir(REMOTE)
    .add_local_dir(ROOT, REMOTE, ignore=lambda path: path.as_posix() not in tracked)
)
app = modal.App("veomni-mxfp4-qat")


@app.function(
    image=image,
    gpu="A100-40GB:2",
    cpu=8,
    memory=32768,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
)
def validate(revision: str):
    report = Path("/tmp/mxfp4-junit.xml")
    gpu_info = subprocess.check_output(["nvidia-smi"], text=True)
    packages = ", ".join(f"{name}={version(name)}" for name in ("torch", "torchao", "transformers"))
    log = [f"VeOmni: {revision}\nPython: {sys.version}\n{packages}\n{gpu_info}\n"]
    print(log[0], end="", flush=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/models/test_mxfp4_qat.py",
        "tests/models/test_mxfp4_qat_fsdp.py",
        "-q",
        "-rs",
        "-s",
        f"--junitxml={report}",
    ]
    # pytest owns the two-rank launch; wrapping it in torchrun would duplicate ranks.
    with subprocess.Popen(command, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.append(line)
        returncode = process.wait()
    return {"returncode": returncode, "log": "".join(log), "junit": report.read_text() if report.exists() else ""}


@app.local_entrypoint()
def main(output_dir: str = "outputs/mxfp4-modal"):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        revision += " (with working-tree changes)"
    result = validate.remote(revision)
    output = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.log").write_text(result["log"])
    (output / "junit.xml").write_text(result["junit"])
    print(f"Saved logs and JUnit report to {output.resolve()}")
    # A dependency/hardware skip is not successful GPU acceptance.
    cases = ET.fromstring(result["junit"]).findall(".//testcase") if result["junit"] else []
    if (
        result["returncode"]
        or len(cases) != 2
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
    ):
        raise RuntimeError("MXFP4 acceptance requires both tests to pass without skips; see the saved report.")
