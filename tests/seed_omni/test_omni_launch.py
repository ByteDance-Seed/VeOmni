"""Launch the fake ``fake_module_a -> fake_module_b`` omni model through the VeOmni entry points.

``configs/seed_omni/fake_model/train/base.yaml`` is driven exactly as a user
would: ``scripts/seed_omni/convert_model.py`` writes the split checkpoint,
``tasks/omni/train_omni.py`` trains it under ``torchrun`` and
``tasks/omni/infer_omni.py`` runs the generation FSM.

Eager inference needs no process group and runs anywhere. Training and
distributed inference need two CUDA devices: the launcher's module overlay
builds ``fake_module_a`` as DDP on ``cuda`` beside an FSDP2 ``fake_module_b``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from veomni.checkpoint import layout
from veomni.utils.device import IS_CUDA_AVAILABLE, get_torch_device

from ..tools.launch_utils import find_free_port


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = REPO_ROOT / "configs" / "seed_omni" / "fake_model"
BASE_YAML = CFG_DIR / "train" / "base.yaml"
MODULES = ("fake_module_a", "fake_module_b")
NUM_SAMPLES = 8

requires_two_cuda_devices = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or get_torch_device().device_count() < 2,
    reason="needs 2 CUDA devices: the fake launcher builds fake_module_a as DDP on cuda",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout + result.stderr)[-8000:]
        pytest.fail(f"{' '.join(cmd)} exited {result.returncode}:\n{output}")
    return result


def _torchrun(script: str, nproc: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={find_free_port()}",
        script,
    ]


@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fake_omni")
    source = root / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "fake_omni", "hidden_size": 8}), encoding="utf-8")
    output = root / "ckpt"
    _run(
        [
            sys.executable,
            "scripts/seed_omni/convert_model.py",
            "--model_path",
            str(source),
            "--output_dir",
            str(output),
        ]
    )
    return output


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fake_omni_data") / "tulu.jsonl"
    rows = [
        {
            "source_name": "tulu-3-sft-mixture",
            "conversations": {
                "messages": [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            },
        }
        for i in range(NUM_SAMPLES)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _infer_cmd(prefix: list[str], checkpoint: Path, modules_yaml: str, output_dir: Path) -> list[str]:
    return [
        *prefix,
        str(BASE_YAML),
        "--model.model_path",
        str(checkpoint),
        "--model.model_config.modules",
        str(CFG_DIR / "infer" / modules_yaml),
        "--infer.prompt",
        "hi",
        "--infer.output_dir",
        str(output_dir),
    ]


def test_eager_inference_launch(fake_checkpoint, tmp_path):
    """All-eager modules load as a bare ``OmniModel`` from the launcher's in-memory config."""
    _run(
        _infer_cmd([sys.executable, "tasks/omni/infer_omni.py"], fake_checkpoint, "modules_infer_eager.yaml", tmp_path)
    )
    # ``infer_type`` in base.yaml is ``infer_gen``; outputs nest under it.
    assert (tmp_path / "infer_gen" / "reply.txt").is_file()
    assert (tmp_path / "infer_gen" / "trace.txt").is_file()


@requires_two_cuda_devices
def test_train_then_resume_launch(fake_checkpoint, fake_data, tmp_path):
    train_cmd = [
        *_torchrun("tasks/omni/train_omni.py", nproc=2),
        str(BASE_YAML),
        "--model.model_path",
        str(fake_checkpoint),
        "--data.train_path",
        str(fake_data),
        "--train.checkpoint.output_dir",
        str(tmp_path),
    ]
    _run(train_cmd)

    # 8 samples over dp=2 with global_batch_size=4 → 2 steps (2 micro-batches each).
    step_root = Path(layout.step_dir(str(tmp_path / "checkpoints"), 2))
    assert layout.checkpoint_is_complete(str(step_root)), f"step 2 is not resumable: {sorted(os.listdir(step_root))}"
    for module in MODULES:
        assert os.path.isdir(layout.weights_dir(str(step_root), module)), module
        assert os.path.isdir(layout.hf_export_dir(str(step_root), module)), module
    for rank in range(2):
        assert os.path.isfile(layout.extra_state_path(str(step_root), rank))
    assert (tmp_path / layout.ASSETS_DIRNAME / "config.json").is_file()

    # Resuming at the last step reloads every module and the job state, then ends.
    _run([*train_cmd, "--train.checkpoint.load_path", "auto"])


@requires_two_cuda_devices
def test_distributed_inference_launch(fake_checkpoint, tmp_path):
    """Any non-eager module routes the request through ``OmniModelRuntime``, which traces the FSM."""
    _run(
        _infer_cmd(
            _torchrun("tasks/omni/infer_omni.py", nproc=2), fake_checkpoint, "modules_infer_fsdp.yaml", tmp_path
        )
    )
    trace = (tmp_path / "infer_gen" / "trace.txt").read_text(encoding="utf-8")
    assert "fake_module_b" in trace, trace
