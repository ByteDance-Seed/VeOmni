"""Dry-run the selective CI distribution installer without any NPU libraries."""

from pathlib import Path

import pytest

from scripts.ci.install_image_fla_npu import install_distribution


def fixture_distribution(tmp_path):
    """Build an inert image package and compatible metadata manifest."""
    package = tmp_path / "image" / "fla_npu"
    info = package.parent / "fla_npu-1.0.0.dist-info"
    package.mkdir(parents=True)
    info.mkdir()
    (package / "custom_aclnn_extension_lib.so").write_bytes(b"fixture")
    (info / "METADATA").write_text("Name: fla_npu\nVersion: 1.0.0\n")
    runtime = dict(python=[3, 12], machine="x86_64", torch="2.10.0+cpu", torch_npu="2.10.0.post2")
    return dict(
        runtime, prefix=str(package.parent), package=str(package), metadata=str(info), version="1.0.0"
    ), runtime


def test_selective_install_does_not_copy_other_image_packages(tmp_path):
    source, runtime = fixture_distribution(tmp_path)
    (Path(source["package"]).parent / "unrelated").mkdir()
    target = tmp_path / "workspace"
    install_distribution(source, target, runtime)
    assert sorted(p.name for p in target.iterdir()) == ["fla_npu", "fla_npu-1.0.0.dist-info"]
    assert (target / "fla_npu" / "custom_aclnn_extension_lib.so").read_bytes() == b"fixture"
    with pytest.raises(FileExistsError):
        install_distribution(source, target, runtime)


@pytest.mark.parametrize("field", ["python", "machine", "torch", "torch_npu"])
def test_incompatible_abi_fails_before_writing(tmp_path, field):
    source, runtime = fixture_distribution(tmp_path)
    runtime[field] = "incompatible"
    target = tmp_path / "workspace"
    with pytest.raises(RuntimeError, match="mismatch"):
        install_distribution(source, target, runtime)
    assert not target.exists()


def test_native_runner_does_not_resync_after_image_overlay():
    """The matched TorchNPU extension must survive until native test execution."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    for name in ("npu_unit_tests.yml", "npu_e2e_test.yml"):
        workflow = yaml.safe_load((root / ".github/workflows" / name).read_text())
        job = next(job for job in workflow["jobs"].values() if "steps" in job)
        steps = job["steps"]
        overlay = next(i for i, step in enumerate(steps) if "install_image_fla_npu.py" in step.get("run", ""))
        assert all("uv run" not in step.get("run", "") for step in steps[overlay:])
        runner = next(step["run"] for step in steps[overlay:] if "scripts.ci.accelerator" in step.get("run", ""))
        assert ".venv/bin/python -m scripts.ci.accelerator" in runner
