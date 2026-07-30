from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import call, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_PATH = REPO_ROOT / "scripts/ci/install_triton_ascend.py"


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_triton_ascend", INSTALLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triton_ascend_configuration_is_reproducible():
    installer = _load_installer()

    installer.validate_configuration()
    assert [(name, version) for name, version, _url, _hash in installer.WHEELS] == [
        ("triton", "3.2.0"),
        ("triton-ascend", "3.2.1"),
    ]


def test_configuration_ignores_unrelated_installer_mentions(tmp_path):
    installer = _load_installer()
    workflows = []
    for workflow_path in installer.WORKFLOWS:
        copied_workflow = tmp_path / workflow_path.name
        copied_workflow.write_text(
            workflow_path.read_text(encoding="utf-8")
            + "\n# See scripts/ci/install_triton_ascend.py.\n"
            + "# uv run --frozen python scripts/ci/install_triton_ascend.py\n",
            encoding="utf-8",
        )
        workflows.append(copied_workflow)

    with patch.object(installer, "WORKFLOWS", tuple(workflows)):
        installer.validate_configuration()


def test_event_paths_rejects_nested_values():
    installer = _load_installer()
    workflow = """\
on:
  pull_request:
    paths:
      nested:
        - scripts/ci/install_triton_ascend.py
"""

    assert installer._event_paths(workflow, "pull_request") == set()


def test_installer_disables_dependency_resolution():
    installer = _load_installer()

    with patch.object(installer.subprocess, "run") as run:
        installer.install_wheels()

    assert run.call_args_list == [
        call(
            ["uv", "pip", "install", "--no-deps", f"{url}#sha256={sha256}"],
            check=True,
        )
        for _name, _version, url, sha256 in installer.WHEELS
    ]


def test_unsupported_python_error_names_cp311_wheels():
    installer = _load_installer()

    with (
        patch.object(installer.platform, "system", return_value="Linux"),
        patch.object(installer.platform, "machine", return_value="x86_64"),
        patch.object(installer.sys, "version_info", (3, 12)),
        pytest.raises(RuntimeError, match="cp311-only"),
    ):
        installer.ensure_supported_platform()
