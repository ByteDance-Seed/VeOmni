from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import call, patch


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
