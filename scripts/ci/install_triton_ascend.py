#!/usr/bin/env python3
"""Install the Triton-Ascend wheels validated by VeOmni's NPU CI."""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse


# Triton-Ascend 3.2.1 exposes exact pins for its development stack as runtime
# dependencies. Installing the two runtime wheels without dependency resolution
# preserves VeOmni's locked NPU environment. Hashes make the direct URLs
# immutable even if their hosts replace an artifact.
WHEELS = (
    (
        "triton",
        "3.2.0",
        "https://files.pythonhosted.org/packages/a7/2e/757d2280d4fefe7d33af7615124e7e298ae7b8e3bc4446cdb8e88b0f9bab/"
        "triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "8009a1fb093ee8546495e96731336a33fb8856a38e45bb4ab6affd6dbc3ba220",
    ),
    (
        "triton-ascend",
        "3.2.1",
        "https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/"
        "triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "34a899d476afbb65351adcd97ee228fe73564788dfa28859fb785f087fad2690",
    ),
)
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    REPO_ROOT / ".github/workflows/npu_e2e_test.yml",
    REPO_ROOT / ".github/workflows/npu_unit_tests.yml",
)
INSTALL_COMMAND = "uv run --frozen python scripts/ci/install_triton_ascend.py"
INSTALLER_PATH = "scripts/ci/install_triton_ascend.py"


def _event_paths(workflow: str, event: str) -> set[str]:
    """Return path filters for one block-style GitHub Actions event."""
    lines = workflow.splitlines()
    event_index = next((index for index, line in enumerate(lines) if line == f"  {event}:"), None)
    if event_index is None:
        return set()

    for index in range(event_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(line) - len(line.lstrip()) <= 2:
            break
        if line != "    paths:":
            continue

        paths = set()
        for path_line in lines[index + 1 :]:
            path_stripped = path_line.strip()
            if path_stripped and not path_stripped.startswith("#") and len(path_line) - len(path_line.lstrip()) <= 4:
                break
            if len(path_line) - len(path_line.lstrip()) == 6 and path_stripped.startswith("- "):
                paths.add(path_stripped[2:].strip("\"'"))
        return paths
    return set()


def validate_configuration() -> None:
    """Keep both NPU workflows on the same immutable installation path."""
    for distribution, _version, url, sha256 in WHEELS:
        if urlparse(url).scheme != "https" or not url.endswith(".whl"):
            raise RuntimeError(f"{distribution} must use a direct HTTPS wheel URL.")
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise RuntimeError(f"{distribution} must use a lowercase SHA256 digest.")

    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        if sum(line.strip() == INSTALL_COMMAND for line in workflow.splitlines()) != 1:
            raise RuntimeError(f"{workflow_path.name} must run the shared installer exactly once.")
        for event in ("push", "pull_request"):
            if INSTALLER_PATH not in _event_paths(workflow, event):
                raise RuntimeError(f"{workflow_path.name} must trigger {event} on installer changes.")
        if "--extra-index-url" in workflow or "triton-ascend.osinfra.cn/pypi/simple" in workflow:
            raise RuntimeError(f"{workflow_path.name} must not resolve Triton-Ascend from an extra index.")


def ensure_supported_platform() -> None:
    """Fail before downloading a wheel on an unsupported host."""
    current = (platform.system(), platform.machine(), sys.version_info[:2])
    if current != ("Linux", "x86_64", (3, 11)):
        system, machine, python_version = current
        version = ".".join(map(str, python_version))
        wheel_names = ", ".join(Path(urlparse(url).path).name for _name, _version, url, _sha256 in WHEELS)
        raise RuntimeError(
            f"The pinned Triton-Ascend wheels are cp311-only ({wheel_names}) and require Linux x86_64 with Python 3.11; "
            f"got {system} {machine} with Python {version}."
        )


def install_wheels() -> None:
    """Overlay the Ascend backend on the matching upstream Triton runtime."""
    for _distribution, _version, url, sha256 in WHEELS:
        subprocess.run(
            ["uv", "pip", "install", "--no-deps", f"{url}#sha256={sha256}"],
            check=True,
        )


def verify_installation() -> None:
    """Check both distributions and the backend file before launching tests."""
    for distribution, expected_version, _url, _sha256 in WHEELS:
        installed_version = metadata.version(distribution)
        if installed_version != expected_version:
            raise RuntimeError(f"Expected {distribution}=={expected_version}, found {installed_version}.")

    backend = Path(metadata.distribution("triton-ascend").locate_file("triton/backends/ascend/__init__.py"))
    if not backend.is_file():
        raise RuntimeError(f"Triton-Ascend backend is missing: {backend}")

    importlib.import_module("triton.backends.ascend")
    print("Verified triton==3.2.0 with triton-ascend==3.2.1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate pinned wheels and workflow integration without installing",
    )
    args = parser.parse_args()

    validate_configuration()
    if args.check_config:
        print("Verified the pinned Triton-Ascend CI configuration.")
        return

    ensure_supported_platform()
    install_wheels()
    importlib.invalidate_caches()
    verify_installation()


if __name__ == "__main__":
    main()
