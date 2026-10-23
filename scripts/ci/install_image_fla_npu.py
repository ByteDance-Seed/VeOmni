"""Install only the CI image's compiled fla_npu distribution in the workspace venv.

The image already contains the matching AscendC OPP libraries. Copying this one
installed distribution avoids rebuilding them and does not expose the image's
other Python packages through PYTHONPATH. Refuse incompatible interpreter/Torch
ABIs before copying any files.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


_SOURCE_QUERY = """
import importlib.metadata as m, json, platform, sys
from pathlib import Path
d = m.distribution('fla_npu')
info = next(f for f in d.files if len(f.parts) == 2 and f.parts[0].endswith('.dist-info') and f.parts[1] == 'METADATA')
print(json.dumps(dict(python=list(sys.version_info[:2]), machine=platform.machine(), prefix=sys.prefix,
    torch=m.version('torch'), torch_npu=m.version('torch-npu'), version=d.version,
    package=str(d.locate_file('fla_npu')), metadata=str(d.locate_file(info).parent))))
"""


def install_distribution(source: dict, target: Path, runtime: dict) -> None:
    """Copy the explicit package and dist-info after checking runtime identity."""
    for key in ("python", "machine", "torch", "torch_npu"):
        if source[key] != runtime[key]:
            raise RuntimeError(f"fla_npu image/workspace {key} mismatch: {source[key]!r} != {runtime[key]!r}")
    package, info = Path(source["package"]), Path(source["metadata"])
    if package.name != "fla_npu" or not info.name.startswith("fla_npu-") or not info.name.endswith(".dist-info"):
        raise ValueError("Expected exactly the fla_npu package and its distribution metadata")
    if any(not p.resolve().is_relative_to(Path(source["prefix"]).resolve()) for p in (package, info)):
        raise ValueError("fla_npu must come from the requested image interpreter prefix")
    if not package.is_dir() or not info.is_dir():
        raise FileNotFoundError("The CI image does not contain the compiled fla_npu distribution")
    if not list(package.glob("custom_aclnn_extension_lib*.so")):
        raise FileNotFoundError("The image fla_npu distribution has no compiled extension")
    destinations = [(p, target / p.name) for p in (package, info)]
    for src, dst in destinations:
        if src.resolve() == dst.resolve():
            raise ValueError("Source image environment and workspace environment must be separate")
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite an existing workspace distribution: {dst}")
    for src, dst in destinations:
        shutil.copytree(src, dst)


def main() -> None:
    """Load image metadata with its interpreter and install into this interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-python", required=True)
    args = parser.parse_args()
    source = json.loads(subprocess.check_output([args.source_python, "-c", _SOURCE_QUERY], text=True))
    runtime = dict(
        python=list(sys.version_info[:2]),
        machine=platform.machine(),
        prefix=sys.prefix,
        torch=metadata.version("torch"),
        torch_npu=metadata.version("torch-npu"),
    )
    install_distribution(source, Path(sysconfig.get_paths()["purelib"]), runtime)
    print(json.dumps({"installed_fla_npu": source["version"], **runtime}, sort_keys=True))


if __name__ == "__main__":
    main()
