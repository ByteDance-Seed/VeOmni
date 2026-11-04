# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bad_line", [None, "unsafe_call", "unsafe_literal", "other_function", "duplicate_line"])
def test_muon_device_api_exceptions_are_line_scoped(tmp_path, bad_line):
    source = ROOT / "veomni/optim/qwen38_muon.py"
    target = tmp_path / "veomni/optim/qwen38_muon.py"
    target.parent.mkdir(parents=True)
    contents = source.read_text(encoding="utf-8")
    backend = "cu" + "da"
    if bad_line == "unsafe_call":
        contents += f"\nunsafe_call = torch.{backend}.synchronize\n"
    elif bad_line == "unsafe_literal":
        contents += f'\nunsafe_literal = "{backend}"\n'
    elif bad_line == "other_function":
        contents += f"\ndef unrelated():\n    backend = torch.backends.{backend}.matmul\n"
    elif bad_line == "duplicate_line":
        line = f"        backend = torch.backends.{backend}.matmul"
        assert line in contents
        contents = contents.replace(line, f"{line}\n{line}", 1)
    target.write_text(contents, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/special_sanity/check_device_api_usage.py"), "-d", str(target.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (bad_line is None), result.stdout + result.stderr


@pytest.mark.parametrize("unsafe", [False, True])
def test_optimizer_factory_device_exceptions_do_not_hide_other_calls(tmp_path, unsafe):
    source = ROOT / "veomni/optim/optimizer.py"
    target = tmp_path / "veomni/optim/optimizer.py"
    target.parent.mkdir(parents=True)
    contents = source.read_text(encoding="utf-8")
    if unsafe:
        contents += "\nunsafe_call = torch." + "cu" + "da.synchronize\n"
    target.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/special_sanity/check_device_api_usage.py"), "-d", str(target.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (not unsafe), result.stdout + result.stderr
