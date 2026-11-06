"""Run the 16-NPU parity tests and reject skips, partial collection, or failures."""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_report(report: Path) -> None:
    """Require all four expected native cases to pass without skips."""
    cases = ET.parse(report).getroot().findall(".//testcase")
    names = [case.get("name") for case in cases]
    if (
        len(cases) != 4
        or len(set(names)) != 4
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
    ):
        raise RuntimeError("Expected exactly four distinct passing cases and no skips")
    expected = {
        f"test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[{mass}-{dtype}]"
        for mass in ("mixed", "zero")
        for dtype in ("fp32", "bf16")
    }
    if set(names) != expected:
        raise RuntimeError("Unexpected test case in weighted loss gate report")


def main() -> None:
    """Run the exact-source weighted loss gate on sixteen available NPUs."""
    os.environ["VEOMNI_TEST_REQUIRE_NPU"] = "1"
    import pytest
    import torch
    import torch_npu  # noqa: F401

    import veomni

    root = Path(__file__).resolve().parents[3]
    if not Path(veomni.__file__).resolve().is_relative_to(root):
        raise RuntimeError("VeOmni import does not match the candidate source")
    if not torch.npu.is_available() or torch.npu.device_count() != 16:
        raise RuntimeError("The gate requires exactly 16 available NPUs")
    with tempfile.TemporaryDirectory(prefix="weighted-fsdp-sp-gate-") as directory:
        report = Path(directory) / "result.xml"
        result = pytest.main(
            [
                str(root / "tests/ops/loss/test_weighted_fsdp_sp_accelerator.py"),
                "-q",
                "-s",
                f"--junitxml={report}",
            ]
        )
        if result != 0:
            raise RuntimeError(f"Weighted loss pytest failed with code {result}")
        validate_report(report)
    print("WEIGHTED_16_NPU_FSDP_SP_GATE_PASSED cases=4 skipped=0", flush=True)


if __name__ == "__main__":
    main()
