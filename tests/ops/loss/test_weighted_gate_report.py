"""The NPU gate must not interpret skips or incomplete test runs as success."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).with_name("run_weighted_fsdp_sp_gate.py")
spec = importlib.util.spec_from_file_location("weighted_gate_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_gate_accepts_four_distinct_passes(tmp_path):
    """Accept a complete report with all four expected passing cases."""
    report = tmp_path / "test.xml"
    cases = "".join(
        f'<testcase name="test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[{i}]"/>'
        for i in ("mixed-fp32", "mixed-bf16", "zero-fp32", "zero-bf16")
    )
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    runner.validate_report(report)


@pytest.mark.parametrize(
    "case",
    [
        '<testcase name="test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[zero-bf16]"><skipped/></testcase>',
        '<testcase name="test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[zero-bf16]"><failure/></testcase>',
        '<testcase name="test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[unknown]"/>',
        "",
    ],
)
def test_gate_rejects_partial_or_skipped_results(tmp_path, case):
    """Reject reports with a missing, skipped, failed or unknown case."""
    report = tmp_path / "test.xml"
    valid = "".join(
        f'<testcase name="test_weighted_loss_fsdp2_sequence_parallel_on_accelerator[{i}]"/>'
        for i in ("mixed-fp32", "mixed-bf16", "zero-fp32")
    )
    report.write_text(f"<testsuites><testsuite>{valid}{case}</testsuite></testsuites>")
    with pytest.raises(RuntimeError):
        runner.validate_report(report)
