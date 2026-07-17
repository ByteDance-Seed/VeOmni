from unittest.mock import patch

import pytest

from .training_utils import is_npu_arch35, resolve_ops_overrides


def _parse_ops_flags(flags: list[str]) -> dict[str, str]:
    prefix = "--model.ops_implementation."
    parsed = {}
    for flag in flags:
        assert flag.startswith(prefix)
        key, value = flag.removeprefix(prefix).split("=", maxsplit=1)
        parsed[key] = value
    return parsed


@pytest.mark.parametrize("model_name", ["qwen3_5", "qwen3_5_moe"])
@patch("tests.tools.training_utils.is_torch_npu_available", return_value=True)
def test_qwen3_5_npu_overrides_select_gated_deltanet_kernels(_mock_npu, model_name):
    overrides = _parse_ops_flags(resolve_ops_overrides(model_name))

    assert overrides["rms_norm_gated_implementation"] == "npu"
    assert overrides["causal_conv1d_implementation"] == "npu"
    assert overrides["chunk_gated_delta_rule_implementation"] == "npu"


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("Ascend910B2", False),
        ("Ascend910_95", True),
        ("Ascend950", True),
    ],
)
@patch("tests.tools.training_utils.get_device_name")
@patch("tests.tools.training_utils.is_torch_npu_available", return_value=True)
def test_is_npu_arch35_detects_unsupported_devices(_mock_npu, mock_device_name, device_name, expected):
    mock_device_name.return_value = device_name

    assert is_npu_arch35() is expected


@patch("tests.tools.training_utils.is_torch_npu_available", return_value=False)
def test_is_npu_arch35_is_false_without_torch_npu(_mock_npu):
    assert is_npu_arch35() is False
