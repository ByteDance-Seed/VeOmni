"""Shared harness for the per-modality e2e parallel-alignment tests.

The per-modality files (`test_e2e_parallel_text.py`,
`test_e2e_parallel_vlm.py`, `test_e2e_parallel_omni.py`,
`test_e2e_parallel_dit.py`) each define their own parametrize lists and
test functions but delegate the actual torchrun + metric-comparison
logic to :func:`main` in this module.

Separating the harness from the per-modality parametrizations keeps
each test file scoped to its model family, which (a) makes it obvious
where to add a new model entry when onboarding, and (b) avoids the
previous situation where touching any one model's parametrize list
re-ran the whole 400+-line file in code review diffs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from veomni.models.auto import build_foundation_model
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_diffusers_available, is_transformers_version_greater_or_equal_to

from ..tools import build_torchrun_cmd, compare_metrics, print_comparison_table
from .utils import prepare_exec_cmd


# ---------------------------------------------------------------------------
# Version / backend gates
# ---------------------------------------------------------------------------

_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")

v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")
dit_only = pytest.mark.skipif(not is_diffusers_available(), reason="Requires diffusers")


# ---------------------------------------------------------------------------
# Default comparison tolerances
# ---------------------------------------------------------------------------

DEFAULT_RTOL = 1e-1
DEFAULT_ATOL = 1e-1


# ---------------------------------------------------------------------------
# Harness implementation
# ---------------------------------------------------------------------------


def _materialize_weights_dir(config_path: str, output_path: str, save_original_format: bool = True) -> Path:
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path, save_original_format=save_original_format)


def main(
    task_name: str,
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    train_path: str,
    max_sp_size: int | None = None,
):
    test_path = f"./{model_name}"
    os.makedirs(test_path, exist_ok=True)

    # Models with stacked 3D expert params (gate_up_proj [E, 2*I, H],
    # down_proj [E, H, I]):
    #
    # - qwen3_5_moe: native HF safetensor format is already stacked.
    #   HF's save_pretrained() with save_original_format=True calls
    #   revert_weight_conversion() that splits them into per-expert keys
    #   (experts.*.gate_proj.weight, etc.), but VeOmni has no runtime
    #   converter for this model. Disable save_original_format to save
    #   in native stacked format.
    #
    # - qwen3_moe (v5): VeOmni registers a runtime CheckpointTensorConverter
    #   that merges per-expert HF keys back to fused format at load time,
    #   so save_original_format=True works correctly.
    save_original_format = model_name != "qwen3_5_moe"
    _materialize_weights_dir(config_path, test_path, save_original_format=save_original_format)

    test_tasks = [task_name]
    command_list = prepare_exec_cmd(
        test_tasks,
        model_name,
        config_path,
        model_path=test_path,
        train_path=train_path,
        output_dir=test_path,
        is_moe=is_moe,
        max_sp_size=max_sp_size,
    )
    res = {}
    log_keys = []
    for task_name, cmd_kwargs in command_list:
        print(f"{'-' * 10} {task_name} {'-' * 10}")
        cmd = build_torchrun_cmd(**cmd_kwargs)
        subprocess.run(cmd, check=True)
        with open(os.path.join(test_path, f"{task_name}/log_dict.json")) as f:
            output = json.load(f)
        if not log_keys:
            log_keys = set(output.keys())
        else:
            assert log_keys == set(output.keys())
        res[task_name] = output

    for key in log_keys:
        print_comparison_table(res, key, title=model_name)
    compare_metrics(res, rtol=rtol, atol=atol)

    shutil.rmtree(test_path)
