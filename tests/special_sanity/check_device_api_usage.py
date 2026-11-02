# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Following codes are inspired from https://github.com/volcengine/verl/blob/main/tests/special_sanity/check_device_api_usage.py

"""
This CI test is used for checking whether device api usage is irregular, suggest using api in `veomni/utils/device.py`.
Search targets include .py files in VeOmni/tasks and VeOmni/veomni.
Some files that must contain ".cuda", "cuda" or "nccl" keyword is pre-defined in whitelist below.

Should be run as follows:

python3 tests/special_sanity/check_device_api_usage.py --directory ./tasks
python3 tests/special_sanity/check_device_api_usage.py --directory ./veomni
"""

import ast
import os
from argparse import ArgumentParser
from pathlib import Path


# directory or file path must contain keyword ".cuda" or "cuda"
CUDA_KEYWORD_CHECK_WHITELIST = [
    "veomni/utils/import_utils.py",
    "veomni/utils/device.py",
    # The optimizer factory identifies the CUDA/NCCL domain for Muon's
    # collective IEEE-FP32 preflight, including CPU-offloaded parameters.
    "veomni/optim/optimizer.py",
    "veomni/utils/helper.py",
    "veomni/distributed/torch_parallelize.py",
    "veomni/models/auto.py",
    "veomni/models/loader.py",
    "veomni/models/checkpoint/weights.py",
    "veomni/models/transformers/flux/encode_flux.py",
    "veomni/arguments/arguments_types.py",
    # Declarative GPU platform detection; this is not kernel device management.
    "veomni/ops/platform/gpu.py",
    "veomni/ops/platform/requirement.py",
    # Magi FA4 is an intrinsically CUDA-only backend. It uses CUDA device
    # contexts and cuda.bindings.runtime for the per-device stack limit, which
    # has no device-agnostic equivalent.
    "veomni/ops/kernels/attention/standard/magi/",
    # DeepGEMM is CUDA-only: these files select the SM90/SM100 recipe and
    # establish operand-device contexts for its device-bound JIT kernels.
    "veomni/ops/kernels/moe_experts/standard/deepgemm.py",
    "veomni/ops/kernels/moe_experts/shared/deepgemm_fp8.py",
    # Kernel tests intentionally select and probe hardware-specific backends.
    "tests/ops/",
    "tests/special_sanity/check_device_api_usage.py",
    "tests/tools/common_utils.py",
    # CPU-only scheduler tests use literal CUDA/NPU backend names as fixtures
    # to validate device leases and launch plans without accessing hardware.
    "tests/ci/test_accelerator.py",
    # EP option tests use literal backend names to verify cross-device rejection.
    "tests/ci/test_ep_hccl_options.py",
    # Gloo failure-propagation tests mock an available CUDA device to verify
    # that the process-group backend still selects CPU tensors.
    "tests/utils/test_dist_utils_group.py",
    # Implicit-CUDA-sync gate. Calls ``torch.cuda.{get,set}_sync_debug_mode``
    # directly because the API is intrinsically CUDA-only and has no
    # ``veomni.utils.device`` equivalent; the test is gated on
    # ``IS_CUDA_AVAILABLE`` so it skips on non-CUDA hosts.
    "tests/models/transformers/test_model_forward_no_implicit_sync.py",
    # Precision and async-offload tests explicitly select CUDA/NPU devices by
    # name to exercise accelerator-specific code paths; the literal "cuda" is
    # a test fixture, not a production device-API call.
    "tests/distributed/test_async_activation_offload.py",
    "tests/models/test_qwen_image_precision.py",
    "tests/precision/qwen_image.py",
    # These Muon tests exercise CUDA-specific precision settings and mocks.
    "tests/optim/test_qwen38_cuda_native.py",
    "tests/optim/test_qwen38_cuda_preflight.py",
    "tests/optim/test_qwen38_muon.py",
    "tests/optim/test_qwen38_muon_device.py",
    # Vendored Ascend Triton kernels for Qwen3.5 gated delta-rule (FLA + Huawei
    # port) are kept byte-identical to upstream; the FLA-origin code references
    # ``.cuda`` in fallback/util paths. Excluded wholesale, matching the ruff
    # exclude in pyproject.toml.
    "veomni/ops/kernels/gated_delta_rule/vendor/",
]

# directory or file path must contain keyword "nccl"
NCCL_KEYWORD_CHECK_WHITELIST = [
    "veomni/utils/device.py",
]

SEARCH_WHITELIST = CUDA_KEYWORD_CHECK_WHITELIST + NCCL_KEYWORD_CHECK_WHITELIST

SEARCH_KEYWORDS = [".cuda", '"cuda"', '"nccl"']

# Only these existing Muon lines need backend-specific precision inspection or
# device dispatch. The existing keyword scan still checks every other line.
LINE_EXCEPTIONS = {
    "veomni/optim/qwen38_muon.py": {
        "_check_cuda_fp32_policy": {"backend = torch.backends.cuda.matmul"},
        "_optimizer_uses_cuda": {
            'return any(param.device.type == "cuda" for group in optimizer.param_groups for param in group["params"])'
        },
        "_strict_matmul": {'elif device_type == "cuda":'},
    }
}


def _remove_scoped_exceptions(path: str, content: str) -> str:
    rules = next(
        (rules for suffix, rules in LINE_EXCEPTIONS.items() if path.endswith(suffix.replace("/", os.sep))), None
    )
    if rules is None:
        return content
    try:
        module = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.splitlines()
    excluded = set()
    for name, permitted_lines in rules.items():
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(functions) != 1:
            continue
        function = functions[0]
        for permitted in permitted_lines:
            matches = [i for i in range(function.lineno - 1, function.end_lineno) if lines[i].strip() == permitted]
            if len(matches) == 1:
                excluded.add(matches[0])
    return "\n".join(line for i, line in enumerate(lines) if i not in excluded)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        path_in_str = str(path.absolute())

        # judge whether current path is in pre-defined search whitelist or not.
        path_in_whitelist = False

        for sw in SEARCH_WHITELIST:
            # for easy debugging in non-linux system
            sw = sw.replace("/", os.sep)
            if sw in path_in_str:
                print(f"[SKIP] File {path_in_str} is in device api usage check whitelist, checking is skipped.")
                path_in_whitelist = True
                break

        if path_in_whitelist:
            continue

        with open(path_in_str, encoding="utf-8") as f:
            file_content = _remove_scoped_exceptions(path_in_str, f.read())

            find_invalid_device_management = False

            for sk in SEARCH_KEYWORDS:
                if sk in file_content:
                    find_invalid_device_management = True
                    break

            print(
                f"[CHECK] File {path_in_str} is detected for device api usage check, check result: "
                f"{'success' if not find_invalid_device_management else f'failed, because detect {sk}'}."
            )

            assert not find_invalid_device_management, (
                f'file {path_in_str} contains .cuda/"cuda"/"nccl" usage, please use api in '
                f"veomni/utils/device.py directly."
            )
