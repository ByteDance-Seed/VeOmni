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

import os
from argparse import ArgumentParser
from pathlib import Path


# directory or file path must contain keyword ".cuda" or "cuda"
CUDA_KEYWORD_CHECK_WHITELIST = [
    "veomni/utils/import_utils.py",
    "veomni/utils/device.py",
    "veomni/ops/group_gemm/utils/benchmark_utils.py",
    "veomni/utils/helper.py",
    "veomni/distributed/torch_parallelize.py",
    "veomni/models/auto.py",
    "veomni/models/loader.py",
    "veomni/models/module_utils.py",
    "veomni/models/seed_omni/auto.py",
    "veomni/models/transformers/flux/encode_flux.py",
    "veomni/models/transformers/llama/modeling_llama.py",
    "veomni/models/transformers/qwen2/modeling_qwen2.py",
    "veomni/models/transformers/qwen2_5_omni/modeling_qwen2_5_omni.py",
    "veomni/models/transformers/qwen2_5vl/modeling_qwen2_5_vl.py",
    "veomni/models/transformers/qwen2_vl/modeling_qwen2_vl.py",
    "veomni/models/transformers/qwen3/modeling_qwen3.py",
    "veomni/models/transformers/qwen3_moe/modeling_qwen3_moe.py",
    "veomni/utils/arguments.py",
    "veomni/ops/group_gemm/utils/device.py",
    "tests/special_senity/check_device_api_usage.py",
    "tests/tools/common_utils.py",
]

# directory or file path must contain keyword "nccl"
NCCL_KEYWORD_CHECK_WHITELIST = [
    "veomni/utils/device.py",
]

SEARCH_WHITELIST = CUDA_KEYWORD_CHECK_WHITELIST + NCCL_KEYWORD_CHECK_WHITELIST

SEARCH_KEYWORDS = [".cuda", '"cuda"', '"nccl"']


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
            file_content = f.read()

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
