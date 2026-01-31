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

import pkg_resources
import pytest

from veomni.utils.import_utils import is_torch_npu_available


def get_package_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        print(f"{package_name}: {version}")
        return version
    except pkg_resources.DistributionNotFound:
        print(f"{package_name} is not installed")
        return None


def check_env():
    torch_version = get_package_version("torch")
    assert torch_version == "2.7.1+cpu"

    torchvision_version = get_package_version("torchvision")
    assert torchvision_version == "0.22.1"

    torch_npu_version = get_package_version("torch-npu")
    assert torch_npu_version == "2.7.1"

    triton_version = get_package_version("triton")
    assert triton_version is None


@pytest.mark.skipif(not is_torch_npu_available(), reason="only npu check test_npu_setup")
def test_veomni_setup():
    check_env()
