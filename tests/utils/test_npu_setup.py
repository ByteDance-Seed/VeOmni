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
