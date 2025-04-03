import pkg_resources


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
    assert torch_version == "2.1.0+cpu"

    torchvision_version = get_package_version("torchvision")
    assert torchvision_version == "0.16.0"

    torch_npu_version = get_package_version("torch-npu")
    assert torch_npu_version == "2.1.0.post6.dev20240716"

    triton_version = get_package_version("triton")
    assert triton_version is None


def test_veomni_setup():
    check_env()
