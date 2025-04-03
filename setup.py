import importlib.metadata
import importlib.util
import os
import re
from typing import List

from setuptools import find_packages, setup


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_torch_npu_available() -> bool:
    return _is_package_available("torch_npu")


def _is_torch_available() -> bool:
    return _is_package_available("torch")


def _is_torch_cuda_available() -> bool:
    if _is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def get_version() -> str:
    with open(os.path.join("veomni", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


CUDA_REQUIRE = ["liger-kernel>=0.4.1,<1.0"]

NPU_REQUIRE = ["torchvision>=0.16.0,<0.16.1"]

EXTRAS_REQUIRE = {"dev": ["pre-commit>=4.0.0,<5.0", "ruff>=0.7.0,<1.0", "pytest>=6.0.0,<8.0", "expecttest>=0.3.0"]}

BASE_REQUIRE = [
    "byted-hdfs-io",
    "datasets>=2.16.0,<=2.21.0",
    "diffusers>=0.30.0,<=0.31.0",
    "packaging>=23.0,<26.0",
    "torchdata>=0.8.0,<1.0",
    "transformers[torch]>=4.46.2,<=4.49.0",
    "tiktoken>=0.9.0",
    "blobfile>=3.0.0",
]


def main():
    # Update install_requires and extras_require
    install_requires = BASE_REQUIRE

    if _is_torch_npu_available():
        install_requires.extend(NPU_REQUIRE)
    elif _is_torch_cuda_available():
        install_requires.extend(CUDA_REQUIRE)

    setup(
        name="veomni",
        version=get_version(),
        python_requires=">=3.8.0",
        packages=find_packages(exclude=["scripts", "tasks", "tests"]),
        url="https://github.com/ByteDance-Seed/VeOmni",
        license="Apache 2.0",
        author="Bytedance - Seed - MLSys",
        author_email="maqianli.fazzie@bytedance.com",
        description="Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework",
        install_requires=install_requires,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
