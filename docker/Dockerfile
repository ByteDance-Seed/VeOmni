# VeOmni commit 10adfab on the NVIDIA A100 / R560 stack.
# NGC 24.08 provides Ubuntu 22.04, Python 3.10, and the CUDA 12.6 toolchain.
# PyTorch 2.7.1 is the newest VeOmni-era framework release with the mature
# FlexAttention path while retaining CUDA 12.6 support on the installed R560.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3@sha256:60cc7c9cf24b423c28f621287779ebe5e52cd39cfd284df04983df52566c1e39
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

LABEL org.opencontainers.image.title="VeOmni A100 CUDA 12.6 training"
LABEL org.opencontainers.image.source="https://github.com/ByteDance-Seed/VeOmni"
LABEL ai.veomni.commit="10adfab65975a4d6d449f860996e206499cf3132"
LABEL ai.veomni.pytorch="2.7.1+cu126"
LABEL ai.veomni.triton="3.3.1"
LABEL ai.veomni.transformers="4.51.3"
LABEL ai.veomni.attention="flash_attention_2,flex_attention"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=32

COPY docker/requirements-a100-cu126.txt /tmp/requirements-a100-cu126.txt

# Install the stable, mutually matched framework trio. Do not install vLLM in
# the training image: its strict framework pins would unnecessarily control
# this environment.
RUN python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        --retries 10 \
        --timeout 120 \
        "torch==2.7.1" \
        "torchvision==0.22.1" \
        "torchaudio==2.7.1"

RUN python -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --retries 10 \
        --timeout 120 \
        --upgrade-strategy only-if-needed \
        -r /tmp/requirements-a100-cu126.txt

# Drop NGC components that either pin the base image's nightly Torch build,
# overlap the official Triton package, or carry malformed wheel metadata.
# They are inference/image-pipeline extras and are not used by VeOmni text
# training. Reinstall Triton after removing pytorch-triton because both
# distributions own the same import paths; Ninja 1.11.1.4 fixes the malformed
# WHEEL metadata shipped by 1.11.1.1.
RUN python -m pip uninstall --yes \
        nvidia-dali-cuda120 \
        nvidia-modelopt \
        nvidia-nvimgcodec-cu12 \
        pytorch-triton \
        torch-tensorrt \
        ninja && \
    python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        --retries 10 \
        --timeout 120 \
        --no-deps \
        --force-reinstall \
        "triton==3.3.1" && \
    python -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --retries 10 \
        --timeout 120 \
        --no-deps \
        "ninja==1.11.1.4"

ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL_SHA256=2d1819070504b6f9d8f117c453815835c6730a90b01aa25dca13f6d2b53c3f1e
LABEL ai.veomni.flash_attention="2.7.4.post1"

# Install the upstream wheel built specifically for CPython 3.10, PyTorch 2.7,
# CUDA 12, and PyTorch's cxx11abiTRUE setting. The digest prevents a replaced
# release asset from silently changing the image; --no-deps keeps the framework
# stack fixed.
RUN python -m pip install \
        --retries 10 \
        --timeout 120 \
        --no-deps \
        --force-reinstall \
        "${FLASH_ATTN_WHEEL_URL}#sha256=${FLASH_ATTN_WHEEL_SHA256}"

COPY . /workspace/VeOmni
WORKDIR /workspace/VeOmni

RUN python -m pip install --no-deps --editable /workspace/VeOmni && \
    python -m pip check && \
    python -c 'import flash_attn, torch, transformers, triton; from transformers.utils import is_torch_flex_attn_available; assert torch.__version__.startswith("2.7.1"), torch.__version__; assert torch.version.cuda.startswith("12.6"), torch.version.cuda; assert torch._C._GLIBCXX_USE_CXX11_ABI is True; assert triton.__version__ == "3.3.1", triton.__version__; assert transformers.__version__ == "4.51.3", transformers.__version__; assert flash_attn.__version__ == "2.7.4.post1", flash_attn.__version__; assert is_torch_flex_attn_available(); print("validated:", torch.__version__, torch.version.cuda, triton.__version__, transformers.__version__, flash_attn.__version__, "flex_attention")' && \
    python -c 'from tasks.train_torch import Arguments; from veomni.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM; print("VeOmni text-training imports validated")'

CMD ["bash"]
