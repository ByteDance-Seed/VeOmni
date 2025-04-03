# Start from the NVIDIA official image (ubuntu-22.04 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""


# Install systemctl and tini
RUN apt-get update && \
apt-get install -y -o Dpkg::Options::="--force-confdef" systemd tini && \
apt-get clean || { echo "Installation failed"; exit 1; }

RUN apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Change pip source
RUN python -m pip install --upgrade pip

# Install torch-2.5.1 + vllm-0.7.3
RUN pip install --no-cache-dir vllm==0.7.3 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 tensordict torchdata \
    transformers>=4.49.0 accelerate datasets peft hf-transfer diffusers \
    codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc qwen-vl-utils wandb ninja liger-kernel \
    pytest yapf py-spy pyext pre-commit ruff packaging

# Install flux
RUN pip install --no-cache-dir byte-flux

# Install flash-attn and triton
RUN pip install --no-cache-dir flash-attn triton>=3.1.0
