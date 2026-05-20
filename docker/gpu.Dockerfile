# Start from the NVIDIA official image (ubuntu-22.04 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""

RUN python3 --version

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini aria2 && \
    apt-get clean


# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/
# Ensure the installed binary is on the `PATH`
ENV PATH="/bin/:$PATH"

# Create the /app folder, copy the repo and grant permission to tiger.
RUN mkdir -p /app
COPY . /app/VeOmni

WORKDIR /app/VeOmni

# Set the uv cache dir
ENV UV_CACHE_DIR=~/.cache/uv

# Run the uv sync commands to install packages for /app/VeOmni repo.
#
# Notes:
#
# 1. Install without flash-attn at first because it needs other packages to build.
#
# 2. The main purpose of this is to warm up the cache. In Arnold, we usually pull
# the repo through the Arnold setting into /home/tiger, so it's recommended to
# exeucte `uv sync` and activates the virtual env under the /home/tiger/VeOmni
# instead of using the one under /app/VeOmni to make it easy to control which
# repo version/branch to use without rebuilding the image. That being said, running
# `uv sync` under /home/tiger/VeOmni should be very fast because the uv cache is
# warmed up and most packages can be linked from the cache instead of downloading.
RUN uv sync --locked --all-packages --extra gpu
