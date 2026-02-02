#!/bin/bash

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

set -x
set -o pipefail

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

NNODES=${NNODES:=1}
if command -v nvidia-smi &> /dev/null && nvidia-smi --list-gpus &> /dev/null; then
  # GPU
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
  fi
  export NCCL_DEBUG=WARN
else
  # NPU
  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(ls -l /dev/davinci* | grep -v "davinci_manager" | wc -l)}
  fi
  # NPU env that may optimize performance
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:='expandable_segments:True'}
fi
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args $@ 2>&1 | tee log.txt
