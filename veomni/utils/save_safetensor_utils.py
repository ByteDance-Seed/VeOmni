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

import os
import time
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_assets, save_model_weights
from veomni.utils import helper
from veomni.utils.import_utils import is_torch_version_greater_than


logger = helper.create_logger(__name__)


def _save_hf_safetensor_distributed(
    model: torch.nn.Module,
    save_path: str,
    fqn_to_index_mapping: Optional[Dict[str, int]],
    model_assets: Optional[Sequence],
):
    """Distributed HuggingFace safetensors save using HuggingFaceStorageWriter (PyTorch >= 2.9).

    All ranks must call this function.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    storage_writer = HuggingFaceStorageWriter(
        path=save_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    logger.info_rank0("Starting distributed HuggingFace safetensors save...")
    dist.barrier()
    start_time = time.time()
    DistributedCheckpointer.save(
        path=save_path,
        state={"model": model},
        storage_writer=storage_writer,
    )
    elapsed_time = time.time() - start_time
    logger.info_rank0(f"Distributed HuggingFace safetensors save took {elapsed_time:.2f}s")

    # Save model assets (config, tokenizer, etc.) on rank 0
    if model_assets and (not dist.is_initialized() or dist.get_rank() == 0):
        save_model_assets(save_path, model_assets)

    logger.info_rank0(f"HuggingFace checkpoint saved at {save_path} successfully!")


def _save_hf_safetensor_legacy(
    save_checkpoint_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    train_architecture: Optional[str],
    output_dir: Optional[str],
):
    """Legacy HuggingFace safetensors save via checkpoint conversion (rank-0 only)."""
    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    if train_architecture == "lora":
        model_state_dict = {k: v for k, v in model_state_dict.items() if "lora" in k}
    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"HuggingFace checkpoint saved at {hf_weights_path} successfully!")


def save_hf_safetensor(
    save_checkpoint_path: Optional[str] = None,
    model_assets: Optional[Sequence] = None,
    ckpt_manager: Optional[str] = None,
    train_architecture: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    save_safetensor_path: Optional[str] = None,
    fqn_to_index_mapping: Optional[Dict[str, int]] = None,
):
    """Save model weights in HuggingFace safetensors format.

    Supports two modes:
    - Distributed mode (PyTorch >= 2.9, ckpt_manager="dcp", non-LoRA): Uses HuggingFaceStorageWriter
      for efficient distributed save directly from the live FSDP model. Must be called on all ranks.
    - Legacy mode: Loads from checkpoint and converts to safetensors on rank 0.

    Args:
        save_checkpoint_path: Path to the distributed checkpoint (legacy mode).
        model_assets: Model assets (e.g., config, tokenizer) to save alongside weights.
        ckpt_manager: Checkpoint manager type.
        train_architecture: Training architecture type. If "lora", uses legacy mode.
        output_dir: Output directory for checkpoint conversion. Required only by omnistore ckpt_manager.
        model: Live FSDP model for distributed save.
        save_safetensor_path: Output path for distributed save.
        fqn_to_index_mapping: Maps FQNs to safetensors file indices for multi-file output.
    """
    use_distributed = (
        is_torch_version_greater_than("2.9")
        and train_architecture != "lora"
        and ckpt_manager == "dcp"
    )

    if use_distributed:
        _save_hf_safetensor_distributed(model, save_safetensor_path, fqn_to_index_mapping, model_assets)
    else:
        # Legacy path is rank-0 only
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        _save_hf_safetensor_legacy(save_checkpoint_path, model_assets, ckpt_manager, train_architecture, output_dir)
