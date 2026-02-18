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

import gc
import os
import shutil
import tempfile
import time
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_assets, save_model_weights
from veomni.utils import helper
from veomni.utils.device import synchronize
from veomni.utils.import_utils import is_torch_version_greater_than


logger = helper.create_logger(__name__)


@torch.no_grad()
def get_model_save_state(
    model: torch.nn.Module,
    fqn_to_index_mapping: Optional[Dict[str, int]],
) -> Dict[str, torch.Tensor]:
    """Build a flat state dict suitable for HuggingFace safetensors saving.

    1. Extracts a flat state dict via ``ModelState`` (FQNs match HF weight_map keys).
    2. Casts float32 tensors to bfloat16 on copies (original model dtypes are preserved).
    3. Filters out tied weights not present in ``fqn_to_index_mapping``.
    """
    from veomni.checkpoint.dcp_checkpointer import ModelState

    # Use flat state dict so DCP FQNs match the original HF weight_map keys
    # (e.g. "model.embed_tokens.weight" instead of "model.model.embed_tokens.weight")
    save_state = ModelState(model).state_dict()

    # Convert float32 tensors to bfloat16 on a copy of the state dict,
    # so the original model parameters remain unchanged.
    converted_state = {}
    for k, v in save_state.items():
        if v.dtype == torch.float32:
            logger.info_rank0(f"Converting {k} from {v.dtype} to torch.bfloat16")
            converted_state[k] = v.to(torch.bfloat16)
        else:
            converted_state[k] = v
    save_state = converted_state

    # Remove tied weights not present in the HF weight_map
    # (e.g. lm_head.weight is tied to model.embed_tokens.weight via tie_word_embeddings)
    if fqn_to_index_mapping is not None:
        filtered_state = {}
        for k, v in save_state.items():
            if k in fqn_to_index_mapping:
                filtered_state[k] = v
            else:
                logger.info_rank0(f"Skipping weight not in HF weight_map: {k}")
        save_state = filtered_state
    else:
        logger.warning_rank0(
            "fqn_to_index_mapping is None, HuggingFaceStorageWriter will save "
            "all model weights into a single safetensors file."
        )

    return save_state


def _save_hf_safetensor_distributed(
    model: torch.nn.Module,
    save_path: str,
    fqn_to_index_mapping: Optional[Dict[str, int]],
    model_assets: Optional[Sequence],
    is_rank_0: bool = False,
):
    """Distributed HuggingFace safetensors save using HuggingFaceStorageWriter (PyTorch >= 2.9).

    All ranks must call this function.

    save_path can be a local path or a mount path. HuggingFaceStorageWriter first writes
    each rank's shard to save_path, then rank 0 consolidates all shards and writes the
    output to a local temp directory (to avoid EOPNOTSUPP on mounted filesystems), and
    finally copies the consolidated files back to save_path.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    storage_writer = HuggingFaceStorageWriter(
        path=save_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # Redirect consolidation output to a local temp dir instead of the mount path to avoid
    # memoryview write issues (EOPNOTSUPP) on mounted filesystems.
    # - Consolidated output goes to local temp dir (rank 0 only, via finish())
    # Note: Only rank 0 executes storage_writer.finish()
    # so consolidated_output_path is only used on rank 0
    local_tmp_dir = None
    if is_rank_0:
        local_tmp_dir = tempfile.mkdtemp(prefix="veomni_hf_save_")
        storage_writer.consolidated_output_path = local_tmp_dir
        logger.info(f"Redirected consolidated_output_path to rank 0 local temp dir: {local_tmp_dir}")

    save_state = get_model_save_state(model, fqn_to_index_mapping)

    logger.info_rank0("Starting distributed HuggingFace safetensors save...")
    if dist.is_initialized():
        dist.barrier()
    start_time = time.time()
    dcp.save(
        state_dict=save_state,
        storage_writer=storage_writer,
    )
    del save_state  # Free copied tensors (e.g. fp32->bf16) to reduce peak memory
    if dist.is_initialized():
        dist.barrier()
    gc.collect()
    helper.empty_cache()
    elapsed_time = time.time() - start_time
    logger.info_rank0(f"Distributed HuggingFace safetensors save took {elapsed_time:.2f}s")

    # Rank 0: copy consolidated files from local temp to mount path, then clean up
    if is_rank_0:
        os.makedirs(save_path, exist_ok=True)
        copy_start = time.time()
        for filename in os.listdir(local_tmp_dir):
            src_file = os.path.join(local_tmp_dir, filename)
            dst_file = os.path.join(save_path, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        copy_elapsed = time.time() - copy_start
        logger.info_rank0(
            f"Copied consolidated safetensors from {local_tmp_dir} to {save_path} in {copy_elapsed:.2f}s"
        )
        # Clean up local temp dir
        shutil.rmtree(local_tmp_dir, ignore_errors=True)

    # Save model assets (config, tokenizer, etc.) on rank 0
    if model_assets and is_rank_0:
        save_model_assets(save_path, model_assets)

    logger.info_rank0(f"HuggingFace checkpoint saved at {save_path} successfully!")


def _save_hf_safetensor_legacy(
    save_checkpoint_path: str,
    save_hf_safetensor_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    train_architecture: Optional[str],
    output_dir: Optional[str],
):
    """Legacy HuggingFace safetensors save via checkpoint conversion (rank-0 only)."""
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    if train_architecture == "lora":
        model_state_dict = {k: v for k, v in model_state_dict.items() if "lora" in k}
    save_model_weights(save_hf_safetensor_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"HuggingFace checkpoint saved at {save_hf_safetensor_path} successfully!")


def save_hf_safetensor(
    save_hf_safetensor_path: Optional[str] = None,
    ckpt_manager: Optional[str] = None,
    model_assets: Optional[Sequence] = None,
    train_architecture: Optional[str] = None,
    # Legacy only
    save_checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    is_rank_0: bool = False,
    # Distributed only
    model: Optional[torch.nn.Module] = None,
    fqn_to_index_mapping: Optional[Dict[str, int]] = None,
):
    """Save model weights in HuggingFace safetensors format.

    This function is self-contained w.r.t. synchronization: it calls ``synchronize()`` at
    entry to flush pending GPU operations before reading tensor data, and calls
    ``dist.barrier()`` before returning to ensure all ranks complete the save. Callers
    do not need to add external synchronization around this function.

    Supports two modes:
    - Distributed mode (PyTorch >= 2.9, ckpt_manager="dcp", non-LoRA): Uses HuggingFaceStorageWriter
      for efficient distributed save directly from the live FSDP model. Must be called on all ranks.
    - Legacy mode: Loads from checkpoint and converts to safetensors on rank 0.

    Args:
        save_hf_safetensor_path: Output path for saved HuggingFace safetensors.
        ckpt_manager: Checkpoint manager type. Used for routing (distributed when "dcp")
            and passed to legacy ``ckpt_to_state_dict``.
        model_assets: Model assets (e.g., config, tokenizer) to save alongside weights.
        train_architecture: Training architecture type. Used for routing (legacy when "lora")
            and to filter LoRA weights in legacy mode.
        save_checkpoint_path: [Legacy only] Path to the distributed checkpoint for conversion.
        output_dir: [Legacy only] Output directory passed to ``ckpt_to_state_dict``.
        is_rank_0: [Legacy only] Whether the current process is global rank 0.
            Legacy save is rank-0 only; non-rank-0 processes return immediately.
            Required by non-dcp checkpoint managers (e.g., omnistore).
        model: [Distributed only] Live FSDP model for distributed save.
        fqn_to_index_mapping: [Distributed only] Maps FQNs to safetensors file indices
            for multi-file output.
    """
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    use_distributed = is_torch_version_greater_than("2.9") and train_architecture != "lora" and ckpt_manager == "dcp"

    # Ensure all GPU operations are complete before reading tensor data for saving
    synchronize()

    # Wait for any pending async DCP save
    if ckpt_manager == "dcp" and DistributedCheckpointer.dcp_save_future is not None:
        logger.info_rank0("Waiting for pending async DCP save to complete before HF safetensor save...")
        DistributedCheckpointer.dcp_save_future.result()
        DistributedCheckpointer.dcp_save_future = None
        if dist.is_initialized():
            dist.barrier()

    if use_distributed:
        _save_hf_safetensor_distributed(model, save_hf_safetensor_path, fqn_to_index_mapping, model_assets, is_rank_0)
    else:
        # Legacy path is rank-0 only; non-rank-0 waits at the barrier below
        if is_rank_0:
            _save_hf_safetensor_legacy(
                save_checkpoint_path,
                save_hf_safetensor_path,
                model_assets,
                ckpt_manager,
                train_architecture,
                output_dir,
            )

    # Ensure all ranks finish saving before anyone proceeds
    if dist.is_initialized():
        dist.barrier()
