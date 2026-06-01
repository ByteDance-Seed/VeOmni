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
import time
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_assets, save_model_weights
from veomni.models.module_utils import _save_state_dict
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
):
    """Distributed HuggingFace safetensors save using HuggingFaceStorageWriter (PyTorch >= 2.9).

    All ranks must call this function.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    # Apply DCP consolidation patch just-in-time for HDFS FUSE compatibility
    # This patches torch.distributed.checkpoint._consolidate_hf_safetensors._process_output_file
    # to use append mode instead of r+b mode, which is required for append-only file systems
    from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch

    apply_dcp_consolidation_patch()

    save_state = get_model_save_state(model, fqn_to_index_mapping)

    # Filter fqn_to_index_mapping to only include keys that exist in save_state.
    # This is necessary when training excludes certain modules (e.g., MTP) but the original
    # fqn_to_index_mapping parsed from model.safetensors.index.json still contains those weights.
    # Without this filtering, HuggingFaceStorageWriter's consolidation phase would create invalid
    # metadata entries (with default values like empty shape and 0 dtype_size) for non-existent
    # weights, resulting in corrupted safetensors output.
    if fqn_to_index_mapping is not None:
        original_mapping_size = len(fqn_to_index_mapping)
        fqn_to_index_mapping = {k: v for k, v in fqn_to_index_mapping.items() if k in save_state}
        if len(fqn_to_index_mapping) < original_mapping_size:
            logger.info_rank0(
                f"Filtered fqn_to_index_mapping from {original_mapping_size} to {len(fqn_to_index_mapping)} keys "
                f"to match actual model weights"
            )

    storage_writer = HuggingFaceStorageWriter(
        path=save_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

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

    # Save model assets (config, tokenizer, etc.) on rank 0
    if model_assets and (not dist.is_initialized() or dist.get_rank() == 0):
        save_model_assets(save_path, model_assets)

    logger.info_rank0(f"HuggingFace checkpoint saved at {save_path} successfully!")


def _save_hf_safetensor_legacy(
    save_checkpoint_path: str,
    save_hf_safetensor_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    output_dir: Optional[str],
):
    """Legacy HuggingFace safetensors save via checkpoint conversion (rank-0 only)."""
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    save_model_weights(save_hf_safetensor_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"HuggingFace checkpoint saved at {save_hf_safetensor_path} successfully!")


def save_hf_safetensor(
    save_hf_safetensor_path: Optional[str] = None,
    ckpt_manager: Optional[str] = None,
    model_assets: Optional[Sequence] = None,
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

    use_distributed = is_torch_version_greater_than("2.9") and ckpt_manager == "dcp"

    # Ensure all GPU operations are complete before reading tensor data for saving
    synchronize()

    # Wait for any pending async DCP save before HF safetensor save
    if ckpt_manager == "dcp":
        DistributedCheckpointer.wait_for_pending_save()

    if use_distributed:
        from veomni.models.checkpoint_tensor_loading import resolve_fqn_to_index_mapping_for_save

        fqn_to_index_mapping = resolve_fqn_to_index_mapping_for_save(model, fqn_to_index_mapping)
        _save_hf_safetensor_distributed(model, save_hf_safetensor_path, fqn_to_index_mapping, model_assets)
    else:
        # Legacy path is rank-0 only; non-rank-0 waits at the barrier below
        if is_rank_0:
            _save_hf_safetensor_legacy(
                save_checkpoint_path,
                save_hf_safetensor_path,
                model_assets,
                ckpt_manager,
                output_dir,
            )

    # Ensure all ranks finish saving before anyone proceeds
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def save_lora_adapter_with_dcp(
    model: torch.nn.Module,
    save_path: str,
    adapter_name: str = "default",
    dcp_subdir: str = ".lora_dcp_tmp",
) -> None:
    """Save LoRA adapter with DCP parallel write and rank-0 consolidation.

    All ranks must call this function. It performs:
    1. Extract LoRA-only state from the live model.
    2. Save with ``dcp.save`` in parallel to a temporary DCP directory.
    3. Consolidate on rank 0 into ``adapter_model.bin`` and ``adapter_config.json``.
    """
    from peft import get_peft_model_state_dict

    synchronize()
    if dist.is_initialized():
        dist.barrier()

    os.makedirs(save_path, exist_ok=True)
    dcp_save_path = os.path.join(save_path, dcp_subdir)
    os.makedirs(dcp_save_path, exist_ok=True)

    lora_state = get_peft_model_state_dict(model)
    lora_state = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in lora_state.items()}
    # ckpt_to_state_dict's DCP conversion path only recognizes keys starting with "model.".
    # Prefix LoRA keys temporarily for DCP save so consolidation can reuse existing conversion logic.
    dcp_lora_state = {k if k.startswith("model.") else f"model.{k}": v for k, v in lora_state.items()}

    storage_writer = dcp.FileSystemWriter(
        dcp_save_path,
        thread_count=16,
        single_file_per_rank=True,
        sync_files=False,
    )
    dcp.save(
        state_dict=dcp_lora_state,
        storage_writer=storage_writer,
    )

    if dist.is_initialized():
        dist.barrier()

    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_0:
        consolidated_state = ckpt_to_state_dict(
            save_checkpoint_path=dcp_save_path,
            ckpt_manager="dcp",
        )
        adapter_model_file = os.path.join(save_path, "adapter_model.bin")
        _save_state_dict(consolidated_state, adapter_model_file, safe_serialization=False)

        if not hasattr(model, "peft_config") or adapter_name not in model.peft_config:
            raise ValueError(f"Cannot find peft config for adapter '{adapter_name}' on model.")
        model.peft_config[adapter_name].save_pretrained(save_path)

        shutil.rmtree(dcp_save_path, ignore_errors=True)
        logger.info_rank0(f"LoRA adapter saved at {save_path} successfully!")

    if dist.is_initialized():
        dist.barrier()

    gc.collect()
    helper.empty_cache()
