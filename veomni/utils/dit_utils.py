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

import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .device import synchronize
from .helper import EnvironMeter as OriginalEnvironMeter


if TYPE_CHECKING:
    from transformers import PretrainedConfig

from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from torch import distributed as dist
from transformers.utils.import_utils import is_safetensors_available

from ..models.module_utils import _save_state_dict
from . import logging
from .helper import empty_cache, get_dtype_size


if is_safetensors_available():
    pass


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


def _compute_wan_seqlens(
    micro_batch: Dict[str, "torch.Tensor"], rmpad: bool, rmpad_with_pos_ids: bool
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Computes the sequence lengths of the current batch.

    Args:
        micro_batch (Dict[str, Tensor]): The current batch.
        rmpad (bool): Whether to remove the padding tokens.
        rmpad_with_pos_ids (bool): Whether to remove the padding tokens using the position ids.
    """
    latent_shape = micro_batch["latents"].shape
    if len(latent_shape) == 5:
        B = latent_shape[0]
    else:
        B = 1
    C, T, H, W = latent_shape[-4:]
    T_out = int((T - 1) / 1 + 1)
    H_out = int((H - 2) / 2 + 1)
    W_out = int((W - 2) / 2 + 1)
    seqlens = B * T_out * H_out * W_out
    return [seqlens]


def _compute_flux_seqlens(micro_batch: Dict[str, "torch.Tensor"]) -> Tuple[List[int], Optional[List[int]]]:
    """
    Computes the sequence lengths of the current batch.

    Args:
        micro_batch (Dict[str, Tensor]): The current batch.
    """
    B, C, H, W = micro_batch.shape
    H_out = int((H - 2) / 2 + 1)
    W_out = int((W - 2) / 2 + 1)
    seqlens = B * H_out * W_out
    return [seqlens]


class EnvironMeter(OriginalEnvironMeter):
    """
    Computes the metrics about the training efficiency.

    Args:
        config (PretrainedConfig): The configuration of the model.
        global_batch_size (int): The global batch size.
        empty_cache_steps (int, optional): The number of steps to empty the cache. Defaults to 500.
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        global_batch_size: int,
        empty_cache_steps: int = 500,
    ) -> None:
        super().__init__(config, global_batch_size, empty_cache_steps=empty_cache_steps)

    def add(self, micro_batch: Dict[str, "torch.Tensor"], model_type: Optional[str] = None) -> None:
        if model_type == "wan":
            seqlens = _compute_wan_seqlens(micro_batch, self.rmpad, self.rmpad_with_pos_ids)
        elif model_type == "flux":
            seqlens = _compute_flux_seqlens(micro_batch)
        else:
            raise ValueError(f"model_type {model_type} not supported")

        self.batch_seqlens.extend(seqlens)


def _get_shard_info(
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> Tuple[bool, int, Dict[str, str]]:
    """
    Gets the shard information, should be executed at rank 0.
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype
        tensor_size = tensor.numel() * get_dtype_size(dtype)  # dtensor's numel == tensor's numel
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

    num_shards = len(shard_list)
    weight_map = OrderedDict()
    is_sharded = None
    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, "os.PathLike"],
    state_dict: Dict[str, "torch.Tensor"],
    global_rank: Optional[int] = None,
    save_dtype: Optional[Union[str, "torch.dtype"]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or dtensor.

    If global_rank is given, it will assume it is executed on all ranks.
    """

    os.makedirs(output_dir, exist_ok=True)
    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)
    full_state_dict = OrderedDict()
    prev_file_name = None
    for name, tensor in state_dict.items():
        if hasattr(tensor.data, "full_tensor"):  # dtensor
            tensor = tensor.data.full_tensor()
        else:
            tensor = tensor.data

        if save_dtype:
            tensor = tensor.to(dtype=getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype)

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            if global_rank is None or global_rank == 0:
                _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)
                full_state_dict = OrderedDict()

            empty_cache()
            if global_rank is not None and dist.is_initialized():  # avoid process hanging
                synchronize()
                dist.barrier()

        if global_rank is None or global_rank == 0:
            full_state_dict[name] = tensor.detach().cpu()

        prev_file_name = weight_map[name]
        del tensor

    if global_rank is None or global_rank == 0:
        if len(full_state_dict):
            _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)

        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }

            index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(f"Model weight splits saved in {output_dir}.")
        else:
            logger.info(f"Model weights saved at {os.path.join(output_dir, prev_file_name)}.")

        if model_assets is not None:
            for model_asset in model_assets:
                if hasattr(model_asset, "save_pretrained"):
                    model_asset.save_pretrained(output_dir)
                else:
                    logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")
