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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

import torch
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils.import_utils import is_safetensors_available

from ..utils import logging
from ..utils.helper import empty_cache, get_dtype_size


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to("meta"), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def _load_state_dict(weights_path: str, **kwargs) -> List["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _find_submodule(module: "nn.Module", name: str) -> Tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._parameters[name].data
    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        module._parameters[name].data.copy_(tensor)


def _dispatch_parameter_cached(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    module_cache: dict,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns parameter to an empty model with module reference caching.

    This is optimized for MoE models where the same module path is accessed multiple times.
    """
    # Check cache first
    if name in module_cache:
        target_module, param_name = module_cache[name]
    else:
        # Cache miss: find module and cache the result
        target_module, param_name = _find_submodule(module, name)
        module_cache[name] = (target_module, param_name)

    # Get the parameter tensor
    orig_tensor = target_module._parameters[param_name].data
    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        target_module._parameters[param_name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        target_module._parameters[param_name].data.copy_(tensor)


def _batch_dispatch_moe_parameters(
    module: "nn.Module",
    moe_weights: List[Tuple[str, "torch.Tensor"]],
    module_cache: dict,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Batch dispatch multiple MoE expert weights to the same parameter.
    
    This is a major optimization for MoE models where multiple expert weights
    (e.g., experts.0.gate_proj, experts.1.gate_proj, ..., experts.99.gate_proj)
    need to be loaded into the same stacked parameter tensor.
    """
    if not moe_weights:
        return
    
    # Get all parameter names from the module for mapping
    module_parameter_names = {name for name, _ in module.named_parameters()}
    
    # Group weights by their mapped parameter name
    param_groups = {}
    for checkpoint_name, tensor in moe_weights:
        mapped_name = _map_moe_expert_key(checkpoint_name, module_parameter_names)
        if mapped_name:
            if mapped_name not in param_groups:
                param_groups[mapped_name] = []
            param_groups[mapped_name].append((checkpoint_name, tensor))
    
    # Process each parameter group
    for param_name, weight_list in param_groups.items():
        if len(weight_list) == 1:
            # Single weight, use normal dispatch
            checkpoint_name, tensor = weight_list[0]
            _dispatch_parameter_cached(module, param_name, tensor, module_cache, dtensor_factory)
        else:
            # Multiple weights for the same parameter - batch process
            _batch_load_moe_parameter(module, param_name, weight_list, module_cache, dtensor_factory)


def _batch_load_moe_parameter(
    module: "nn.Module",
    param_name: str,
    weight_list: List[Tuple[str, "torch.Tensor"]],
    module_cache: dict,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Efficiently load multiple expert weights into a single stacked parameter.
    
    This function extracts the expert index from checkpoint names and places
    each weight at the correct position in the stacked parameter tensor.
    """
    # Get the target parameter tensor
    if param_name in module_cache:
        target_module, param_name_short = module_cache[param_name]
    else:
        target_module, param_name_short = _find_submodule(module, param_name)
        module_cache[param_name] = (target_module, param_name_short)
    
    orig_tensor = target_module._parameters[param_name_short].data
    
    # Process each expert weight
    for checkpoint_name, tensor in weight_list:
        # Extract expert index from checkpoint name (e.g., "experts.99.gate_proj.weight" -> 99)
        import re
        match = re.search(r'\.experts\.(\d+)\.', checkpoint_name)
        if match:
            expert_idx = int(match.group(1))
            tensor = tensor.to(orig_tensor)
            
            # Copy to the correct expert position in the stacked tensor
            if hasattr(orig_tensor, "device_mesh"):  # dtensor
                if orig_tensor.device.type == "cpu":
                    raise ValueError("Cannot load dtensor on CPU.")
                
                device_mesh = getattr(orig_tensor, "device_mesh")
                placements = getattr(orig_tensor, "placements")
                # For dtensor, we need to handle this carefully
                orig_tensor[expert_idx].copy_(tensor)
            else:
                # Direct copy to the expert slice
                orig_tensor[expert_idx].copy_(tensor)


def _map_moe_expert_key(name: str, parameter_names: set) -> Optional[str]:
    """
    Maps MoE expert keys from checkpoint format to model format.
    
    Args:
        name: Key from checkpoint (e.g., 'model.layers.0.mlp.experts.0.gate_proj.weight')
        parameter_names: Set of parameter names in the model
        
    Returns:
        Mapped key name if found, None otherwise
    """
    # Check if this is a MoE expert key (contains 'experts.N.')
    if 'experts.' in name and '.experts.' in name:
        # Use a simple and robust approach: regex to remove expert index and .weight
        if name.endswith('.weight'):
            # Remove .weight suffix
            name_no_weight = name[:-7]
            # Remove expert index (any number after 'experts.')
            import re

            mapped_name = re.sub(r'\.experts\.\d+\.', '.experts.', name_no_weight)
            
            # Try with model. prefix first (since that's what the model actually has)
            if mapped_name in parameter_names:
                return mapped_name
            
            # Try without model. prefix as fallback
            if mapped_name.startswith('model.'):
                mapped_name_no_prefix = mapped_name[6:]
                if mapped_name_no_prefix in parameter_names:
                    return mapped_name_no_prefix
            
            # If stacked parameter doesn't exist, we need to create a mapping
            # from individual expert parameters to the stacked parameter
            # This is the key insight: when checkpoint has individual experts but model expects stacked
            expert_pattern = re.search(r'\.experts\.(\d+)\.', name)
            if expert_pattern:
                # Extract the parameter type (e.g., 'gate_proj', 'up_proj', 'down_proj')
                param_type = name_no_weight.split('.')[-1]
                
                # Create the expected stacked parameter name
                layer_base = '.'.join(name_no_weight.split('.')[:-2])  # e.g., 'model.layers.1.mlp'
                stacked_param_name = f"{layer_base}.experts.{param_type}"

                # Check if this stacked parameter exists in the model
                if stacked_param_name in parameter_names:
                    return stacked_param_name
                
                # Also check without model. prefix
                stacked_param_name_no_prefix = stacked_param_name[6:] if stacked_param_name.startswith('model.') else stacked_param_name
                if stacked_param_name_no_prefix in parameter_names:
                    return stacked_param_name_no_prefix

                return None

    return None


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name].data
    module._buffers[name] = buffer.to(orig_tensor)


def _init_parameter(
    module: "nn.Module",
    name: str,
) -> None:
    """
    Initializes parameter in model.
    """
    pieces = name.split(".")
    init_func = None
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if hasattr(module, "_init_weights"):
            init_func = getattr(module, "_init_weights")

        module = getattr(module, piece)

    if init_func is None:
        raise ValueError(f"Cannot retrieve `_init_weights` function in the parents of {module}.")

    module.apply(init_func)


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names = {name for name, _ in model.named_parameters()}

    model.to_empty(device=init_device)
    
    # Initialize module cache for MoE optimization
    module_cache = {}
    
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        # Collect MoE weights for batch processing
        moe_weights = []
        
        for name, tensor in state_dict_iterator:
            if name in buffer_dict.keys():  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                parameter_names.remove(name)
                _dispatch_parameter(model, name, tensor, dtensor_factory)
            else:
                # Check if this is a MoE expert weight
                mapped_name = _map_moe_expert_key(name, parameter_names)
                if mapped_name and mapped_name in parameter_names:
                    moe_weights.append((name, tensor))
                elif 'experts.' in name and '.experts.' in name:
                    # Still try to add it to moe_weights for processing
                    if mapped_name:  # If we have a mapped name, use it
                        moe_weights.append((name, tensor))
                else:
                    # This is a truly unexpected key
                    logger.info_rank0(f"Unexpected key in state dict: {name}.")

        # Batch process MoE weights for maximum efficiency
        if moe_weights:
            _batch_dispatch_moe_parameters(model, moe_weights, module_cache, dtensor_factory)

            # Remove mapped MoE parameters from parameter_names after processing
            # This ensures we don't remove them prematurely when multiple experts map to the same parameter
            processed_moe_params = set()
            for checkpoint_name, _ in moe_weights:
                mapped_name = _map_moe_expert_key(checkpoint_name, parameter_names)
                if mapped_name and mapped_name in parameter_names:
                    processed_moe_params.add(mapped_name)

            # Remove all processed MoE parameters at once
            for param_name in processed_moe_params:
                parameter_names.discard(param_name)

        del state_dict_iterator
        empty_cache()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)

    if len(parameter_names) > 0:
        logger.info_rank0(f"Find missing key(s) in state dict: {parameter_names}, initialize them.")
        for name in parameter_names:
            _init_parameter(model, name)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
        except Exception as e:
            logger.info_rank0(f"Failed to tie embeddings: {e}")


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

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
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


def _save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save function.
    """
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


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
                torch.cuda.synchronize()
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


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")
