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


import itertools
import json
import math
import os
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

import torch


try:
    from hdfs_io import copy  # for internal use only
except ImportError:
    from ..utils.hdfs_io import copy
from safetensors import safe_open
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import get_device_type, synchronize
from ..utils.helper import empty_cache, get_cache_dir, get_dtype_size
from ..utils.import_utils import is_diffusers_available
from .checkpoint_tensor_loading import (
    checkpoint_converter_is_dim0_zero_pad,
    get_checkpoint_tensor_converter,
    maybe_convert_checkpoint_tensor,
)


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..distributed.parallel_plan import ParallelPlan

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
        getattr(torch, "float8_e8m0fnu", None),
    )
    if dtype is not None
)


def _requires_byte_broadcast(dtype: "torch.dtype") -> bool:
    return dtype in _FLOAT8_DTYPES


def _num_storage_bytes(shape: Tuple[int, ...], dtype: "torch.dtype") -> int:
    if dtype in _FLOAT8_DTYPES or dtype == torch.bool:
        return math.prod(shape)
    elem_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    return math.prod(shape) * elem_size


def _view_as_bytes(tensor: "torch.Tensor") -> "torch.Tensor":
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def _view_from_bytes(tensor: "torch.Tensor", dtype: "torch.dtype", shape: Tuple[int, ...]) -> "torch.Tensor":
    return tensor.view(dtype).reshape(shape)


def _is_all_zero_tensor(tensor: "torch.Tensor") -> bool:
    if tensor.is_meta:
        return False
    try:
        return bool(torch.count_nonzero(tensor).item() == 0)
    except NotImplementedError:
        return False


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
            # When we have a case of tensor2 = tensor1, it would call the set_attr
            # of param, which in turn would call the register_parameter API.
            # In this case, the new param is already on meta-device, since it was moved
            # previously when it was initialized. Hence, when resetting, you can
            # directly assign that tensor instead of re-init. If you re-init you would
            # lose the relationship.
            module._parameters[name] = (
                param
                if param.device == torch.device("meta")
                else param_cls(module._parameters[name].to("meta"), **kwargs)
            )

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


@dataclass
class BroadcastMetadata:
    done: bool
    name: Optional[str]
    shape: Optional["torch.Size"]
    dtype: Optional["torch.dtype"]


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

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    if is_diffusers_available():
        from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
        from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME

        resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
        if resolved_weight_file:
            return [StateDictIterator(resolved_weight_file)]

        resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
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
    parallel_plan: Optional["ParallelPlan"] = None,
    dtensor_to_cpu: bool = False,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    full_param_name = name
    module, local_name = _find_submodule(module, name)
    orig_tensor = module._parameters[local_name].data

    # Handle parameter slicing according to parallel_plan, now only ExtraParallel-aware
    if parallel_plan is not None:
        tensor = parallel_plan.shard_tensor(tensor, full_param_name, orig_tensor.shape)

    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if dtensor_factory is None:
            raise ValueError("dtensor parameter requires a dtensor_factory.")
        device_mesh = orig_tensor.device_mesh
        placements = orig_tensor.placements
        sharded_tensor = dtensor_factory(tensor.to(dtype=orig_tensor.dtype), device_mesh, placements)
        if dtensor_to_cpu:
            sharded_tensor = sharded_tensor.to("cpu")
        module._parameters[local_name].data.copy_(sharded_tensor)
    else:  # not dtensor
        tensor = tensor.to(orig_tensor)
        module._parameters[local_name].data.copy_(tensor)


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    dtensor_to_cpu: bool = False,
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name]

    if hasattr(orig_tensor, "device_mesh"):  # dtensor buffer
        if dtensor_factory is None:
            raise ValueError("dtensor buffer requires a dtensor_factory.")

        device_mesh = orig_tensor.device_mesh
        placements = orig_tensor.placements
        sharded_buffer = dtensor_factory(buffer.to(dtype=orig_tensor.dtype), device_mesh, placements)
        if dtensor_to_cpu:
            sharded_buffer = sharded_buffer.to("cpu")
        module._buffers[name] = sharded_buffer
    else:
        module._buffers[name].copy_(buffer.to(device=orig_tensor.device, dtype=orig_tensor.dtype))


def _get_communication_device(init_device: Literal["cpu", "cuda", "npu"]) -> torch.device:
    if init_device == "cpu":
        return torch.device(get_device_type())
    return torch.device(init_device)


def _init_parameter(
    module: "nn.Module",
    name: str,
) -> None:
    """
    Initializes parameter in model.
    """
    pieces = name.split(".")
    if any(p.startswith("lora_") for p in pieces):
        from ..lora.weight_loading import init_lora_parameter

        init_lora_parameter(module, name)
        return
    init_func = None
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if hasattr(module, "_init_weights"):
            init_func = module._init_weights

        module = getattr(module, piece)

    if init_func is None:
        raise ValueError(f"Cannot retrieve `_init_weights` function in the parents of {module}.")

    module.apply(init_func)


def _convert_weight_key(key: str, model: "PreTrainedModel") -> str:
    """
    Convert a single state dict key using the model's checkpoint conversion mapping.

    For example, in the InternVL, we have _checkpoint_conversion_mapping = {"^model": "language_model"}

    This is to adapt to the big breaking change introduced in HF transformers 4.52:
    https://github.com/huggingface/transformers/pull/38385
    """
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return key

    for pattern, replacement in model._checkpoint_conversion_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        converted_key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return converted_key

    return key


def _param_larger_than(shape: Tuple[int, ...], dtype: torch.dtype, max_load_broadcast_size: float = 20.0) -> bool:
    """
    Check if a parameter is large enough to be sharded.
    """
    param_size = _num_storage_bytes(tuple(shape), dtype)
    return param_size >= max_load_broadcast_size * (1024**3)


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    **kwargs,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    # Get parallel plan if available -- via the runtime helper which
    # detects PEFT and prepends ``base_model.model.`` to every plan
    # pattern so EP-aware ``parallel_plan.shard_tensor`` matches the
    # full PEFT-namespaced ``full_param_name`` it'll see below.
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        from ..distributed.parallel_plan import get_runtime_parallel_plan

        parallel_plan = get_runtime_parallel_plan(model)

    # Build LoRA key remapping when loading a base checkpoint into a PEFT-wrapped model.
    # Maps bare base-model param names to PEFT-namespaced FQNs, e.g.:
    #   "layers.0.self_attn.q_proj.weight" -> "base_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    # Keys not found in the map receive a plain "base_model.model." prefix.
    is_peft_model = kwargs.get("is_peft_model", False)
    adapter_path = kwargs.get("adapter_path", None)
    if is_peft_model:
        from ..lora.weight_loading import build_lora_key_overrides

        lora_key_overrides = build_lora_key_overrides(model)

    def _apply_peft_override(bare_name: str) -> str:
        """Map a *bare* base-model FQN to its PEFT-wrapped destination.

        Applied AFTER ``maybe_convert_checkpoint_tensor`` so converter-produced
        merged keys (e.g. ``model.layers.0.mlp.experts.gate_up_proj`` from the
        Qwen3-MoE per-expert -> fused converter) also flow through the
        ``base_layer.weight`` rename when the experts module is wrapped by
        ``LoraSharedExperts`` / ``LoraIndependentExperts``. Keys without an
        override entry receive the plain ``base_model.model.`` prefix.
        """
        if not is_peft_model:
            return bare_name
        return lora_key_overrides.get(bare_name, "base_model.model." + bare_name)

    converter = get_checkpoint_tensor_converter(model)
    if converter is None and is_peft_model and hasattr(model, "get_base_model"):
        converter = get_checkpoint_tensor_converter(model.get_base_model())
    state_dict_iterators = _load_state_dict(weights_path)

    def _dispatch_kv(name: str, tensor: "torch.Tensor") -> None:
        if name in buffer_dict.keys():  # persistent buffers
            buffer_dict[name] = tensor.clone()
        elif name in parameter_names_to_load:
            parameter_names_to_load.remove(name)
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
        else:
            logger.info_rank0(f"Unexpected key in state dict: {name}.")

    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            name = _convert_weight_key(name, model)
            converted = maybe_convert_checkpoint_tensor(name, tensor, converter)
            if converted is None:
                continue
            _dispatch_kv(_apply_peft_override(converted.name), converted.tensor)

        del state_dict_iterator
        empty_cache()

    if converter is not None:
        for result in converter.finalize():
            _dispatch_kv(_apply_peft_override(result.name), result.tensor)

    if is_peft_model and adapter_path:
        # Load LoRA adapter weights when an adapter_path is provided; otherwise
        # they are initialised in post_process_after_weight_loading. The native
        # VeOmniLoraModel reads the PEFT-format file without importing peft.
        from ..lora.weight_loading import load_lora_weights

        load_lora_weights(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parameter_names_to_load=parameter_names_to_load,
            # EP-aware slicing: ``LoraIndependentExperts`` LoRA tensors are shrunk
            # from ``[E, ...]`` to ``[E_local, ...]`` inside ``_dispatch_parameter``
            # before the DTensor copy, or the propagation asserts on shape mismatch.
            parallel_plan=parallel_plan,
        )

    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )

    fqn_to_index_mapping = kwargs.get("fqn_to_index_mapping")
    if fqn_to_index_mapping is not None:
        from .checkpoint_tensor_loading import prepare_fqn_to_index_mapping_for_model

        prepare_fqn_to_index_mapping_for_model(model, fqn_to_index_mapping)


def _resolve_safetensors_shards(weights_path: str, **kwargs) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Resolve a (sharded) safetensors checkpoint for per-key streaming reads.

    Returns ``(key_to_file, file_to_path)`` where ``key_to_file`` maps each
    checkpoint tensor name to its shard **basename** and ``file_to_path`` maps
    that basename to the resolved local file path. Handles both the sharded
    (``model.safetensors.index.json`` + ``model-0000x-of-0000y.safetensors``)
    and single-file (``model.safetensors``) layouts.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_index = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_index:
        with open(resolved_index) as fh:
            weight_map = json.load(fh)["weight_map"]  # key -> shard basename
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_index, **kwargs)
        file_to_path = {os.path.basename(p): p for p in shard_files}
        return weight_map, file_to_path

    resolved_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_file:
        base = os.path.basename(resolved_file)
        with safe_open(resolved_file, framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
        return dict.fromkeys(keys, base), {base: resolved_file}

    raise ValueError(f"ep_sharded_stream_load: no safetensors checkpoint found under {weights_path}.")


@torch.no_grad()
def load_model_weights_ep_sharded(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    **kwargs,
) -> None:
    """Per-rank ExtraParallel-slice streaming loader (opt-in alternative to
    :func:`load_model_weights`).

    For parameters the model's ``get_parallel_plan`` marks ExtraParallel-sharded
    (e.g. MoE experts, ``Shard(0)`` over the ``ep`` mesh), this reads **only this
    rank's dim-0 slice** straight from the safetensors shard via
    ``safe_open(...).get_slice()[start:end]`` -- instead of reading the whole
    ``[E, ...]`` tensor and slicing it in host memory (what
    :func:`load_model_weights` does via ``ParallelPlan.shard_tensor``).

    Why it is faster / lighter for large MoE checkpoints:
      * per-rank disk/HDFS bytes for expert tensors drop from the whole set to
        ``1/ep`` of it (in aggregate the expert bytes are read ~once across the
        ``ep`` ranks, in parallel), vs every rank reading the full expert set and
        discarding ``(ep-1)/ep``;
      * peak host RAM drops accordingly -- the full ``[E, ...]`` tensor is never
        materialised, only the ``[E/ep, ...]`` local shard.

    Dense (non-ExtraParallel) params and buffers are read whole (small relative
    to experts) and dispatched exactly as in :func:`load_model_weights` (FSDP's
    ``distribute_tensor`` still shards them). The dispatch of a sliced expert is
    identical to the whole-tensor path's second half: pass the already-sliced
    ``[E/ep, ...]`` with ``parallel_plan=None`` so it is not sliced again, then
    the FSDP ``dtensor_factory`` shards it over the ``ep_fsdp`` sub-mesh.

    Raises ``NotImplementedError`` when the checkpoint/model is unsupported: PEFT,
    no ExtraParallel ``get_parallel_plan``, or a checkpoint-tensor converter whose
    transform is not a pure dim-0 zero-pad (a fusion converter needs the whole
    tensor set). A converter that only zero-pads dim-0 stays streamable -- each
    rank reads its real-row slice and zero-fills the tail -- and opts in via the
    optional ``CheckpointTensorConverter.is_dim0_zero_pad`` capability (see
    :func:`checkpoint_converter_is_dim0_zero_pad`).
    """
    if kwargs.get("is_peft_model", False):
        raise NotImplementedError("ep_sharded_stream_load does not support PEFT models.")
    # A checkpoint-tensor converter generally needs the whole tensor set (e.g.
    # per-expert-key fusion) which streaming can't provide. The one streamable
    # exception is a *pure dim-0 zero-pad* converter, which opts in via the
    # optional ``is_dim0_zero_pad`` capability; that is enforced per-key below
    # (dim-0 zero-pad -> stream + tail zero-fill; anything else -> bail).
    converter = get_checkpoint_tensor_converter(model)
    get_plan = getattr(model, "get_parallel_plan", None)
    parallel_plan = get_plan() if get_plan is not None else None
    if parallel_plan is None or not getattr(parallel_plan, "extra_parallel_plan", None):
        raise NotImplementedError("ep_sharded_stream_load requires a model with an ExtraParallel parallel_plan.")

    # This streaming loader reads each rank's ExtraParallel slice with a dim-0
    # ``get_slice()[start:end]`` -- the same dim-0 assumption upstream's
    # ``ParallelPlan._slice_shard_tensor`` makes (every VeOmni ExtraParallel plan
    # today is ``Shard(0)``: MoE experts, embed parallel). If a plan ever shards a
    # non-zero dim, a blind dim-0 slice would silently corrupt weights, so bail to
    # the whole-tensor loader (which handles arbitrary ``Shard(dim)`` via DTensor).
    for _pname, _pplan in parallel_plan.extra_parallel_plan.items():
        for _fqn_pattern, _placement in _pplan.items():
            _dim = getattr(_placement, "dim", None)
            if _dim is not None and _dim != 0:
                raise NotImplementedError(
                    f"ep_sharded_stream_load only supports dim-0 ExtraParallel sharding (Shard(0)); "
                    f"'{_pname}' pattern '{_fqn_pattern}' uses Shard({_dim})."
                )

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    param_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    parameter_names_to_load = set(param_shapes.keys())
    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    parallel_state = get_parallel_state()
    key_to_file, file_to_path = _resolve_safetensors_shards(weights_path, **kwargs)

    keys_by_file: Dict[str, List[str]] = {}
    for key, fname in key_to_file.items():
        keys_by_file.setdefault(fname, []).append(key)

    n_ep = n_dense = n_buf = 0
    for fname in tqdm(
        sorted(keys_by_file),
        desc="Streaming EP-sharded checkpoint",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    ):
        with safe_open(file_to_path[fname], framework="pt", device="cpu") as f:
            for raw_name in keys_by_file[fname]:
                name = _convert_weight_key(raw_name, model)
                if name in buffer_dict:  # persistent buffers: read whole
                    buffer_dict[name] = f.get_tensor(raw_name).clone()
                    n_buf += 1
                    continue
                if name not in parameter_names_to_load:
                    if converter is not None and converter.can_handle(name):
                        raise NotImplementedError(
                            "ep_sharded_stream_load does not support checkpoint tensor conversion for "
                            f"unexpected key '{name}'. Use load_model_weights or rank0_load_and_broadcast_weights "
                            "for FP8/int8 scaled checkpoints."
                        )
                    logger.info_rank0(f"Unexpected key in state dict: {name}.")
                    continue

                shard_group = parallel_plan._get_shard_parameter_groupname(name)
                if shard_group is not None:
                    # ExtraParallel (e.g. EP / embed) param -> read only this rank's
                    # dim-0 slice. ``param_shapes[name][0]`` is the per-rank chunk, so
                    # the model's full dim0 is ``expected_full0 = target0 * para_size``.
                    # If a converter declares this key a pure dim-0 zero-pad, the model
                    # may have more rows than the checkpoint (``real0``): read the
                    # overlap and zero-fill the tail. Otherwise the checkpoint must
                    # match the model exactly.
                    zero_pad = checkpoint_converter_is_dim0_zero_pad(converter, name)
                    if converter is not None and converter.can_handle(name) and not zero_pad:
                        # A converter that fuses/reshapes this key can't be streamed.
                        raise NotImplementedError(
                            f"ep_sharded_stream_load: converter applies a non-dim0-zero-pad "
                            f"transform to ExtraParallel key '{name}'."
                        )
                    target0 = param_shapes[name][0]
                    para_size = (
                        parallel_state.extra_parallel_sizes[shard_group]
                        if parallel_state.extra_parallel_enabled(shard_group)
                        else 1
                    )
                    para_rank = (
                        parallel_state.extra_parallel_rank(shard_group)
                        if parallel_state.extra_parallel_enabled(shard_group)
                        else 0
                    )
                    sl = f.get_slice(raw_name)
                    real0 = sl.get_shape()[0]
                    expected_full0 = target0 * para_size
                    if real0 > expected_full0:
                        raise RuntimeError(f"{name}: checkpoint dim0={real0} exceeds model dim0={expected_full0}.")
                    if not zero_pad and real0 != expected_full0:
                        # No dim-0 zero-pad converter -> checkpoint must match exactly;
                        # a silent zero-fill here would hide a real shape mismatch.
                        raise RuntimeError(
                            f"{name}: checkpoint dim0={real0} != model dim0={expected_full0} "
                            f"(no dim-0 zero-pad converter for this key)."
                        )
                    start = para_rank * target0
                    end = start + target0
                    # Real checkpoint rows for this rank are [start, real0); clamp BOTH
                    # ends to real0 so a rank lying entirely in the zero-pad region
                    # (start >= real0) reads an in-bounds empty slice (sl[real0:real0])
                    # instead of relying on out-of-bounds ``get_slice`` semantics, which
                    # vary by safetensors version. Missing tail rows are zero-filled.
                    read_start = min(start, real0)
                    read_end = min(end, real0)
                    tensor = sl[read_start:read_end]
                    if tensor.shape[0] < target0:  # trailing zero rows (dim-0 zero-pad converter)
                        pad = torch.zeros((target0 - tensor.shape[0], *tuple(tensor.shape[1:])), dtype=tensor.dtype)
                        tensor = torch.cat([tensor, pad], dim=0)
                    # Already the local slice -> parallel_plan=None (do not slice
                    # again); dtensor_factory then shards over the ep_fsdp sub-mesh
                    # exactly as the whole-tensor path's post-shard_tensor half.
                    _dispatch_parameter(model, name, tensor, dtensor_factory, None, dtensor_to_cpu)
                    n_ep += 1
                else:
                    tensor = f.get_tensor(raw_name)
                    converted = maybe_convert_checkpoint_tensor(name, tensor, converter)
                    if converted is None:
                        raise NotImplementedError(
                            "ep_sharded_stream_load does not support checkpoint tensor conversion that buffers "
                            f"dense key '{name}'. Use load_model_weights or rank0_load_and_broadcast_weights "
                            "for FP8/int8 scaled checkpoints."
                        )
                    _dispatch_parameter(
                        model, converted.name, converted.tensor, dtensor_factory, parallel_plan, dtensor_to_cpu
                    )
                    n_dense += 1
                parameter_names_to_load.discard(name)
        empty_cache()

    logger.info_rank0(
        f"ep_sharded_stream_load: read {n_ep} ExtraParallel-sliced, {n_dense} dense, {n_buf} buffer tensors/rank."
    )
    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )


@torch.no_grad()
def rank0_load_and_broadcast_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    cpu_load_param_name: List[str] = None,
    max_load_broadcast_size: float = 20.0,  # in GB
    **kwargs,
):
    """
    This functions serves as the same purpose as `load_model_weights`
    but reduces disk I/O by broadcasting weights from rank0.
    In comparison, `load_model_weights` would require every GPU to go through the entire model weights on disk.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning_once("Distributed environment not initialized, falling back to load_model_weights.")
        return load_model_weights(model, weights_path, init_device, dtensor_factory, **kwargs)

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    # Get parallel plan if available -- routed through the runtime helper
    # so PEFT-prefix bridging happens once per call (see ``load_model_weights``).
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        from ..distributed.parallel_plan import get_runtime_parallel_plan

        parallel_plan = get_runtime_parallel_plan(model)

    # Build LoRA key remapping when loading a base checkpoint into a PEFT-wrapped model.
    # non-lora-layer: xxx.xxx -> base_model.model.xxx.xxx
    # lora-layer: xxx.xxx.weight -> base_model.model.xxx.xxx.base_layer.weight
    is_peft_model = kwargs.get("is_peft_model", False)
    adapter_path = kwargs.get("adapter_path", None)
    if is_peft_model:
        from ..lora.weight_loading import build_lora_key_overrides

        lora_key_overrides = build_lora_key_overrides(model)

    converter = get_checkpoint_tensor_converter(model)
    if converter is None and is_peft_model and hasattr(model, "get_base_model"):
        converter = get_checkpoint_tensor_converter(model.get_base_model())
    global_rank = get_parallel_state().global_rank
    torch_device = _get_communication_device(init_device)

    def _broadcast_and_dispatch(name, shape, dtype, tensor):
        """Broadcast a single (name, tensor) from rank0 and dispatch it."""
        logger.info_rank0(f"rank0_load_and_broadcast_weights: broadcasting {name=}")
        broadcast_as_bytes = _requires_byte_broadcast(dtype)
        if global_rank == 0:
            tensor = tensor.to(torch_device, non_blocking=True)
            broadcast_tensor = _view_as_bytes(tensor) if broadcast_as_bytes else tensor
        else:
            if broadcast_as_bytes:
                broadcast_tensor = torch.empty(
                    _num_storage_bytes(shape, dtype), dtype=torch.uint8, device=torch_device
                )
            else:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
                broadcast_tensor = tensor

        start_time = time.perf_counter()
        dist.broadcast(broadcast_tensor, src=0)
        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
        )
        if broadcast_as_bytes:
            tensor = _view_from_bytes(broadcast_tensor, dtype, shape)

        if name in buffer_dict:
            buffer_dict[name] = tensor.detach().clone()
        elif name in parameter_names_to_load:
            parameter_names_to_load.discard(name)
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
        else:
            if global_rank == 0:
                logger.info_rank0(f"Unexpected key in state dict: {name}.")
        del tensor

    # P2P chunk transfer parameter
    extra_parallel_shard_dst_cache: Dict[str, List[List[int]]] = {}
    p2p_chunk_tag_counter = 0

    def _next_chunk_p2p_tag() -> int:
        nonlocal p2p_chunk_tag_counter
        p2p_chunk_tag_counter += 1
        return p2p_chunk_tag_counter

    def _get_extra_parallel_shard_dst_ranks(
        parallel_state: Any,
        para_group_name: str,
        para_size: int,
        device: torch.device,
    ) -> List[List[int]]:
        """
        Build chunk_id -> [global ranks] table via one all_gather (cached for this load).

        Each global rank reports its extra_parallel_rank for `para_group_name`; ranks whose
        rank is in [0, para_size) are grouped by shard index.
        """
        if para_group_name in extra_parallel_shard_dst_cache:
            return extra_parallel_shard_dst_cache[para_group_name]
        if not dist.is_initialized():
            raise RuntimeError("_get_extra_parallel_shard_dst_ranks requires initialized process group")
        world_size = dist.get_world_size()
        mine = torch.tensor(
            [parallel_state.extra_parallel_rank(para_group_name)],
            dtype=torch.long,
            device=device,
        )

        gathered = [torch.empty_like(mine) for _ in range(world_size)]
        dist.all_gather(gathered, mine)
        table: List[List[int]] = [[] for _ in range(para_size)]
        for r in range(world_size):
            epr = int(gathered[r].item())
            assert 0 <= epr < para_size, (
                f"Global rank {r} reports extra_parallel_rank {epr} for parallel group {para_group_name}, which is out of range [0, {para_size})."
            )
            table[epr].append(r)
        extra_parallel_shard_dst_cache[para_group_name] = table
        return table

    def _chunk_and_broadcast_and_dispatch(name, shape, dtype, tensor):
        """Broadcast a single (name, tensor) from rank0 and dispatch it."""
        logger.info_rank0(f"rank0_load_and_broadcast_weights: chunking and broadcasting {name=}")

        assert name not in buffer_dict, f"Buffer {name} should not be chunked."
        assert name in parameter_names_to_load, f"Unexpected key in state dict: {name}."

        if global_rank == 0:
            assert tensor.device == torch.device("cpu"), "Large parameter should be loaded to CPU first."

        start_time = time.perf_counter()

        para_group_name = parallel_plan._get_shard_parameter_groupname(name)
        assert para_group_name is not None, (
            f"Parameter {name} can not be chunked as it is not part of any parallel group."
        )

        parallel_state = get_parallel_state()
        assert parallel_state.extra_parallel_enabled(para_group_name), (
            f"Parallel group {para_group_name} is not enabled for extra parallel."
        )

        module, local_name = _find_submodule(model, name)
        shard_tensor = module._parameters[local_name].data
        target_shape = shard_tensor.shape

        para_size = parallel_state.extra_parallel_sizes[para_group_name]
        assert len(shape) >= 1, f"Original parameter {name} to be chunked has shape {shape}, which is 0-dim."
        assert len(target_shape) >= 1, f"Shard parameter {name} to get chunk has shape {target_shape}, which is 0-dim."

        if global_rank == 0:
            if shape[0] % para_size != 0:
                logger.info_rank0(
                    f"Parallel group {para_group_name} size {para_size} does not divide original parameter shape {shape} at dim 0."
                )

                target_size = list(shard_tensor.size())
                target_size[0] *= para_size
                target_size = torch.Size(target_size)
                loaded_size = tensor.size()
                pad_size = tuple(
                    (0, target_dim - loaded_dim) for target_dim, loaded_dim in zip(target_size, loaded_size)
                )
                pad_size = tuple(itertools.chain(*(pad_size[::-1])))

                logger.info_rank0(
                    f"Shard parameter shape = {target_shape}, Loaded parameter shape = {tensor.shape}, pad_size = {pad_size}"
                )

                tensor = torch.nn.functional.pad(tensor, pad_size, value=0.0)

            assert tensor.shape[0] // para_size == target_shape[0], (
                f"Parallel {para_group_name}: padded shape {shape} at dim 0 // {para_size} == {tensor.shape[0] // para_size}, not equal to {target_shape[0]}"
            )

            chunk_loaded_data = list(tensor.chunk(para_size, dim=0))

        is_shard_tensor_dtensor = hasattr(shard_tensor, "device_mesh")
        if is_shard_tensor_dtensor:
            device_mesh = shard_tensor.device_mesh
            placements = shard_tensor.placements
            shard_comm_device = torch.device(device_mesh.device_type)
        else:
            device_mesh = None
            placements = None
            shard_comm_device = _get_communication_device(init_device)

        chunk_broadcast_as_bytes = _requires_byte_broadcast(shard_tensor.dtype)
        if chunk_broadcast_as_bytes:
            broadcast_buffer = torch.empty(
                _num_storage_bytes(tuple(shard_tensor.shape), shard_tensor.dtype),
                dtype=torch.uint8,
                device=shard_comm_device,
            )
        else:
            broadcast_buffer = torch.empty(
                shard_tensor.shape,
                dtype=shard_tensor.dtype,
                device=shard_comm_device,
            )
        shard_dst_ranks = _get_extra_parallel_shard_dst_ranks(
            parallel_state, para_group_name, para_size, shard_comm_device
        )
        for chunk_id in range(para_size):
            # For example:
            #   if we have two params, ranks = [0, 1, 2, 3, 4, 5, 6, 7], then
            #   at extra_parallel_1 with para_size = 2,
            #     when chunk_id = 0, send_seq = [2, 4, 6], p2p tag = [1, 1, 1]
            #     when chunk_id = 1, send_seq = [1, 3, 5, 7], p2p tag = [2, 2, 2, 2]
            #   at extra_parallel_2 with para_size = 4,
            #     when chunk_id = 0, send_seq = [4], p2p tag = [3]
            #     when chunk_id = 1, send_seq = [1, 5], p2p tag = [4, 4]
            #     when chunk_id = 2, send_seq = [2, 6], p2p tag = [5, 5]
            #     when chunk_id = 3, send_seq = [3, 7], p2p tag = [6, 6]

            if dist.get_rank() == 0:
                if chunk_broadcast_as_bytes:
                    chunk_tensor = chunk_loaded_data[chunk_id].to(device=shard_comm_device, dtype=shard_tensor.dtype)
                    broadcast_buffer.copy_(_view_as_bytes(chunk_tensor))
                else:
                    broadcast_buffer.copy_(chunk_loaded_data[chunk_id].contiguous())
            dst_ranks = sorted(shard_dst_ranks[chunk_id])
            # One tag increment per chunk_id on every rank so the counter stays aligned.
            tag = _next_chunk_p2p_tag()
            send_seq = [d for d in dst_ranks if d != 0]
            extra_para_local_rank = parallel_state.extra_parallel_rank(para_group_name)

            if global_rank == 0:
                for dst in send_seq:
                    dist.send(broadcast_buffer, dst=dst, tag=tag)

                if global_rank in dst_ranks:
                    dispatch_buffer = (
                        _view_from_bytes(broadcast_buffer, shard_tensor.dtype, tuple(shard_tensor.shape))
                        if chunk_broadcast_as_bytes
                        else broadcast_buffer
                    )
                    if is_shard_tensor_dtensor:
                        chunk_tensor = dtensor_factory(dispatch_buffer, device_mesh, placements).contiguous()
                        if dtensor_to_cpu:
                            chunk_tensor = chunk_tensor.to("cpu")
                        shard_tensor.copy_(chunk_tensor)
                    else:
                        shard_tensor.copy_(dispatch_buffer.to(shard_tensor.device).contiguous())

            elif global_rank in send_seq:
                if is_shard_tensor_dtensor:
                    assert device_mesh.mesh.tolist() == dst_ranks, (
                        f"Device mesh {device_mesh.mesh.tolist()} does not match dst ranks {dst_ranks}."
                    )

                assert extra_para_local_rank == chunk_id, f"Rank {global_rank} is not the shard {chunk_id} rank."
                dist.recv(broadcast_buffer, src=0, tag=tag)

                dispatch_buffer = (
                    _view_from_bytes(broadcast_buffer, shard_tensor.dtype, tuple(shard_tensor.shape))
                    if chunk_broadcast_as_bytes
                    else broadcast_buffer
                )
                if is_shard_tensor_dtensor:
                    chunk_tensor = dtensor_factory(dispatch_buffer, device_mesh, placements).contiguous()
                    if dtensor_to_cpu:
                        chunk_tensor = chunk_tensor.to("cpu")
                    shard_tensor.copy_(chunk_tensor)
                else:
                    shard_tensor.copy_(dispatch_buffer.to(shard_tensor.device).contiguous())

        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, chunk and broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
        )

        parameter_names_to_load.discard(name)
        del tensor

    # --- Broadcast shard count ---
    state_dict_iterators = _load_state_dict(weights_path) if global_rank == 0 else None
    shard_count = len(state_dict_iterators) if global_rank == 0 else 0
    logger.info_rank0(f"rank0_load_and_broadcast_weights: {shard_count=} ")
    shard_count_tensor = torch.tensor(
        [shard_count],
        dtype=torch.int64,
        device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
    )
    dist.broadcast(shard_count_tensor, src=0)
    shard_count = int(shard_count_tensor.item())

    if global_rank == 0:
        shard_iterable = enumerate(
            tqdm(
                state_dict_iterators,
                desc="Loading checkpoint shards",
                disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
            )
        )
    else:
        shard_iterable = enumerate(range(shard_count))

    # iterate safetensor files; each file would have a iterator to read weight keys and tensors
    for _shard_idx, shard_payload in shard_iterable:
        state_dict_iterator = shard_payload if global_rank == 0 else None
        iterator = iter(state_dict_iterator) if global_rank == 0 else None

        while True:
            tensor: Optional["torch.Tensor"] = None

            if global_rank == 0:
                while True:
                    # Inner loop: rank0 keeps reading tensors until the converter
                    # produces a result or the shard is exhausted. The converter may
                    # return None to indicate "still accumulating" (e.g. collecting
                    # per-expert MoE tensors), so we continue without broadcasting.
                    try:
                        key, tensor = next(iterator)  # type: ignore[arg-type]
                        key = _convert_weight_key(key, model)
                        converted = maybe_convert_checkpoint_tensor(key, tensor, converter)
                        if converted is None:
                            continue
                        key, tensor = converted.name, converted.tensor
                        # PEFT override is applied AFTER the converter so that
                        # converter-produced merged keys (e.g. Qwen3-MoE
                        # per-expert -> ``...experts.gate_up_proj``) also get
                        # mapped to their PEFT-wrapped ``...base_layer.weight``
                        # destination when the experts module is wrapped by
                        # ``LoraSharedExperts`` / ``LoraIndependentExperts``.
                        # Bare-key lookup intentionally: ``lora_key_overrides``
                        # is keyed by base-model FQNs (no ``base_model.model.``
                        # prefix), and the converter is given the bare key so
                        # MoE per-expert pattern matches still fire.
                        if is_peft_model:
                            key = lora_key_overrides.get(key, "base_model.model." + key)
                        logger.info_rank0(f"loading {key=}")
                        if _is_all_zero_tensor(tensor):
                            logger.warning_rank0(
                                f"Detected tensor with all-zero values when reading safetensor: {key=}"
                            )
                        metadata = BroadcastMetadata(False, key, tensor.shape, tensor.dtype)
                        break
                    except StopIteration:
                        metadata = BroadcastMetadata(True, None, None, None)
                        break
            else:
                metadata = BroadcastMetadata(False, None, None, None)

            metadata_list = [metadata]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            if metadata.done:
                break

            name = metadata.name
            shape = metadata.shape
            dtype = metadata.dtype

            if name is None or shape is None or dtype is None:
                raise RuntimeError("Received incomplete broadcast metadata.")
            if (
                (
                    (cpu_load_param_name is not None and name in cpu_load_param_name)
                    or _param_larger_than(shape, dtype, max_load_broadcast_size=max_load_broadcast_size)
                )
                and name in parameter_names_to_load
                and (parallel_plan is not None and parallel_plan._get_shard_parameter_groupname(name) is not None)
            ):
                _chunk_and_broadcast_and_dispatch(name, shape, dtype, tensor)
            else:
                _broadcast_and_dispatch(name, shape, dtype, tensor)

        if global_rank == 0:
            del state_dict_iterator

        empty_cache()

    # --- Flush converter (broadcast any finalized tensors) ---
    if converter is not None:
        finalized = converter.finalize() if global_rank == 0 else []
        fin_count_tensor = torch.tensor(
            [len(finalized)],
            dtype=torch.int64,
            device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
        )
        dist.broadcast(fin_count_tensor, src=0)
        fin_count = int(fin_count_tensor.item())

        for i in range(fin_count):
            if global_rank == 0:
                result = finalized[i]
                # Same post-converter PEFT override as the streaming loop
                # above -- finalize() may emit merged keys after every shard
                # has been read, e.g. the last gate/up pair for the
                # Qwen3-MoE per-expert -> fused converter.
                fin_name = result.name
                if is_peft_model:
                    fin_name = lora_key_overrides.get(fin_name, "base_model.model." + fin_name)
                metadata = BroadcastMetadata(False, fin_name, result.tensor.shape, result.tensor.dtype)
                tensor = result.tensor
            else:
                metadata = BroadcastMetadata(False, None, None, None)
                tensor = None

            metadata_list = [metadata]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            name = metadata.name
            shape = metadata.shape
            dtype = metadata.dtype
            if name is None or shape is None or dtype is None:
                raise RuntimeError("Received incomplete broadcast metadata from finalize.")
            _broadcast_and_dispatch(name, shape, dtype, tensor)

    if is_peft_model and adapter_path:
        # Rank-0 reads the PEFT-format adapter file (natively, no peft import)
        # and broadcasts each tensor to all ranks. The runtime plan is forwarded
        # so EP-sharded LoRA tensors (registered by
        # ``_extend_plan_for_moe_lora_independent`` for ``LoraIndependentExperts``)
        # get sliced from the disk-side ``[E, ...]`` shape down to the local
        # ``[E_local, ...]`` shape inside ``_dispatch_parameter`` before the
        # DTensor ``.copy_()`` -- without this the copy asserts on a global-shape
        # mismatch (the ep_size=2 + ``mode=="independent"`` failure mode).
        from ..lora.weight_loading import rank0_load_and_broadcast_lora_weights

        rank0_load_and_broadcast_lora_weights(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parameter_names_to_load=parameter_names_to_load,
            parallel_plan=parallel_plan,
        )

    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )

    fqn_to_index_mapping = kwargs.get("fqn_to_index_mapping")
    if fqn_to_index_mapping is not None:
        from .checkpoint_tensor_loading import prepare_fqn_to_index_mapping_for_model

        prepare_fqn_to_index_mapping_for_model(model, fqn_to_index_mapping)


def post_process_after_weight_loading(
    model: Union["nn.Module", "PreTrainedModel"],
    buffer_dict,
    parameter_names_left: Optional[set[str]] = None,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    dtensor_to_cpu: bool = False,
):
    """
    shared logic after weight loading that handles buffer, missing weight keys and tied embedding weights.
    """
    parameter_names_left = parameter_names_left or set()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu)
    if parameter_names_left:
        logger.info_rank0(f"Find missing key(s) in state dict: {parameter_names_left}, initialize them.")
        for name in sorted(parameter_names_left):
            _init_parameter(model, name)

    # to_empty() leaves embeddings untied (except under FSDP2 swap-tensor);
    # re-tie only when the config asks for it. Nested multimodal layouts can
    # disable tying on either side (InternVL on inner, Qwen3VLMoe on outer with
    # inner silent), so AND both. Treat unset as True so a silent side does not
    # override an explicit True, but require at least one side to set the flag
    # -- if neither does, default to False (matches HF v5).
    text_config = (
        model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
    )
    if (
        (hasattr(model.config, "tie_word_embeddings") or hasattr(text_config, "tie_word_embeddings"))
        and getattr(model.config, "tie_word_embeddings", True)
        and getattr(text_config, "tie_word_embeddings", True)
    ):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
        except Exception as e:
            logger.info_rank0(f"Failed to tie embeddings: {e}")
            raise RuntimeError("Failed to tie input/output embeddings") from e


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
    if output_dir.startswith("hdfs://"):
        hdfs_dir = output_dir
        hdfs_upper_dir = output_dir.rstrip("/")
        hdfs_upper_dir = hdfs_upper_dir[: hdfs_upper_dir.rfind("/")]
        output_dir = get_cache_dir(output_dir)
    else:
        hdfs_dir = None

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

        if hdfs_dir is not None:
            copy(output_dir, hdfs_upper_dir)
            logger.info(f"Model weights uploaded to {hdfs_dir}.")


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    if output_dir.startswith("hdfs://"):
        hdfs_dir = output_dir
        hdfs_upper_dir = output_dir.rstrip("/")
        hdfs_upper_dir = hdfs_upper_dir[: hdfs_upper_dir.rfind("/")]
        output_dir = get_cache_dir(output_dir)
    else:
        hdfs_dir = None

    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")

    if hdfs_dir is not None:
        copy(output_dir, hdfs_upper_dir)
        logger.info(f"Model config and tokenizer uploaded to {hdfs_dir}.")


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)
