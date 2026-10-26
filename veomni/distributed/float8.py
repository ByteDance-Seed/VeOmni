# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Native tensorwise float8 compute with unchanged FP32 parameter state."""

import re
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn as nn

from ..arguments import Float8TrainingConfig, MixedPrecisionConfig
from ..utils import device, logging
from .utils import check_fqn_match


logger = logging.get_logger(__name__)


def validate_float8_training(
    precision: MixedPrecisionConfig,
    parallel_state,
    *,
    cpu_offload: bool = False,
    compile_enabled: bool = False,
    is_peft_model: bool = False,
) -> None:
    """Reject unsupported runtime combinations before conversion or FSDP wrapping."""
    if not precision.float8.enable:
        return
    if not precision.enable or precision.param_dtype != "bfloat16" or precision.reduce_dtype != "float32":
        raise ValueError("float8 requires enabled mixed precision with bfloat16 parameters and float32 reductions.")
    if parallel_state.dp_mode != "fsdp2":
        raise ValueError("float8 training requires FSDP2.")
    if parallel_state.sp_enabled:
        raise ValueError("float8 training does not yet support sequence/context parallelism.")
    if parallel_state.any_extra_parallel_enabled:
        raise ValueError("float8 training does not yet support ExtraParallel.")
    if parallel_state.tp_size != 1:
        raise ValueError("float8 training does not yet support tensor parallelism.")
    if parallel_state.pp_size != 1:
        raise ValueError("float8 training does not yet support pipeline parallelism.")
    if parallel_state.dp_replicate_size != 1:
        raise ValueError("float8 training does not yet support HSDP replication.")
    if cpu_offload:
        raise ValueError("float8 training does not yet support FSDP CPU offload.")
    if compile_enabled:
        raise ValueError("float8 training with torch.compile is not yet validated.")
    if is_peft_model:
        raise ValueError("float8 training does not yet support LoRA/PEFT.")
    if not device.IS_CUDA_AVAILABLE or torch.version.hip or device.get_gpu_compute_capability() < 89:
        raise ValueError("float8 training requires an NVIDIA CUDA SM89 or newer device.")


def convert_float8_linears(model: nn.Module, config: Float8TrainingConfig) -> list[str]:
    """Convert only matched plain Linear modules, retaining weight/bias identity and keys.

    Call before FSDP wraps the model, including when parameters are on meta.
    Tensorwise dynamic scales are derived during every operation, so no scales
    or other quantization buffers are added to the checkpoint.
    """
    if not config.enable:
        return []
    if getattr(getattr(model, "config", None), "model_type", None) not in ("qwen3", "llama"):
        raise ValueError("float8 training currently supports dense Qwen3 and Llama models only.")

    aliases = Counter(id(param) for _, param in model.named_parameters(remove_duplicate=False))
    selected = []
    matched_patterns = set()
    for name, module in model.named_modules(remove_duplicate=False):
        matches = {pattern for pattern in config.target_modules if check_fqn_match(pattern, name)}
        if not matches:
            continue
        matched_patterns.update(matches)
        if type(module) is not nn.Linear:
            raise ValueError(
                f"float8 target {name!r} must be a plain torch.nn.Linear; subclasses/LoRA are unsupported."
            )
        if (
            re.fullmatch(r"model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)", name)
            is None
        ):
            raise ValueError(
                f"float8 target {name!r} is not a supported decoder MLP/attention projection. "
                "The LM head is unsupported because training loss reads its weight directly."
            )
        if module.in_features % 16 or module.out_features % 16:
            raise ValueError(f"float8 target {name!r} requires in_features and out_features divisible by 16.")
        if any(aliases[id(param)] > 1 for param in module.parameters(recurse=False)):
            raise ValueError(f"float8 target {name!r} has shared parameters or module aliases.")
        if any(param.dtype != torch.float32 for param in module.parameters(recurse=False)):
            raise ValueError(f"float8 target {name!r} requires FP32 master parameters before FSDP wrapping.")
        selected.append(name)
    missing = sorted(set(config.target_modules) - matched_patterns)
    if missing:
        raise ValueError(f"float8 target_modules matched no module for: {missing}.")

    try:
        installed_version = version("torchao")
    except PackageNotFoundError as exc:
        raise ImportError(
            "float8 training requires the GPU dependency torchao==0.18.0: uv sync --extra gpu --dev"
        ) from exc
    if installed_version != "0.18.0":
        raise ImportError(f"float8 training requires torchao==0.18.0; found {installed_version}.")

    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    # Weight-gradient GEMMs contract over flattened token rows, which may not
    # be divisible by 16 for packed/ragged batches. Native zero padding keeps
    # those batches valid without changing the returned shapes or using emulation.
    recipe = replace(Float8LinearConfig.from_recipe_name("tensorwise"), pad_inner_dim=True)
    # Keep FSDP parameter AllGather and the master parameters in their existing
    # dtypes; this feature changes only the Linear forward/backward GEMMs.
    assert not recipe.enable_fsdp_float8_all_gather and not recipe.emulate
    selected_names = set(selected)
    convert_to_float8_training(model, config=recipe, module_filter_fn=lambda module, name: name in selected_names)
    logger.info_rank0(f"Enabled tensorwise float8 compute for {len(selected)} Linear modules: {selected}")
    return selected
