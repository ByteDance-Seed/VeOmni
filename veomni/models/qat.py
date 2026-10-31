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

"""Dense Qwen3 MXFP4 QAT preparation and distributed validation.

Training keeps ordinary parameters and state-dict keys, with quantization
derived on each forward.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from importlib.metadata import version

import torch
from torch import nn


_RECIPE = "torchao_mxfp4"
_TORCHAO_VERSION = "0.18.0"
_PROJECTION = re.compile(r"model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)")


def _nested_mxfp4(config):
    for name in getattr(config, "sub_configs", {}):
        child = getattr(config, name, None)
        selections = getattr(child, "ops_implementation", None) or {}
        if selections.get("qat_implementation") == _RECIPE or _nested_mxfp4(child):
            return True
    return False


def _qat_ops(model):
    config = getattr(model, "config", None)
    if _nested_mxfp4(config):
        raise ValueError("torchao_mxfp4 currently supports only standalone Dense Qwen3, not nested model configs.")
    ops = getattr(config, "ops_config", None)
    # The HF backend restores native config classes, retaining their selections.
    if ops is None and (getattr(config, "ops_implementation", None) or {}).get("qat_implementation") == _RECIPE:
        from ..arguments.arguments_types import OpsImplementationConfig

        ops = OpsImplementationConfig(**config.ops_implementation)
    if getattr(ops, "qat_implementation", None) != _RECIPE:
        return None
    if config.model_type != "qwen3":
        raise ValueError("torchao_mxfp4 currently supports only standalone Dense Qwen3.")
    return ops


def _torchao_api():
    try:
        installed = version("torchao")
    except ModuleNotFoundError as exc:
        raise ImportError("torchao_mxfp4 requires torchao==0.18.0; install VeOmni's gpu extra.") from exc
    if installed != _TORCHAO_VERSION:
        raise ImportError(f"torchao_mxfp4 requires torchao=={_TORCHAO_VERSION}, got {installed}.")
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.prototype.qat.mx import MXFakeQuantizedLinear
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float4_e2m1fn_x2,
        weight_dtype=torch.float4_e2m1fn_x2,
        block_size=32,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED,
    )
    return quantize_, QATConfig, config, MXFakeQuantizedLinear


def _selected_projections(model, patterns, prepared_type):
    modules = dict(model.named_modules())
    selected = {}
    for pattern in patterns:
        matches = {name: module for name, module in modules.items() if fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"QAT target pattern {pattern!r} matched no modules.")
        selected.update(matches)
    parameter_names = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(param), []).append(name)
    for name, module in selected.items():
        if not _PROJECTION.fullmatch(name) or type(module) not in (nn.Linear, prepared_type):
            raise ValueError(f"Unsupported QAT target {name!r}; select plain Qwen3 decoder projections.")
        if module.bias is not None:
            raise ValueError(f"QAT target {name!r} has a bias; torchao MXFP4 QAT does not compute bias gradients.")
        if module.in_features % 32:
            raise ValueError(f"QAT target {name!r} needs in_features divisible by the MXFP4 block size (32).")
        if len(parameter_names[id(module.weight)]) != 1:
            raise ValueError(f"QAT target {name!r} shares a parameter; shared projection weights are unsupported.")
    return selected


def prepare_mxfp4_qat(model: nn.Module) -> nn.Module:
    """Replace selected Qwen3 projections before FSDP wrapping, including on meta.

    Ordinary models do not import torchao. Distributed training uses CUDA.
    """
    ops = _qat_ops(model)
    if ops is None:
        return model
    quantize, QATConfig, recipe, prepared_type = _torchao_api()
    selected = _selected_projections(model, ops.qat_target_modules, prepared_type)
    quantize(
        model,
        QATConfig(recipe, step="prepare"),
        filter_fn=lambda module, name: name in selected and type(module) is nn.Linear,
    )
    return model


def validate_mxfp4_qat_parallelism(model, mixed_precision, compile_config=None, **kwargs) -> None:
    """Check the initial CUDA FSDP2 recipe before mutating or sharding a model."""
    ops = _qat_ops(model)
    if ops is None:
        return

    _validate_mxfp4_config(mixed_precision, compile_config, **kwargs)
    _validate_mxfp4_device()
    _, _, _, prepared_type = _torchao_api()
    selected = _selected_projections(model, ops.qat_target_modules, prepared_type)
    if any(type(module) is not prepared_type for module in selected.values()):
        raise ValueError("Build the QAT model with build_foundation_model before applying FSDP2.")
    if any(module.weight.dtype != torch.float32 for module in selected.values()):
        raise ValueError(
            "torchao_mxfp4 requires FP32 master parameters; use build_parallelize_model or model.float()."
        )


def _validate_mxfp4_config(mixed_precision, compile_config=None, **kwargs) -> None:
    """Check topology, precision and runtime options for the initial QAT recipe."""
    from ..distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if ps.dp_mode != "fsdp2" or ps.sp_enabled or ps.tp_enabled or ps.pp_enabled or ps.any_extra_parallel_enabled:
        raise ValueError("torchao_mxfp4 supports FSDP2 without SP/CP, TP, PP or ExtraParallel.")
    if ps.dp_replicate_enabled:
        raise ValueError("torchao_mxfp4 HSDP has not been enabled; use a single sharding mesh.")
    if (
        not mixed_precision.enable
        or mixed_precision.param_dtype != "bfloat16"
        or mixed_precision.reduce_dtype != "float32"
        or not mixed_precision.cast_forward_inputs
    ):
        raise ValueError(
            "torchao_mxfp4 requires enabled BF16 parameter / FP32 reduction precision with input casting."
        )
    if kwargs.get("is_peft_model") or kwargs.get("enable_fsdp_offload") or getattr(compile_config, "enable", False):
        raise ValueError("torchao_mxfp4 does not yet support LoRA, FSDP CPU offload or torch.compile.")


def _validate_mxfp4_device() -> None:
    """Require native BF16 on the current rank's NVIDIA GPU."""
    from ..utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability

    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        raise ValueError("Distributed torchao_mxfp4 training requires NVIDIA CUDA.")
    if get_gpu_compute_capability() < 80:
        raise ValueError("Distributed torchao_mxfp4 training requires native BF16 support (SM80 or newer).")
