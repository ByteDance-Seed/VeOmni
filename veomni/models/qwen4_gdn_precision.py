"""Precision policy for the two Qwen4-Exp Gated DeltaNet state parameters.

These parameters are small but sensitive.  They must be promoted on the meta
model *before* loading a checkpoint, not after its values have been rounded.
"""

import re

import torch
from torch import nn


_GDN_CLASS_NAME = "Qwen4ExpTextGatedDeltaNet"
_GDN_PARAMETER_NAMES = ("A_log", "dt_bias")
_GDN_FQN = re.compile(r"(?:^|\.)linear_attn\.(?:A_log|dt_bias)$")


def is_qwen4_exp_config(config) -> bool:
    return getattr(config, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}


def expected_qwen4_gdn_layer_ids(config):
    """Return configured GDN layer ids, or None when coverage is unspecified."""
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        return None
    return {index for index, kind in enumerate(layer_types) if kind == "linear_attention"}


def qwen4_exp_model_config(model):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    config = getattr(model, "config", None)
    return config if is_qwen4_exp_config(config) else None


def is_qwen4_exp_model(model) -> bool:
    return qwen4_exp_model_config(model) is not None


def is_qwen4_gdn_parameter(name: str) -> bool:
    return bool(_GDN_FQN.search(name))


def validate_qwen4_gdn_pairs(names, config=None) -> None:
    """A full model export must not omit either member of a GDN pair."""
    groups = {}
    for name in names:
        if is_qwen4_gdn_parameter(name):
            prefix, _, leaf = name.rpartition(".")
            groups.setdefault(prefix, set()).add(leaf)
    expected = expected_qwen4_gdn_layer_ids(config)
    if not groups and expected == set():
        return
    if groups and expected == set():
        raise RuntimeError("Qwen4 config declares no GDN layers but GDN parameters are present")
    if not groups or any(leaves != set(_GDN_PARAMETER_NAMES) for leaves in groups.values()):
        raise RuntimeError("Qwen4 full export requires complete A_log/dt_bias pairs for every GDN layer")
    if expected is not None:
        found = set()
        for prefix in groups:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn$", prefix)
            if match is not None:
                found.add(int(match.group(1)))
        if expected != found:
            raise RuntimeError(
                f"Qwen4 GDN layer coverage differs from config: expected={sorted(expected)}, found={sorted(found)}"
            )


def qwen4_gdn_modules(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if type(m).__name__ == _GDN_CLASS_NAME]


def qwen4_gdn_parameters(module: nn.Module) -> set[nn.Parameter]:
    params = set()
    for name in _GDN_PARAMETER_NAMES:
        parameter = module._parameters.get(name)
        if parameter is None or parameter.dtype != torch.float32:
            raise RuntimeError(f"Qwen4 GDN {name} must be a direct FP32 parameter before FSDP wrapping")
        params.add(parameter)
    return params


def prepare_qwen4_gdn_precision(model: nn.Module, *, initialize_materialized: bool = False) -> None:
    """Prepare FP32 targets before weight load, preserving Parameter identity.

    For a scratch model built directly on a device, the original BF16 random
    initialization cannot be recovered. Reinitialize the two parameters using
    the Qwen4 scratch distributions in FP32 before any training begins.
    """
    if not is_qwen4_exp_config(getattr(model, "config", None)):
        return
    modules = qwen4_gdn_modules(model)
    if not modules:
        if expected_qwen4_gdn_layer_ids(model.config) == set():
            return
        raise RuntimeError("Qwen4-Exp model has no recognizable GDN modules")
    if expected_qwen4_gdn_layer_ids(model.config) == set():
        raise RuntimeError("Qwen4-Exp config declares no GDN layers but GDN modules are present")
    for module in modules:
        for name in _GDN_PARAMETER_NAMES:
            parameter = module._parameters.get(name)
            if parameter is None:
                raise RuntimeError(f"Qwen4 GDN is missing parameter {name}")
            if parameter.dtype == torch.float32:
                continue
            if parameter.device.type != "meta":
                if initialize_materialized:
                    value = torch.empty(parameter.shape, dtype=torch.float32, device=parameter.device)
                    if name == "dt_bias":
                        value.fill_(1)
                    else:
                        value.uniform_(0.01, 16).log_()
                    parameter.data = value
                    continue
                raise RuntimeError(
                    f"Cannot promote materialized Qwen4 GDN {name}: BF16 rounding has already occurred. "
                    "Construct on meta and load the original FP32 checkpoint instead."
                )
            parameter.data = parameter.data.to(dtype=torch.float32)


def preserve_qwen4_gdn_fp32(model_assets) -> bool:
    return has_qwen4_exp_assets(model_assets)


def has_qwen4_exp_assets(model_assets) -> bool:
    return bool(model_assets and any(is_qwen4_exp_config(asset) for asset in model_assets))
