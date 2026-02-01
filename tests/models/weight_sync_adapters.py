"""
Temporary weight-sync adapters for models where HF and VeOmni state dict layouts differ.
These will be removed in a future version when layouts are aligned; use only in tests.
"""

from typing import Callable, Union

import torch


# Registry for model-specific sync functions. Add/remove entries when adding new models
# or when a model no longer needs custom sync (e.g. layout aligned).
SYNC_WEIGHT_REGISTRY: dict[str, Callable] = {}


def get_sync_weight_func(model_key: str) -> Union[Callable, None]:
    """Return the sync weight function for a model key, or None if not needed."""
    return SYNC_WEIGHT_REGISTRY.get(model_key, None)


def sync_weight_qwen3moe(config, state_dict_source, veomni_model):
    """
    Align HF state dict to VeOmni Qwen3MoE layout (experts stacked per module).
    Temporary adapter; will be removed when HF/VeOmni layouts are aligned.
    """
    layer_num = config.num_hidden_layers
    expert_num = config.num_experts

    hf_model_state_dict = state_dict_source
    veomni_model_state_dict = veomni_model.state_dict()
    # copy weights
    for i in hf_model_state_dict.keys():
        if i in veomni_model_state_dict.keys():
            veomni_model_state_dict[i] = hf_model_state_dict[i]

    # Align experts between HF and VeOmni:
    # VeOmni: experts.{module_name} stacked [num_experts, ...]
    # HF: experts.{expert_id}.{module_name} per expert
    for layer_id in range(layer_num):
        for module_name in ["gate_proj", "up_proj", "down_proj"]:
            expert_weights = []
            for expert_id in range(expert_num):
                key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{module_name}.weight"
                expert_weights.append(hf_model_state_dict[key])

            veomni_module_name = f"model.layers.{layer_id}.mlp.experts.{module_name}"
            veomni_model_state_dict[veomni_module_name] = torch.stack(expert_weights, dim=0)

    veomni_model.load_state_dict(veomni_model_state_dict)

    for i in hf_model_state_dict.keys():
        if i in veomni_model_state_dict.keys():
            try:
                assert veomni_model_state_dict[i].equal(hf_model_state_dict[i])
            except AssertionError:
                raise AssertionError(f"tensor is not the same after init. key={i}")
    return veomni_model


def sync_weight_deepseek_v3(config, state_dict_source, veomni_model):
    """
    Align HF state dict to VeOmni DeepseekV3 layout (experts stacked per module).
    Temporary adapter; will be removed when HF/VeOmni layouts are aligned.
    """
    layer_num = config.num_hidden_layers
    expert_num = config.n_routed_experts
    first_k_dense_replace = config.first_k_dense_replace

    hf_model_state_dict = state_dict_source
    veomni_model_state_dict = veomni_model.state_dict()
    # copy weights
    for i in hf_model_state_dict.keys():
        if i in veomni_model_state_dict.keys():
            veomni_model_state_dict[i] = hf_model_state_dict[i]

    # Align experts between HF and VeOmni:
    # VeOmni: experts.{module_name} stacked [num_experts, ...]
    # HF: experts.{expert_id}.{module_name} per expert
    for layer_id in range(first_k_dense_replace, layer_num):
        for module_name in ["gate_proj", "up_proj", "down_proj"]:
            expert_weights = []
            for expert_id in range(expert_num):
                key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{module_name}.weight"
                expert_weights.append(hf_model_state_dict[key])

            veomni_module_name = f"model.layers.{layer_id}.mlp.experts.{module_name}"
            veomni_model_state_dict[veomni_module_name] = torch.stack(expert_weights, dim=0)

    veomni_model.load_state_dict(veomni_model_state_dict)

    for i in hf_model_state_dict.keys():
        if i in veomni_model_state_dict.keys():
            try:
                assert veomni_model_state_dict[i].equal(hf_model_state_dict[i])
            except AssertionError:
                raise AssertionError(f"tensor is not the same after init. key={i}")
    return veomni_model


# Register adapters (remove entry when adapter is no longer needed)
SYNC_WEIGHT_REGISTRY["qwen3_moe"] = sync_weight_qwen3moe
SYNC_WEIGHT_REGISTRY["deepseek_v3"] = sync_weight_deepseek_v3
