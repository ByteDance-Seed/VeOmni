"""Utility functions for FSDP2 integration tests."""

import os
from typing import Literal, Optional, Union

import torch
from transformers import PreTrainedModel

from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.loader import MODELING_REGISTRY
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type


# =============================================================================
# Patch Management Functions (testing only)
# =============================================================================


def unapply_veomni_loss_patch() -> None:
    """Unpatch VeOmni loss implementation and restore HuggingFace defaults."""
    from transformers.loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss

    from veomni.ops import fused_cross_entropy

    fused_cross_entropy._cross_entropy = None

    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    LOSS_MAPPING["ForConditionalGeneration"] = ForCausalLMLoss


def unapply_veomni_attention_patch() -> None:
    """Unpatch VeOmni attention implementation and restore HuggingFace defaults."""
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from veomni.ops import flash_attn

    flash_attn._flash_attention_forward = None

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_3", flash_attention_forward)


def unapply_all_veomni_patches() -> None:
    """Unpatch all VeOmni ops patches to restore HuggingFace defaults."""
    unapply_veomni_loss_patch()
    unapply_veomni_attention_patch()


def verify_model_backend(
    model: PreTrainedModel,
    expected_backend: Literal["hf", "veomni"],
) -> None:
    """Verify that the model class matches the expected backend.

    Args:
        model: The model to check.
        expected_backend: The expected backend ("hf" or "veomni").

    Raises:
        AssertionError: If the model backend doesn't match the expected backend.
    """
    model_type = getattr(model.config, "model_type", None)
    modeling_backend_env = os.environ.get("MODELING_BACKEND", "")
    is_registered_in_veomni = model_type in MODELING_REGISTRY.valid_keys()

    if expected_backend == "hf":
        assert modeling_backend_env == "hf", (
            f"Expected MODELING_BACKEND='hf' for HuggingFace backend, "
            f"but got MODELING_BACKEND='{modeling_backend_env}'."
        )
    else:
        assert modeling_backend_env == "veomni", (
            f"Expected MODELING_BACKEND='veomni' for VeOmni backend, but got MODELING_BACKEND='{modeling_backend_env}'."
        )
        assert is_registered_in_veomni, (
            f"Model type '{model_type}' is not registered in VeOmni's MODELING_REGISTRY. "
            f"Registered model types: {list(MODELING_REGISTRY.valid_keys())}"
        )


# Fixed training constants
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0


def build_fsdp_model_optim(
    config_path: str,
    weights_path: Optional[str] = None,
    attn_implementation: str = "flash_attention_2",
    dp_size: int = 1,
    ulysses_sp_size: int = 1,
    torch_dtype: str = "float32",
    enable_mixed_precision: bool = True,
):
    """Build FSDP2-wrapped VeOmni model and optimizer with optional Ulysses SP.

    Args:
        config_path: Path to model configuration.
        weights_path: Path to pretrained weights.
        attn_implementation: Attention implementation type.
        dp_size: Data parallel size.
        ulysses_sp_size: Ulysses sequence parallel size.
        torch_dtype: Data type for model parameters.
        enable_mixed_precision: Whether to enable mixed precision training.
    """
    os.environ["MODELING_BACKEND"] = "veomni"

    init_parallel_state(
        dp_size=dp_size,
        ulysses_size=ulysses_sp_size,
        dp_mode="fsdp2",
        device_type=get_device_type(),
    )

    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        init_device="meta",
    )

    # Verify that the model is from VeOmni, not HuggingFace.
    verify_model_backend(model, expected_backend="veomni")

    model = build_parallelize_model(
        model,
        weights_path=weights_path,
        enable_full_shard=True,
        enable_mixed_precision=enable_mixed_precision,
        init_device="meta",
        fsdp_kwargs={},
        basic_modules=getattr(model, "_no_split_modules", []),
        enable_reentrant=False,
        enable_forward_prefetch=False,
        enable_gradient_checkpointing=True,
    )

    optimizer = build_optimizer(
        model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,
        optimizer_type="adamw",
        no_decay_modules=[],
        no_decay_params=[],
    )

    return model, optimizer


def mean_global_loss_baseline(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    micro_batch_token_len: dict[str, torch.Tensor],
    micro_batches_token_len: dict[str, torch.Tensor],
):
    """Calculate global mean loss for single-GPU baseline (non-distributed).

    Averages loss by token count: loss = loss * cur_tokens / total_tokens
    """
    loss_dict = {}

    if isinstance(losses, torch.Tensor):
        losses = {"foundation_loss": losses}

    first_loss = next(iter(losses.values()))
    loss_bwd = torch.tensor(0.0, device=first_loss.device)

    for key, cur_loss in losses.items():
        loss_name = key.split("_loss")[0]
        cur_token_len = micro_batch_token_len[f"{loss_name}_tokens"]
        all_reduced_len = micro_batches_token_len[f"{loss_name}_tokens"].item()

        if all_reduced_len != 0:
            cur_loss = cur_loss * cur_token_len / all_reduced_len
        else:
            if not torch.allclose(cur_loss, torch.zeros_like(cur_loss)):
                raise ValueError(f"all_reduced_len for {loss_name}_tokens is 0, but cur_loss is not 0: {cur_loss}")

        loss_bwd += cur_loss
        loss_dict[key] = cur_loss.item()

    return loss_bwd, loss_dict
