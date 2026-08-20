"""Shared helpers for BAGEL tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from veomni.arguments import (
    OmniArguments,
    OmniDataArguments,
    OmniModelRuntimeArguments,
    build_module_runtime_args,
    build_omni_model_runtime,
)
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import PackedConversation, preprocess_mot_inputs
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem


ToyCase = Literal["ce_only", "vit_ce", "mse_only", "mixed"]

ALIGN_ATOL = 2e-2
ALIGN_RTOL = 2e-2
ALIGN_GRAD_ATOL = 5e-2
ALIGN_GRAD_RTOL = 5e-2


def bagel_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "Bagel" / "bagel_7b_mot"


def config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def model_cls(model_type: str):
    """Return the VeOmni-accelerated module class (training / graph hooks)."""
    return OMNI_ACCELERATED_MODEL_REGISTRY[model_type]()


def native_model_cls(model_type: str):
    """Return the HF-native module class (eager inference weights)."""
    return OMNI_MODEL_REGISTRY[model_type]()


def load_module_runtime_args(
    *,
    model_path: str = "",
    modules_path: Path,
    for_inference: bool = False,
) -> dict:
    base = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path or ".",
            model_config={"modules": str(modules_path)},
        ),
        data=OmniDataArguments(train_path=""),
    )._to_module_global_args()
    return build_module_runtime_args(
        global_args=base,
        model_path=model_path,
        modules=str(modules_path),
        for_inference=for_inference,
    )


def load_omni_config(
    *,
    model_path: str = "",
    modules_path: Path,
    train_graph_path: Path | None = None,
    infer_graph_path: Path | None = None,
    generation_kwargs: dict | None = None,
) -> OmniConfig:
    model_path = model_path or "."
    model_config = {"modules": str(modules_path)}
    if train_graph_path is not None:
        model_config["train_graph"] = str(train_graph_path)
    base = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config=model_config,
        ),
        data=OmniDataArguments(train_path="."),
    )._to_module_global_args()
    return build_omni_model_runtime(
        global_args=base,
        model_path=model_path,
        train_modules=str(modules_path),
        train_graph=str(train_graph_path) if train_graph_path else None,
        infer_graph=str(infer_graph_path) if infer_graph_path else None,
        generation_kwargs=generation_kwargs,
    ).to_hf_config()


def tiny_bagel_qwen2_cfg() -> dict:
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )


def tiny_align_qwen2_cfg() -> dict:
    config = tiny_bagel_qwen2_cfg()
    config["num_key_value_heads"] = 2
    return config


def _clone_meta(meta: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in meta.items():
        cloned[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return cloned


def clone_conversation(conversation: list[list[ConversationItem]]) -> list[list[ConversationItem]]:
    """Deep-copy a packed conversation so eager and accelerated can mutate independently."""
    cloned: list[list[ConversationItem]] = []
    for sample in conversation:
        cloned_sample: list[ConversationItem] = []
        for item in sample:
            value = item.value
            if torch.is_tensor(value):
                requires_grad = value.requires_grad
                value = value.detach().clone()
                if requires_grad:
                    value = value.requires_grad_()
            cloned_sample.append(
                ConversationItem(
                    type=item.type,
                    value=value,
                    role=item.role,
                    source=item.source,
                    meta=_clone_meta(item.meta),
                )
            )
        cloned.append(cloned_sample)
    return cloned


def _embed(
    generator: torch.Generator,
    length: int,
    hidden_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> torch.Tensor:
    value = torch.randn(length, hidden_size, generator=generator, device=device, dtype=dtype)
    if requires_grad:
        value = value.requires_grad_()
    return value


def build_toy_conversation(
    case: ToyCase,
    *,
    hidden_size: int,
    vocab_size: int,
    patch_latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 17,
) -> list[list[ConversationItem]]:
    """Build one packed sample covering CE-only, vit-CE, MSE-only, or mixed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    items: list[ConversationItem] = []

    if case in {"vit_ce", "mixed"}:
        items.append(
            ConversationItem(
                type="image",
                value=_embed(generator, 4, hidden_size, device=device, dtype=dtype),
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
                meta={_IMG_TAG_KEY: "und"},
            )
        )

    if case in {"ce_only", "vit_ce", "mixed"}:
        labels = torch.randint(3, vocab_size, (8,), generator=generator, device=device)
        items.append(
            ConversationItem(
                type="text",
                value=_embed(generator, 8, hidden_size, device=device, dtype=dtype),
                role="assistant",
                meta={"labels": labels},
            )
        )

    if case in {"mse_only", "mixed"}:
        items.append(
            ConversationItem(
                type="image",
                value=_embed(generator, 6, hidden_size, device=device, dtype=dtype),
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={
                    _IMG_TAG_KEY: "gen",
                    "flow_velocity_target": torch.randn(
                        6,
                        patch_latent_dim,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    ),
                },
            )
        )

    return [items]


def scatter_mot_hidden(
    conversation: list[list[ConversationItem]],
    packed: PackedConversation,
    hidden_states: torch.Tensor,
) -> None:
    """Write packed MoT hidden states back onto the original carrier items."""
    for span in packed.spans:
        span_hidden = hidden_states[span.start : span.start + span.length]
        offset = 0
        for item, length in zip(span.items, span.lengths, strict=True):
            item.value = span_hidden[offset : offset + length]
            offset += length


def run_eager_mot(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
) -> torch.Tensor:
    packed = preprocess_mot_inputs(
        conversation,
        device=model.device,
        dtype=model.dtype,
        hidden_size=int(model.config.hidden_size),
    )
    if packed is None:
        raise ValueError("toy conversation produced no packable tokens")
    outputs = model(
        packed_sequence=packed.packed_sequence,
        packed_position_ids=packed.packed_position_ids,
        packed_token_type_ids=packed.packed_token_type_ids,
        packed_attention_metadata=packed.packed_attention_metadata,
    )
    hidden_states = outputs["hidden_states"]
    scatter_mot_hidden(conversation, packed, hidden_states)
    return hidden_states


def run_accelerated_mot(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
) -> torch.Tensor:
    inputs = model.forward_pre(conversation_list=conversation)
    outputs = model(**inputs)
    hidden_states = outputs["hidden_states"]
    model.forward_post(**outputs)
    return hidden_states


def conversation_ce_loss(
    conversation: list[list[ConversationItem]],
    lm_head: torch.nn.Linear,
) -> torch.Tensor | None:
    hidden_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for item in (item for sample in conversation for item in sample):
        labels = item.meta.get("labels")
        if item.type != "text" or not torch.is_tensor(item.value) or not torch.is_tensor(labels):
            continue
        hidden = item.value.squeeze(0) if item.value.dim() == 3 else item.value
        hidden_parts.append(hidden)
        label_parts.append(labels.to(device=hidden.device))
    if not hidden_parts:
        return None
    hidden = torch.cat(hidden_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    shift_labels = F.pad(labels[..., 1:].contiguous(), (0, 1), value=-100)
    logits = lm_head(hidden.float())
    return F.cross_entropy(logits, shift_labels, ignore_index=-100)


def conversation_mse_loss(
    conversation: list[list[ConversationItem]],
    llm2vae: torch.nn.Linear,
) -> torch.Tensor | None:
    velocity_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for item in (item for sample in conversation for item in sample):
        target = item.meta.get("flow_velocity_target")
        if not torch.is_tensor(item.value) or not torch.is_tensor(target):
            continue
        hidden = item.value.squeeze(0) if item.value.dim() == 3 else item.value
        velocity_parts.append(llm2vae(hidden.float()))
        target_parts.append(target.to(device=hidden.device, dtype=torch.float32))
    if not velocity_parts:
        return None
    mse = (torch.cat(velocity_parts, dim=0) - torch.cat(target_parts, dim=0)).square()
    return mse.mean(dim=-1).mean()
