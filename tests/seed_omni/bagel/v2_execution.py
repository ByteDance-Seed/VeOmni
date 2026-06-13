"""BAGEL V2 graph/module execution helpers for parity cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.bagel.transformers.bagel import make_reference_image
from tests.seed_omni.parity_suite.core import to_device
from tests.seed_omni.parity_suite.v2.model import (
    ModuleNode,
    run_module_nodes,
)
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modeling_omni import OmniModel


BAGEL_ORACLE_SOURCE = "bagel_official_oracle"
BAGEL_MODULE_NAMES = (
    "bagel_text_encoder",
    "bagel_qwen2_mot",
    "bagel_flow_connector",
    "bagel_siglip_navit",
    "bagel_vae",
)
IMAGE_EDIT_CONTEXT_NODES: tuple[ModuleNode, ...] = (
    ("bagel_vae", "encode"),
    ("bagel_siglip_navit", "generate"),
)
PROMPT_ENCODE_NODES: tuple[ModuleNode, ...] = (("bagel_text_encoder", "prompt_encode"),)
TEXT_UND_GENERATE_NODES: tuple[ModuleNode, ...] = (("bagel_qwen2_mot", "generate"),)
TEXT_UND_TOKEN_NODES: tuple[ModuleNode, ...] = (("bagel_text_encoder", "token_generate"),)
IMAGE_EDIT_LATENT_PROMPT_NODES: tuple[ModuleNode, ...] = (
    ("bagel_flow_connector", "embed_latent"),
    ("bagel_qwen2_mot", "generate"),
    ("bagel_text_encoder", "prompt_encode"),
)
IMAGE_FLOW_NODES: tuple[ModuleNode, ...] = (
    ("bagel_flow_connector", "embed_latent"),
    ("bagel_qwen2_mot", "generate"),
    ("bagel_flow_connector", "decode_velocity"),
)


def run_v2_infer_module(
    model: OmniModel,
    driver_case: str,
    reference_output: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    generation_kwargs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "conversation_list": build_conversation_from_canonical(reference_output["canonical"], device=device)
    }
    observations: dict[tuple[str, str], list[dict[str, Any]]] = {}

    bagel_modules = {name: model.get_module(name) for name in BAGEL_MODULE_NAMES}

    if driver_case == "image_edit":
        run_module_nodes(
            IMAGE_EDIT_CONTEXT_NODES,
            modules=bagel_modules,
            ctx=ctx,
            observations=observations,
            whitelist=whitelist,
            state="prompt_encode",
            generation_kwargs=generation_kwargs,
        )

    run_module_nodes(
        PROMPT_ENCODE_NODES,
        modules=bagel_modules,
        ctx=ctx,
        observations=observations,
        whitelist=whitelist,
        state="prompt_encode",
        generation_kwargs=generation_kwargs,
    )

    if driver_case == "image_edit":
        run_module_nodes(
            IMAGE_EDIT_LATENT_PROMPT_NODES,
            modules=bagel_modules,
            ctx=ctx,
            observations=observations,
            whitelist=whitelist,
            state="prompt_encode",
            generation_kwargs=generation_kwargs,
        )
        run_module_nodes(
            IMAGE_FLOW_NODES,
            modules=bagel_modules,
            ctx=ctx,
            observations=observations,
            whitelist=whitelist,
            state="image_flow",
            generation_kwargs=generation_kwargs,
        )
        return {"observations": observations, "ctx": ctx, "trace": ["module:prompt_encode", "module:image_flow"]}

    run_module_nodes(
        TEXT_UND_GENERATE_NODES,
        modules=bagel_modules,
        ctx=ctx,
        observations=observations,
        whitelist=whitelist,
        state="prompt_encode",
        generation_kwargs=generation_kwargs,
    )

    if driver_case == "image_gen":
        run_module_nodes(
            IMAGE_FLOW_NODES,
            modules=bagel_modules,
            ctx=ctx,
            observations=observations,
            whitelist=whitelist,
            state="image_flow",
            generation_kwargs=generation_kwargs,
        )
        return {"observations": observations, "ctx": ctx, "trace": ["module:prompt_encode", "module:image_flow"]}

    run_module_nodes(
        TEXT_UND_TOKEN_NODES,
        modules=bagel_modules,
        ctx=ctx,
        observations=observations,
        whitelist=whitelist,
        state="prompt_encode",
        generation_kwargs=generation_kwargs,
    )
    return {"observations": observations, "ctx": ctx, "trace": ["module:prompt_encode"]}


def build_conversation_from_canonical(canonical: Mapping[str, Any], *, device: torch.device) -> list[ConversationItem]:
    if canonical.get("kind") in {"image_gen", "image_edit"}:
        return _build_image_conversation_from_canonical(canonical, device=device)
    return _build_text_conversation_from_canonical(canonical, device=device)


def _build_text_conversation_from_canonical(
    canonical: Mapping[str, Any], *, device: torch.device
) -> list[ConversationItem]:
    return [_build_text_und_item(canonical, device=device)]


def _build_text_und_item(canonical: Mapping[str, Any], *, device: torch.device) -> ConversationItem:
    prompt = to_device(canonical["prompt_input"], device)
    start = to_device(canonical["start_input"], device)
    return ConversationItem(
        type="text",
        value=prompt["packed_text_ids"].detach().to(dtype=torch.long),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "text",
            "raw_text": canonical["prompt"],
            "position_ids": prompt["packed_text_position_ids"].detach().to(dtype=torch.long),
            "sequence_indexes": prompt["packed_text_indexes"].detach().to(dtype=torch.long),
            "context_indexes": prompt["packed_key_value_indexes"].detach().to(dtype=torch.long),
            "token_lens": prompt["text_token_lens"].detach().to(dtype=torch.int32),
            "key_value_lens_before": prompt["key_values_lens"].detach().to(dtype=torch.int32),
            "key_value_lens_after": canonical["kv_lens_after_prompt"],
            "rope_after": canonical["ropes_after_prompt"],
            "next_token": {
                "input_ids": start["packed_start_tokens"].detach().to(dtype=torch.long),
                "query_lens": canonical["query_lens"].to(device=device, dtype=torch.int32),
                "position_ids": start["packed_query_position_ids"].detach().to(dtype=torch.long),
                "query_indexes": canonical["packed_query_indexes"].to(device=device, dtype=torch.long),
                "key_value_lens": start["key_values_lens"].detach().to(dtype=torch.int32),
                "context_indexes": canonical["packed_key_value_indexes_for_step"].to(device=device, dtype=torch.long),
            },
        },
    )


def _build_image_conversation_from_canonical(
    canonical: Mapping[str, Any], *, device: torch.device
) -> list[ConversationItem]:
    latent_item = _build_image_latent_item(canonical, device=device)
    text_item = _build_image_prompt_item(canonical, device=device)
    if canonical.get("kind") == "image_edit":
        return [_build_image_vae_input_item(canonical), latent_item, text_item]
    return [latent_item, text_item]


def _build_image_latent_item(canonical: Mapping[str, Any], *, device: torch.device) -> ConversationItem:
    latent = to_device(canonical["latent_input"], device)
    timesteps = to_device(canonical["timesteps"], device)
    return ConversationItem(
        type="image",
        value=latent["packed_init_noises"].detach().to(dtype=torch.float32),
        role="assistant",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_gen_latent",
            "raw_image_size": [canonical["image_height"], canonical["image_width"]],
            "text_token_ids": latent["packed_text_ids"].detach().to(dtype=torch.long),
            "text_indexes": latent["packed_text_indexes"].detach().to(dtype=torch.long),
            "vae_token_indexes": latent["packed_vae_token_indexes"].detach().to(dtype=torch.long),
            "vae_position_ids": latent["packed_vae_position_ids"].detach().to(dtype=torch.long),
            "query_lens": latent["packed_seqlens"].detach().to(dtype=torch.int32),
            "position_ids": latent["packed_position_ids"].detach().to(dtype=torch.long),
            "sequence_indexes": latent["packed_indexes"].detach().to(dtype=torch.long),
            "key_value_lens": latent["key_values_lens"].detach().to(dtype=torch.int32),
            "context_indexes": latent["packed_key_value_indexes"].detach().to(dtype=torch.long),
            "rope_after_prompt": canonical["ropes_after_prompt"],
            "timesteps": timesteps["timesteps"].to(device=device, dtype=torch.float32),
            "dts": timesteps["dts"].to(device=device, dtype=torch.float32),
            "max_flow_steps": int(canonical.get("max_flow_steps", 1)),
            "cfg_text_scale": 1.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.0, 1.0],
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
        },
    )


def _build_image_prompt_item(canonical: Mapping[str, Any], *, device: torch.device) -> ConversationItem:
    prompt = to_device(canonical["prompt_input"], device)
    return ConversationItem(
        type="text",
        value=prompt["packed_text_ids"].detach().to(dtype=torch.long),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "text",
            "raw_text": canonical["prompt"],
            "image_generation_prompt": canonical.get("kind") == "image_edit",
            "position_ids": prompt["packed_text_position_ids"].detach().to(dtype=torch.long),
            "sequence_indexes": prompt["packed_text_indexes"].detach().to(dtype=torch.long),
            "context_indexes": prompt["packed_key_value_indexes"].detach().to(dtype=torch.long),
            "token_lens": prompt["text_token_lens"].detach().to(dtype=torch.int32),
            "key_value_lens_before": prompt["key_values_lens"].detach().to(dtype=torch.int32),
            "key_value_lens_after": canonical["kv_lens_after_prompt"],
            "rope_after": canonical["ropes_after_prompt"],
        },
    )


def _build_image_vae_input_item(canonical: Mapping[str, Any]) -> ConversationItem:
    input_width = int(canonical["input_width"])
    input_height = int(canonical["input_height"])
    return ConversationItem(
        type="image",
        value=make_reference_image(input_width, input_height),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_vae_input",
            "enable_vae_context": True,
            "raw_image_size": [input_width, input_height],
        },
    )


__all__ = [
    "build_conversation_from_canonical",
    "run_v2_infer_module",
]
