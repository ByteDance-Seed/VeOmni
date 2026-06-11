"""Test-only adapters for BAGEL official parity fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from veomni.models.seed_omni.conversation import ConversationItem


TEXT_FIXTURE_CASE_ID = "text_only_one_step_logits"
TEXT_IMAGE_UND_FIXTURE_CASE_ID = "text_image_understanding_one_step_logits"
IMAGE_GEN_FIXTURE_CASE_ID = "image_generation_one_step_velocity"
IMAGE_EDIT_FIXTURE_CASE_ID = "image_edit_vae_context_one_step"


def adapt_text_only_fixture(fixture: dict[str, Any]) -> list[ConversationItem]:
    """Convert an official text-only fixture into V2 conversation items.

    This adapter is deliberately outside BAGEL runtime modules. It may read
    official fixture fields, but its output uses ordinary ``ConversationItem``
    objects and neutral metadata names.
    """

    metadata = fixture.get("metadata", {})
    if metadata.get("case_id") != TEXT_FIXTURE_CASE_ID:
        raise ValueError(f"Unsupported BAGEL fixture case: {metadata.get('case_id')!r}")

    prompt = fixture["raw_input"]["prompt"]
    prompt_fields = fixture["prepared"]["prompt"]
    start_fields = fixture["prepared"]["start"]

    token_ids = prompt_fields["packed_text_ids"].clone().detach().to(dtype=torch.long)
    position_ids = prompt_fields["packed_text_position_ids"].clone().detach().to(dtype=torch.long)
    sequence_indexes = prompt_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long)
    context_indexes = prompt_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long)

    return [
        ConversationItem(
            type="text",
            value=token_ids,
            role="user",
            source="bagel_official_fixture",
            meta={
                "bagel_role": "text",
                "raw_text": prompt,
                "position_ids": position_ids,
                "sequence_indexes": sequence_indexes,
                "context_indexes": context_indexes,
                "token_lens": prompt_fields["text_token_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_before": prompt_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_after": fixture["prepared"]["kv_lens_after_prompt"],
                "rope_after": fixture["prepared"]["ropes_after_prompt"],
                "next_token": {
                    "input_ids": start_fields["packed_start_tokens"].clone().detach().to(dtype=torch.long),
                    "query_lens": fixture["prepared"]["query_lens"].clone().detach().to(dtype=torch.int32),
                    "position_ids": start_fields["packed_query_position_ids"].clone().detach().to(dtype=torch.long),
                    "query_indexes": fixture["prepared"]["packed_query_indexes"].clone().detach().to(dtype=torch.long),
                    "key_value_lens": start_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                    "context_indexes": fixture["prepared"]["packed_key_value_indexes_for_step"]
                    .clone()
                    .detach()
                    .to(dtype=torch.long),
                },
                "expected": {
                    "hidden_state": fixture["one_step"]["hidden_state"],
                    "logits": fixture["one_step"]["logits"],
                    "greedy_token": fixture["one_step"]["greedy_token"],
                },
            },
        )
    ]


def adapt_text_image_und_fixture(fixture: dict[str, Any], *, use_raw_image: bool = False) -> list[ConversationItem]:
    """Convert an official text+image understanding fixture into V2 conversation items."""

    metadata = fixture.get("metadata", {})
    if metadata.get("case_id") != TEXT_IMAGE_UND_FIXTURE_CASE_ID:
        raise ValueError(f"Unsupported BAGEL fixture case: {metadata.get('case_id')!r}")

    image_fields = fixture["prepared"]["image"]
    prompt_fields = fixture["prepared"]["prompt"]
    start_fields = fixture["prepared"]["start"]

    raw_image = _make_fixture_image(fixture["raw_input"]["image_size"]) if use_raw_image else None
    image_meta: dict[str, Any] = {
        "bagel_role": "image_und",
        "raw_image_size": fixture["raw_input"]["image_size"],
    }
    if not use_raw_image:
        image_meta.update(
            {
                "image_token_ids": image_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
                "image_text_indexes": image_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
                "vit_position_ids": image_fields["packed_vit_position_ids"].clone().detach().to(dtype=torch.long),
                "vit_token_indexes": image_fields["packed_vit_token_indexes"].clone().detach().to(dtype=torch.long),
                "vit_token_lens": image_fields["vit_token_seqlens"].clone().detach().to(dtype=torch.int32),
                "position_ids": image_fields["packed_position_ids"].clone().detach().to(dtype=torch.long),
                "sequence_indexes": image_fields["packed_indexes"].clone().detach().to(dtype=torch.long),
                "context_indexes": image_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "query_lens": image_fields["packed_seqlens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_before": image_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                "is_causal": False,
            }
        )

    return [
        ConversationItem(
            type="image",
            value=raw_image if use_raw_image else image_fields["packed_vit_tokens"].clone().detach(),
            role="user",
            source="bagel_official_fixture",
            meta=image_meta,
        ),
        ConversationItem(
            type="text",
            value=prompt_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
            role="user",
            source="bagel_official_fixture",
            meta={
                "bagel_role": "text",
                "raw_text": fixture["raw_input"]["prompt"],
                "position_ids": prompt_fields["packed_text_position_ids"].clone().detach().to(dtype=torch.long),
                "sequence_indexes": prompt_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
                "context_indexes": prompt_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "token_lens": prompt_fields["text_token_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_before": prompt_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_after": fixture["prepared"]["kv_lens_after_prompt"],
                "rope_after": fixture["prepared"]["ropes_after_prompt"],
                "is_causal": True,
                "next_token": {
                    "input_ids": start_fields["packed_start_tokens"].clone().detach().to(dtype=torch.long),
                    "query_lens": fixture["prepared"]["query_lens"].clone().detach().to(dtype=torch.int32),
                    "position_ids": start_fields["packed_query_position_ids"].clone().detach().to(dtype=torch.long),
                    "query_indexes": fixture["prepared"]["packed_query_indexes"].clone().detach().to(dtype=torch.long),
                    "key_value_lens": start_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                    "context_indexes": fixture["prepared"]["packed_key_value_indexes_for_step"]
                    .clone()
                    .detach()
                    .to(dtype=torch.long),
                },
                "expected": {
                    "hidden_state": fixture["one_step"]["hidden_state"],
                    "logits": fixture["one_step"]["logits"],
                    "greedy_token": fixture["one_step"]["greedy_token"],
                },
            },
        ),
    ]


def _make_fixture_image(image_size: list[int]) -> Image.Image:
    width, height = image_size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(rgb)


def adapt_image_gen_fixture(fixture: dict[str, Any]) -> list[ConversationItem]:
    """Convert an official image-generation fixture into V2 conversation items."""

    metadata = fixture.get("metadata", {})
    if metadata.get("case_id") != IMAGE_GEN_FIXTURE_CASE_ID:
        raise ValueError(f"Unsupported BAGEL fixture case: {metadata.get('case_id')!r}")

    prompt_fields = fixture["prepared"]["prompt"]
    latent_fields = fixture["prepared"]["latent"]
    cfg_text_fields = fixture["prepared"].get("cfg_text")
    cfg_img_fields = fixture["prepared"].get("cfg_img")
    timestep_fields = fixture["prepared"]["timesteps"]
    raw_input = fixture["raw_input"]
    cfg_interval = raw_input.get("cfg_interval", [0.0, 1.0])

    return [
        ConversationItem(
            type="image",
            value=fixture["one_step"]["x_t0"].clone().detach(),
            role="assistant",
            source="bagel_official_fixture",
            meta={
                "bagel_role": "image_gen_latent",
                "raw_image_size": fixture["raw_input"]["image_size"],
                "text_token_ids": latent_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
                "text_indexes": latent_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
                "vae_token_indexes": latent_fields["packed_vae_token_indexes"].clone().detach().to(dtype=torch.long),
                "vae_position_ids": latent_fields["packed_vae_position_ids"].clone().detach().to(dtype=torch.long),
                "query_lens": latent_fields["packed_seqlens"].clone().detach().to(dtype=torch.int32),
                "position_ids": latent_fields["packed_position_ids"].clone().detach().to(dtype=torch.long),
                "sequence_indexes": latent_fields["packed_indexes"].clone().detach().to(dtype=torch.long),
                "context_indexes": latent_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "key_value_lens": latent_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                "timestep": timestep_fields["timestep"].clone().detach(),
                "dt": timestep_fields["dt"].clone().detach(),
                "timesteps": timestep_fields["timesteps"].clone().detach(),
                "dts": timestep_fields["dts"].clone().detach(),
                "cfg_text_scale": float(raw_input.get("cfg_text_scale", 1.0)),
                "cfg_img_scale": float(raw_input.get("cfg_img_scale", 1.0)),
                "cfg_interval": list(cfg_interval),
                "cfg_renorm_min": float(raw_input.get("cfg_renorm_min", 0.0)),
                "cfg_renorm_type": raw_input.get("cfg_renorm_type", "global"),
                "cfg_text_position_ids": None
                if cfg_text_fields is None
                else cfg_text_fields["cfg_packed_position_ids"].clone().detach().to(dtype=torch.long),
                "cfg_text_sequence_indexes": None
                if cfg_text_fields is None
                else cfg_text_fields["cfg_packed_query_indexes"].clone().detach().to(dtype=torch.long),
                "cfg_text_key_value_lens": None
                if cfg_text_fields is None
                else cfg_text_fields["cfg_key_values_lens"].clone().detach().to(dtype=torch.int32),
                "cfg_text_context_indexes": None
                if cfg_text_fields is None
                else cfg_text_fields["cfg_packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "cfg_img_position_ids": None
                if cfg_img_fields is None
                else cfg_img_fields["cfg_packed_position_ids"].clone().detach().to(dtype=torch.long),
                "cfg_img_sequence_indexes": None
                if cfg_img_fields is None
                else cfg_img_fields["cfg_packed_query_indexes"].clone().detach().to(dtype=torch.long),
                "cfg_img_key_value_lens": None
                if cfg_img_fields is None
                else cfg_img_fields["cfg_key_values_lens"].clone().detach().to(dtype=torch.int32),
                "cfg_img_context_indexes": None
                if cfg_img_fields is None
                else cfg_img_fields["cfg_packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "flow_step_index": 0,
                "max_flow_steps": 1,
                "expected": {
                    "latent_embeds": fixture["one_step"]["latent_embeds"],
                    "packed_sequence": fixture["one_step"]["packed_sequence"],
                    "hidden_state": fixture["one_step"]["hidden_state"],
                    "velocity": fixture["one_step"]["velocity"],
                    "x_t1": fixture["one_step"]["x_t1"],
                },
            },
        ),
        ConversationItem(
            type="text",
            value=prompt_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
            role="user",
            source="bagel_official_fixture",
            meta={
                "bagel_role": "text",
                "raw_text": fixture["raw_input"]["prompt"],
                "image_generation_prompt": True,
                "position_ids": prompt_fields["packed_text_position_ids"].clone().detach().to(dtype=torch.long),
                "sequence_indexes": prompt_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
                "context_indexes": prompt_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
                "token_lens": prompt_fields["text_token_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_before": prompt_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
                "key_value_lens_after": fixture["prepared"]["kv_lens_after_prompt"],
                "rope_after": fixture["prepared"]["ropes_after_prompt"],
                "is_causal": True,
            },
        ),
    ]


def adapt_image_edit_fixture(fixture: dict[str, Any]) -> list[ConversationItem]:
    """Convert an official input-image VAE-context fixture into V2 conversation items."""

    metadata = fixture.get("metadata", {})
    if metadata.get("case_id") != IMAGE_EDIT_FIXTURE_CASE_ID:
        raise ValueError(f"Unsupported BAGEL fixture case: {metadata.get('case_id')!r}")

    prompt_fields = fixture["prepared"]["prompt"]
    latent_fields = fixture["prepared"]["latent"]
    timestep_fields = fixture["prepared"]["timesteps"]
    raw_input = fixture["raw_input"]

    input_image_item = ConversationItem(
        type="image",
        value=_make_fixture_image(raw_input["input_image_size"]),
        role="user",
        source="bagel_official_fixture",
        meta={
            "bagel_role": "image_vae_input",
            "enable_vae_context": True,
            "raw_image_size": raw_input["input_image_size"],
        },
    )
    image_gen_item = ConversationItem(
        type="image",
        value=fixture["one_step"]["x_t0"].clone().detach(),
        role="assistant",
        source="bagel_official_fixture",
        meta={
            "bagel_role": "image_gen_latent",
            "raw_image_size": raw_input["image_size"],
            "text_token_ids": latent_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
            "text_indexes": latent_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
            "vae_token_indexes": latent_fields["packed_vae_token_indexes"].clone().detach().to(dtype=torch.long),
            "vae_position_ids": latent_fields["packed_vae_position_ids"].clone().detach().to(dtype=torch.long),
            "query_lens": latent_fields["packed_seqlens"].clone().detach().to(dtype=torch.int32),
            "position_ids": latent_fields["packed_position_ids"].clone().detach().to(dtype=torch.long),
            "sequence_indexes": latent_fields["packed_indexes"].clone().detach().to(dtype=torch.long),
            "context_indexes": latent_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
            "key_value_lens": latent_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
            "timestep": timestep_fields["timestep"].clone().detach(),
            "dt": timestep_fields["dt"].clone().detach(),
            "timesteps": timestep_fields["timesteps"].clone().detach(),
            "dts": timestep_fields["dts"].clone().detach(),
            "flow_step_index": 0,
            "max_flow_steps": 1,
        },
    )
    prompt_item = ConversationItem(
        type="text",
        value=prompt_fields["packed_text_ids"].clone().detach().to(dtype=torch.long),
        role="user",
        source="bagel_official_fixture",
        meta={
            "bagel_role": "text",
            "raw_text": raw_input["prompt"],
            "image_generation_prompt": True,
            "position_ids": prompt_fields["packed_text_position_ids"].clone().detach().to(dtype=torch.long),
            "sequence_indexes": prompt_fields["packed_text_indexes"].clone().detach().to(dtype=torch.long),
            "context_indexes": prompt_fields["packed_key_value_indexes"].clone().detach().to(dtype=torch.long),
            "token_lens": prompt_fields["text_token_lens"].clone().detach().to(dtype=torch.int32),
            "key_value_lens_before": prompt_fields["key_values_lens"].clone().detach().to(dtype=torch.int32),
            "key_value_lens_after": fixture["prepared"]["kv_lens_after_prompt"].clone().detach().to(dtype=torch.int32),
            "rope_after": fixture["prepared"]["ropes_after_prompt"].clone().detach().to(dtype=torch.long),
            "is_causal": True,
        },
    )
    return [input_image_item, image_gen_item, prompt_item]


def assert_text_fixture_schema(fixture: dict[str, Any]) -> None:
    """Validate the minimal schema needed by the text-only adapter."""

    metadata = fixture["metadata"]
    if metadata["case_id"] != TEXT_FIXTURE_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in ("raw_input", "rng_state", "tokenizer", "prepared", "cache_after_prefill", "one_step"):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

    prompt_fields = fixture["prepared"]["prompt"]
    for name in (
        "packed_text_ids",
        "packed_text_position_ids",
        "text_token_lens",
        "packed_text_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in prompt_fields:
            raise AssertionError(f"Fixture prompt section missing tensor: {name}")

    start_fields = fixture["prepared"]["start"]
    for name in (
        "packed_start_tokens",
        "packed_query_position_ids",
        "key_values_lens",
        "packed_key_value_indexes",
    ):
        if name not in start_fields:
            raise AssertionError(f"Fixture start section missing tensor: {name}")

    one_step = fixture["one_step"]
    for name in ("hidden_state", "logits", "greedy_token", "cache_after_step"):
        if name not in one_step:
            raise AssertionError(f"Fixture one_step section missing field: {name}")

    for name in ("query_lens", "packed_query_indexes", "packed_key_value_indexes_for_step"):
        if name not in fixture["prepared"]:
            raise AssertionError(f"Fixture prepared section missing tensor: {name}")


def assert_image_gen_fixture_schema(fixture: dict[str, Any]) -> None:
    """Validate the minimal schema needed by the image-generation adapter."""

    metadata = fixture["metadata"]
    if metadata["case_id"] != IMAGE_GEN_FIXTURE_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in ("raw_input", "rng_state", "tokenizer", "prepared", "cache_after_prompt", "one_step"):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

    prompt_fields = fixture["prepared"]["prompt"]
    for name in (
        "packed_text_ids",
        "packed_text_position_ids",
        "text_token_lens",
        "packed_text_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in prompt_fields:
            raise AssertionError(f"Fixture prompt section missing tensor: {name}")

    latent_fields = fixture["prepared"]["latent"]
    for name in (
        "packed_text_ids",
        "packed_text_indexes",
        "packed_init_noises",
        "packed_vae_position_ids",
        "packed_vae_token_indexes",
        "packed_seqlens",
        "packed_position_ids",
        "key_values_lens",
        "packed_indexes",
        "packed_key_value_indexes",
    ):
        if name not in latent_fields:
            raise AssertionError(f"Fixture latent section missing tensor: {name}")

    timestep_fields = fixture["prepared"]["timesteps"]
    for name in ("timesteps", "dts", "timestep", "dt"):
        if name not in timestep_fields:
            raise AssertionError(f"Fixture timesteps section missing tensor: {name}")

    one_step = fixture["one_step"]
    for name in ("x_t0", "packed_sequence", "latent_embeds", "hidden_state", "velocity", "x_t1"):
        if name not in one_step:
            raise AssertionError(f"Fixture one_step section missing tensor: {name}")


def assert_image_edit_fixture_schema(fixture: dict[str, Any]) -> None:
    """Validate the minimal schema needed by the input-image VAE-context adapter."""

    metadata = fixture["metadata"]
    if metadata["case_id"] != IMAGE_EDIT_FIXTURE_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in (
        "raw_input",
        "rng_state",
        "rng_state_before_vae",
        "prepared",
        "cache_after_vae",
        "cache_after_vit",
        "cache_after_prompt",
        "vae_context",
        "one_step",
    ):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

    for name in ("packed_latents", "latent_embeds", "packed_sequence"):
        if name not in fixture["vae_context"]:
            raise AssertionError(f"Fixture vae_context section missing tensor: {name}")

    for name in ("x_t0", "packed_sequence", "latent_embeds", "hidden_state", "velocity", "x_t1"):
        if name not in fixture["one_step"]:
            raise AssertionError(f"Fixture one_step section missing tensor: {name}")


def assert_text_image_fixture_schema(fixture: dict[str, Any]) -> None:
    """Validate the minimal schema needed by the text+image understanding adapter."""

    metadata = fixture["metadata"]
    if metadata["case_id"] != TEXT_IMAGE_UND_FIXTURE_CASE_ID:
        raise AssertionError(f"Unexpected case_id: {metadata['case_id']!r}")

    for section in (
        "raw_input",
        "rng_state",
        "tokenizer",
        "prepared",
        "cache_after_image",
        "cache_after_prompt",
        "one_step",
    ):
        if section not in fixture:
            raise AssertionError(f"Fixture missing section: {section}")

    image_fields = fixture["prepared"]["image"]
    for name in (
        "packed_text_ids",
        "packed_text_indexes",
        "vit_token_seqlens",
        "packed_vit_tokens",
        "packed_vit_position_ids",
        "packed_vit_token_indexes",
        "packed_position_ids",
        "packed_seqlens",
        "packed_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in image_fields:
            raise AssertionError(f"Fixture image section missing tensor: {name}")

    image_embeds = fixture["prepared"]["image_embeds"]
    for name in ("image_embeds", "cu_seqlens", "max_seqlen"):
        if name not in image_embeds:
            raise AssertionError(f"Fixture image_embeds section missing field: {name}")

    prompt_fields = fixture["prepared"]["prompt"]
    for name in (
        "packed_text_ids",
        "packed_text_position_ids",
        "text_token_lens",
        "packed_text_indexes",
        "packed_key_value_indexes",
        "key_values_lens",
    ):
        if name not in prompt_fields:
            raise AssertionError(f"Fixture prompt section missing tensor: {name}")

    for name in ("query_lens", "packed_query_indexes", "packed_key_value_indexes_for_step"):
        if name not in fixture["prepared"]:
            raise AssertionError(f"Fixture prepared section missing tensor: {name}")


__all__ = [
    "IMAGE_GEN_FIXTURE_CASE_ID",
    "IMAGE_EDIT_FIXTURE_CASE_ID",
    "TEXT_FIXTURE_CASE_ID",
    "TEXT_IMAGE_UND_FIXTURE_CASE_ID",
    "adapt_image_edit_fixture",
    "adapt_image_gen_fixture",
    "adapt_text_image_und_fixture",
    "adapt_text_only_fixture",
    "assert_image_edit_fixture_schema",
    "assert_image_gen_fixture_schema",
    "assert_text_image_fixture_schema",
    "assert_text_fixture_schema",
]
