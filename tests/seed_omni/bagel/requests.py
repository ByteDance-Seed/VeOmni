"""BAGEL V2 request handlers for parity cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.bagel.fixtures import latent_position_ids, synthetic_latent_fixture, synthetic_vit_fixture
from tests.seed_omni.parity_suite.core import make_reference_image, to_device
from tests.seed_omni.parity_suite.v2.request import ConversationRequestBuilder, V2RequestContext
from veomni.models.seed_omni.conversation import ConversationItem


BAGEL_ORACLE_SOURCE = "bagel_official_oracle"


def build_text_und_request(ctx: V2RequestContext) -> dict[str, Any]:
    builder = ConversationRequestBuilder(ctx.canonical, device=ctx.device)
    return builder.request(_build_text_und_item(builder, ctx.canonical))


def build_text_image_und_request(ctx: V2RequestContext) -> dict[str, Any]:
    builder = ConversationRequestBuilder(ctx.canonical, device=ctx.device)
    return builder.request(
        _build_image_und_item(builder, ctx.canonical),
        _build_text_und_item(builder, ctx.canonical),
    )


def build_image_gen_request(ctx: V2RequestContext) -> dict[str, Any]:
    builder = ConversationRequestBuilder(ctx.canonical, device=ctx.device)
    return builder.request(
        _build_image_prompt_item(builder, ctx.canonical),
    )


def build_image_edit_request(ctx: V2RequestContext) -> dict[str, Any]:
    builder = ConversationRequestBuilder(ctx.canonical, device=ctx.device)
    return builder.request(
        _build_image_vae_input_item(builder, ctx.canonical),
        _build_image_prompt_item(builder, ctx.canonical, image_generation_prompt=True),
    )


def build_train_request(ctx: V2RequestContext) -> dict[str, Any]:
    if ctx.reference_output is not None:
        return _build_train_graph_request_from_raw_fixture(ctx.canonical, device=ctx.device)
    loss_mode = ctx.case.recipe.reference.get("loss_mode")
    if loss_mode is None:
        raise ValueError(f"BAGEL training recipe {ctx.case.recipe.id!r} must declare a loss_mode.")
    return _build_train_graph_request_from_stimulus(
        ctx.stimulus,
        loss_mode=str(loss_mode),
        device=ctx.device,
    )


def _build_train_graph_request_from_raw_fixture(
    canonical: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    fixture = to_device(canonical["train_fixture"], device)
    builder = ConversationRequestBuilder(fixture, device=device)
    return builder.batched_request(*_build_train_sample_from_fixture(builder, fixture, device=device))


def _build_train_graph_request_from_stimulus(
    stimulus: Mapping[str, Any],
    *,
    loss_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt = str(stimulus["prompt"])
    builder = ConversationRequestBuilder({}, device=device)
    sample: list[ConversationItem] = []
    if loss_mode == "text_image_ce":
        vit_fixture = synthetic_vit_fixture(device=device)
        vit_tokens = vit_fixture["vit_tokens"]
        sample.append(
            builder.image(
                vit_tokens,
                role="user",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_und",
                    "bagel_train_exact_no_boundaries": True,
                    "vit_position_ids": vit_fixture["vit_position_ids"],
                    "vit_token_lens": vit_fixture["vit_token_lens"],
                    "bagel_train_position_ids": torch.zeros(int(vit_tokens.shape[0]), device=device, dtype=torch.long),
                },
            )
        )

    sample.append(
        builder.text(
            prompt,
            role="user" if loss_mode == "mse_only" else "assistant",
            source=BAGEL_ORACLE_SOURCE,
            meta={"bagel_role": "text"},
        )
    )

    if loss_mode in {"ce_mse", "mse_only"}:
        latent_fixture = synthetic_latent_fixture(device=device)
        target_latent = latent_fixture["target_latent"]
        latent_grid = latent_fixture["latent_grid"]
        sample.append(
            builder.image(
                target_latent,
                role="assistant",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_gen_target",
                    "bagel_train_exact_no_boundaries": True,
                    "bagel_vae_latents_ready": True,
                    "padded_latent": target_latent,
                    "patchified_vae_latent_shape": latent_grid,
                    "packed_latent_position_ids": latent_position_ids(
                        latent_grid[0], latent_grid[1], max_latent_size=64, device=device
                    ),
                    "flow_timesteps": latent_fixture["flow_timesteps"],
                    "flow_noise": latent_fixture["flow_noise"],
                    "timestep_shift": float(stimulus.get("timestep_shift", 3.0)),
                },
            )
        )
    return builder.batched_request(*sample)


def _build_train_sample_from_fixture(
    builder: ConversationRequestBuilder,
    fixture: Mapping[str, Any],
    *,
    device: torch.device,
) -> list[ConversationItem]:
    sample: list[ConversationItem] = []
    if "vit_tokens" in fixture:
        sample.append(
            builder.image(
                builder.path("vit_tokens", dtype=torch.float32),
                role="user",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_und",
                    "bagel_train_exact_no_boundaries": True,
                    "vit_position_ids": builder.path("vit_position_ids", dtype=torch.long),
                    "vit_token_lens": builder.path("vit_token_lens", dtype=torch.int32),
                    "bagel_train_position_ids": torch.zeros(
                        int(fixture["vit_tokens"].shape[0]), device=device, dtype=torch.long
                    ),
                },
            )
        )

    compute_ce = bool(fixture.get("compute_ce", True))
    sample.append(
        builder.text(
            builder.path("text_token_ids", dtype=torch.long),
            role="assistant" if compute_ce else "user",
            source=BAGEL_ORACLE_SOURCE,
            meta={"bagel_role": "text", "bagel_train_exact_text_ids": True},
        )
    )

    if str(fixture["loss_mode"]) in {"ce_mse", "mse_only"} and "target_latent" in fixture:
        latent_grid = tuple(int(value) for value in fixture["latent_grid"])
        text_len = int(fixture["text_token_ids"].numel())
        sample.append(
            builder.image(
                builder.path("target_latent", dtype=torch.float32),
                role="assistant",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_gen_target",
                    "bagel_train_exact_no_boundaries": True,
                    "bagel_vae_latents_ready": True,
                    "padded_latent": builder.path("target_latent", dtype=torch.float32),
                    "patchified_vae_latent_shape": latent_grid,
                    "packed_latent_position_ids": latent_position_ids(
                        latent_grid[0],
                        latent_grid[1],
                        max_latent_size=int(fixture.get("max_latent_size", 64)),
                        device=device,
                    ),
                    "bagel_train_position_ids": torch.full(
                        (latent_grid[0] * latent_grid[1],), text_len, device=device, dtype=torch.long
                    ),
                    "flow_timesteps": builder.path("flow_timesteps", dtype=torch.float32),
                    "flow_noise": builder.path("flow_noise", dtype=torch.float32),
                    "timestep_shift": float(fixture.get("timestep_shift", 3.0)),
                },
            )
        )
    return sample


def _build_text_und_item(builder: ConversationRequestBuilder, canonical: Mapping[str, Any]) -> ConversationItem:
    return builder.text(
        builder.path("prompt_input.packed_text_ids", dtype=torch.long),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "text",
            "raw_text": canonical["prompt"],
        },
    )


def _build_image_und_item(builder: ConversationRequestBuilder, canonical: Mapping[str, Any]) -> ConversationItem:
    if canonical.get("use_raw_image", False):
        return _build_raw_image_und_item(builder, canonical)
    return _build_prepared_image_und_item(builder, canonical)


def _build_raw_image_und_item(builder: ConversationRequestBuilder, canonical: Mapping[str, Any]) -> ConversationItem:
    return builder.image(
        builder.literal(make_reference_image(int(canonical["image_width"]), int(canonical["image_height"]))),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_und",
            "raw_image_size": [canonical["image_width"], canonical["image_height"]],
        },
    )


def _build_prepared_image_und_item(
    builder: ConversationRequestBuilder,
    canonical: Mapping[str, Any],
) -> ConversationItem:
    return builder.image(
        builder.path("image_input.packed_vit_tokens", dtype=torch.float32),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_und",
            "raw_image_size": [canonical["image_width"], canonical["image_height"]],
            "vit_position_ids": builder.path("image_input.packed_vit_position_ids", dtype=torch.long),
            "vit_token_lens": builder.path("image_input.vit_token_seqlens", dtype=torch.int32),
        },
    )


def _build_image_latent_item(builder: ConversationRequestBuilder, canonical: Mapping[str, Any]) -> ConversationItem:
    return builder.image(
        builder.path("latent_input.packed_init_noises", dtype=torch.float32),
        role="assistant",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_gen_latent",
            "raw_image_size": [canonical["image_height"], canonical["image_width"]],
            "text_token_ids": builder.path("latent_input.packed_text_ids", dtype=torch.long),
            "text_indexes": builder.path("latent_input.packed_text_indexes", dtype=torch.long),
            "vae_token_indexes": builder.path("latent_input.packed_vae_token_indexes", dtype=torch.long),
            "vae_position_ids": builder.path("latent_input.packed_vae_position_ids", dtype=torch.long),
            "query_lens": builder.path("latent_input.packed_seqlens", dtype=torch.int32),
            "position_ids": builder.path("latent_input.packed_position_ids", dtype=torch.long),
            "sequence_indexes": builder.path("latent_input.packed_indexes", dtype=torch.long),
            "key_value_lens": builder.path("latent_input.key_values_lens", dtype=torch.int32),
            "context_indexes": builder.path("latent_input.packed_key_value_indexes", dtype=torch.long),
            "rope_after_prompt": builder.path("ropes_after_prompt", device=False),
            "timesteps": builder.path("timesteps.timesteps", dtype=torch.float32),
            "dts": builder.path("timesteps.dts", dtype=torch.float32),
            "max_flow_steps": int(canonical.get("max_flow_steps", 1)),
            "cfg_text_scale": 1.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.0, 1.0],
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
        },
    )


def _build_image_prompt_item(
    builder: ConversationRequestBuilder,
    canonical: Mapping[str, Any],
    *,
    image_generation_prompt: bool = False,
) -> ConversationItem:
    return builder.text(
        builder.path("prompt_input.packed_text_ids", dtype=torch.long),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "text",
            "raw_text": canonical["prompt"],
            "image_generation_prompt": image_generation_prompt,
        },
    )


def _build_image_vae_input_item(builder: ConversationRequestBuilder, canonical: Mapping[str, Any]) -> ConversationItem:
    input_width = int(canonical["input_width"])
    input_height = int(canonical["input_height"])
    return builder.image(
        builder.literal(make_reference_image(input_width, input_height)),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_vae_input",
            "enable_vae_context": True,
            "raw_image_size": [input_width, input_height],
        },
    )


__all__ = [
    "BAGEL_ORACLE_SOURCE",
    "build_image_edit_request",
    "build_image_gen_request",
    "build_text_image_und_request",
    "build_text_und_request",
    "build_train_request",
]
