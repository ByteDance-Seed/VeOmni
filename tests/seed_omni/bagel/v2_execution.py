"""BAGEL V2 graph/module execution helpers for parity cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.bagel.reference_model import make_reference_image
from tests.seed_omni.parity_suite.core import to_device
from veomni.models.seed_omni.conversation import ConversationItem


BAGEL_ORACLE_SOURCE = "bagel_official_oracle"


def build_infer_request_from_canonical(canonical: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    builder = _CONVERSATION_BUILDERS.get(str(canonical.get("kind", "")), _build_text_conversation_from_canonical)
    return {"conversation_list": builder(canonical, device=device)}


def build_train_graph_request_from_raw_fixture(
    canonical: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    fixture = to_device(canonical["train_fixture"], device)
    return {"conversation_list": [_build_train_sample_from_fixture(fixture, device=device)]}


def build_train_graph_request_from_stimulus(
    stimulus: Mapping[str, Any],
    *,
    loss_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt = str(stimulus["prompt"])
    sample: list[ConversationItem] = []
    if loss_mode == "text_image_ce":
        vit_tokens, vit_position_ids, vit_token_lens = _synthetic_vit_fixture(device=device)
        sample.append(
            ConversationItem(
                type="image",
                value=vit_tokens,
                role="user",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_und",
                    "bagel_train_exact_no_boundaries": True,
                    "vit_position_ids": vit_position_ids,
                    "vit_token_lens": vit_token_lens,
                    "bagel_train_position_ids": torch.zeros(int(vit_tokens.shape[0]), device=device, dtype=torch.long),
                },
            )
        )

    sample.append(
        ConversationItem(
            type="text",
            value=prompt,
            role="user" if loss_mode == "mse_only" else "assistant",
            source=BAGEL_ORACLE_SOURCE,
            meta={"bagel_role": "text"},
        )
    )

    if loss_mode in {"ce_mse", "mse_only"}:
        target_latent, latent_grid, flow_noise, flow_timesteps = _synthetic_latent_fixture(device=device)
        sample.append(
            ConversationItem(
                type="image",
                value=target_latent,
                role="assistant",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_gen_target",
                    "bagel_train_exact_no_boundaries": True,
                    "bagel_vae_latents_ready": True,
                    "padded_latent": target_latent,
                    "patchified_vae_latent_shape": latent_grid,
                    "packed_latent_position_ids": _latent_position_ids(
                        latent_grid[0], latent_grid[1], max_latent_size=64, device=device
                    ),
                    "flow_timesteps": flow_timesteps,
                    "flow_noise": flow_noise,
                    "timestep_shift": float(stimulus.get("timestep_shift", 3.0)),
                },
            )
        )
    return {"conversation_list": [sample]}


def _build_train_sample_from_fixture(fixture: Mapping[str, Any], *, device: torch.device) -> list[ConversationItem]:
    sample: list[ConversationItem] = []
    if "vit_tokens" in fixture:
        sample.append(
            ConversationItem(
                type="image",
                value=fixture["vit_tokens"].detach().to(device=device, dtype=torch.float32),
                role="user",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_und",
                    "bagel_train_exact_no_boundaries": True,
                    "vit_position_ids": fixture["vit_position_ids"].detach().to(device=device, dtype=torch.long),
                    "vit_token_lens": fixture["vit_token_lens"].detach().to(device=device, dtype=torch.int32),
                    "bagel_train_position_ids": torch.zeros(
                        int(fixture["vit_tokens"].shape[0]), device=device, dtype=torch.long
                    ),
                },
            )
        )

    compute_ce = bool(fixture.get("compute_ce", True))
    sample.append(
        ConversationItem(
            type="text",
            value=fixture["text_token_ids"].detach().to(device=device, dtype=torch.long),
            role="assistant" if compute_ce else "user",
            source=BAGEL_ORACLE_SOURCE,
            meta={"bagel_role": "text", "bagel_train_exact_text_ids": True},
        )
    )

    if str(fixture["loss_mode"]) in {"ce_mse", "mse_only"} and "target_latent" in fixture:
        latent_grid = tuple(int(value) for value in fixture["latent_grid"])
        text_len = int(fixture["text_token_ids"].numel())
        sample.append(
            ConversationItem(
                type="image",
                value=fixture["target_latent"].detach().to(device=device, dtype=torch.float32),
                role="assistant",
                source=BAGEL_ORACLE_SOURCE,
                meta={
                    "bagel_role": "image_gen_target",
                    "bagel_train_exact_no_boundaries": True,
                    "bagel_vae_latents_ready": True,
                    "padded_latent": fixture["target_latent"].detach().to(device=device, dtype=torch.float32),
                    "patchified_vae_latent_shape": latent_grid,
                    "packed_latent_position_ids": _latent_position_ids(
                        latent_grid[0],
                        latent_grid[1],
                        max_latent_size=int(fixture.get("max_latent_size", 64)),
                        device=device,
                    ),
                    "bagel_train_position_ids": torch.full(
                        (latent_grid[0] * latent_grid[1],), text_len, device=device, dtype=torch.long
                    ),
                    "flow_timesteps": fixture["flow_timesteps"].detach().to(device=device, dtype=torch.float32),
                    "flow_noise": fixture["flow_noise"].detach().to(device=device, dtype=torch.float32),
                    "timestep_shift": float(fixture.get("timestep_shift", 3.0)),
                },
            )
        )
    return sample


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


def _build_text_image_und_conversation_from_canonical(
    canonical: Mapping[str, Any], *, device: torch.device
) -> list[ConversationItem]:
    return [
        _build_image_und_item(canonical, device=device),
        _build_text_und_item(canonical, device=device),
    ]


def _build_image_und_item(canonical: Mapping[str, Any], *, device: torch.device) -> ConversationItem:
    if canonical.get("use_raw_image", False):
        return _build_raw_image_und_item(canonical)
    return _build_prepared_image_und_item(canonical, device=device)


def _build_raw_image_und_item(canonical: Mapping[str, Any]) -> ConversationItem:
    return ConversationItem(
        type="image",
        value=make_reference_image(int(canonical["image_width"]), int(canonical["image_height"])),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_und",
            "raw_image_size": [canonical["image_width"], canonical["image_height"]],
        },
    )


def _build_prepared_image_und_item(canonical: Mapping[str, Any], *, device: torch.device) -> ConversationItem:
    image = to_device(canonical["image_input"], device)
    return ConversationItem(
        type="image",
        value=image["packed_vit_tokens"].detach().to(dtype=torch.float32),
        role="user",
        source=BAGEL_ORACLE_SOURCE,
        meta={
            "bagel_role": "image_und",
            "raw_image_size": [canonical["image_width"], canonical["image_height"]],
            "image_token_ids": image["packed_text_ids"].detach().to(dtype=torch.long),
            "image_text_indexes": image["packed_text_indexes"].detach().to(dtype=torch.long),
            "vit_position_ids": image["packed_vit_position_ids"].detach().to(dtype=torch.long),
            "vit_token_indexes": image["packed_vit_token_indexes"].detach().to(dtype=torch.long),
            "vit_token_lens": image["vit_token_seqlens"].detach().to(dtype=torch.int32),
            "position_ids": image["packed_position_ids"].detach().to(dtype=torch.long),
            "sequence_indexes": image["packed_indexes"].detach().to(dtype=torch.long),
            "context_indexes": image["packed_key_value_indexes"].detach().to(dtype=torch.long),
            "query_lens": image["packed_seqlens"].detach().to(dtype=torch.int32),
            "key_value_lens_before": image["key_values_lens"].detach().to(dtype=torch.int32),
            "key_value_lens_after": canonical["kv_lens_after_image"],
            "rope_after": canonical["ropes_after_image"],
            "is_causal": False,
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


_CONVERSATION_BUILDERS = {
    "text_image_und": _build_text_image_und_conversation_from_canonical,
    "image_gen": _build_image_conversation_from_canonical,
    "image_edit": _build_image_conversation_from_canonical,
}


def _synthetic_latent_fixture(
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor, torch.Tensor]:
    latent_grid = (2, 2)
    latent_channels = 16
    latent_patch_size = 2
    h, w = latent_grid
    target_latent = torch.linspace(
        -0.75,
        0.75,
        steps=latent_channels * h * latent_patch_size * w * latent_patch_size,
        device=device,
        dtype=torch.float32,
    ).reshape(1, latent_channels, h * latent_patch_size, w * latent_patch_size)
    num_vae_tokens = h * w
    patch_dim = latent_channels * latent_patch_size * latent_patch_size
    flow_noise = torch.linspace(
        -0.25, 0.25, steps=num_vae_tokens * patch_dim, device=device, dtype=torch.float32
    ).reshape(num_vae_tokens, patch_dim)
    flow_timesteps = torch.linspace(-0.5, 0.5, steps=num_vae_tokens, device=device)
    return target_latent, latent_grid, flow_noise, flow_timesteps


def _synthetic_vit_fixture(*, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vit_tokens = 2
    vit_patch_dim = 3 * 14 * 14
    return (
        torch.linspace(-1.0, 1.0, steps=vit_tokens * vit_patch_dim, device=device, dtype=torch.float32).reshape(
            vit_tokens, vit_patch_dim
        ),
        torch.arange(vit_tokens, device=device, dtype=torch.long),
        torch.tensor([vit_tokens], device=device, dtype=torch.int32),
    )


def _latent_position_ids(height: int, width: int, *, max_latent_size: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(height, device=device, dtype=torch.long)[:, None] * max_latent_size
    cols = torch.arange(width, device=device, dtype=torch.long)[None]
    return (rows + cols).flatten()


__all__ = [
    "build_infer_request_from_canonical",
    "build_train_graph_request_from_raw_fixture",
    "build_train_graph_request_from_stimulus",
]
