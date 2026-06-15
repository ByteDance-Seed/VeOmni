"""BAGEL reference-side execution helpers for parity cases."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel.transformers.bagel import ReferenceImageTransform, make_reference_image
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel.qwen2_navit import NaiveCache
from tests.seed_omni.parity_suite.core import to_cpu, to_device
from tests.seed_omni.parity_suite.core.utilities import autocast_for_dtype, patched_randn_like, sample_named_grad
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext


def run_reference(
    driver_case: str,
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    if driver_case == "train_ce_mse":
        return _run_train_reference(ref_model, inputs, context)
    if driver_case == "text_image_und":
        return _run_text_image_reference(ref_model, inputs, context)
    if driver_case == "image_edit":
        return _run_image_edit_reference(ref_model, inputs, context)
    if driver_case == "image_gen":
        return _run_image_gen_reference(ref_model, inputs, context)
    return _run_text_reference(ref_model, inputs, context)


def _run_text_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    prompt = str(inputs["prompt"])
    model = ref_model.model
    tokenizer = ref_model.tokenizer
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    prompt_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = to_device(prompt_input, device)
    past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)

    start_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    start_input = to_device(start_input, device)
    curr_tokens = start_input["packed_start_tokens"]
    key_values_lens = start_input["key_values_lens"]
    packed_key_value_indexes = start_input["packed_key_value_indexes"]
    packed_query_position_ids = start_input["packed_query_position_ids"]

    packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
    query_lens = torch.ones_like(curr_tokens)
    packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
        0,
        len(key_values_lens),
        device=key_values_lens.device,
        dtype=key_values_lens.dtype,
    )
    packed_key_value_indexes_for_step = _step_key_value_indexes(packed_key_value_indexes, key_values_lens)
    extra_inputs = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=query_lens,
        packed_query_position_ids=packed_query_position_ids,
        packed_query_indexes=packed_query_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes_for_step,
        update_past_key_values=True,
        is_causal=True,
        **extra_inputs,
    )
    logits = model.language_model.lm_head(output.packed_query_sequence)
    canonical = {
        "prompt": prompt,
        "prompt_input": to_cpu(prompt_input),
        "kv_lens_after_prompt": list(kv_lens),
        "ropes_after_prompt": list(ropes),
        "start_input": to_cpu(start_input),
        "packed_query_indexes": packed_query_indexes.detach().cpu(),
        "packed_key_value_indexes_for_step": packed_key_value_indexes_for_step.detach().cpu(),
        "query_lens": query_lens.detach().cpu(),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "hidden_state": output.packed_query_sequence.detach().cpu(),
            "logits": logits.detach().cpu(),
            "greedy_token": torch.argmax(logits, dim=-1).detach().cpu(),
        },
    }
    context.record_extra("canonical", canonical)
    return result


def _run_text_image_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    prompt = str(inputs["prompt"])
    model = ref_model.model
    tokenizer = ref_model.tokenizer
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    raw_image = make_reference_image(int(inputs.get("image_width", 448)), int(inputs.get("image_height", 336)))
    vit_transform = ReferenceImageTransform(
        max_image_size=int(inputs.get("vit_max_image_size", 980)),
        min_image_size=int(inputs.get("vit_min_image_size", 378)),
        image_stride=int(inputs.get("vit_image_stride", 14)),
        max_pixels=int(inputs.get("vit_max_pixels", 14 * 14 * 9 * 1024)),
    )
    preprocessed_image = vit_transform(raw_image.convert("RGB"))

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    image_input, kv_lens_after_image, ropes_after_image = model.prepare_vit_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[raw_image.convert("RGB")],
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    image_input = to_device(image_input, device)
    with autocast_for_dtype(device, dtype):
        image_embeds = _official_image_embeds(model, image_input)
        past_key_values = model.forward_cache_update_vit(past_key_values, **image_input)

    prompt_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=kv_lens_after_image,
        curr_rope=ropes_after_image,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = to_device(prompt_input, device)
    with autocast_for_dtype(device, dtype):
        past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)

    start_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    start_input = to_device(start_input, device)
    curr_tokens = start_input["packed_start_tokens"]
    key_values_lens = start_input["key_values_lens"]
    packed_key_value_indexes = start_input["packed_key_value_indexes"]
    packed_query_position_ids = start_input["packed_query_position_ids"]

    packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
    query_lens = torch.ones_like(curr_tokens)
    packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
        0,
        len(key_values_lens),
        device=key_values_lens.device,
        dtype=key_values_lens.dtype,
    )
    packed_key_value_indexes_for_step = _step_key_value_indexes(packed_key_value_indexes, key_values_lens)
    extra_inputs = {"mode": "und"} if model.use_moe else {}
    with autocast_for_dtype(device, dtype):
        output = model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes_for_step,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        logits = model.language_model.lm_head(output.packed_query_sequence)
    canonical = {
        "kind": "text_image_und",
        "use_raw_image": bool(inputs.get("use_raw_image", False)),
        "prompt": prompt,
        "image_width": raw_image.width,
        "image_height": raw_image.height,
        "image_input": to_cpu(image_input),
        "kv_lens_after_image": list(kv_lens_after_image),
        "ropes_after_image": list(ropes_after_image),
        "prompt_input": to_cpu(prompt_input),
        "kv_lens_after_prompt": list(kv_lens),
        "ropes_after_prompt": list(ropes),
        "start_input": to_cpu(start_input),
        "packed_query_indexes": packed_query_indexes.detach().cpu(),
        "packed_key_value_indexes_for_step": packed_key_value_indexes_for_step.detach().cpu(),
        "query_lens": query_lens.detach().cpu(),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "hidden_state": output.packed_query_sequence.detach().cpu(),
            "logits": logits.detach().cpu(),
            "greedy_token": torch.argmax(logits, dim=-1).detach().cpu(),
            "image_embeds_sample": _sample_tensor(image_embeds).detach().cpu(),
            "preprocessed_image_size": torch.tensor(
                [int(preprocessed_image.shape[-1]), int(preprocessed_image.shape[-2])],
                dtype=torch.long,
            ),
        },
    }
    context.record_extra("canonical", canonical)
    return result


def _run_image_gen_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    prompt = str(inputs["prompt"])
    model = ref_model.model
    tokenizer = ref_model.tokenizer
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]
    prompt_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = to_device(prompt_input, device)
    past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)

    image_height = int(inputs.get("image_height", 1024))
    image_width = int(inputs.get("image_width", 1024))
    flow_step = _run_reference_image_flow_step(
        model,
        new_token_ids,
        inputs,
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        image_size=(image_height, image_width),
        past_key_values=past_key_values,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    canonical = {
        "kind": "image_gen",
        "prompt": prompt,
        "prompt_input": to_cpu(prompt_input),
        "kv_lens_after_prompt": list(kv_lens),
        "ropes_after_prompt": list(ropes),
        "latent_input": to_cpu(flow_step["latent_input"]),
        "timesteps": to_cpu(flow_step["timesteps"]),
        "image_height": image_height,
        "image_width": image_width,
        "max_flow_steps": int(inputs.get("max_flow_steps", 1)),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "base_velocity": flow_step["base_velocity"].detach().cpu(),
            "cfg_text_velocity": None
            if flow_step["cfg_text_velocity"] is None
            else flow_step["cfg_text_velocity"].detach().cpu(),
            "cfg_img_velocity": None
            if flow_step["cfg_img_velocity"] is None
            else flow_step["cfg_img_velocity"].detach().cpu(),
            "velocity": [tensor.detach().cpu() for tensor in flow_step["velocity_steps"]],
            "x_t": [tensor.detach().cpu() for tensor in flow_step["x_t_steps"]],
            "latent_embeds_sample": flow_step["latent_embeds_sample"].detach().cpu(),
            "packed_sequence_sample": flow_step["packed_sequence_sample"].detach().cpu(),
            "image_hidden_state_sample": flow_step["hidden_state_sample"].detach().cpu(),
            "cfg_text_hidden_state_sample": None
            if flow_step["cfg_text_hidden_state_sample"] is None
            else flow_step["cfg_text_hidden_state_sample"].detach().cpu(),
            "cfg_img_hidden_state_sample": None
            if flow_step["cfg_img_hidden_state_sample"] is None
            else flow_step["cfg_img_hidden_state_sample"].detach().cpu(),
            "generated_image_count": torch.tensor(1, dtype=torch.long),
            "generated_image_size": torch.tensor([image_height, image_width], dtype=torch.long),
        },
    }
    context.record_extra("canonical", canonical)
    return result


def _run_image_edit_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    prompt = str(inputs["prompt"])
    model = ref_model.model
    vae_model = ref_model.vae_model
    if vae_model is None:
        raise ValueError("BAGEL image-edit reference requires a loaded VAE model.")
    tokenizer = ref_model.tokenizer
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    raw_image = make_reference_image(int(inputs.get("input_width", 384)), int(inputs.get("input_height", 256)))
    vae_transform = ReferenceImageTransform(
        max_image_size=int(inputs.get("vae_max_image_size", 1024)),
        min_image_size=int(inputs.get("vae_min_image_size", 512)),
        image_stride=int(inputs.get("vae_image_stride", 16)),
        max_pixels=int(inputs.get("vae_max_pixels", 14 * 14 * 9 * 1024)),
    )
    vit_transform = ReferenceImageTransform(
        max_image_size=int(inputs.get("vit_max_image_size", 980)),
        min_image_size=int(inputs.get("vit_min_image_size", 378)),
        image_stride=int(inputs.get("vit_image_stride", 14)),
        max_pixels=int(inputs.get("vit_max_pixels", 14 * 14 * 9 * 1024)),
    )
    context_image = vae_transform.resize_transform(raw_image.convert("RGB"))
    image_shape = context_image.size[::-1]

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    curr_kvlens = [0]
    curr_rope = [0]

    vae_input, curr_kvlens, curr_rope = model.prepare_vae_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[context_image],
        transforms=vae_transform,
        new_token_ids=new_token_ids,
    )
    vae_input = to_device(vae_input, device)
    with autocast_for_dtype(device, dtype):
        vae_context = _forward_vae_context(
            model,
            vae_model,
            past_key_values=past_key_values,
            vae_input=vae_input,
        )
    past_key_values = vae_context["past_key_values"]

    vit_input, curr_kvlens, curr_rope = model.prepare_vit_images(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        images=[context_image],
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    vit_input = to_device(vit_input, device)
    with autocast_for_dtype(device, dtype):
        image_embeds = _official_image_embeds(model, vit_input)
        past_key_values = model.forward_cache_update_vit(past_key_values, **vit_input)

    prompt_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_input = to_device(prompt_input, device)
    with autocast_for_dtype(device, dtype):
        past_key_values = model.forward_cache_update_text(past_key_values, **prompt_input)

    flow_step = _run_reference_image_flow_step(
        model,
        new_token_ids,
        inputs,
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        image_size=image_shape,
        past_key_values=past_key_values,
        device=device,
        dtype=dtype,
    )
    canonical = {
        "kind": "image_edit",
        "prompt": prompt,
        "input_width": raw_image.width,
        "input_height": raw_image.height,
        "prompt_input": to_cpu(prompt_input),
        "kv_lens_after_prompt": list(kv_lens),
        "ropes_after_prompt": list(ropes),
        "latent_input": to_cpu(flow_step["latent_input"]),
        "timesteps": to_cpu(flow_step["timesteps"]),
        "image_height": int(image_shape[0]),
        "image_width": int(image_shape[1]),
        "max_flow_steps": int(inputs.get("max_flow_steps", 1)),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "base_velocity": flow_step["base_velocity"].detach().cpu(),
            "cfg_text_velocity": None
            if flow_step["cfg_text_velocity"] is None
            else flow_step["cfg_text_velocity"].detach().cpu(),
            "cfg_img_velocity": None
            if flow_step["cfg_img_velocity"] is None
            else flow_step["cfg_img_velocity"].detach().cpu(),
            "velocity": [tensor.detach().cpu() for tensor in flow_step["velocity_steps"]],
            "x_t": [tensor.detach().cpu() for tensor in flow_step["x_t_steps"]],
            "image_embeds_sample": _sample_tensor(image_embeds).detach().cpu(),
            "vae_context_latents": vae_context["packed_latents"].detach().cpu(),
            "vae_context_latent_embeds_sample": vae_context["latent_embeds_sample"].detach().cpu(),
            "vae_context_packed_sequence_sample": vae_context["packed_sequence_sample"].detach().cpu(),
            "latent_embeds_sample": flow_step["latent_embeds_sample"].detach().cpu(),
            "packed_sequence_sample": flow_step["packed_sequence_sample"].detach().cpu(),
            "image_hidden_state_sample": flow_step["hidden_state_sample"].detach().cpu(),
            "cfg_text_hidden_state_sample": None
            if flow_step["cfg_text_hidden_state_sample"] is None
            else flow_step["cfg_text_hidden_state_sample"].detach().cpu(),
            "cfg_img_hidden_state_sample": None
            if flow_step["cfg_img_hidden_state_sample"] is None
            else flow_step["cfg_img_hidden_state_sample"].detach().cpu(),
        },
    }
    context.record_extra("canonical", canonical)
    return result


def _run_train_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    model = ref_model.model
    tokenizer = ref_model.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    loss_mode = str(inputs.get("loss_mode", "ce_mse"))
    batch = _build_train_batch(
        tokenizer,
        str(inputs["prompt"]),
        device=device,
        dtype=dtype,
        timestep_shift=float(inputs.get("timestep_shift", 3.0)),
        loss_mode=loss_mode,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    official_batch = {key: value for key, value in batch.items() if key not in {"fixed_noise", "shifted_timesteps"}}
    noise_context = patched_randn_like(batch["fixed_noise"]) if "fixed_noise" in batch else nullcontext()
    with torch.enable_grad(), autocast_for_dtype(device, dtype), noise_context:
        output = model(**official_batch)
        ce = output.get("ce")
        mse = output.get("mse")
        if loss_mode in {"ce_mse", "ce_only", "text_image_ce"} and ce is None:
            raise RuntimeError(f"BAGEL {loss_mode} reference requires CE output.")
        if loss_mode in {"ce_mse", "mse_only"} and mse is None:
            raise RuntimeError(f"BAGEL {loss_mode} reference requires MSE output.")
        ce_loss = None if ce is None else ce.mean()
        mse_loss = None if mse is None else mse.mean()
        loss_terms = [term for term in (ce_loss, mse_loss) if term is not None]
        if not loss_terms:
            raise RuntimeError(f"BAGEL {loss_mode} reference produced no loss terms.")
        loss = sum(loss_terms)
    loss.backward()

    canonical = {"kind": f"train_{loss_mode}", "train_batch": to_cpu(batch)}
    result = {
        "canonical": canonical,
        "reference": {
            "train_total_loss": loss.detach().cpu(),
        },
    }
    if ce_loss is not None:
        result["reference"]["train_ce_loss"] = ce_loss.detach().cpu()
        result["reference"]["train_grad_early_q_proj"] = sample_named_grad(
            model,
            "language_model.model.layers.0.self_attn.q_proj.weight",
        )
        result["reference"]["train_grad_lm_head_rows"] = sample_named_grad(
            model,
            "language_model.lm_head.weight",
            rows=torch.unique(batch["packed_label_ids"].detach().cpu()).to(dtype=torch.long),
        )
        if "packed_vit_tokens" in batch:
            result["reference"]["train_grad_siglip_q_proj"] = sample_named_grad(
                model,
                "vit_model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            )
    if mse_loss is not None:
        result["reference"]["train_mse_loss"] = mse_loss.detach().cpu()
        result["reference"]["train_grad_gen_q_proj"] = sample_named_grad(
            model,
            "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
        )
        result["reference"]["train_grad_llm2vae"] = sample_named_grad(model, "llm2vae.weight")
    context.record_extra("canonical", canonical)
    return result


def _build_train_batch(
    tokenizer: Any,
    text: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    timestep_shift: float,
    loss_mode: str,
) -> dict[str, Any]:
    token_ids = torch.tensor(tokenizer.encode(text), device=device, dtype=torch.long)
    if loss_mode == "text_image_ce":
        return _prepare_text_image_ce_batch(token_ids, device=device, dtype=dtype)
    batch, fixed_noise = _prepare_ce_mse_batch(token_ids, device=device, dtype=dtype)
    _add_flow_noise_controls(batch, fixed_noise=fixed_noise, timestep_shift=timestep_shift)
    if loss_mode == "ce_only":
        _remove_mse_fields(batch)
    elif loss_mode == "mse_only":
        _remove_ce_fields(batch)
    elif loss_mode != "ce_mse":
        raise ValueError(f"Unsupported BAGEL train loss_mode: {loss_mode!r}")
    return batch


def _add_flow_noise_controls(batch: dict[str, Any], *, fixed_noise: torch.Tensor, timestep_shift: float) -> None:
    batch["fixed_noise"] = fixed_noise
    shifted_timesteps = torch.sigmoid(batch["packed_timesteps"])
    batch["shifted_timesteps"] = timestep_shift * shifted_timesteps / (1 + (timestep_shift - 1) * shifted_timesteps)


def _remove_mse_fields(batch: dict[str, Any]) -> None:
    _remove_fields(
        batch,
        (
            "padded_latent",
            "patchified_vae_latent_shapes",
            "packed_latent_position_ids",
            "packed_vae_token_indexes",
            "packed_timesteps",
            "mse_loss_indexes",
            "fixed_noise",
            "shifted_timesteps",
        ),
    )


def _remove_ce_fields(batch: dict[str, Any]) -> None:
    _remove_fields(batch, ("ce_loss_indexes", "packed_label_ids"))


def _remove_fields(batch: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        batch.pop(key, None)


def _step_key_value_indexes(packed_key_value_indexes: torch.Tensor, key_values_lens: torch.Tensor) -> torch.Tensor:
    unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for idx, indexes in enumerate(unpacked):
        unpacked[idx] = indexes + idx
    return torch.cat(unpacked, dim=0)


def _prepare_ce_mse_batch(
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor]:
    latent_grid = (2, 2)
    latent_channels = 16
    latent_patch_size = 2
    max_latent_size = 64
    h, w = latent_grid
    latent = torch.linspace(
        -0.75,
        0.75,
        steps=latent_channels * h * latent_patch_size * w * latent_patch_size,
        device=device,
        dtype=dtype,
    ).reshape(1, latent_channels, h * latent_patch_size, w * latent_patch_size)
    num_vae_tokens = h * w
    text_fields = _base_text_fields(token_ids)
    vae_indexes = torch.arange(int(token_ids.numel()), int(token_ids.numel()) + num_vae_tokens, device=device)
    sequence_length = int(token_ids.numel()) + num_vae_tokens
    batch: dict[str, Any] = {
        "sequence_length": sequence_length,
        **text_fields,
        "sample_lens": [sequence_length],
        "packed_position_ids": torch.cat(
            [
                torch.arange(int(token_ids.numel()), device=device, dtype=torch.long),
                torch.full((num_vae_tokens,), int(token_ids.numel()), device=device, dtype=torch.long),
            ]
        ),
        "nested_attention_masks": [_causal_attention_mask(sequence_length, device)],
        "padded_latent": latent,
        "patchified_vae_latent_shapes": [(h, w)],
        "packed_latent_position_ids": _latent_position_ids(h, w, max_latent_size=max_latent_size, device=device),
        "packed_vae_token_indexes": vae_indexes.to(dtype=torch.long),
        "packed_timesteps": torch.linspace(-0.5, 0.5, steps=num_vae_tokens, device=device),
        "mse_loss_indexes": torch.zeros(sequence_length, device=device, dtype=torch.bool),
    }
    batch["mse_loss_indexes"][batch["packed_vae_token_indexes"]] = True
    _apply_ce(batch, batch["packed_text_indexes"], token_ids)
    patch_dim = latent_channels * latent_patch_size * latent_patch_size
    fixed_noise = torch.linspace(-0.25, 0.25, steps=num_vae_tokens * patch_dim, device=device, dtype=dtype).reshape(
        num_vae_tokens, patch_dim
    )
    return batch, fixed_noise


def _prepare_text_image_ce_batch(
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    vit_tokens = 2
    vit_patch_dim = 3 * 14 * 14
    packed_vit_tokens = torch.linspace(
        -1.0,
        1.0,
        steps=vit_tokens * vit_patch_dim,
        device=device,
        dtype=dtype,
    ).reshape(vit_tokens, vit_patch_dim)
    text_fields = _base_text_fields(token_ids, start_index=vit_tokens)
    sequence_length = vit_tokens + int(token_ids.numel())
    batch: dict[str, Any] = {
        "sequence_length": sequence_length,
        **text_fields,
        "sample_lens": [sequence_length],
        "packed_position_ids": torch.cat(
            [
                torch.zeros(vit_tokens, device=device, dtype=torch.long),
                torch.arange(1, int(token_ids.numel()) + 1, device=device, dtype=torch.long),
            ]
        ),
        "nested_attention_masks": [_causal_attention_mask(sequence_length, device)],
        "packed_vit_tokens": packed_vit_tokens,
        "packed_vit_token_indexes": torch.arange(vit_tokens, device=device, dtype=torch.long),
        "packed_vit_position_ids": torch.arange(vit_tokens, device=device, dtype=torch.long),
        "vit_token_seqlens": torch.tensor([vit_tokens], device=device, dtype=torch.int32),
    }
    _apply_ce(batch, text_fields["packed_text_indexes"], token_ids)
    return batch


def _base_text_fields(token_ids: torch.Tensor, *, start_index: int = 0) -> dict[str, torch.Tensor]:
    length = int(token_ids.numel())
    return {
        "packed_text_ids": token_ids,
        "packed_text_indexes": torch.arange(
            start_index, start_index + length, device=token_ids.device, dtype=torch.long
        ),
        "packed_position_ids": torch.arange(
            start_index, start_index + length, device=token_ids.device, dtype=torch.long
        ),
    }


def _apply_ce(batch: dict[str, Any], text_indexes: torch.Tensor, token_ids: torch.Tensor) -> None:
    ce_loss_indexes = torch.zeros(int(batch["sequence_length"]), device=token_ids.device, dtype=torch.bool)
    ce_loss_indexes[text_indexes[:-1]] = True
    batch["ce_loss_indexes"] = ce_loss_indexes
    batch["packed_label_ids"] = token_ids[1:].clone()


def _causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def _latent_position_ids(height: int, width: int, *, max_latent_size: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(height, device=device, dtype=torch.long)[:, None] * max_latent_size
    cols = torch.arange(width, device=device, dtype=torch.long)[None]
    return (rows + cols).flatten()


def _first_flow_timestep(num_timesteps: int, timestep_shift: float, device: torch.device) -> dict[str, torch.Tensor]:
    if num_timesteps < 2:
        raise ValueError("BAGEL image generation requires num_timesteps >= 2.")
    timesteps_full = torch.linspace(1, 0, num_timesteps, device=device, dtype=torch.float32)
    timesteps_shifted = timestep_shift * timesteps_full / (1 + (timestep_shift - 1) * timesteps_full)
    dts = timesteps_shifted[:-1] - timesteps_shifted[1:]
    return {
        "timesteps": timesteps_shifted[:-1],
        "dts": dts,
        "timestep": timesteps_shifted[:1],
        "dt": dts[:1],
    }


def _cfg_active(inputs: Mapping[str, Any], timestep: torch.Tensor) -> bool:
    if float(inputs.get("cfg_text_scale", 1.0)) <= 1.0 and float(inputs.get("cfg_img_scale", 1.0)) <= 1.0:
        return False
    cfg_interval = inputs.get("cfg_interval", [0.0, 1.0])
    start, end = float(cfg_interval[0]), float(cfg_interval[1])
    t_value = float(timestep.detach().reshape(-1)[0].item())
    return t_value > start and t_value <= end


def _combine_cfg_velocity(
    base_velocity: torch.Tensor,
    cfg_text_velocity: torch.Tensor | None,
    cfg_img_velocity: torch.Tensor | None,
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_renorm_min: float,
    cfg_renorm_type: str,
) -> torch.Tensor:
    if cfg_text_scale <= 1.0:
        return base_velocity
    if cfg_text_velocity is None:
        raise ValueError("cfg_text_velocity is required when cfg_text_scale > 1.0.")
    if cfg_img_scale > 1.0 and cfg_img_velocity is None:
        raise ValueError("cfg_img_velocity is required when cfg_img_scale > 1.0.")

    if cfg_renorm_type == "text_channel":
        text_guided = cfg_text_velocity + cfg_text_scale * (base_velocity - cfg_text_velocity)
        norm_base = torch.norm(base_velocity, dim=-1, keepdim=True)
        norm_text_guided = torch.norm(text_guided, dim=-1, keepdim=True)
        scale = (norm_base / (norm_text_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        text_guided = text_guided * scale
        if cfg_img_scale > 1.0:
            return cfg_img_velocity + cfg_img_scale * (text_guided - cfg_img_velocity)
        return text_guided

    text_guided = cfg_text_velocity + cfg_text_scale * (base_velocity - cfg_text_velocity)
    if cfg_img_scale > 1.0:
        guided = cfg_img_velocity + cfg_img_scale * (text_guided - cfg_img_velocity)
    else:
        guided = text_guided
    if cfg_renorm_type == "global":
        norm_base = torch.norm(base_velocity)
        norm_guided = torch.norm(guided)
    elif cfg_renorm_type == "channel":
        norm_base = torch.norm(base_velocity, dim=-1, keepdim=True)
        norm_guided = torch.norm(guided, dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"{cfg_renorm_type} is not supported")
    scale = (norm_base / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    return guided * scale


def _run_reference_image_flow_step(
    model: Any,
    new_token_ids: Mapping[str, int],
    inputs: Mapping[str, Any],
    *,
    curr_kvlens: list[int],
    curr_rope: list[int],
    image_size: tuple[int, int],
    past_key_values: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    cfg_text_scale = float(inputs.get("cfg_text_scale", 1.0))
    cfg_img_scale = float(inputs.get("cfg_img_scale", 1.0))
    if cfg_img_scale > 1.0 and cfg_text_scale <= 1.0:
        raise ValueError(
            "Official BAGEL applies CFG-image after CFG-text; use cfg_text_scale > 1.0 with cfg_img_scale."
        )
    latent_input = model.prepare_vae_latent(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        image_sizes=[image_size],
        new_token_ids=new_token_ids,
    )
    latent_input = to_device(latent_input, device)
    timestep_fields = _first_flow_timestep(
        int(inputs.get("num_timesteps", 50)),
        float(inputs.get("timestep_shift", 3.0)),
        device,
    )
    x_t0 = latent_input["packed_init_noises"]
    cfg_text_input = None
    cfg_text_past_key_values = None
    if cfg_text_scale > 1.0:
        cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        cfg_text_input = model.prepare_vae_latent_cfg(curr_kvlens=[0], curr_rope=[0], image_sizes=[image_size])
        cfg_text_input = to_device(cfg_text_input, device)
    cfg_img_input = None
    cfg_img_past_key_values = None
    if cfg_img_scale > 1.0:
        cfg_img_past_key_values = past_key_values
        cfg_img_input = model.prepare_vae_latent_cfg(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            image_sizes=[image_size],
        )
        cfg_img_input = to_device(cfg_img_input, device)
    with autocast_for_dtype(device, dtype):
        flow_output = _forward_flow_base(
            model,
            x_t=x_t0,
            timestep=timestep_fields["timestep"],
            latent_input=latent_input,
            past_key_values=past_key_values,
        )
        cfg_text_flow_output = (
            _forward_flow_base(
                model,
                x_t=x_t0,
                timestep=timestep_fields["timestep"],
                latent_input=latent_input,
                past_key_values=cfg_text_past_key_values,
                packed_position_ids=cfg_text_input["cfg_packed_position_ids"] if cfg_text_input is not None else None,
                packed_indexes=cfg_text_input["cfg_packed_query_indexes"] if cfg_text_input is not None else None,
                key_values_lens=cfg_text_input["cfg_key_values_lens"] if cfg_text_input is not None else None,
                packed_key_value_indexes=(
                    cfg_text_input["cfg_packed_key_value_indexes"] if cfg_text_input is not None else None
                ),
            )
            if cfg_text_scale > 1.0 and _cfg_active(inputs, timestep_fields["timestep"])
            else None
        )
        cfg_img_flow_output = (
            _forward_flow_base(
                model,
                x_t=x_t0,
                timestep=timestep_fields["timestep"],
                latent_input=latent_input,
                past_key_values=cfg_img_past_key_values,
                packed_position_ids=cfg_img_input["cfg_packed_position_ids"] if cfg_img_input is not None else None,
                packed_indexes=cfg_img_input["cfg_packed_query_indexes"] if cfg_img_input is not None else None,
                key_values_lens=cfg_img_input["cfg_key_values_lens"] if cfg_img_input is not None else None,
                packed_key_value_indexes=(
                    cfg_img_input["cfg_packed_key_value_indexes"] if cfg_img_input is not None else None
                ),
            )
            if cfg_img_scale > 1.0 and _cfg_active(inputs, timestep_fields["timestep"])
            else None
        )
    base_velocity = flow_output["velocity"]
    cfg_text_velocity = None if cfg_text_flow_output is None else cfg_text_flow_output["velocity"]
    cfg_img_velocity = None if cfg_img_flow_output is None else cfg_img_flow_output["velocity"]
    velocity = _combine_cfg_velocity(
        base_velocity,
        cfg_text_velocity,
        cfg_img_velocity,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_renorm_min=float(inputs.get("cfg_renorm_min", 0.0)),
        cfg_renorm_type=str(inputs.get("cfg_renorm_type", "global")),
    )
    x_t_final = x_t0 - velocity.to(x_t0.device) * timestep_fields["dt"][0]
    # Accumulate per-step velocity / x_t so a multi-step trajectory can be compared
    # step-by-step against the V2 observe sink (which records one record per FSM
    # ``image_flow`` step). Each entry mirrors V2's ``bagel_last_velocity`` (the
    # CFG-combined velocity used for the Euler step) and ``bagel_last_x_t`` (the
    # latent after that Euler update), in the same order.
    step_velocities = [velocity]
    step_x_ts = [x_t_final]
    max_flow_steps = min(int(inputs.get("max_flow_steps", 1)), int(timestep_fields["dts"].numel()))
    for step_index in range(1, max_flow_steps):
        timestep = timestep_fields["timesteps"][step_index : step_index + 1]
        with autocast_for_dtype(device, dtype):
            flow_output = _forward_flow_base(
                model,
                x_t=x_t_final,
                timestep=timestep,
                latent_input=latent_input,
                past_key_values=past_key_values,
            )
            cfg_text_flow_output = (
                _forward_flow_base(
                    model,
                    x_t=x_t_final,
                    timestep=timestep,
                    latent_input=latent_input,
                    past_key_values=cfg_text_past_key_values,
                    packed_position_ids=cfg_text_input["cfg_packed_position_ids"]
                    if cfg_text_input is not None
                    else None,
                    packed_indexes=cfg_text_input["cfg_packed_query_indexes"] if cfg_text_input is not None else None,
                    key_values_lens=cfg_text_input["cfg_key_values_lens"] if cfg_text_input is not None else None,
                    packed_key_value_indexes=(
                        cfg_text_input["cfg_packed_key_value_indexes"] if cfg_text_input is not None else None
                    ),
                )
                if cfg_text_scale > 1.0 and _cfg_active(inputs, timestep)
                else None
            )
            cfg_img_flow_output = (
                _forward_flow_base(
                    model,
                    x_t=x_t_final,
                    timestep=timestep,
                    latent_input=latent_input,
                    past_key_values=cfg_img_past_key_values,
                    packed_position_ids=cfg_img_input["cfg_packed_position_ids"]
                    if cfg_img_input is not None
                    else None,
                    packed_indexes=cfg_img_input["cfg_packed_query_indexes"] if cfg_img_input is not None else None,
                    key_values_lens=cfg_img_input["cfg_key_values_lens"] if cfg_img_input is not None else None,
                    packed_key_value_indexes=(
                        cfg_img_input["cfg_packed_key_value_indexes"] if cfg_img_input is not None else None
                    ),
                )
                if cfg_img_scale > 1.0 and _cfg_active(inputs, timestep)
                else None
            )
        base_velocity = flow_output["velocity"]
        cfg_text_velocity = None if cfg_text_flow_output is None else cfg_text_flow_output["velocity"]
        cfg_img_velocity = None if cfg_img_flow_output is None else cfg_img_flow_output["velocity"]
        velocity = _combine_cfg_velocity(
            base_velocity,
            cfg_text_velocity,
            cfg_img_velocity,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_renorm_min=float(inputs.get("cfg_renorm_min", 0.0)),
            cfg_renorm_type=str(inputs.get("cfg_renorm_type", "global")),
        )
        x_t_final = x_t_final - velocity.to(x_t_final.device) * timestep_fields["dts"][step_index]
        step_velocities.append(velocity)
        step_x_ts.append(x_t_final)
    return {
        "latent_input": latent_input,
        "timesteps": timestep_fields,
        "base_velocity": base_velocity,
        "cfg_text_velocity": cfg_text_velocity,
        "cfg_img_velocity": cfg_img_velocity,
        "velocity": velocity,
        "x_t": x_t_final,
        "velocity_steps": step_velocities,
        "x_t_steps": step_x_ts,
        "latent_embeds_sample": _sample_tensor(flow_output["latent_embeds"]),
        "packed_sequence_sample": _sample_tensor(flow_output["packed_sequence"]),
        "hidden_state_sample": _sample_tensor(flow_output["hidden_state"]),
        "cfg_text_hidden_state_sample": None
        if cfg_text_flow_output is None
        else _sample_tensor(cfg_text_flow_output["hidden_state"]),
        "cfg_img_hidden_state_sample": None
        if cfg_img_flow_output is None
        else _sample_tensor(cfg_img_flow_output["hidden_state"]),
    }


def _forward_flow_base(
    model: Any,
    *,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    latent_input: Mapping[str, torch.Tensor],
    past_key_values: Any,
    packed_position_ids: torch.Tensor | None = None,
    packed_indexes: torch.Tensor | None = None,
    key_values_lens: torch.Tensor | None = None,
    packed_key_value_indexes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    packed_text_ids = latent_input["packed_text_ids"]
    packed_text_indexes = latent_input["packed_text_indexes"]
    packed_vae_token_indexes = latent_input["packed_vae_token_indexes"]
    packed_vae_position_ids = latent_input["packed_vae_position_ids"]
    packed_seqlens = latent_input["packed_seqlens"]

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), model.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    packed_timestep = torch.full((x_t.shape[0],), float(timestep.item()), device=x_t.device, dtype=x_t.dtype)
    latent_embeds = (
        model.vae2llm(x_t) + model.time_embedder(packed_timestep) + model.latent_pos_embed(packed_vae_position_ids)
    )
    if latent_embeds.dtype != packed_sequence.dtype:
        latent_embeds = latent_embeds.to(packed_sequence.dtype)
    packed_sequence[packed_vae_token_indexes] = latent_embeds

    extra_inputs = {}
    if model.use_moe:
        extra_inputs = {
            "mode": "gen",
            "packed_vae_token_indexes": packed_vae_token_indexes,
            "packed_text_indexes": packed_text_indexes,
        }
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=packed_position_ids
        if packed_position_ids is not None
        else latent_input["packed_position_ids"],
        packed_query_indexes=packed_indexes if packed_indexes is not None else latent_input["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=key_values_lens if key_values_lens is not None else latent_input["key_values_lens"],
        packed_key_value_indexes=(
            packed_key_value_indexes
            if packed_key_value_indexes is not None
            else latent_input["packed_key_value_indexes"]
        ),
        update_past_key_values=False,
        is_causal=False,
        **extra_inputs,
    )
    velocity = model.llm2vae(output.packed_query_sequence)[packed_vae_token_indexes]
    return {
        "packed_sequence": packed_sequence.detach(),
        "latent_embeds": latent_embeds.detach(),
        "hidden_state": output.packed_query_sequence.detach(),
        "velocity": velocity.detach(),
    }


def _forward_vae_context(
    model: Any,
    vae_model: nn.Module,
    *,
    past_key_values: Any,
    vae_input: Mapping[str, Any],
) -> dict[str, Any]:
    packed_text_ids = vae_input["packed_text_ids"]
    packed_text_indexes = vae_input["packed_text_indexes"]
    packed_vae_token_indexes = vae_input["packed_vae_token_indexes"]
    packed_seqlens = vae_input["packed_seqlens"]

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), model.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    padded_latent = vae_model.encode(vae_input["padded_images"])
    packed_latents = []
    patch_size = model.latent_patch_size
    for latent, (height, width) in zip(padded_latent, vae_input["patchified_vae_latent_shapes"]):
        latent = latent[:, : height * patch_size, : width * patch_size].reshape(
            model.latent_channel,
            height,
            patch_size,
            width,
            patch_size,
        )
        packed_latents.append(
            torch.einsum("chpwq->hwpqc", latent).reshape(-1, patch_size * patch_size * model.latent_channel)
        )
    packed_latent = torch.cat(packed_latents, dim=0)

    latent_embeds = (
        model.vae2llm(packed_latent)
        + model.time_embedder(vae_input["packed_timesteps"])
        + model.latent_pos_embed(vae_input["packed_vae_position_ids"])
    )
    if latent_embeds.dtype != packed_sequence.dtype:
        latent_embeds = latent_embeds.to(packed_sequence.dtype)
    packed_sequence[packed_vae_token_indexes] = latent_embeds

    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=vae_input["packed_position_ids"],
        packed_query_indexes=vae_input["packed_indexes"],
        past_key_values=past_key_values,
        key_values_lens=vae_input["key_values_lens"],
        packed_key_value_indexes=vae_input["packed_key_value_indexes"],
        update_past_key_values=True,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=packed_vae_token_indexes,
        packed_text_indexes=packed_text_indexes,
    )
    return {
        "past_key_values": output.past_key_values,
        "packed_latents": packed_latent.detach(),
        "latent_embeds": latent_embeds.detach(),
        "latent_embeds_sample": _sample_tensor(latent_embeds),
        "packed_sequence": packed_sequence.detach(),
        "packed_sequence_sample": _sample_tensor(packed_sequence),
    }


def _official_image_embeds(model: Any, image_input: Mapping[str, torch.Tensor]) -> torch.Tensor:
    vit_token_seqlens = image_input["vit_token_seqlens"].detach().to(dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
    max_seqlen = int(torch.max(vit_token_seqlens).item())
    image_embeds = model.vit_model(
        packed_pixel_values=image_input["packed_vit_tokens"],
        packed_flattened_position_ids=image_input["packed_vit_position_ids"],
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    image_embeds = model.connector(image_embeds)
    return image_embeds + model.vit_pos_embed(image_input["packed_vit_position_ids"])


def _sample_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 2:
        return value.detach()[:4, :4]
    return value.detach()[:16]


__all__ = [
    "run_reference",
]
