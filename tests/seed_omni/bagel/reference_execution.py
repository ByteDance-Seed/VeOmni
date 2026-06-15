"""BAGEL reference-side execution helpers for parity cases."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel.transformers.bagel import ReferenceImageTransform, make_reference_image
from tests.seed_omni.bagel.transformers.vendor.inference import InterleaveInferencer
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel.qwen2_navit import NaiveCache
from tests.seed_omni.parity_suite.core import ParityReport, to_cpu, to_device
from tests.seed_omni.parity_suite.core.utilities import autocast_for_dtype, patched_randn_like, sample_named_grad
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext


def run_reference_recipe(
    reference_case: str,
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    if reference_case == "train_ce_mse":
        return _run_train_reference(ref_model, inputs, context)
    if reference_case == "text_image_und":
        return _run_text_image_reference(ref_model, inputs, context)
    if reference_case == "image_edit":
        return _run_image_edit_reference(ref_model, inputs, context)
    if reference_case == "image_gen":
        return _run_image_gen_reference(ref_model, inputs, context)
    return _run_text_reference(ref_model, inputs, context)


def run_reference_only_recipe(*, recipe_id: str, run_kind: str, case_id: str) -> ParityReport:
    if run_kind != "reference_smoke":
        raise NotImplementedError(f"Unsupported BAGEL reference run kind: {run_kind!r}")
    if recipe_id != "transformers_reference_smoke":
        raise NotImplementedError(f"Unsupported BAGEL reference recipe: {recipe_id!r}")
    _run_transformers_reference_smoke()
    return ParityReport(case_id=case_id, probes=())


def _run_transformers_reference_smoke() -> None:
    from tests.seed_omni.bagel.transformers import (
        AutoEncoderParams,
        Bagel,
        BagelConfig,
        BagelOfficialReference,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen2Model,
        SiglipVisionConfig,
        SiglipVisionModel,
        load_vendored_model,
    )

    if Bagel.config_class is not BagelConfig:
        raise AssertionError("BAGEL official reference did not expose BagelConfig.")
    if not issubclass(Qwen2Config, Qwen2Model.config_class):
        raise AssertionError("BAGEL Qwen2Config is not compatible with Qwen2Model.")
    if not issubclass(Qwen2Config, Qwen2ForCausalLM.config_class):
        raise AssertionError("BAGEL Qwen2Config is not compatible with Qwen2ForCausalLM.")
    if SiglipVisionModel.config_class is not SiglipVisionConfig:
        raise AssertionError("BAGEL SigLIP reference did not expose SiglipVisionConfig.")
    if AutoEncoderParams.__name__ != "AutoEncoderParams":
        raise AssertionError("BAGEL autoencoder parameters are not exposed.")
    if not callable(load_vendored_model):
        raise AssertionError("BAGEL vendored reference loader is not callable.")

    llm_config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_module="Qwen2MoTDecoderLayer",
        qk_norm=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    reference = BagelOfficialReference.from_configs(
        llm_config=llm_config,
        visual_gen=False,
        visual_und=False,
        init_on_meta=True,
    )
    if not isinstance(reference.model, Bagel):
        raise AssertionError("BAGEL official reference did not assemble a Bagel model.")
    if reference.config.llm_config is not llm_config:
        raise AssertionError("BAGEL official reference did not preserve the provided LLM config.")
    if reference.config.visual_gen:
        raise AssertionError("BAGEL text-only reference unexpectedly enabled visual generation.")
    if reference.config.visual_und:
        raise AssertionError("BAGEL text-only reference unexpectedly enabled visual understanding.")


def _make_interleave_inferencer(
    ref_model: nn.Module,
    *,
    vae_transform: Any | None = None,
    vit_transform: Any | None = None,
) -> InterleaveInferencer:
    return InterleaveInferencer(
        model=ref_model.model,
        vae_model=ref_model.vae_model,
        tokenizer=ref_model.tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=ref_model.new_token_ids,
    )


def _update_context_text_for_capture(
    inferencer: InterleaveInferencer,
    text: str,
    gen_context: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    generation_input, kv_lens, ropes = inferencer.model.prepare_prompts(
        curr_kvlens=gen_context["kv_lens"],
        curr_rope=gen_context["ropes"],
        prompts=[text],
        tokenizer=inferencer.tokenizer,
        new_token_ids=inferencer.new_token_ids,
    )
    generation_input = to_device(generation_input, device)
    maybe_autocast = nullcontext() if dtype is None else autocast_for_dtype(device, dtype)
    with maybe_autocast:
        past_key_values = inferencer.model.forward_cache_update_text(
            gen_context["past_key_values"],
            **generation_input,
        )
    return generation_input, {
        "kv_lens": kv_lens,
        "ropes": ropes,
        "past_key_values": past_key_values,
    }


def _update_context_image_for_capture(
    inferencer: InterleaveInferencer,
    image: Any,
    gen_context: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    vae: bool,
    vit: bool,
) -> dict[str, Any]:
    if not (vae or vit):
        raise ValueError("BAGEL image context update requires vae or vit.")

    model = inferencer.model
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    result: dict[str, Any] = {}

    if vae:
        vae_input, kv_lens, ropes = model.prepare_vae_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=inferencer.vae_transform,
            new_token_ids=inferencer.new_token_ids,
        )
        vae_input = to_device(vae_input, device)
        with autocast_for_dtype(device, dtype):
            vae_context = _capture_official_cache_update_vae(
                model,
                inferencer.vae_model,
                past_key_values=past_key_values,
                vae_input=vae_input,
            )
        past_key_values = vae_context["past_key_values"]
        result["vae_input"] = vae_input
        result["vae_context"] = vae_context

    if vit:
        vit_input, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=inferencer.vit_transform,
            new_token_ids=inferencer.new_token_ids,
        )
        vit_input = to_device(vit_input, device)
        with autocast_for_dtype(device, dtype):
            vit_context = _capture_official_cache_update_vit(
                model,
                past_key_values=past_key_values,
                vit_input=vit_input,
            )
            past_key_values = vit_context["past_key_values"]
        result["vit_input"] = vit_input
        result["image_embeds"] = vit_context["image_embeds"]

    result["gen_context"] = {
        "kv_lens": kv_lens,
        "ropes": ropes,
        "past_key_values": past_key_values,
    }
    return result


def _run_text_reference(
    ref_model: nn.Module,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
) -> dict[str, Any]:
    prompt = str(inputs["prompt"])
    model = ref_model.model
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device

    inferencer = _make_interleave_inferencer(ref_model)
    gen_context = inferencer.init_gen_context()
    prompt_input, gen_context = _update_context_text_for_capture(inferencer, prompt, gen_context, device=device)
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    past_key_values = gen_context["past_key_values"]

    start_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    start_input = to_device(start_input, device)
    text_step = _capture_official_generate_text(
        model,
        start_input,
        past_key_values=past_key_values,
        new_token_ids=new_token_ids,
        inputs=inputs,
    )
    canonical = {
        "prompt": prompt,
        "prompt_input": to_cpu(prompt_input),
        "kv_lens_after_prompt": list(kv_lens),
        "ropes_after_prompt": list(ropes),
        "start_input": to_cpu(start_input),
        "packed_query_indexes": text_step["packed_query_indexes"].detach().cpu(),
        "packed_key_value_indexes_for_step": text_step["packed_key_value_indexes"].detach().cpu(),
        "query_lens": text_step["query_lens"].detach().cpu(),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "hidden_state": text_step["hidden_state"].detach().cpu(),
            "logits": text_step["logits"].detach().cpu(),
            "greedy_token": text_step["greedy_token"].detach().cpu(),
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

    inferencer = _make_interleave_inferencer(ref_model, vit_transform=vit_transform)
    gen_context = inferencer.init_gen_context()
    image_update = _update_context_image_for_capture(
        inferencer,
        raw_image.convert("RGB"),
        gen_context,
        device=device,
        dtype=dtype,
        vae=False,
        vit=True,
    )
    gen_context = image_update["gen_context"]
    image_input = image_update["vit_input"]
    image_embeds = image_update["image_embeds"]
    kv_lens_after_image = list(gen_context["kv_lens"])
    ropes_after_image = list(gen_context["ropes"])

    prompt_input, gen_context = _update_context_text_for_capture(
        inferencer,
        prompt,
        gen_context,
        device=device,
        dtype=dtype,
    )
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    past_key_values = gen_context["past_key_values"]

    start_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    start_input = to_device(start_input, device)
    text_step = _capture_official_generate_text(
        model,
        start_input,
        past_key_values=past_key_values,
        new_token_ids=new_token_ids,
        inputs=inputs,
        device=device,
        dtype=dtype,
    )
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
        "packed_query_indexes": text_step["packed_query_indexes"].detach().cpu(),
        "packed_key_value_indexes_for_step": text_step["packed_key_value_indexes"].detach().cpu(),
        "query_lens": text_step["query_lens"].detach().cpu(),
    }
    result = {
        "canonical": canonical,
        "reference": {
            "hidden_state": text_step["hidden_state"].detach().cpu(),
            "logits": text_step["logits"].detach().cpu(),
            "greedy_token": text_step["greedy_token"].detach().cpu(),
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
    new_token_ids = ref_model.new_token_ids
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inferencer = _make_interleave_inferencer(ref_model)
    gen_context = inferencer.init_gen_context()
    cfg_img_context = deepcopy(gen_context)

    # Mirror official InterleaveInferencer.interleave_inference for a text-to-image span:
    # CFG-text keeps the pre-text context, while CFG-image receives the text prompt.
    cfg_text_context = deepcopy(gen_context)
    prompt_input, gen_context = _update_context_text_for_capture(
        inferencer,
        prompt,
        gen_context,
        device=device,
        dtype=dtype,
    )
    _, cfg_img_context = _update_context_text_for_capture(
        inferencer,
        prompt,
        cfg_img_context,
        device=device,
        dtype=dtype,
    )
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    past_key_values = gen_context["past_key_values"]

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
        dtype=dtype,
        cfg_text_context=cfg_text_context,
        cfg_img_context=cfg_img_context,
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

    inferencer = _make_interleave_inferencer(ref_model, vae_transform=vae_transform, vit_transform=vit_transform)
    gen_context = inferencer.init_gen_context()
    cfg_img_context = deepcopy(gen_context)
    image_update = _update_context_image_for_capture(
        inferencer,
        context_image,
        gen_context,
        device=device,
        dtype=dtype,
        vae=True,
        vit=True,
    )
    gen_context = image_update["gen_context"]
    vae_context = image_update["vae_context"]
    image_embeds = image_update["image_embeds"]

    # Mirror official image-edit interleave: after the image span, CFG-text keeps
    # image-only context; CFG-image receives only the text prompt.
    cfg_text_context = deepcopy(gen_context)
    prompt_input, gen_context = _update_context_text_for_capture(
        inferencer,
        prompt,
        gen_context,
        device=device,
        dtype=dtype,
    )
    _, cfg_img_context = _update_context_text_for_capture(
        inferencer,
        prompt,
        cfg_img_context,
        device=device,
        dtype=dtype,
    )
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    past_key_values = gen_context["past_key_values"]

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
        cfg_text_context=cfg_text_context,
        cfg_img_context=cfg_img_context,
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


def _capture_official_generate_text(
    model: Any,
    start_input: Mapping[str, torch.Tensor],
    *,
    past_key_values: Any,
    new_token_ids: Mapping[str, int],
    inputs: Mapping[str, Any],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    original_forward_inference = model.language_model.forward_inference
    capture: dict[str, torch.Tensor] = {}

    def wrapped_forward_inference(**forward_kwargs: Any) -> Any:
        output = original_forward_inference(**forward_kwargs)
        logits = model.language_model.lm_head(output.packed_query_sequence)
        capture.update(
            {
                "hidden_state": output.packed_query_sequence.detach(),
                "logits": logits.detach(),
                "greedy_token": torch.argmax(logits, dim=-1).detach(),
                "packed_query_indexes": forward_kwargs["packed_query_indexes"].detach(),
                "packed_key_value_indexes": forward_kwargs["packed_key_value_indexes"].detach(),
                "query_lens": forward_kwargs["query_lens"].detach(),
            }
        )
        return output

    model.language_model.forward_inference = wrapped_forward_inference
    context = nullcontext() if device is None or dtype is None else autocast_for_dtype(device, dtype)
    try:
        with context:
            model.generate_text(
                past_key_values=past_key_values,
                max_length=1,
                do_sample=bool(inputs.get("do_sample", False)),
                temperature=float(inputs.get("temperature", inputs.get("text_temperature", 1.0))),
                end_token_id=new_token_ids["eos_token_id"],
                **start_input,
            )
    finally:
        model.language_model.forward_inference = original_forward_inference
    if not capture:
        raise RuntimeError("Official BAGEL text generation produced no captured step.")
    return capture


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
    cfg_text_context: Mapping[str, Any] | None = None,
    cfg_img_context: Mapping[str, Any] | None = None,
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
    cfg_text_input, cfg_text_past_key_values = _prepare_cfg_latent_input(
        model,
        cfg_text_context,
        image_size=image_size,
        device=device,
        default_kvlens=[0],
        default_ropes=[0],
        default_past_key_values=NaiveCache(model.config.llm_config.num_hidden_layers),
    )
    cfg_img_input, cfg_img_past_key_values = _prepare_cfg_latent_input(
        model,
        cfg_img_context,
        image_size=image_size,
        device=device,
        default_kvlens=curr_kvlens,
        default_ropes=curr_rope,
        default_past_key_values=past_key_values,
    )
    captures = _capture_official_generate_image(
        model,
        latent_input,
        past_key_values=past_key_values,
        cfg_text_input=cfg_text_input,
        cfg_text_past_key_values=cfg_text_past_key_values,
        cfg_img_input=cfg_img_input,
        cfg_img_past_key_values=cfg_img_past_key_values,
        num_timesteps=int(inputs.get("num_timesteps", 50)),
        timestep_shift=float(inputs.get("timestep_shift", 3.0)),
        cfg_interval=inputs.get("cfg_interval", [0.0, 1.0]),
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_renorm_min=float(inputs.get("cfg_renorm_min", 0.0)),
        cfg_renorm_type=str(inputs.get("cfg_renorm_type", "global")),
        enable_taylorseer=bool(inputs.get("enable_taylorseer", False)),
        max_flow_steps=int(inputs.get("max_flow_steps", 1)),
        timestep_fields=timestep_fields,
        device=device,
        dtype=dtype,
    )
    if not captures:
        raise RuntimeError("Official BAGEL image generation produced no captured flow steps.")
    first_capture = captures[0]
    last_capture = captures[-1]
    return {
        "latent_input": latent_input,
        "timesteps": timestep_fields,
        "base_velocity": first_capture["base_velocity"],
        "cfg_text_velocity": first_capture["cfg_text_velocity"],
        "cfg_img_velocity": first_capture["cfg_img_velocity"],
        "velocity": last_capture["velocity"],
        "x_t": last_capture["x_t"],
        "velocity_steps": [capture["velocity"] for capture in captures],
        "x_t_steps": [capture["x_t"] for capture in captures],
        "latent_embeds_sample": _sample_tensor(first_capture["latent_embeds"]),
        "packed_sequence_sample": _sample_tensor(first_capture["packed_sequence"]),
        "hidden_state_sample": _sample_tensor(first_capture["hidden_state"]),
        "cfg_text_hidden_state_sample": None
        if first_capture["cfg_text_hidden_state"] is None
        else _sample_tensor(first_capture["cfg_text_hidden_state"]),
        "cfg_img_hidden_state_sample": None
        if first_capture["cfg_img_hidden_state"] is None
        else _sample_tensor(first_capture["cfg_img_hidden_state"]),
    }


def _prepare_cfg_latent_input(
    model: Any,
    context: Mapping[str, Any] | None,
    *,
    image_size: tuple[int, int],
    device: torch.device,
    default_kvlens: list[int],
    default_ropes: list[int],
    default_past_key_values: Any,
) -> tuple[dict[str, torch.Tensor], Any]:
    if context is None:
        curr_kvlens = default_kvlens
        curr_rope = default_ropes
        past_key_values = default_past_key_values
    else:
        curr_kvlens = context["kv_lens"]
        curr_rope = context["ropes"]
        past_key_values = context["past_key_values"]
    cfg_input = model.prepare_vae_latent_cfg(
        curr_kvlens=curr_kvlens,
        curr_rope=curr_rope,
        image_sizes=[image_size],
    )
    return to_device(cfg_input, device), past_key_values


class _StopFlowCapture(Exception):
    pass


def _capture_official_generate_image(
    model: Any,
    latent_input: Mapping[str, torch.Tensor],
    *,
    past_key_values: Any,
    cfg_text_input: Mapping[str, torch.Tensor],
    cfg_text_past_key_values: Any,
    cfg_img_input: Mapping[str, torch.Tensor],
    cfg_img_past_key_values: Any,
    num_timesteps: int,
    timestep_shift: float,
    cfg_interval: Any,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_renorm_min: float,
    cfg_renorm_type: str,
    enable_taylorseer: bool,
    max_flow_steps: int,
    timestep_fields: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    max_capture_steps = min(max_flow_steps, int(timestep_fields["dts"].numel()))
    original_forward_flow = model._forward_flow

    def wrapped_forward_flow(**flow_kwargs: Any) -> torch.Tensor:
        if len(captures) >= max_capture_steps:
            raise _StopFlowCapture
        step_index = len(captures)
        original_forward_inference = model.language_model.forward_inference
        call_records: list[dict[str, torch.Tensor]] = []
        packed_vae_token_indexes = flow_kwargs["packed_vae_token_indexes"]

        def wrapped_forward_inference(**forward_kwargs: Any) -> Any:
            output = original_forward_inference(**forward_kwargs)
            hidden_state = output.packed_query_sequence.detach()
            call_records.append(
                {
                    "packed_sequence": forward_kwargs["packed_query_sequence"].detach(),
                    "hidden_state": hidden_state,
                    "velocity": model.llm2vae(hidden_state)[packed_vae_token_indexes].detach(),
                }
            )
            return output

        model.language_model.forward_inference = wrapped_forward_inference
        try:
            velocity = original_forward_flow(**flow_kwargs)
        finally:
            model.language_model.forward_inference = original_forward_inference

        base_record = call_records[0]
        cfg_text_record = call_records[1] if flow_kwargs.get("cfg_text_scale", 1.0) > 1.0 else None
        cfg_img_offset = 2 if cfg_text_record is not None else 1
        cfg_img_record = call_records[cfg_img_offset] if flow_kwargs.get("cfg_img_scale", 1.0) > 1.0 else None
        x_t = flow_kwargs["x_t"]
        comparable_velocity = velocity.to(dtype=base_record["velocity"].dtype)
        x_t_after = x_t - comparable_velocity.to(x_t.device) * timestep_fields["dts"][step_index]
        captures.append(
            {
                "base_velocity": base_record["velocity"],
                "cfg_text_velocity": None if cfg_text_record is None else cfg_text_record["velocity"],
                "cfg_img_velocity": None if cfg_img_record is None else cfg_img_record["velocity"],
                "velocity": comparable_velocity.detach(),
                "x_t": x_t_after.detach(),
                "latent_embeds": base_record["packed_sequence"][packed_vae_token_indexes].detach(),
                "packed_sequence": base_record["packed_sequence"],
                "hidden_state": base_record["hidden_state"],
                "cfg_text_hidden_state": None if cfg_text_record is None else cfg_text_record["hidden_state"],
                "cfg_img_hidden_state": None if cfg_img_record is None else cfg_img_record["hidden_state"],
            }
        )
        return velocity

    model._forward_flow = wrapped_forward_flow
    try:
        with autocast_for_dtype(device, dtype):
            model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                **latent_input,
                cfg_text_packed_position_ids=cfg_text_input["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text_input["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text_input["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text_input["cfg_packed_key_value_indexes"],
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_packed_position_ids=cfg_img_input["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_input["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_input["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_input["cfg_packed_key_value_indexes"],
                cfg_img_past_key_values=cfg_img_past_key_values,
                enable_taylorseer=enable_taylorseer,
            )
    except _StopFlowCapture:
        pass
    finally:
        model._forward_flow = original_forward_flow
        if enable_taylorseer:
            model.language_model.model.enable_taylorseer = False
    return captures


def _capture_official_cache_update_vae(
    model: Any,
    vae_model: nn.Module,
    *,
    past_key_values: Any,
    vae_input: Mapping[str, Any],
) -> dict[str, Any]:
    packed_vae_token_indexes = vae_input["packed_vae_token_indexes"]
    original_forward_inference = model.language_model.forward_inference
    capture: dict[str, Any] = {}

    def capture_vae2llm_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        capture["packed_latents"] = args[0].detach()

    def wrapped_forward_inference(**forward_kwargs: Any) -> Any:
        output = original_forward_inference(**forward_kwargs)
        packed_sequence = forward_kwargs["packed_query_sequence"].detach()
        capture["packed_sequence"] = packed_sequence
        capture["latent_embeds"] = packed_sequence[packed_vae_token_indexes].detach()
        return output

    hook = model.vae2llm.register_forward_pre_hook(capture_vae2llm_input)
    model.language_model.forward_inference = wrapped_forward_inference
    try:
        past_key_values = model.forward_cache_update_vae(vae_model, past_key_values, **vae_input)
    finally:
        hook.remove()
        model.language_model.forward_inference = original_forward_inference
    if "packed_latents" not in capture or "packed_sequence" not in capture:
        raise RuntimeError("Official BAGEL VAE context update produced no captured tensors.")
    return {
        "past_key_values": past_key_values,
        "packed_latents": capture["packed_latents"],
        "latent_embeds": capture["latent_embeds"],
        "latent_embeds_sample": _sample_tensor(capture["latent_embeds"]),
        "packed_sequence": capture["packed_sequence"],
        "packed_sequence_sample": _sample_tensor(capture["packed_sequence"]),
    }


def _capture_official_cache_update_vit(
    model: Any,
    *,
    past_key_values: Any,
    vit_input: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    original_forward_inference = model.language_model.forward_inference
    packed_vit_token_indexes = vit_input["packed_vit_token_indexes"]
    capture: dict[str, torch.Tensor] = {}

    def wrapped_forward_inference(**forward_kwargs: Any) -> Any:
        output = original_forward_inference(**forward_kwargs)
        packed_sequence = forward_kwargs["packed_query_sequence"].detach()
        capture["image_embeds"] = packed_sequence[packed_vit_token_indexes].detach()
        return output

    model.language_model.forward_inference = wrapped_forward_inference
    try:
        past_key_values = model.forward_cache_update_vit(past_key_values, **vit_input)
    finally:
        model.language_model.forward_inference = original_forward_inference
    if "image_embeds" not in capture:
        raise RuntimeError("Official BAGEL ViT context update produced no captured image embeddings.")
    return {"past_key_values": past_key_values, "image_embeds": capture["image_embeds"]}


def _sample_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 2:
        return value.detach()[:4, :4]
    return value.detach()[:16]


__all__ = [
    "run_reference_only_recipe",
    "run_reference_recipe",
]
