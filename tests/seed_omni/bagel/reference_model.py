"""BAGEL parity reference model and official oracle execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tests.seed_omni.bagel.fixtures import latent_position_ids, synthetic_latent_fixture, synthetic_vit_fixture
from tests.seed_omni.bagel.transformers.vendor.data.data_utils import add_special_tokens
from tests.seed_omni.bagel.transformers.vendor.data.transforms import ImageTransform
from tests.seed_omni.bagel.transformers.vendor.inference import InterleaveInferencer
from tests.seed_omni.bagel.transformers.vendor.modeling.autoencoder import AutoEncoderParams, load_ae
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel.qwen2_navit import NaiveCache
from tests.seed_omni.bagel.transformers.vendor.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from tests.seed_omni.parity_suite.core import (
    ParityReport,
    autocast_for_dtype,
    make_reference_image,
    patched_randn_like,
    resolve_torch_dtype,
    sample_named_grad,
    to_cpu,
    to_device,
)
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.contract import make_reference_run_output
from tests.seed_omni.parity_suite.reference.model import (
    ParityReferenceModel,
    empty_init_context,
    load_safetensors_weights,
)


class BagelModel(ParityReferenceModel):
    """Own the official BAGEL assembly so captures do not import vendor internals."""

    config_class = BagelConfig
    model_type = "bagel"

    def __init__(
        self,
        model: Bagel,
        *,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
        vae_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids or {}
        self.config = model.config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        *,
        llm_config: Qwen2Config,
        vit_config: SiglipVisionConfig | None = None,
        vae_config: Any | None = None,
        visual_gen: bool,
        visual_und: bool,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
        init_on_meta: bool = True,
        latent_patch_size: int | None = None,
        max_latent_size: int | None = None,
        timestep_shift: float | None = None,
        vit_max_num_patch_per_side: int | None = None,
        connector_act: str | None = None,
    ) -> BagelModel:
        _ensure_default_rope_init()
        _normalize_llm_config(llm_config)
        config = BagelConfig(
            visual_gen=visual_gen,
            visual_und=visual_und,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            **_optional_bagel_config_kwargs(
                latent_patch_size=latent_patch_size,
                max_latent_size=max_latent_size,
                timestep_shift=timestep_shift,
                vit_max_num_patch_per_side=vit_max_num_patch_per_side,
                connector_act=connector_act,
            ),
        )
        context = empty_init_context() if init_on_meta else nullcontext()
        with context:
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config) if visual_und and vit_config is not None else None
            model = Bagel(language_model, vit_model, config)
            if visual_und and vit_model is not None:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=init_on_meta)
        return cls(model, tokenizer=tokenizer, new_token_ids=new_token_ids)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        config: BagelConfig | None = None,
        visual_gen: bool = True,
        visual_und: bool = True,
        init_on_meta: bool = True,
        torch_dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        load_weights: bool = True,
        latent_patch_size: int = 2,
        max_latent_size: int = 64,
        timestep_shift: float = 3.0,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        **kwargs: Any,
    ) -> BagelModel:
        del model_args, config, kwargs
        root = Path(pretrained_model_name_or_path)
        dtype = resolve_torch_dtype(torch_dtype)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        llm_config = Qwen2Config.from_json_file(str(root / "llm_config.json"))
        _normalize_llm_config(llm_config)

        tokenizer = Qwen2Tokenizer.from_pretrained(str(root), local_files_only=True)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens > 0:
            llm_config.vocab_size = len(tokenizer)

        vit_config = SiglipVisionConfig.from_json_file(str(root / "vit_config.json")) if visual_und else None
        if vit_config is not None:
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1
        reference = cls.from_configs(
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=_default_vae_config() if visual_gen else None,
            visual_gen=visual_gen,
            visual_und=visual_und,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            init_on_meta=init_on_meta,
            latent_patch_size=latent_patch_size,
            max_latent_size=max_latent_size,
            timestep_shift=timestep_shift,
            vit_max_num_patch_per_side=vit_max_num_patch_per_side,
            connector_act=connector_act,
        )
        if load_weights:
            load_safetensors_weights(
                reference.model,
                root / "ema.safetensors",
                include_prefixes=_bagel_weight_prefixes(visual_gen, visual_und),
                device=target_device,
                dtype=dtype,
            )
        reference.model.to(device=target_device)
        if visual_und:
            vae_model, _ = load_ae(str(root / "ae.safetensors"))
            reference.vae_model = vae_model.to(device=target_device, dtype=dtype).eval()
        reference.eval()
        return reference

    def run_reference_train(self, inputs: Mapping[str, Any], context: ReferenceCaptureContext) -> dict[str, Any]:
        return _run_train_reference(self, inputs, context)

    def run_reference_text_und(self, inputs: Mapping[str, Any], context: ReferenceCaptureContext) -> dict[str, Any]:
        return _run_text_reference(self, inputs, context)

    def run_reference_text_image_und(
        self, inputs: Mapping[str, Any], context: ReferenceCaptureContext
    ) -> dict[str, Any]:
        return _run_text_image_reference(self, inputs, context)

    def run_reference_image_gen(self, inputs: Mapping[str, Any], context: ReferenceCaptureContext) -> dict[str, Any]:
        return _run_image_gen_reference(self, inputs, context)

    def run_reference_image_edit(self, inputs: Mapping[str, Any], context: ReferenceCaptureContext) -> dict[str, Any]:
        return _run_image_edit_reference(self, inputs, context)


def _bagel_weight_prefixes(visual_gen: bool, visual_und: bool) -> tuple[str, ...]:
    prefixes: tuple[str, ...] = ("language_model",)
    if visual_gen:
        prefixes = ("language_model.", "vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.")
    if visual_und:
        prefixes = (
            *prefixes,
            "vit_model.",
            "connector.",
            "vit_pos_embed.",
        )
    return prefixes


def _normalize_llm_config(llm_config: Qwen2Config) -> None:
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "bos_token_id", 0)


def _ensure_default_rope_init() -> None:
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _default_vae_config() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


def _optional_bagel_config_kwargs(
    *,
    latent_patch_size: int | None,
    max_latent_size: int | None,
    timestep_shift: float | None,
    vit_max_num_patch_per_side: int | None,
    connector_act: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if latent_patch_size is not None:
        values["latent_patch_size"] = latent_patch_size
    if max_latent_size is not None:
        values["max_latent_size"] = max_latent_size
    if timestep_shift is not None:
        values["timestep_shift"] = timestep_shift
    if vit_max_num_patch_per_side is not None:
        values["vit_max_num_patch_per_side"] = vit_max_num_patch_per_side
    if connector_act is not None:
        values["connector_act"] = connector_act
    return values


def _compute_default_rope_parameters(config: Any, device: torch.device | None = None, **kwargs: Any):
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def run_reference_only_recipe(*, recipe_id: str, run_kind: str, case_id: str) -> ParityReport:
    if run_kind != "reference_smoke":
        raise NotImplementedError(f"Unsupported BAGEL reference run kind: {run_kind!r}")
    if recipe_id != "transformers_reference_smoke":
        raise NotImplementedError(f"Unsupported BAGEL reference recipe: {recipe_id!r}")
    _run_transformers_reference_smoke()
    return ParityReport(case_id=case_id, probes=())


def _run_transformers_reference_smoke() -> None:
    from transformers import AutoConfig, AutoModel

    from tests.seed_omni.bagel.transformers import (
        AutoEncoderParams,
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen2Model,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    BagelModel.register_auto_model()

    if Bagel.config_class is not BagelConfig:
        raise AssertionError("BAGEL official reference did not expose BagelConfig.")
    if not isinstance(AutoConfig.for_model("bagel"), BagelConfig):
        raise AssertionError("BAGEL config is not registered with AutoConfig.")
    if AutoModel._model_mapping[BagelConfig] is not BagelModel:
        raise AssertionError("BAGEL model is not registered with AutoModel.")
    if not issubclass(Qwen2Config, Qwen2Model.config_class):
        raise AssertionError("BAGEL Qwen2Config is not compatible with Qwen2Model.")
    if not issubclass(Qwen2Config, Qwen2ForCausalLM.config_class):
        raise AssertionError("BAGEL Qwen2Config is not compatible with Qwen2ForCausalLM.")
    if SiglipVisionModel.config_class is not SiglipVisionConfig:
        raise AssertionError("BAGEL SigLIP reference did not expose SiglipVisionConfig.")
    if AutoEncoderParams.__name__ != "AutoEncoderParams":
        raise AssertionError("BAGEL autoencoder parameters are not exposed.")

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
    reference = BagelModel.from_configs(
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
    result = make_reference_run_output(
        canonical,
        {
            "hidden_state": text_step["hidden_state"].detach().cpu(),
            "logits": text_step["logits"].detach().cpu(),
            "greedy_token": text_step["greedy_token"].detach().cpu(),
        },
    )
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
    vit_transform = ImageTransform(
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
    result = make_reference_run_output(
        canonical,
        {
            "hidden_state": text_step["hidden_state"].detach().cpu(),
            "logits": text_step["logits"].detach().cpu(),
            "greedy_token": text_step["greedy_token"].detach().cpu(),
            "image_embeds_sample": _sample_tensor(image_embeds).detach().cpu(),
            "preprocessed_image_size": torch.tensor(
                [int(preprocessed_image.shape[-1]), int(preprocessed_image.shape[-2])],
                dtype=torch.long,
            ),
        },
    )
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
    result = make_reference_run_output(
        canonical,
        {
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
    )
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
    vae_transform = ImageTransform(
        max_image_size=int(inputs.get("vae_max_image_size", 1024)),
        min_image_size=int(inputs.get("vae_min_image_size", 512)),
        image_stride=int(inputs.get("vae_image_stride", 16)),
        max_pixels=int(inputs.get("vae_max_pixels", 14 * 14 * 9 * 1024)),
    )
    vit_transform = ImageTransform(
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
    result = make_reference_run_output(
        canonical,
        {
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
    )
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
    fixture = _build_train_fixture(
        tokenizer,
        str(inputs["prompt"]),
        device=device,
        dtype=dtype,
        timestep_shift=float(inputs.get("timestep_shift", 3.0)),
        loss_mode=loss_mode,
    )
    batch = _build_train_batch_from_fixture(fixture, device=device, dtype=dtype)
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

    canonical = {"kind": str(fixture["kind"]), "train_fixture": to_cpu(fixture)}
    reference: dict[str, Any] = {
        "train_total_loss": loss.detach().cpu(),
    }
    if ce_loss is not None:
        reference["train_ce_loss"] = ce_loss.detach().cpu()
        reference["train_grad_early_q_proj"] = sample_named_grad(
            model,
            "language_model.model.layers.0.self_attn.q_proj.weight",
        )
        reference["train_grad_lm_head_rows"] = sample_named_grad(
            model,
            "language_model.lm_head.weight",
            rows=torch.unique(batch["packed_label_ids"].detach().cpu()).to(dtype=torch.long),
        )
        if "packed_vit_tokens" in batch:
            reference["train_grad_siglip_q_proj"] = sample_named_grad(
                model,
                "vit_model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            )
    if mse_loss is not None:
        reference["train_mse_loss"] = mse_loss.detach().cpu()
        reference["train_grad_gen_q_proj"] = sample_named_grad(
            model,
            "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
        )
        reference["train_grad_llm2vae"] = sample_named_grad(model, "llm2vae.weight")
    result = make_reference_run_output(canonical, reference)
    context.record_extra("canonical", canonical)
    return result


def _build_train_fixture(
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
        fixture = _prepare_text_image_ce_fixture(token_ids, device=device, dtype=dtype)
    else:
        fixture = _prepare_ce_mse_fixture(token_ids, device=device, dtype=dtype)
    if loss_mode == "mse_only":
        fixture["compute_ce"] = False
    elif loss_mode not in {"ce_mse", "ce_only", "text_image_ce"}:
        raise ValueError(f"Unsupported BAGEL train loss_mode: {loss_mode!r}")
    fixture["kind"] = f"train_{loss_mode}"
    fixture["loss_mode"] = loss_mode
    fixture["prompt"] = text
    fixture["timestep_shift"] = float(timestep_shift)
    return fixture


def _build_train_batch_from_fixture(
    fixture: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    loss_mode = str(fixture["loss_mode"])
    token_ids = fixture["text_token_ids"].to(device=device, dtype=torch.long)
    if loss_mode == "text_image_ce":
        return _prepare_text_image_ce_batch_from_fixture(fixture, token_ids, device=device, dtype=dtype)
    batch, fixed_noise = _prepare_ce_mse_batch_from_fixture(fixture, token_ids, device=device, dtype=dtype)
    _add_flow_noise_controls(batch, fixed_noise=fixed_noise, timestep_shift=float(fixture.get("timestep_shift", 3.0)))
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


def _prepare_ce_mse_fixture(
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "text_token_ids": token_ids,
        "compute_ce": True,
        **synthetic_latent_fixture(device=device, dtype=dtype),
    }


def _prepare_text_image_ce_fixture(
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "text_token_ids": token_ids,
        "compute_ce": True,
        **synthetic_vit_fixture(device=device, dtype=dtype),
    }


def _prepare_ce_mse_batch_from_fixture(
    fixture: Mapping[str, Any],
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor]:
    latent = fixture["target_latent"].to(device=device, dtype=dtype)
    h, w = tuple(int(value) for value in fixture["latent_grid"])
    max_latent_size = int(fixture.get("max_latent_size", 64))
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
        "packed_latent_position_ids": latent_position_ids(h, w, max_latent_size=max_latent_size, device=device),
        "packed_vae_token_indexes": vae_indexes.to(dtype=torch.long),
        "packed_timesteps": fixture["flow_timesteps"].to(device=device, dtype=torch.float32),
        "mse_loss_indexes": torch.zeros(sequence_length, device=device, dtype=torch.bool),
    }
    batch["mse_loss_indexes"][batch["packed_vae_token_indexes"]] = True
    if bool(fixture.get("compute_ce", True)):
        _apply_ce(batch, batch["packed_text_indexes"], token_ids)
    return batch, fixture["flow_noise"].to(device=device, dtype=dtype)


def _prepare_text_image_ce_batch_from_fixture(
    fixture: Mapping[str, Any],
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    packed_vit_tokens = fixture["vit_tokens"].to(device=device, dtype=dtype)
    vit_tokens = int(packed_vit_tokens.shape[0])
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
        "packed_vit_position_ids": fixture["vit_position_ids"].to(device=device, dtype=torch.long),
        "vit_token_seqlens": fixture["vit_token_lens"].to(device=device, dtype=torch.int32),
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
    "BagelModel",
    "ImageTransform",
    "make_reference_image",
    "run_reference_only_recipe",
]
