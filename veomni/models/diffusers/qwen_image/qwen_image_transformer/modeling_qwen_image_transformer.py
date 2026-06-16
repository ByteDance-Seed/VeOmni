from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as _QwenImageTransformer2DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
)
from diffusers.utils import apply_lora_scale
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)
from .....utils import logging
from .configuration_qwen_image_transformer import QWEN_IMAGE_INIT_SIGNATURE, QwenImageTransformer2DModelConfig


logger = logging.get_logger(__name__)


def _pad_seq(x: torch.Tensor, dim: int, pad_size: int, value: float = 0) -> torch.Tensor:
    """Right-pad ``x`` along ``dim`` with ``pad_size`` entries of ``value``."""
    if pad_size == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    if value == 0:
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    else:
        pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


class QwenImageSPAttnProcessor:
    """Joint dual-stream attention processor with Ulysses sequence parallelism.

    Mirrors diffusers' ``QwenDoubleStreamAttnProcessor2_0`` but, when SP is
    enabled, gathers the full sequence / scatters heads (all-to-all) on each
    stream independently before the joint attention, then performs the inverse
    all-to-all on the outputs. Because each stream is gathered independently and
    concatenated in the original ``[text, image]`` order, the full joint
    attention mask stays valid without any reordering.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenImageSPAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,  # Image stream
        encoder_hidden_states: torch.Tensor = None,  # Text stream
        encoder_hidden_states_mask: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenImageSPAttnProcessor requires encoder_hidden_states (text stream)")

        sp_enabled = get_parallel_state().sp_enabled
        ulysses_group = get_parallel_state().ulysses_group if sp_enabled else None

        # QKV projections for both streams.
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Ulysses all-to-all: gather full sequence, scatter heads. Done per
        # stream and BEFORE RoPE so RoPE is applied on the full (un-sliced)
        # positions using the full frequency tables.
        if sp_enabled:
            img_query = gather_seq_scatter_heads(img_query, seq_dim=1, head_dim=2, group=ulysses_group)
            img_key = gather_seq_scatter_heads(img_key, seq_dim=1, head_dim=2, group=ulysses_group)
            img_value = gather_seq_scatter_heads(img_value, seq_dim=1, head_dim=2, group=ulysses_group)
            txt_query = gather_seq_scatter_heads(txt_query, seq_dim=1, head_dim=2, group=ulysses_group)
            txt_key = gather_seq_scatter_heads(txt_key, seq_dim=1, head_dim=2, group=ulysses_group)
            txt_value = gather_seq_scatter_heads(txt_value, seq_dim=1, head_dim=2, group=ulysses_group)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Full (post-gather) text length used to split the joint output.
        seq_txt = txt_query.shape[1]

        # Joint attention, order [text, image].
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # joint_hidden_states: (B, joint_seq, heads_local, head_dim)
        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]

        # Inverse all-to-all: scatter sequence, gather heads back (per stream).
        if sp_enabled:
            txt_attn_output = gather_heads_scatter_seq(txt_attn_output, seq_dim=1, head_dim=2, group=ulysses_group)
            img_attn_output = gather_heads_scatter_seq(img_attn_output, seq_dim=1, head_dim=2, group=ulysses_group)

        txt_attn_output = txt_attn_output.flatten(2, 3).to(joint_query.dtype)
        img_attn_output = img_attn_output.flatten(2, 3).to(joint_query.dtype)

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


@apply_lora_scale("attention_kwargs")
def QwenImageTransformer2DModel_forward(
    self: _QwenImageTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
):
    """SP-aware ``QwenImageTransformer2DModel.forward``.

    Structurally identical to the diffusers forward, with Ulysses SP inserted:
    both the image and text streams are sliced across SP ranks before the
    transformer blocks (in-model slicing -- the model receives the *full*
    sequence), the joint self-attention performs all-to-all inside
    :class:`QwenImageSPAttnProcessor`, and the image stream is gathered back
    before the output head so that the prediction is identical across SP ranks.
    """
    sp_enabled = get_parallel_state().sp_enabled

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

    batch_size, image_seq_len = hidden_states.shape[:2]
    txt_seq_len_full = encoder_hidden_states.shape[1]

    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}

    if sp_enabled:
        sp_group = get_parallel_state().sp_group
        sp_size = get_parallel_state().sp_size
        num_heads = self.config.num_attention_heads
        if num_heads % sp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by the sequence-parallel size ({sp_size})."
            )

        img_pad = (sp_size - image_seq_len % sp_size) % sp_size
        txt_pad = (sp_size - txt_seq_len_full % sp_size) % sp_size

        # Build the (padded) joint attention mask on FULL lengths so that, after
        # the per-stream all-to-all inside attention, padded positions on both
        # streams are masked out. Order matches the processor: [text, image].
        if encoder_hidden_states_mask is not None:
            text_mask = encoder_hidden_states_mask.to(torch.bool)
        else:
            text_mask = torch.ones((batch_size, txt_seq_len_full), dtype=torch.bool, device=hidden_states.device)
        text_mask = _pad_seq(text_mask, dim=1, pad_size=txt_pad, value=False)
        image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
        image_mask = _pad_seq(image_mask, dim=1, pad_size=img_pad, value=False)
        block_attention_kwargs["attention_mask"] = torch.cat([text_mask, image_mask], dim=1)

        # Pad streams + RoPE to a multiple of sp_size, then slice across ranks.
        hidden_states = _pad_seq(hidden_states, dim=1, pad_size=img_pad, value=0)
        encoder_hidden_states = _pad_seq(encoder_hidden_states, dim=1, pad_size=txt_pad, value=0)
        img_freqs, txt_freqs = image_rotary_emb
        img_freqs = _pad_seq(img_freqs, dim=0, pad_size=img_pad, value=0)
        txt_freqs = _pad_seq(txt_freqs, dim=0, pad_size=txt_pad, value=0)
        image_rotary_emb = (img_freqs, txt_freqs)
        if modulate_index is not None:
            modulate_index = _pad_seq(modulate_index, dim=1, pad_size=img_pad, value=0)

        hidden_states = slice_input_tensor(hidden_states, dim=1, group=sp_group)
        encoder_hidden_states = slice_input_tensor(encoder_hidden_states, dim=1, group=sp_group)
        if modulate_index is not None:
            modulate_index = slice_input_tensor(modulate_index, dim=1, group=sp_group)
    elif encoder_hidden_states_mask is not None:
        image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
        block_attention_kwargs["attention_mask"] = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                None,
                temb,
                image_rotary_emb,
                block_attention_kwargs,
                modulate_index,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    # Gather the image stream and drop padding before the output head, so every
    # SP rank holds the full prediction and the loss is identical across ranks.
    if sp_enabled:
        hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
        if hidden_states.shape[1] != image_seq_len:
            hidden_states = hidden_states[:, :image_seq_len]

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def apply_veomni_qwen_image_transformer_patch() -> None:
    """Monkey-patch ``_QwenImageTransformer2DModel.forward`` with Ulysses SP support.

    The patched forward is structurally identical to the diffusers forward but
    slices the image/text streams across SP ranks before the transformer blocks
    and gathers the image stream back before the output head. SP only takes
    effect when ``get_parallel_state().sp_enabled`` is True; otherwise the
    forward and :class:`QwenImageSPAttnProcessor` behave exactly like the
    upstream diffusers implementation.
    """
    _QwenImageTransformer2DModel.forward = QwenImageTransformer2DModel_forward
    logger.info_rank0("Applied VeOmni SP patch to QwenImageTransformer2DModel.forward.")


@dataclass
class QwenImageModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class _QwenImageTransformerInitShim(_QwenImageTransformer2DModel):
    """Avoid constructing the large diffusers default model during PreTrainedModel init."""

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class QwenImageTransformer2DModel(PreTrainedModel, _QwenImageTransformerInitShim):
    config_class = QwenImageTransformer2DModelConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]

    def __init__(self, config: QwenImageTransformer2DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        if hasattr(self, "_internal_dict"):
            del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _QwenImageTransformer2DModel.__init__(self, **config.to_diffuser_dict())
        self.config: QwenImageTransformer2DModelConfig = config
        self.config.tie_word_embeddings = False

        # Install the Ulysses-SP joint-attention processor on every block. It is
        # a no-op (plain diffusers joint attention) when SP is disabled.
        sp_processor = QwenImageSPAttnProcessor()
        for block in self.transformer_blocks:
            block.attn.set_processor(sp_processor)

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            if length is None:
                return []
            return [None] * length
        if isinstance(value, list):
            return value
        if length is not None and isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == length:
            return [value[idx : idx + 1] for idx in range(length)]
        return [value]

    @staticmethod
    def _normalize_img_shapes(sample_img_shapes: Any) -> list[list[tuple[int, int, int]]]:
        if isinstance(sample_img_shapes, torch.Tensor):
            sample_img_shapes = sample_img_shapes.detach().cpu().tolist()

        def _is_fhw(value: Any) -> bool:
            return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(x, int) for x in value)

        if _is_fhw(sample_img_shapes):
            return [[tuple(int(x) for x in sample_img_shapes)]]

        if isinstance(sample_img_shapes, list):
            if len(sample_img_shapes) == 1 and _is_fhw(sample_img_shapes[0]):
                return [[tuple(int(x) for x in sample_img_shapes[0])]]
            if len(sample_img_shapes) == 1 and isinstance(sample_img_shapes[0], list):
                nested = sample_img_shapes[0]
                if all(_is_fhw(item) for item in nested):
                    return [[tuple(int(x) for x in item) for item in nested]]
            if all(_is_fhw(item) for item in sample_img_shapes):
                return [[tuple(int(x) for x in item) for item in sample_img_shapes]]

        raise ValueError(f"Unsupported img_shapes format: {sample_img_shapes}")

    def predict_noise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        img_shapes: Any = None,
        guidance: torch.Tensor | None = None,
        additional_t_cond: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        """Single raw noise prediction via the diffusers backbone.

        This is the sole wrapper around the underlying diffusers ``forward``:
        it casts inputs to the model dtype and forwards everything else
        through. Both the supervised loss path and the inference / RL path
        in :meth:`forward` go through this method to avoid duplicated logic.
        """
        param_dtype = self.dtype
        return _QwenImageTransformer2DModel.forward(
            self,
            hidden_states=hidden_states.to(dtype=param_dtype),
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(dtype=param_dtype),
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=img_shapes,
            guidance=guidance,
            additional_t_cond=additional_t_cond,
            return_dict=return_dict,
        )

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor] | None = None,
        img_shapes: Any = None,
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | list[torch.Tensor] | None = None,
        additional_t_cond: torch.Tensor | list[torch.Tensor] | None = None,
        latents: torch.Tensor | list[torch.Tensor] | None = None,
        return_dict: bool = True,
    ):
        if training_target is None:
            return self.predict_noise(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                img_shapes=img_shapes,
                guidance=guidance,
                additional_t_cond=additional_t_cond,
                return_dict=return_dict,
            )

        if img_shapes is None:
            raise ValueError("Qwen-Image supervised training forward requires `img_shapes`.")

        hidden_states_list = self._as_list(hidden_states)
        sample_count = len(hidden_states_list)
        timestep_list = self._as_list(timestep, sample_count)
        encoder_hidden_states_list = self._as_list(encoder_hidden_states, sample_count)
        target_list = self._as_list(training_target, sample_count)
        img_shapes_list = self._as_list(img_shapes, sample_count)
        mask_list = self._as_list(encoder_hidden_states_mask, sample_count)
        guidance_list = self._as_list(guidance, sample_count)
        additional_t_cond_list = self._as_list(additional_t_cond, sample_count)

        per_sample_losses = []
        predictions = []
        for (
            sample_hs,
            sample_ts,
            sample_enc_hs,
            sample_target,
            sample_img_shapes,
            sample_mask,
            sample_guidance,
            sample_add_t_cond,
        ) in zip(
            hidden_states_list,
            timestep_list,
            encoder_hidden_states_list,
            target_list,
            img_shapes_list,
            mask_list,
            guidance_list,
            additional_t_cond_list,
        ):
            prediction = self.predict_noise(
                hidden_states=sample_hs,
                timestep=sample_ts,
                encoder_hidden_states=sample_enc_hs,
                encoder_hidden_states_mask=sample_mask,
                img_shapes=self._normalize_img_shapes(sample_img_shapes),
                guidance=sample_guidance,
                additional_t_cond=sample_add_t_cond,
            )[0]
            predictions.append(prediction)
            per_sample_loss = F.mse_loss(prediction.float(), sample_target.float(), reduction="none")
            per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)

        loss = torch.stack(per_sample_losses).mean()
        return QwenImageModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _QwenImageTransformer2DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        diffusers_model = _QwenImageTransformer2DModel.from_pretrained(path, **kwargs)
        diffusers_model.__class__ = cls

        valid_keys = set(QWEN_IMAGE_INIT_SIGNATURE.parameters) - {"self"}
        diffusers_cfg = dict(diffusers_model.config)
        veomni_cfg = cls.config_class(**{k: v for k, v in diffusers_cfg.items() if k in valid_keys})
        diffusers_model.config = veomni_cfg
        diffusers_model.config.tie_word_embeddings = False

        # __init__ is bypassed by from_pretrained, so install the SP processor here too.
        sp_processor = QwenImageSPAttnProcessor()
        for block in diffusers_model.transformer_blocks:
            block.attn.set_processor(sp_processor)
        return diffusers_model
