from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as _WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import (
    _get_added_kv_projections,
    _get_qkv_projections,
)
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor_scale_grad,
)
from .....utils import logging
from .configuration_wan_transformer import WanTransformer3DModelConfig


logger = logging.get_logger(__name__)

_VEOMNI_SP_ATTN_IMPLS = frozenset(
    {
        "veomni_flash_attention_2_with_sp",
        "veomni_flash_attention_3_with_sp",
        "veomni_flash_attention_4_with_sp",
    }
)


def _get_flash_attn_fn(attn_implementation: str):
    """Return the flash-attention callable matching *attn_implementation*."""
    if attn_implementation in ("veomni_flash_attention_2_with_sp", "flash_attention_2"):
        from flash_attn import flash_attn_func

        return flash_attn_func
    elif attn_implementation in ("veomni_flash_attention_3_with_sp", "flash_attention_3"):
        from flash_attn_interface import flash_attn_func

        return flash_attn_func
    elif attn_implementation == "veomni_flash_attention_4_with_sp":
        from flash_attn.cute import flash_attn_func

        return flash_attn_func
    return None


class WanSPAttnProcessor:
    """Flash-attention processor with Ulysses sequence-parallelism for WanTransformer.

    For self-attention this processor performs the standard Ulysses AllToAll before
    and after flash attention so that each rank holds a full-sequence slice with
    scattered heads.  For cross-attention the query slice is kept local while K/V
    are replicated across ranks, which is equivalent to skipping the AllToAll.

    The ``apply_rotary_emb`` logic is intentionally identical to the one inside
    ``WanAttnProcessor`` so that RoPE is applied correctly to the local sequence
    slice *before* the AllToAll.
    """

    def __init__(self, flash_attn_fn):
        self.flash_attn_fn = flash_attn_fn

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Identical to the inner ``apply_rotary_emb`` in diffusers WanAttnProcessor."""
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        # I2V: the first part of encoder_hidden_states holds image context.
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape to (B, seq, heads, head_dim)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply RoPE on the per-rank sequence slice *before* Ulysses AllToAll so
        # that each token gets the embedding for its global position index.
        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, *rotary_emb)
            key = self._apply_rotary_emb(key, *rotary_emb)

        # Ulysses SP for self-attention: scatter heads, gather full sequence.
        # Cross-attention skips AllToAll because encoder states are replicated.
        ulysses_enabled = get_parallel_state().ulysses_enabled
        if ulysses_enabled and not is_cross_attention:
            ulysses_group = get_parallel_state().ulysses_group
            query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2, group=ulysses_group)
            key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2, group=ulysses_group)
            value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=ulysses_group)

        # I2V: additional cross-attention over image tokens (no Ulysses SP needed).
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = self.flash_attn_fn(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states_out = self.flash_attn_fn(query, key, value)

        # Inverse AllToAll: scatter sequence, gather heads back.
        if ulysses_enabled and not is_cross_attention:
            hidden_states_out = gather_heads_scatter_seq(hidden_states_out, seq_dim=1, head_dim=2, group=ulysses_group)

        hidden_states_out = hidden_states_out.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states_out = hidden_states_out + hidden_states_img

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out


def apply_veomni_wan_sp_patch() -> None:
    """Monkey-patch ``_WanTransformer3DModel.forward`` with Ulysses SP support.

    The patch is structurally identical to the original diffusers forward but
    inserts two SP operations:

    1. **Sequence slice** (with gradient scaling) after patchification – each SP
       rank processes a contiguous chunk of video tokens.
    2. **Sequence gather** before the output head – all ranks see the full output
       so that the loss is identical across SP ranks.

    Slicing only takes effect when BOTH ``ulysses_enabled`` is True AND an SP-aware
    attention implementation (``veomni_flash_attention_*_with_sp``) is configured.
    Without the SP attention processor the required AllToAll in self-attention
    would be absent, making sequence slicing incorrect.
    """
    _original_forward = _WanTransformer3DModel.forward

    def _sp_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary position embeddings for the full sequence
        rotary_emb = self.rope(hidden_states)
        # 2. Patch embedding: (B, C, F, H, W) → (B, seq, inner_dim)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 3. Condition embedding
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # SP: slice the sequence only when a SP-aware attention processor is
        # installed. Without ``WanSPAttnProcessor`` the self-attention lacks the
        # AllToAll that is required for Ulysses SP to be correct.
        attn_impl = getattr(self.config, "_attn_implementation", None)
        _sp_active = get_parallel_state().ulysses_enabled and (attn_impl in _VEOMNI_SP_ATTN_IMPLS)

        if _sp_active:
            hidden_states = slice_input_tensor_scale_grad(hidden_states, dim=1)

            # Slice rotary embeddings to the local rank's positions (no gradient).
            freqs_cos, freqs_sin = rotary_emb
            ulysses_size = get_parallel_state().ulysses_size
            ulysses_rank = get_parallel_state().ulysses_rank
            seq_len = freqs_cos.shape[1]
            chunk = seq_len // ulysses_size
            freqs_cos = freqs_cos[:, ulysses_rank * chunk : (ulysses_rank + 1) * chunk]
            freqs_sin = freqs_sin[:, ulysses_rank * chunk : (ulysses_rank + 1) * chunk]
            rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # SP: gather before output head – every rank holds the full sequence so
        # that the loss is identical across SP ranks.
        if _sp_active:
            hidden_states = gather_outputs(hidden_states, gather_dim=1)

        # 5. Output: norm → projection → unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # Return a plain tensor to simplify the DiT wrapper's forward.
        return output

    _WanTransformer3DModel.forward = _sp_forward
    logger.info_rank0("Applied VeOmni SP patch to WanTransformer3DModel.forward.")


def _setup_sp_attention(model: _WanTransformer3DModel, attn_implementation: str) -> None:
    """Install ``WanSPAttnProcessor`` in every transformer block of *model*."""
    flash_attn_fn = _get_flash_attn_fn(attn_implementation)
    if flash_attn_fn is None:
        logger.warning_rank0(
            f"WanTransformer3DModel: could not resolve flash-attn function for "
            f"{attn_implementation!r}. Falling back to default attention processor."
        )
        return
    sp_processor = WanSPAttnProcessor(flash_attn_fn=flash_attn_fn)
    for block in model.blocks:
        block.attn1.set_processor(sp_processor)
        block.attn2.set_processor(sp_processor)
    logger.info_rank0(f"WanTransformer3DModel: installed SP-aware attention ({attn_implementation}).")


@dataclass
class WanModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class WanTransformer3DModel(PreTrainedModel, _WanTransformer3DModel):
    config_class = WanTransformer3DModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: WanTransformer3DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        # Remove VeOmni-specific kwargs before passing to the diffusers init.
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _WanTransformer3DModel.__init__(self, **config.to_diffuser_dict())
        self.config: WanTransformer3DModelConfig = config
        self.config.tie_word_embeddings = False

        # Install the SP-aware attention processor when an SP implementation is
        # requested.  The patched forward (applied at module import time) will
        # activate sequence slicing/gathering only when this processor is present.
        attn_impl = getattr(self.config, "_attn_implementation", None)
        if attn_impl in _VEOMNI_SP_ATTN_IMPLS:
            _setup_sp_attention(self, attn_impl)
        elif attn_impl is not None:
            logger.info_rank0(
                f"WanTransformer3DModel: attn_implementation={attn_impl!r} does not enable "
                "SP-aware attention. Use veomni_flash_attention_*_with_sp for SP training."
            )

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def forward(
        self,
        latents: torch.Tensor,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        training_target: torch.Tensor,
    ):
        per_sample_losses = []
        predictions = []
        for hidden_state, ts, enc_hs, target in zip(hidden_states, timestep, encoder_hidden_states, training_target):
            # Call the SP-patched diffusers forward for each sample.
            prediction = _WanTransformer3DModel.forward(
                self, hidden_states=hidden_state, timestep=ts, encoder_hidden_states=enc_hs
            )
            predictions.append(prediction)
            per_sample_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
            per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)
        loss = torch.stack(per_sample_losses).mean()
        return WanModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _WanTransformer3DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return _WanTransformer3DModel.from_pretrained(path, **kwargs)
