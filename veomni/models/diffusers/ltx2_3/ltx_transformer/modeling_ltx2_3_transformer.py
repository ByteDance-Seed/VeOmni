from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, replace

import torch
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)
from .....utils import logging
from .configuration_ltx2_3_transformer import LTXVideoTransformerModelConfig, diffusers_version


logger = logging.get_logger(__name__)

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors(time=8, height=32, width=32)
AUDIO_MEL_BINS = 16
AUDIO_CHANNELS = 8
DEFAULT_FPS = 24

_VEOMNI_SP_ATTN_IMPLS = frozenset(
    {
        "veomni_flash_attention_2_with_sp",
        "veomni_flash_attention_3_with_sp",
        "veomni_flash_attention_4_with_sp",
    }
)


def LTXSPAttention_forward(
    self: Attention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pe: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool = False,
    *,
    sp_valid_length: int | None = None,
    sp_context_length: int | None = None,
) -> torch.Tensor:
    is_cross_attention = context is not None

    # Audio<->video cross-attention attends to the other modality, which is sliced across SP
    # ranks like x. Gather it (and its key RoPE) back to the full, unpadded sequence.
    context_length = sp_context_length
    if is_cross_attention and context_length is not None and get_parallel_state().sp_enabled:
        context = gather_outputs(context, gather_dim=1, padding_dim=1, unpad_dim_size=context_length)
        if k_pe is not None:
            k_pe = tuple(gather_outputs(t, gather_dim=2, padding_dim=2, unpad_dim_size=context_length) for t in k_pe)

    context = x if context is None else context
    use_attention = not all_perturbed
    v = self.to_v(context)

    if not use_attention:
        out = v
    else:
        q = self.to_q(x)
        k = self.to_k(context)
        q, k = self.preattention_function(q, k, self, mask, pe, k_pe)

        sp_enabled = get_parallel_state().sp_enabled and not is_cross_attention

        if sp_enabled:
            heads = self.heads
            dim_head = self.dim_head
            ulysses_group = get_parallel_state().ulysses_group
            ulysses_size = get_parallel_state().ulysses_size

            q = q.unflatten(-1, (heads, dim_head))
            k = k.unflatten(-1, (heads, dim_head))
            v_sp = v.unflatten(-1, (heads, dim_head))

            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=ulysses_group)
            k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, group=ulysses_group)
            v_sp = gather_seq_scatter_heads(v_sp, seq_dim=1, head_dim=2, group=ulysses_group)

            # Drop the Ulysses tail padding so no backend attends to it; restored below.
            padded_length = q.shape[1]
            valid_length = sp_valid_length or padded_length
            q, k, v_sp = q[:, :valid_length], k[:, :valid_length], v_sp[:, :valid_length]

            sp_heads = heads // ulysses_size

            q = q.flatten(2, 3)
            k = k.flatten(2, 3)
            v_sp = v_sp.flatten(2, 3)
        else:
            sp_heads = self.heads
            v_sp = v

        if mask is None:
            out = self.attention_function(q, k, v_sp, sp_heads)
        else:
            out = self.masked_attention_function(q, k, v_sp, sp_heads, mask)

        if sp_enabled:
            out = out.unflatten(-1, (sp_heads, self.dim_head))
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, padded_length - valid_length))
            out = gather_heads_scatter_seq(out, seq_dim=1, head_dim=2, group=ulysses_group)
            out = out.flatten(2, 3)

        if perturbation_mask is not None:
            out = out * perturbation_mask + v * (1 - perturbation_mask)

    if self.to_gate_logits is not None:
        out = self.gated_attention_function(x, out, self)

    return self.to_out(out)


def _slice_transformer_args_for_sp(args, group):
    """Slice every per-token field of ``args`` the way ``slice_input_tensor`` slices ``x``.

    ``slice_input_tensor`` pads to a multiple of the SP size, so RoPE, per-token timesteps and
    the AV cross-attention AdaLN inputs must be sliced the same way to stay aligned with ``x``.
    Tensors whose sequence axis is not the token axis (e.g. per-sample or per-prompt) are kept.
    """
    num_tokens = args.x.shape[1]

    def seq(t, dim):
        return slice_input_tensor(t, dim=dim, group=group) if t.shape[dim] == num_tokens else t

    updates = {"x": seq(args.x, 1), "positional_embeddings": tuple(seq(t, 2) for t in args.positional_embeddings)}
    if args.cross_positional_embeddings is not None:
        updates["cross_positional_embeddings"] = tuple(seq(t, 2) for t in args.cross_positional_embeddings)
    for name in ("timesteps", "prompt_timestep", "cross_scale_shift_timestep", "cross_gate_timestep"):
        t = getattr(args, name)
        if t is not None and t.ndim > 1:
            updates[name] = seq(t, 1)
    return replace(args, **updates)


def LTXVideoModel_forward(
    self: LTXModel,
    video: Modality | None,
    audio: Modality | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
    audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

    use_sp = get_parallel_state().sp_enabled
    video_len = video_args.x.shape[1] if video_args is not None else None
    audio_len = audio_args.x.shape[1] if audio_args is not None else None
    if use_sp:
        sp_group = get_parallel_state().sp_group
        if video_args is not None:
            video_args = _slice_transformer_args_for_sp(video_args, sp_group)
        if audio_args is not None:
            audio_args = _slice_transformer_args_for_sp(audio_args, sp_group)

    video_out, audio_out = self._process_transformer_blocks(
        video=video_args,
        audio=audio_args,
        perturbations=perturbations,
        sp_video_length=video_len if use_sp else None,
        sp_audio_length=audio_len if use_sp else None,
    )

    if use_sp and video_out is not None:
        video_out = replace(
            video_out, x=gather_outputs(video_out.x, gather_dim=1, padding_dim=1, unpad_dim_size=video_len)
        )
    if use_sp and audio_out is not None:
        audio_out = replace(
            audio_out, x=gather_outputs(audio_out.x, gather_dim=1, padding_dim=1, unpad_dim_size=audio_len)
        )

    vx = (
        self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
        )
        if video_out is not None
        else None
    )
    ax = (
        self._process_output(
            self.audio_scale_shift_table,
            self.audio_norm_out,
            self.audio_proj_out,
            audio_out.x,
            audio_out.embedded_timestep,
        )
        if audio_out is not None
        else None
    )
    return vx, ax


@dataclass
class LTXVideoModelOutput(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: list[torch.FloatTensor] | None = None
    audio_predictions: list[torch.FloatTensor] | None = None


class _LTXModelInitShim(LTXModel):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class LTXVideoTransformerModel(PreTrainedModel, _LTXModelInitShim):
    config_class = LTXVideoTransformerModelConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["BasicAVTransformerBlock"]
    _checkpoint_conversion_mapping = {
        "^model\\.diffusion_model\\.": "",
    }

    def __init__(self, config: LTXVideoTransformerModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        caption_projection = None
        if not config.caption_proj_before_connector:
            caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels,
                hidden_size=inner_dim,
            )

        audio_caption_projection = None
        if config.with_audio and not config.caption_proj_before_connector:
            audio_inner_dim = config.audio_num_attention_heads * config.audio_attention_head_dim
            audio_caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels,
                hidden_size=audio_inner_dim,
            )

        model_type = LTXModelType.AudioVideo if config.with_audio else LTXModelType.VideoOnly

        LTXModel.__init__(
            self,
            model_type=model_type,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            cross_attention_dim=config.cross_attention_dim,
            norm_eps=config.norm_eps,
            positional_embedding_theta=config.positional_embedding_theta,
            positional_embedding_max_pos=config.positional_embedding_max_pos,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            use_middle_indices_grid=config.use_middle_indices_grid,
            audio_num_attention_heads=config.audio_num_attention_heads,
            audio_attention_head_dim=config.audio_attention_head_dim,
            audio_in_channels=config.audio_in_channels,
            audio_out_channels=config.audio_out_channels,
            audio_cross_attention_dim=config.audio_cross_attention_dim,
            audio_positional_embedding_max_pos=config.audio_positional_embedding_max_pos,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            rope_type=LTXRopeType(config.rope_type),
            double_precision_rope=config.frequencies_precision == "float64",
            apply_gated_attention=config.apply_gated_attention,
            caption_projection=caption_projection,
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.cross_attention_adaln,
        )
        self.config: LTXVideoTransformerModelConfig = config
        self.config.tie_word_embeddings = False
        self.gradient_checkpointing = False

    def _init_weights(self, module: torch.nn.Module) -> None:
        for attr in (
            "scale_shift_table",
            "audio_scale_shift_table",
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_video",
            "prompt_scale_shift_table",
            "audio_prompt_scale_shift_table",
        ):
            param = getattr(module, attr, None)
            if param is not None and isinstance(param, torch.nn.Parameter):
                torch.nn.init.zeros_(param)
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        timestep: list[torch.Tensor],
        encoder_hidden_states: list[torch.Tensor],
        video_loss_mask: list[torch.Tensor] | None = None,
        context_mask: list[torch.Tensor] | None = None,
        audio_hidden_states: list[torch.Tensor] | None = None,
        audio_timestep: list[torch.Tensor] | None = None,
        audio_encoder_hidden_states: list[torch.Tensor] | None = None,
        fps: list[float] | None = None,
        training_target: list[torch.Tensor] | None = None,
        audio_training_target: list[torch.Tensor] | None = None,
        audio_loss_mask: list[torch.Tensor] | None = None,
        ref_seq_len: list[int] | None = None,
    ):
        video_patchifier = VideoLatentPatchifier(patch_size=1)
        audio_patchifier = AudioPatchifier(patch_size=1) if self.config.with_audio else None
        predictions = []
        audio_predictions = []

        model_device = next(self.parameters()).device
        compute_dtype = next(self.parameters()).dtype

        for sample_idx, (hidden_state, ts, enc_hs) in enumerate(zip(hidden_states, timestep, encoder_hidden_states)):
            hidden_state = hidden_state.to(device=model_device, dtype=compute_dtype)
            ts = ts.to(device=model_device, dtype=compute_dtype)
            enc_hs = enc_hs.to(device=model_device, dtype=compute_dtype)

            latent_shape = VideoLatentShape.from_torch_shape(hidden_state.shape)

            latent_tokens = video_patchifier.patchify(hidden_state)
            num_tokens = latent_tokens.shape[1]

            sigma = ts
            if sigma.ndim == 0:
                sigma = sigma.unsqueeze(0)

            sample_vlm = video_loss_mask[sample_idx] if video_loss_mask is not None else None
            if sample_vlm is not None:
                sample_vlm = sample_vlm.to(device=model_device, dtype=torch.bool)
                timesteps = torch.where(
                    sample_vlm.unsqueeze(0),
                    sigma.unsqueeze(1).expand(1, num_tokens),
                    torch.zeros(1, num_tokens, device=sigma.device, dtype=compute_dtype),
                )
            else:
                timesteps = sigma.unsqueeze(1).expand(1, num_tokens)

            sample_ref_seq_len = ref_seq_len[sample_idx] if ref_seq_len is not None else 0

            if sample_ref_seq_len > 0:
                B_hs, C_hs, F_total, H_hs, W_hs = hidden_state.shape
                ref_frames = sample_ref_seq_len // (H_hs * W_hs)

                ref_hs = hidden_state[:, :, :ref_frames, :, :]
                target_hs = hidden_state[:, :, ref_frames:, :, :]

                ref_shape = VideoLatentShape.from_torch_shape(ref_hs.shape)
                target_shape = VideoLatentShape.from_torch_shape(target_hs.shape)

                ref_coords = video_patchifier.get_patch_grid_bounds(ref_shape, device=hidden_state.device)
                ref_pos = get_pixel_coords(ref_coords, VIDEO_SCALE_FACTORS, causal_fix=True)

                target_coords = video_patchifier.get_patch_grid_bounds(target_shape, device=hidden_state.device)
                target_pos = get_pixel_coords(target_coords, VIDEO_SCALE_FACTORS, causal_fix=True)

                positions = torch.cat([ref_pos, target_pos], dim=2)
            else:
                latent_coords = video_patchifier.get_patch_grid_bounds(latent_shape, device=hidden_state.device)
                positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, causal_fix=True)

            positions = positions.to(device=model_device, dtype=compute_dtype)
            sample_fps = fps[sample_idx] if fps is not None else DEFAULT_FPS
            positions[:, 0, ...] = positions[:, 0, ...] / sample_fps

            sample_ctx_mask = context_mask[sample_idx] if context_mask is not None else None
            if sample_ctx_mask is not None:
                sample_ctx_mask = sample_ctx_mask.to(device=model_device)

            video_modality = Modality(
                latent=latent_tokens,
                sigma=sigma,
                timesteps=timesteps,
                positions=positions,
                context=enc_hs,
                context_mask=sample_ctx_mask,
            )

            audio_modality = None
            if self.config.with_audio and audio_hidden_states is not None and audio_timestep is not None:
                sample_audio_hs = audio_hidden_states[sample_idx]
                sample_audio_ts = audio_timestep[sample_idx]
                sample_audio_hs = sample_audio_hs.to(device=model_device, dtype=compute_dtype)
                sample_audio_ts = sample_audio_ts.to(device=model_device, dtype=compute_dtype)

                audio_latent_shape = AudioLatentShape.from_torch_shape(sample_audio_hs.shape)
                audio_tokens = audio_patchifier.patchify(sample_audio_hs)
                audio_num_tokens = audio_tokens.shape[1]

                audio_sigma = sample_audio_ts
                if audio_sigma.ndim == 0:
                    audio_sigma = audio_sigma.unsqueeze(0)
                audio_timesteps_per_token = audio_sigma.unsqueeze(1).expand(1, audio_num_tokens)

                audio_coords = audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=model_device)
                audio_positions = audio_coords.to(device=model_device, dtype=compute_dtype)

                audio_enc_hs = (
                    audio_encoder_hidden_states[sample_idx] if audio_encoder_hidden_states is not None else enc_hs
                )
                audio_enc_hs = audio_enc_hs.to(device=model_device, dtype=compute_dtype)

                audio_modality = Modality(
                    latent=audio_tokens,
                    sigma=audio_sigma,
                    timesteps=audio_timesteps_per_token,
                    positions=audio_positions,
                    context=audio_enc_hs,
                    context_mask=sample_ctx_mask,
                )

            perturbations = BatchedPerturbationConfig.empty(1)

            vx, ax = LTXModel.forward(self, video=video_modality, audio=audio_modality, perturbations=perturbations)

            prediction = video_patchifier.unpatchify(vx, latent_shape)
            predictions.append(prediction)

            if ax is not None and audio_patchifier is not None:
                audio_pred = audio_patchifier.unpatchify(ax, audio_latent_shape)
                audio_predictions.append(audio_pred)

        loss = None
        if training_target is not None:
            loss_dict = compute_ltx2_loss(
                predictions,
                training_target,
                video_loss_mask,
                audio_predictions=audio_predictions if audio_predictions else None,
                audio_training_targets=audio_training_target,
            )
            loss = loss_dict

        return LTXVideoModelOutput(
            loss=loss,
            predictions=predictions,
            audio_predictions=audio_predictions if audio_predictions else None,
        )

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)

        config_dict = self.config.to_diffuser_dict()
        config_dict["_class_name"] = "LTXVideoTransformerModel"
        config_dict["_diffusers_version"] = diffusers_version

        PreTrainedModel.save_pretrained(self, path, **kwargs)

        config_path = os.path.join(path, "config.json")
        os.makedirs(path, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from ....loader import get_model_config

        kwargs.pop("trust_remote_code", None)
        config = get_model_config(path, **kwargs)
        model = cls._from_config(config)
        return model


def compute_ltx2_loss(
    predictions: list[torch.Tensor],
    training_targets: list[torch.Tensor],
    video_loss_masks: list[torch.Tensor] | None,
    audio_predictions: list[torch.Tensor] | None = None,
    audio_training_targets: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    per_sample_losses = []

    for i, (prediction, target) in enumerate(zip(predictions, training_targets)):
        prediction = prediction.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        per_element_loss = (prediction - target).pow(2)

        B, C, F, H, W = prediction.shape

        sample_vlm = None
        if video_loss_masks is not None and i < len(video_loss_masks):
            sample_vlm = video_loss_masks[i]

        if sample_vlm is not None:
            sample_vlm = sample_vlm.to(device=prediction.device, dtype=torch.bool)
            mask_batch = sample_vlm.numel() // (F * H * W)
            assert mask_batch == B, f"Loss mask batch size {mask_batch} != prediction batch size {B}"
            loss_mask = sample_vlm.view(B, 1, F, H, W).float()
            masked_loss = per_element_loss * loss_mask
            valid_count = loss_mask.reshape(B, -1).sum(dim=-1).clamp(min=1e-8)
            per_sample_loss = masked_loss.reshape(B, C, -1).sum(dim=-1).mean(dim=-1) / valid_count
        else:
            per_sample_loss = per_element_loss.reshape(B, -1).mean(dim=1)

        if audio_predictions is not None and audio_training_targets is not None and i < len(audio_predictions):
            audio_pred = audio_predictions[i].to(dtype=torch.float32)
            audio_target = audio_training_targets[i].to(dtype=torch.float32)
            audio_loss = (audio_pred - audio_target).pow(2).mean(dim=tuple(range(1, audio_pred.dim())))
            per_sample_loss = per_sample_loss + audio_loss

        per_sample_losses.append(per_sample_loss)

    loss = torch.stack(per_sample_losses).mean()
    return {"mse_loss": loss}


def apply_veomni_ltx_transformer_patch() -> None:
    Attention.forward = LTXSPAttention_forward
    LTXModel.forward = LTXVideoModel_forward
    logger.info_rank0("Applied VeOmni SP patch to LTXModel.forward and Attention.forward.")
