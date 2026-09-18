"""HF PreTrainedModel wrapper for MiniMaxH3 DiT.

Wraps the raw MiniMaxH3DiT nn.Module (see ..minimax_h3_core.minimax_h3_dit) and
handles:
- HF from_pretrained / _from_config loading
- Forward pass with unpatchify + negation + loss computation
- _no_split_modules for FSDP2

Loss is computed INTERNALLY (model pops training_target from kwargs)
because VeOmni's DiTTrainer does not pop/process training targets externally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...packing import DiffusionBatchOutput
from ..minimax_h3_core.batch_packing import pack_samples
from ..minimax_h3_core.minimax_h3_dit import MiniMaxH3Attention, MiniMaxH3DiT, unpack_audio, unpatchify_video
from .configuration_minimax_h3_transformer import MiniMaxH3DiTModelConfig


@dataclass
class MiniMaxH3DiTOutput(ModelOutput):
    loss: dict | None = None
    predictions: list | None = None


class MiniMaxH3DiTModel(PreTrainedModel):
    config_class = MiniMaxH3DiTModelConfig
    supports_gradient_checkpointing = True
    supports_remove_padding = True
    _supports_sdpa = True  # HF checks this before the load-time packing hook runs.
    _no_split_modules = ["MiniMaxH3DiTBlock"]

    _checkpoint_conversion_mapping = {"^": "dit."}

    def __init__(self, config: MiniMaxH3DiTModelConfig, **kwargs):
        super().__init__(config)
        self.gradient_checkpointing = False  # enable HF-compatible GC attr
        self.use_remove_padding = False
        self.dit = MiniMaxH3DiT(
            num_layers=config.num_layers,
            token_refiner_num_layers=config.token_refiner_num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            ffn_hidden_size=config.ffn_hidden_size,
            latents_dim=config.latents_dim,
            audio_latents_dim=config.audio_latents_dim,
            patch_size=config.patch_size,
            text_dim=config.text_dim,
            timestep_input_dim=config.timestep_input_dim,
            time_embed_hidden_size=config.time_embed_hidden_size,
            time_embed_dim=config.time_embed_dim,
            adaln_out_features=config.adaln_out_features,
            final_adaln_out_features=config.final_adaln_out_features,
            rope_inv_freq_len=config.rope_inv_freq_len,
            norm_eps=config.norm_eps,
            qk_norm_eps=config.qk_norm_eps,
            final_norm_eps=config.final_norm_eps,
        )

    def configure_remove_padding(self, *, attn_implementation: str):
        """Configure only this instance; the existing parameter/wrap hierarchy is unchanged."""
        from .....ops.kernels.attention.flash import _load_veomni_local_flash_kernel

        if (
            self.config.latents_dim != 24
            or self.config.audio_latents_dim != 32
            or tuple(self.config.patch_size) != (1, 2, 2)
        ):
            raise ValueError("H3 remove-padding requires the native 24-video/32-audio patch geometry.")
        if attn_implementation in ("eager", "sdpa"):
            kernel = None
        elif attn_implementation in ("veomni_flash_attention_2_with_sp", "veomni_flash_attention_3_with_sp"):
            kernel = _load_veomni_local_flash_kernel(attn_implementation).flash_attn_varlen_func
        else:
            raise ValueError(f"Unsupported H3 remove-padding backend: {attn_implementation}")
        for module in self.dit.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.packed_sdpa = kernel is None
                module.varlen_kernel = kernel
        self.dit.use_varlen_attention = kernel is not None
        self.dit.use_remove_padding = True
        self.use_remove_padding = True

    def _forward_samples(self, samples):
        packed_inputs, row_counts = pack_samples(samples)
        video, audio = self.dit(**packed_inputs)
        video_parts = video.split([v for v, _ in row_counts])
        audio_parts = audio.split([a for _, a in row_counts])
        outputs = [
            self._finish_outputs(v, a, **sample["targets"], **sample["metadata"])
            for sample, v, a in zip(samples, video_parts, audio_parts)
        ]
        if any((out.loss is None) != (outputs[0].loss is None) for out in outputs):
            raise ValueError("All H3 samples must consistently supply or omit training targets.")
        losses = (
            None
            if outputs[0].loss is None
            else {key: torch.stack([out.loss[key] for out in outputs]) for key in outputs[0].loss}
        )
        return DiffusionBatchOutput(
            sample_predictions=[{"video": out.predictions[0], "audio": out.predictions[1]} for out in outputs],
            sample_losses=losses,
        )

    def forward(
        self,
        x=None,
        audio_x=None,
        img_position_ids=None,
        unique_timesteps=None,
        inverse_indices=None,
        update_mask=None,
        token_tags=None,
        prompt_embeds=None,
        img_pos_info=None,
        audio_pos_info=None,
        text_pos_info=None,
        img_pos_for_infer_output_info=None,
        packed_seq_params=None,
        refiner_packed_seq_params=None,
        use_gradient_checkpointing=False,
        training_target=None,
        training_target_audio=None,
        video_latent_shape=None,
        audio_latent_shape=None,
        sample_inputs=None,
        **kwargs,
    ) -> MiniMaxH3DiTOutput | DiffusionBatchOutput:
        """Forward pass with unpatchify + negation + internal loss computation.

        Accepts all keys from condition_model.process_condition().
        Pops training_target* internally (not via trainer) and computes MSE loss.

        video_latent_shape: (T_v, latent_h//2, latent_w//2) for unpatchify_video
        audio_latent_shape: (audio_channel, T_a) for unpack_audio
        """
        if sample_inputs is not None:
            if not self.use_remove_padding:
                raise ValueError("sample_inputs requires use_remove_padding to be configured at model load.")
            return self._forward_samples(sample_inputs)

        # Pop metadata keys (trainer may also pop them)
        skip_mask_out_condition = kwargs.pop("skip_mask_out_condition", False)
        cond_rows = kwargs.pop("cond_rows", 0)
        scheduler_video = kwargs.pop("scheduler_video", None)
        scheduler_audio = kwargs.pop("scheduler_audio", None)

        # Run DiT — skip_mask_out_condition=True, update_mask=None
        video_tokens, audio_tokens = self.dit(
            x=x,
            audio_x=audio_x,
            img_position_ids=img_position_ids,
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
            update_mask=None if skip_mask_out_condition else update_mask,
            token_tags=token_tags,
            prompt_embeds=prompt_embeds,
            img_pos_info=img_pos_info,
            audio_pos_info=audio_pos_info,
            text_pos_info=text_pos_info,
            img_pos_for_infer_output_info=img_pos_for_infer_output_info,
            packed_seq_params=packed_seq_params,
            refiner_packed_seq_params=refiner_packed_seq_params,
            use_gradient_checkpointing=use_gradient_checkpointing,
            skip_mask_out_condition=skip_mask_out_condition,
        )

        return self._finish_outputs(
            video_tokens,
            audio_tokens,
            cond_rows=cond_rows,
            video_latent_shape=video_latent_shape,
            audio_latent_shape=audio_latent_shape,
            training_target=training_target,
            training_target_audio=training_target_audio,
            scheduler_video=scheduler_video,
            scheduler_audio=scheduler_audio,
            **kwargs,
        )

    @staticmethod
    def _finish_outputs(
        video_tokens,
        audio_tokens,
        *,
        cond_rows=0,
        video_latent_shape=None,
        audio_latent_shape=None,
        training_target=None,
        training_target_audio=None,
        scheduler_video=None,
        scheduler_audio=None,
        **kwargs,
    ):
        # Slice off condition rows (v_video_rows[cond_rows_count:])
        if cond_rows > 0:
            video_tokens = video_tokens[cond_rows:]
            # No ref audio in FL2VA, so audio_tokens don't need slicing

        # Unpatchify per-token output → dense latent tensors.
        # unpatchify_video expects patchify_video's input shape (full latent
        # T/H/W: f, h, w = video_latents.shape[2:]);
        # packed stores the halved H/W (latent_h_patched), so double back.
        T_v, hp, wp = video_latent_shape
        video_logits = unpatchify_video(video_tokens, T_v, hp * 2, wp * 2)

        C, Ta = audio_latent_shape
        audio_logits = unpack_audio(audio_tokens, C, Ta)

        # Negate: model_fn returns -v, matching training_target = noise - clean
        video_pred = -video_logits
        audio_pred = -audio_logits

        loss = None
        if training_target is not None and training_target_audio is not None:
            loss_video = F.mse_loss(video_pred.float(), training_target.float())
            loss_audio = F.mse_loss(audio_pred.float(), training_target_audio.float())

            # Apply timestep-dependent training weights
            # t_video / t_audio passed directly from process_condition (t = 1 - sigma)
            t_video_val = kwargs.pop("t_video", None)
            t_audio_val = kwargs.pop("t_audio", None)

            if scheduler_video is not None and t_video_val is not None:
                sigma_v = 1.0 - float(t_video_val)
                ts_v = torch.tensor(sigma_v * scheduler_video.num_train_timesteps, device=video_pred.device)
                weight_v = scheduler_video.training_weight(ts_v)
                loss_video = loss_video * weight_v

            if scheduler_audio is not None and t_audio_val is not None:
                sigma_a = 1.0 - float(t_audio_val)
                ts_a = torch.tensor(sigma_a * scheduler_audio.num_train_timesteps, device=audio_pred.device)
                weight_a = scheduler_audio.training_weight(ts_a)
                loss_audio = loss_audio * weight_a

            loss = {"mse_video": loss_video, "mse_audio": loss_audio}

        return MiniMaxH3DiTOutput(
            loss=loss,
            predictions=[video_pred, audio_pred],
        )
