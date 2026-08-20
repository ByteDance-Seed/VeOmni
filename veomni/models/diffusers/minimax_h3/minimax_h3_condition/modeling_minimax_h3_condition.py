"""MiniMaxH3 Condition Model — VAE encoding + text encoding + noise/pack for DiT training.

Two-stage training flow:
  1. offline_embedding: get_condition() encodes raw data → saves parquet
  2. offline_training: skip_encoder_load=True, process_condition() adds noise + packs
  3. online_training: get_condition() + process_condition() in one step

Implements the VeOmni ConditionModel contract:
  - get_condition(**micro_batch) → dict[str, list] (per-sample lists)
  - process_condition(**cached) → dict for model.forward()
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import torch
from transformers import PreTrainedModel

from .....utils import logging
from .configuration_minimax_h3_condition import MiniMaxH3ConditionModelConfig


logger = logging.get_logger(__name__)

# VAE constraints
_MINIMAX_H3_FRAME_RATE = 24
_MINIMAX_H3_TIME_DIVISION_FACTOR = 17
_MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


class MiniMaxH3ConditionModel(PreTrainedModel):
    config_class = MiniMaxH3ConditionModelConfig

    def __init__(self, config: MiniMaxH3ConditionModelConfig, **kwargs):
        super().__init__(config)
        self._video_vae = None
        self._audio_vae = None
        self._text_encoder = None
        self._processor = None
        self._tokenizer = None
        self._scheduler_video = None
        self._scheduler_audio = None

        if not config.skip_encoder_load:
            self._init_encoders(config)

        self._init_schedulers(config)

    # ── Initialization ────────────────────────────────────────────────

    def _init_encoders(self, config: MiniMaxH3ConditionModelConfig):
        base = config.base_model_path

        from veomni.models.module_utils import init_empty_weights

        from ..minimax_h3_core.minimax_h3_audio_vae import MiniMaxH3AudioVAE
        from ..minimax_h3_core.minimax_h3_text_encoder import MiniMaxH3TextEncoder
        from ..minimax_h3_core.minimax_h3_video_vae import MiniMaxH3VideoVAE

        with init_empty_weights():
            self._text_encoder = MiniMaxH3TextEncoder(num_retained_layers=config.text_encoder_num_retained_layers)
            self._video_vae = MiniMaxH3VideoVAE()
            self._audio_vae = MiniMaxH3AudioVAE()

        # Processor / tokenizer
        from transformers import AutoProcessor

        proc_path = f"{base}/{config.processor_subfolder}" if base else None
        if proc_path:
            try:
                self._processor = AutoProcessor.from_pretrained(proc_path)
                self._tokenizer = self._processor.tokenizer
            except Exception:
                logger.warning("Failed to load Qwen3-VL processor from %s", proc_path)

        # Load weights if base path provided
        if base:
            self._load_encoder_weights(base, config)

    def _load_encoder_weights(self, base: str, config: MiniMaxH3ConditionModelConfig):
        # ── Text Encoder ──
        te_path = f"{base}/{config.text_encoder_subfolder}"
        if os.path.isdir(te_path):

            def te_filter(k):
                n = config.text_encoder_num_retained_layers
                return (
                    k.startswith("lm_head.")
                    or k.startswith("model.language_model.norm.")
                    or any(k.startswith(f"model.language_model.layers.{i}.") for i in range(n, 100))
                )

            n = self._stream_load_weights(self._text_encoder, te_path, filter_fn=te_filter)
            logger.info_rank0("Text encoder loaded (%d tensors).", n)

        # ── Video VAE ──
        vv_path = f"{base}/{config.video_vae_subfolder}"
        if os.path.isdir(vv_path):
            n = self._stream_load_weights(self._video_vae, vv_path, key_converter=self._convert_video_vae_keys)
            logger.info_rank0("Video VAE loaded (%d tensors).", n)

        # ── Audio VAE ──
        av_path = f"{base}/{config.audio_vae_subfolder}"
        if os.path.isdir(av_path):
            n = self._stream_load_weights(self._audio_vae, av_path, key_converter=self._convert_audio_vae_keys)
            logger.info_rank0("Audio VAE loaded (%d tensors).", n)

        # Cast encoders to bf16: the init context casts buffers too (e.g.
        # vision rotary inv_freq). VeOmni meta-init only replaces parameters,
        # leaving buffers at their construction dtype (fp32), which changes
        # rotary precision.
        for enc in [self._text_encoder, self._video_vae, self._audio_vae]:
            if enc is not None:
                enc.to(torch.bfloat16)
                enc.requires_grad_(False)

    @staticmethod
    def _stream_load_weights(model, path: str, filter_fn=None, key_converter=None) -> int:
        """Stream safetensors files into model, bf16, assign=True.

        safe_open per file → get_tensor → cast torch_dtype (bf16) →
        state_dict_converter → model.load_state_dict(state_dict, assign=True).
        """
        from collections import OrderedDict

        from veomni.models.module_utils import StateDictIterator

        total = 0
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".safetensors"):
                continue
            state = OrderedDict()
            for k, v in StateDictIterator(os.path.join(path, fname)):
                if filter_fn is not None and filter_fn(k):
                    continue
                if key_converter is not None:
                    k2 = key_converter({k: v})
                    if not k2:
                        continue
                    k, v = next(iter(k2.items()))
                state[k] = v.to(torch.bfloat16)
            if state:
                model.load_state_dict(state, assign=True, strict=False)
                total += len(state)
        return total

    @staticmethod
    def _convert_video_vae_keys(state: OrderedDict) -> OrderedDict:
        """Convert VideoVAE state dict keys to VeOmni format."""
        converted = OrderedDict()
        for k, v in state.items():
            # decoder.register_tokens -> decoder.register_tokens.weight
            if "register_tokens" in k and not k.endswith(".weight"):
                k = k + ".weight"
            # *.scale1, *.scale2 -> *.scale1.weight, *.scale2.weight
            if k.endswith(".scale1") or k.endswith(".scale2"):
                k = k + ".weight"
            # Skip mask_token (not in model)
            if "mask_token" in k:
                continue
            converted[k] = v
        return converted

    @staticmethod
    def _convert_audio_vae_keys(state: OrderedDict) -> OrderedDict:
        """Convert AudioVAE state dict keys to VeOmni format."""
        converted = OrderedDict()
        for k, v in state.items():
            # q_bias, v_bias, zero_k_bias -> q_bias.weight, etc.
            if k.endswith(".q_bias") or k.endswith(".v_bias") or k.endswith(".zero_k_bias"):
                k = k + ".weight"
            converted[k] = v
        return converted

    def _init_schedulers(self, config: MiniMaxH3ConditionModelConfig):
        from ..minimax_h3_core.flow_match_scheduler import FlowMatchScheduler

        self._scheduler_video = FlowMatchScheduler(shift=config.sigma_shift_video)
        self._scheduler_video.set_timesteps(config.num_train_timesteps, training=True)

        self._scheduler_audio = FlowMatchScheduler(shift=config.sigma_shift_audio)
        self._scheduler_audio.set_timesteps(config.num_train_timesteps, training=True)

    # ── get_condition (online encoding) ───────────────────────────────

    @torch.no_grad()
    def get_condition(
        self,
        inputs: list[str],
        videos: list[list[torch.Tensor]],
        audios: list | None = None,
        images: list | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Encode raw data → per-sample lists of latents / embeddings / packed info.

        Input (from DataCollator via DiTTrainer.preforward):
          inputs: list[str]                   — text prompts
          videos: list[list[torch.Tensor]]    — each sample: list of frame tensors [C,H,W]
          audios: list[(waveform, sr)] | None — each sample: audio (waveform[C,T], sample_rate)
          images: list | None                 — external keyframe images (FL2VA: unused)

        Returns:
          prompt_embeds: list[tensor [L, 5120]]
          input_latents: list[tensor [1, 24, T_v, H_lat, W_lat]]
          audio_input_latents: list[tensor [C, 32, T_a]]
          keyframe_cond_anchor: list[tensor [cond_rows, 96]] or None
          packed: list[dict]
          imgvid_cond_noise_aug: list[float]
          audio_cond_noise_aug: list[float]
          use_gradient_checkpointing: list[bool]
        """
        device = next(self._video_vae.parameters()).device
        cfg = self.config

        results = {
            "prompt_embeds": [],
            "input_latents": [],
            "audio_input_latents": [],
            "keyframe_cond_anchor": [],
            "packed": [],
            "imgvid_cond_noise_aug": [],
            "audio_cond_noise_aug": [],
            "use_gradient_checkpointing": [],
        }

        num_samples = len(inputs)
        for i in range(num_samples):
            prompt = inputs[i]
            video_frames = videos[i]
            num_frames = len(video_frames)

            # 1. Validate frame count for VAE temporal grouping
            if num_frames % _MINIMAX_H3_TIME_DIVISION_FACTOR != _MINIMAX_H3_TIME_DIVISION_REMAINDER:
                raise ValueError(f"Sample {i}: num_frames={num_frames}, expected 17n+5 (e.g. 73)")

            # 2. Extract keyframes from video (FL2VA: first + last frame)
            # Prefer transform-provided PIL keyframes (native uint8, exact match
            # with keyframes = data["video"][0/-1]); fall back to
            # rebuilding from frame tensors only when the transform lacks them.
            if cfg.use_keyframe_condition and cfg.keyframe_indices:
                if images is not None and i < len(images) and images[i]:
                    keyframe_images = list(images[i])
                else:
                    from PIL import Image

                    keyframe_images = []
                    for idx in cfg.keyframe_indices:
                        frame_tensor = video_frames[idx]
                        if frame_tensor.dim() == 3:
                            frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                            keyframe_images.append(Image.fromarray(frame_np))
            else:
                keyframe_images = None

            # 3. Encode text (Qwen3VL)
            prompt_embeds, text_token_tags = self._encode_text(prompt, keyframe_images, device)

            # 4. Encode video (VAE)
            video_tensor = torch.stack(video_frames).permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]
            input_latent = self._encode_video(video_tensor, device)

            # 5. Encode audio (VAE)
            if audios is not None and i < len(audios) and audios[i] is not None:
                audio_latent = self._encode_audio(audios[i], num_frames, device)
            else:
                # Placeholder: silent audio latent
                audio_latent = self._make_silent_audio_latent(num_frames, device)

            # 6. Encode keyframe as condition anchors
            if cfg.use_keyframe_condition and keyframe_images:
                if len(keyframe_images) != len(cfg.keyframe_indices):
                    raise ValueError(
                        f"Sample {i}: got {len(keyframe_images)} keyframe images, "
                        f"expected {len(cfg.keyframe_indices)} for keyframe_indices={cfg.keyframe_indices}"
                    )
                # Transform-provided keyframes carry their source indices;
                # validate them, not only the count.
                sample_keyframe_indices = kwargs.get("keyframe_indices")
                if sample_keyframe_indices is not None and i < len(sample_keyframe_indices):
                    if list(sample_keyframe_indices[i]) != list(cfg.keyframe_indices):
                        raise ValueError(
                            f"Sample {i}: keyframes were extracted at indices "
                            f"{sample_keyframe_indices[i]}, expected {cfg.keyframe_indices}"
                        )
                keyframe_cond_anchor = self._encode_keyframe_cond(keyframe_images, device, input_latent.shape[2])
            else:
                if cfg.use_keyframe_condition and cfg.keyframe_indices:
                    raise ValueError(f"Sample {i}: keyframe condition is enabled but no keyframe image was produced")
                keyframe_cond_anchor = None

            # 7. Build packed sequence (FL2VA layout)
            T_v = input_latent.shape[2]
            H_lat = input_latent.shape[3]
            W_lat = input_latent.shape[4]
            audio_ch = audio_latent.shape[0]
            T_a = audio_latent.shape[2]

            from ..minimax_h3_core.packed_sequence import build_packed_fl2va

            packed = build_packed_fl2va(
                text_len=prompt_embeds.shape[0],
                latent_t=T_v,
                latent_h=H_lat,
                latent_w=W_lat,
                audio_t=T_a,
                keyframe_indices=cfg.keyframe_indices if cfg.use_keyframe_condition else [],
                audio_channel=audio_ch,
                text_token_tags=text_token_tags,
            )

            results["prompt_embeds"].append(prompt_embeds)
            results["input_latents"].append(input_latent)
            results["audio_input_latents"].append(audio_latent)
            results["keyframe_cond_anchor"].append(keyframe_cond_anchor)
            results["packed"].append(packed)
            results["imgvid_cond_noise_aug"].append(cfg.imgvid_cond_noise_aug)
            results["audio_cond_noise_aug"].append(cfg.audio_cond_noise_aug)
            results["use_gradient_checkpointing"].append(True)

        return results

    def _encode_text(self, prompt: str, keyframe_images, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt + keyframe images via Qwen3VL → (last_hidden_state[0], text_token_tags)."""
        from ..minimax_h3_core.minimax_h3_text_encoder import image_token_counts, presentation_fl2va

        if keyframe_images and self._tokenizer:
            pixel_values, image_grid_thw, counts = image_token_counts(self._processor, keyframe_images)
            input_ids, text_token_tags = presentation_fl2va(self._tokenizer, prompt, counts)
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
            image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)

            prompt_embeds = self._text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            # Text-only: use t2va presentation
            from ..minimax_h3_core.minimax_h3_text_encoder import presentation_t2va

            input_ids, text_token_tags = presentation_t2va(self._tokenizer, prompt)
            input_ids = input_ids.unsqueeze(0).to(device)
            prompt_embeds = self._text_encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )

        return prompt_embeds.to(device), text_token_tags

    def _encode_video(self, video_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode video tensor [1,3,T,H,W] → latent [1,24,T_v,H/16,W/16].

        Match InputVideoEmbedder: preprocess_video produces float32
        [0,1] frames, encode_video(frames, dtype=bf16) casts inside, output
        cast to bf16.
        """
        video = video_tensor.to(device=device, dtype=torch.float32)
        latent = self._video_vae.encode_video(video, dtype=torch.bfloat16)
        return latent.to(torch.bfloat16)

    def _encode_audio(self, audio_input, num_frames: int, device: torch.device) -> torch.Tensor:
        """Encode audio → latent [C, 32, T_a].

        The data loader (minimax_h3_online transform) already
        trims/pads the waveform to int(num_frames/24 * original_sr) at the
        ORIGINAL sample rate. Here we only do the InputAudioEmbedder steps:
        convert_to_stereo(float) → resample to 32kHz → encode_audio(dtype=bf16).
        """
        waveform, sample_rate = audio_input
        waveform = torch.as_tensor(waveform)
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        # convert_to_stereo: duplicate mono, keep stereo
        if waveform.size(-2) == 1:
            waveform = waveform.repeat(2, 1)
        waveform = waveform.float()
        if sample_rate != 32000:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)
        latent = self._audio_vae.encode_audio(waveform[:2].to(device), dtype=torch.bfloat16)
        return latent.to(torch.bfloat16)

    @staticmethod
    def _make_silent_audio_latent(num_frames: int, device: torch.device) -> torch.Tensor:
        """Create a zero-filled audio latent tensor as placeholder."""
        T_a = round(num_frames / _MINIMAX_H3_FRAME_RATE * 40)
        return torch.zeros(2, 32, T_a, device=device, dtype=torch.bfloat16)

    def _encode_keyframe_cond(self, keyframe_images, device: torch.device, video_latent_t: int) -> torch.Tensor:
        """Encode keyframe images → condition anchor [cond_rows, 96] with noise augmentation.

        Matches KeyframeEncoder exactly:
        - img float32 [0,1] → encode_video(dtype=bf16, process_image=True)
        - clean rows cast to bf16
        - noise = generate_noise((1,24,T_v+len(keyframes),H,W), seed, "cpu",
          rand_torch_dtype=bf16)[:, :, :1] → patchify → bf16, ONE noise shared across keyframes
        - ts = tensor(0.999, bf16); anchor = ts*clean + (1-ts)*noise per keyframe
        """
        from ..minimax_h3_core.minimax_h3_dit import patchify_video

        all_cond_rows = []
        for img in keyframe_images:
            img_tensor = self._pil_to_tensor(img).to(device=device, dtype=torch.float32)  # [1,3,H,W]
            z_norm = self._video_vae.encode_video(
                img_tensor, dtype=torch.bfloat16, process_image=True
            )  # [1,24,1,H/16,W/16]
            rows = patchify_video(z_norm)
            all_cond_rows.append(rows)

        clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=device, dtype=torch.bfloat16)

        if self.config.imgvid_cond_noise_aug == 1.0:
            return clean_cond_rows.contiguous()

        ts = torch.tensor(self.config.imgvid_cond_noise_aug, dtype=torch.bfloat16, device=device)
        latent_h, latent_w = int(z_norm.shape[3]), int(z_norm.shape[4])
        noise = torch.randn(
            (1, 24, video_latent_t + len(keyframe_images), latent_h, latent_w),
            device="cpu",
            dtype=torch.bfloat16,
        )[:, :, :1]
        noise_rows = patchify_video(noise.to(device=device, dtype=torch.bfloat16))
        frame_rows = (latent_h // 2) * (latent_w // 2)
        parts = [
            ts * clean_cond_rows[i * frame_rows : (i + 1) * frame_rows] + (1.0 - ts) * noise_rows
            for i in range(len(keyframe_images))
        ]
        return torch.cat(parts, dim=0).contiguous()

    @staticmethod
    def _pil_to_tensor(img) -> torch.Tensor:
        """PIL → float32 [0,1] tensor [1,3,H,W] (preprocess_image min_value=0)."""
        import numpy as np

        arr = np.array(img, dtype=np.float32)
        return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0) * (1.0 / 255.0)

    # ── process_condition (add noise + pack) ──────────────────────────

    @torch.no_grad()
    def process_condition(
        self,
        input_latents: list[torch.Tensor],
        audio_input_latents: list[torch.Tensor],
        prompt_embeds: list[torch.Tensor],
        packed: list[dict],
        keyframe_cond_anchor: list[torch.Tensor | None] | None = None,
        imgvid_cond_noise_aug: float = 0.999,
        audio_cond_noise_aug: float = 1.0,
        use_gradient_checkpointing: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Add noise + pack latents into model.forward() inputs.

        Per-sample (batch is list of samples, this processes sample 0):

        1. Sample shared timestep_id ~ Uniform(0, 999)
        2. Compute sigma_video, sigma_audio from respective schedulers
        3. Add noise: noised = (1-sigma)*clean + sigma*noise
        4. Target = noise - clean (velocity)
        5. Patchify/pack into padded sequence buffers
        6. Per-token timesteps with unique/inverse mapping

        Returns dict with all keys for model.forward() + training_target*.
        """
        # DiTDataCollator wraps every value (including scalars) in lists via
        # batch[key].append(feature[key]). Unwrap scalars that arrive as
        # single-element lists.
        if isinstance(imgvid_cond_noise_aug, list):
            imgvid_cond_noise_aug = imgvid_cond_noise_aug[0]
        if isinstance(audio_cond_noise_aug, list):
            audio_cond_noise_aug = audio_cond_noise_aug[0]
        if isinstance(use_gradient_checkpointing, list):
            use_gradient_checkpointing = use_gradient_checkpointing[0]

        # Process first sample (batch=1 per micro_batch)
        clean_video = input_latents[0]
        clean_audio = audio_input_latents[0]
        prompt = prompt_embeds[0]
        pk = packed[0]
        cond_anchor = keyframe_cond_anchor[0] if keyframe_cond_anchor else None

        device = clean_video.device
        dtype = clean_video.dtype

        cfg = self.config
        seq_len = pk["seq_len"]
        audio_ch = pk["audio_channel"]

        # 1. Sample timestep
        timestep_id_int = int(torch.randint(0, cfg.num_train_timesteps, (1,), device=device).item())
        # Match the precision path: timestep = scheduler.timesteps[id].to(dtype=model_dtype),
        # passed directly to scheduler.add_noise() / scheduler.training_weight().
        # Lower dtype precision (e.g. bfloat16) shifts the argmin index used by add_noise
        # and training_weight, changing the effective sigma and weight.
        model_dtype = clean_video.dtype
        ts_v = self._scheduler_video.timesteps[timestep_id_int].to(dtype=model_dtype)
        ts_a = self._scheduler_audio.timesteps[timestep_id_int].to(dtype=model_dtype)

        # t = 1.0 - timestep / num_train_timesteps
        t_video = 1.0 - ts_v.float().item() / self._scheduler_video.num_train_timesteps
        t_audio = 1.0 - ts_a.float().item() / self._scheduler_audio.num_train_timesteps

        # 2. Add noise + compute velocity target
        video_noise = torch.randn_like(clean_video)
        video_noised = self._scheduler_video.add_noise(clean_video, video_noise, ts_v)
        training_target = self._scheduler_video.training_target(clean_video, video_noise, ts_v)

        audio_noise = torch.randn_like(clean_audio)
        audio_noised = self._scheduler_audio.add_noise(clean_audio, audio_noise, ts_a)
        training_target_audio = self._scheduler_audio.training_target(clean_audio, audio_noise, ts_a)

        # 3. Patchify + pack
        from ..minimax_h3_core.minimax_h3_dit import pack_audio, patchify_video

        video_rows = patchify_video(video_noised)  # [Tv*hp*wp, 96]
        audio_rows = pack_audio(audio_noised)  # [C*Ta, 32]

        # Move packed index tensors to device once (defensive: _to_device
        # in preforward should have done this, but NPU indexing requires it).
        img_pos = pk["img_pos"].to(device)
        cond_rows = pk["cond_rows"]
        audio_pos = pk["audio_pos"].to(device)

        # Packed x: [1, seq_len, feats]
        x = torch.zeros(1, seq_len, video_rows.shape[-1], device=device, dtype=dtype)
        x[0, img_pos[cond_rows:]] = video_rows.to(dtype)
        if cond_anchor is not None:
            x[0, img_pos[:cond_rows]] = cond_anchor.to(dtype)

        audio_x = torch.zeros(1, seq_len, audio_rows.shape[-1], device=device, dtype=dtype)
        audio_x[0, audio_pos] = audio_rows.to(dtype)

        # 4. Per-token timesteps (t = 1 - sigma format)
        timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
        timesteps[audio_pos] = t_audio
        if cond_rows > 0:
            # Condition tokens: always near-clean (t ≈ 1.0, sigma ≈ 0)
            timesteps[img_pos[:cond_rows]] = max(float(t_video), float(cfg.imgvid_cond_noise_aug))
        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

        # 5. Refiner cu_seqlens (text-only, no padding for refiner)
        text_len = pk["text_len"]
        refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32)

        # 6. Shape info for unpatchify (use keys NOT popped by trainer)
        T_v = pk["latent_t"]
        T_a = pk["audio_t"]

        video_latent_shape = (T_v, pk["latent_h_patched"], pk["latent_w_patched"])
        audio_latent_shape = (audio_ch, T_a)

        return {
            "x": x,
            "audio_x": audio_x,
            "img_position_ids": pk["img_position_ids"].to(device),
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "update_mask": None,  # skip_mask_out_condition=True, handled in modeling.py
            "token_tags": pk["token_tags"].to(device),
            "prompt_embeds": prompt,
            "img_pos_info": {"position_ids": pk["img_pos"].to(device)},
            "audio_pos_info": {"position_ids": pk["audio_pos"].to(device)},
            "text_pos_info": {"position_ids": pk["text_pos"].to(device)},
            "img_pos_for_infer_output_info": {"position_ids": pk["img_pos"].to(device)},
            "packed_seq_params": {
                "cu_seqlens_q": pk["cu_seqlens"].to(device),
                "max_seqlen_q": int(pk["cu_seqlens"][1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": refiner_cu.to(device),
                "max_seqlen_q": text_len,
            },
            "skip_mask_out_condition": True,
            "cond_rows": cond_rows,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "video_latent_shape": video_latent_shape,
            "audio_latent_shape": audio_latent_shape,
            "training_target": training_target,
            "training_target_audio": training_target_audio,
            # Pass scheduler references and raw t values so modeling.py can compute training_weight
            "scheduler_video": self._scheduler_video,
            "scheduler_audio": self._scheduler_audio,
            "t_video": t_video,
            "t_audio": t_audio,
        }
