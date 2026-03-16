from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.video_processor import VideoProcessor
from torchvision.transforms import InterpolationMode, functional
from transformers import AutoTokenizer, PreTrainedModel, UMT5EncoderModel

from .....utils import logging
from .configuration_wan_condition import WanTransformer3DConditionModelConfig


logger = logging.get_logger(__name__)


# T2V only
class WanTransformer3DConditionModel(PreTrainedModel):
    config_class = WanTransformer3DConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: WanTransformer3DConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.video_processor = None
        self.negative_prompt_embeds = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        # Persistent per-SP-group RNG generators; initialized lazily on first use.
        self._sp_cuda_gen: torch.Generator | None = None
        self._sp_cpu_gen: torch.Generator | None = None
        self._load_components()

    @property
    def _execution_device(self):
        return self.vae.device

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Wan condition components from {base}.")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        if self.meta_init:
            self.vae = AutoencoderKLWan.from_config(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        else:
            self.vae = AutoencoderKLWan.from_pretrained(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.scale_factor_spatial)
        self._prepare_negative_prompt_embeds()
        if self.meta_init:
            del self.text_encoder

    @torch.no_grad()
    def _prepare_negative_prompt_embeds(self):
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=[self.config.cfg_negative_prompt],
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.negative_prompt_embeds = prompt_embeds[0].unsqueeze(0)

    def _encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        # resize video to max size
        height, width = video.shape[-2:]

        size = min(self.config.video_max_size, min(width, height))
        video = functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)
        video = self.video_processor.preprocess_video(video)
        video = video.to(device=self.vae.device, dtype=self.vae.dtype)

        # save mean & logvar
        posterior: DiagonalGaussianDistribution = self.vae.encode(video).latent_dist

        return posterior.parameters

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        return (latents - latents_mean) / latents_std

    @torch.no_grad()
    def _get_t5_prompt_embeds(self, **kwargs):
        return WanPipeline._get_t5_prompt_embeds(self, **kwargs)

    @torch.no_grad()
    def get_condition(self, inputs, videos, **kwargs) -> dict[str, Any]:
        """
        inputs: list[str], a list of samples of prompts
        videos: list[list[torch.Tensor]] a list of samples of videos
        """
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=inputs,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )  # bs, seqlen, dim
        context_list = [u.unsqueeze(0) for u in prompt_embeds]

        latents_list: list[list[torch.Tensor]] = []
        for sample_videos in videos:
            assert len(sample_videos) == 1, "Only one video per sample is supported for T2V"
            latents_list.append(self._encode_video_to_latents(sample_videos[0]))  # 1, c, f, h, w

        return {"latents": latents_list, "context": context_list}

    def _sp_generators(self, device: torch.device) -> tuple[torch.Generator | None, torch.Generator | None]:
        """Return persistent (CUDA, CPU) generators synchronized across the SP group.

        **Initialization (first call only):** a seed is drawn on SP rank-0 and
        broadcast to every peer in the Ulysses group so all ranks start from an
        identical state.

        **Subsequent calls:** the generators are returned as-is and continue to
        advance naturally — exactly like a normal RNG.  Because all SP ranks in the
        same group make the same sequence of ``randn_like / randint / rand`` calls
        inside ``process_condition``, their generator states remain in lock-step
        automatically, with genuine randomness across training steps.

        If Ulysses SP is not active, ``(None, None)`` is returned so callers fall
        back to the global RNG (unchanged behaviour from before SP was added).
        """
        if self._sp_cuda_gen is not None:
            # Already initialized; just hand back the live generators.
            return self._sp_cuda_gen, self._sp_cpu_gen

        import torch.distributed as dist

        from .....distributed.parallel_state import get_parallel_state

        if not get_parallel_state().sp_enabled:
            return None, None

        # First-time setup: broadcast the initial seed from SP rank-0 so every
        # rank in the group starts from the exact same generator state.
        # The collective backend (NCCL / HCCL) only supports device tensors, so
        # we always create the seed on the current accelerator device regardless
        # of where the latents live (they may still be on CPU at this point).
        from .....utils.device import get_device_type

        accel_device = torch.device(get_device_type())
        seed = torch.randint(0, 2**31, (), dtype=torch.int64, device=accel_device)
        dist.broadcast(
            seed,
            src=dist.get_global_rank(get_parallel_state().sp_group, 0),
            group=get_parallel_state().sp_group,
        )
        seed_val = seed.item()

        self._sp_cuda_gen = torch.Generator(device=accel_device)
        self._sp_cuda_gen.manual_seed(seed_val)

        self._sp_cpu_gen = torch.Generator(device="cpu")
        self._sp_cpu_gen.manual_seed(seed_val)

        return self._sp_cuda_gen, self._sp_cpu_gen

    def process_condition(self, latents: list[torch.Tensor], context: list[torch.Tensor]) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents[0].device)
            self._timesteps_ready = True

        # Obtain persistent generators that are synchronized across the SP group.
        # All SP ranks start from the same seed (set once) and advance in lock-step
        # because they always call the same random ops in the same order, so
        # every step naturally produces different but identical-across-ranks samples.
        cuda_gen, cpu_gen = self._sp_generators(latents[0].device)

        packed_conditions: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "training_target": [],
            "latents": [],
        }
        for sample_latents, sample_context in zip(latents, context):
            latents = DiagonalGaussianDistribution(sample_latents).mode()
            latents = self._normalize_latents(latents)
            # When cuda_gen is active (SP mode) we generate on the generator's
            # own device (accelerator) and then move the result to the latent
            # device.  torch.randn / torch.randint require that the generator
            # device matches the output device, so we cannot pass latents.device
            # directly when the generator lives on a different device.
            gen_device = cuda_gen.device if cuda_gen is not None else latents.device
            noise = torch.randn(latents.shape, dtype=latents.dtype, device=gen_device, generator=cuda_gen).to(
                latents.device
            )
            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (latents.shape[0],),
                device=gen_device,
                generator=cuda_gen,
            ).to(latents.device)
            timestep = self.scheduler.timesteps[timestep_ids].to(device=latents.device, dtype=latents.dtype)
            noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
            training_target = noise - latents

            use_negative_context = torch.rand((), generator=cpu_gen) < self.config.cfg_negative_prob
            if use_negative_context:
                sample_context = self.negative_prompt_embeds.to(device=latents.device, dtype=sample_context.dtype)
            else:
                sample_context = sample_context.to(latents.device)

            packed_conditions["hidden_states"].append(noisy_latents)
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(sample_context)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(latents)

        return packed_conditions
