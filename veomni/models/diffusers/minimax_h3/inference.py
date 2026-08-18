"""MiniMax H3 audio/video inference pipeline.

Ported from the minimax_h3_audio_video pipeline (MiniMaxH3Pipeline) and the
base pipeline machinery (BasePipeline / PipelineUnit / PipelineUnitRunner).
Framework-specific dependencies are replaced by VeOmni equivalents:

- model loading (ModelPool / ModelConfig): VeOmni registries
  (MODEL_CONFIG_REGISTRY / MODELING_REGISTRY) for the condition model
  (text encoder + VAEs + schedulers) and build_foundation_model for the DiT.
- get_device_type: veomni.utils.device.get_device_type
- audio utils (convert_to_stereo / resample_waveform): ported below.
- attention / gradient checkpointing: veomni minimax_h3_core (not used here).
- load_models_to_device: VeOmni models have no VRAM-management hooks, so the
  pipeline stages models by name explicitly (listed models move to
  self.device, all other children move to CPU).

Known deviations:
- self.dit is the HF wrapper MiniMaxH3DiTModel; model_fn calls its
  forward and takes predictions (same raw DiT forward + cond-row slicing +
  unpatchify + negation). ref2va audio references are NOT supported by the
  wrapper (raises NotImplementedError); t2va and fl2va paths are numerically
  equivalent.
- FlowMatchScheduler.set_timesteps ignores the shift kwarg (VeOmni scheduler
  pins shift at construction; default shifts are 12.0 video / 3.0 audio).
"""

from __future__ import annotations

import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image
from tqdm import tqdm

from veomni.models import build_foundation_model
from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from veomni.utils.device import get_device_type

from .minimax_h3_core.flow_match_scheduler import FlowMatchScheduler
from .minimax_h3_core.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from .minimax_h3_core.minimax_h3_dit import pack_audio, patchify_video
from .minimax_h3_core.minimax_h3_text_encoder import (
    MiniMaxH3TextEncoder,
    image_token_counts,
    presentation_fl2va,
    presentation_ref2va,
    presentation_t2va,
    sample_qwen_video_frames,
    video_token_counts,
)
from .minimax_h3_core.minimax_h3_video_vae import MiniMaxH3VideoVAE
from .minimax_h3_transformer.modeling_minimax_h3_transformer import MiniMaxH3DiTModel


# ── Audio utils ──────────────────────────────────────────────────────


def convert_to_stereo(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to stereo.
    Supports [C, T] or [B, C, T]. Duplicate mono, keep stereo.
    """
    if audio_tensor.size(-2) == 1:
        return audio_tensor.repeat(1, 2, 1) if audio_tensor.dim() == 3 else audio_tensor.repeat(2, 1)
    return audio_tensor


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample waveform to target sample rate if needed."""
    if source_rate == target_rate:
        return waveform
    import torchaudio

    resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return resampled.to(dtype=waveform.dtype)


# ── Base pipeline machinery ──────────────────────────────────────────


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(set(params))
        return params

    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}

    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(
        self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict
    ) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega
            )
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


class BasePipeline(torch.nn.Module):
    def __init__(
        self,
        device=None,
        torch_dtype=torch.bfloat16,
        height_division_factor=32,
        width_division_factor=32,
        time_division_factor=17,
        time_division_remainder=5,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device if device is not None else get_device_type()
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # VRAM management (VeOmni models have no onload/offload hooks,
        # load_models_to_device stages children explicitly instead).
        self.vram_management_enabled = False
        # Pipeline Unit Runner
        self.unit_runner = PipelineUnitRunner()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None, verbose=1):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            )
            if verbose > 0:
                print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            if verbose > 0:
                print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    num_frames - self.time_division_remainder + self.time_division_factor - 1
                ) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                if verbose > 0:
                    print(
                        f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                    )
            return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [
            self.preprocess_image(
                image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [
            self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value)
            for image in vae_output
        ]
        return video

    def output_audio_format_check(self, audio_output):
        # output standard foramt: [C, T], output dtype: float()
        # remove batch dim
        if audio_output.ndim == 3:
            audio_output = audio_output.squeeze(0)
        return audio_output.float().cpu()

    def load_models_to_device(self, model_names):
        # VeOmni models have no onload/offload VRAM-management hooks; stage
        # children by name instead: listed models go to self.device,
        # everything else to CPU. "condition_model" is only a handle to the
        # encoder container and is skipped (it holds the same text encoder /
        # VAE modules as their named children). A self-managed DiT (block
        # offload) stages its own weights.
        for name, model in self.named_children():
            if name == "condition_model":
                continue
            if model is not None and getattr(model, "_self_managed", False):
                continue
            if model is not None:
                model.to(self.device if name in model_names else "cpu")

    def generate_noise(
        self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None
    ):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        timestep = scheduler.timesteps[progress_id]
        if inpaint_mask is not None:
            noise_pred_expected = scheduler.return_to_timestep(
                scheduler.timesteps[progress_id], latents, input_latents
            )
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)
        latents_next = scheduler.step(noise_pred, timestep, latents)
        return latents_next

    def blend_with_mask(self, base, addition, mask):
        return base * (1 - mask) + addition * mask

    def check_vram_management_state(self):
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and module.vram_management_enabled:
                vram_management_enabled = True
        return vram_management_enabled

    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        # Positive side forward
        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)
        if cfg_scale != 1.0:
            # Negative side forward
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)
            if isinstance(noise_pred_posi, tuple):
                # Separately handling different output types of latents, eg. video and audio latents.
                noise_pred = tuple(
                    n_nega + cfg_scale * (n_posi - n_nega) for n_posi, n_nega in zip(noise_pred_posi, noise_pred_nega)
                )
            else:
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        return noise_pred


# ── Pipeline ─────────────────────────────────────────────────────────


class MiniMaxH3Pipeline(BasePipeline):
    def __init__(self, device=None, torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=17,
            time_division_remainder=5,
        )
        self.scheduler = FlowMatchScheduler(shift=12.0)
        self.scheduler_audio = FlowMatchScheduler(shift=3.0)
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3DiTModel = None
        self.video_vae: MiniMaxH3VideoVAE = None
        self.audio_vae: MiniMaxH3AudioVAE = None
        self.tokenizer = None
        self.processor = None
        self.imgvid_cond_noise_aug = 0.999
        self.audio_cond_noise_aug = 1.0
        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_NoiseInitializer(),
            MiniMaxH3Unit_InputVideoEmbedder(),
            MiniMaxH3Unit_InputAudioEmbedder(),
            MiniMaxH3Unit_KeyframeEncoder(),
            MiniMaxH3Unit_ReferenceEncoder(),
            MiniMaxH3Unit_PromptEmbedder(),
            MiniMaxH3Unit_PackedSequenceBuilder(),
        ]
        self.model_fn = model_fn_minimax_h3
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        condition_model_path: str = None,
        condition_model_cfg: dict = None,
        transformer_config_path: str = None,
        transformer_weights_path: str = None,
        transformer_config_kwargs: dict = None,
        ops_implementation=None,
    ):
        """Load the pipeline through VeOmni model loading paths.

        - condition model (text encoder + video/audio VAE + schedulers +
          tokenizer/processor) via MODEL_CONFIG_REGISTRY / MODELING_REGISTRY,
          same as DiTTrainer._build_condition_model.
        - DiT via build_foundation_model. init_device=<device>: single-process
          parallel state reports global_rank=-1, so "cpu" would trigger
          empty_init and leave the weights on meta.
        """
        pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch_dtype)

        # Condition model: encoders are always needed at inference time, so
        # force load even when the config was written for offline training.
        condition_model_cfg = dict(condition_model_cfg or {})
        condition_model_cfg.setdefault("skip_encoder_load", False)
        # Video VAE weights live in video_vae/source (checkpoint layout); the
        # config default subfolder "video_vae" has no safetensors directly.
        condition_model_cfg.setdefault("video_vae_subfolder", "video_vae/source")
        config_class = MODEL_CONFIG_REGISTRY["MiniMaxH3ConditionModel"]()
        condition_cfg = config_class.from_pretrained(condition_model_path, **condition_model_cfg)
        model_class = MODELING_REGISTRY["MiniMaxH3ConditionModel"]()
        condition_model = model_class._from_config(condition_cfg)
        condition_model.eval()

        # Keep a handle to the condition model, but do NOT register it as an
        # nn.Module child: load_models_to_device() stages children by name and
        # the condition model contains all encoders (moving it would undo the
        # per-encoder staging).
        pipe.__dict__["condition_model"] = condition_model
        pipe.text_encoder = condition_model._text_encoder
        pipe.video_vae = condition_model._video_vae
        pipe.audio_vae = condition_model._audio_vae
        pipe.tokenizer = condition_model._tokenizer
        pipe.processor = condition_model._processor
        pipe.scheduler = condition_model._scheduler_video
        pipe.scheduler_audio = condition_model._scheduler_audio
        pipe.imgvid_cond_noise_aug = condition_cfg.imgvid_cond_noise_aug
        pipe.audio_cond_noise_aug = condition_cfg.audio_cond_noise_aug

        # The 50-layer DiT (~60GB bf16) alone nearly fills the NPU. Encoders
        # must not sit on it while the DiT allocates; the pipeline units
        # re-stage each model per use during __call__.
        pipe.load_models_to_device([])
        # Release the allocator pool so the DiT allocation starts from a
        # clean state instead of a pool reserved for the offloaded encoders.
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()

        # DiT (HF wrapper MiniMaxH3DiTModel). The 61.7GB of
        # weights do not fit on the NPU (~61.3GB total), so they live on
        # CPU and the DiT stages its blocks in two halves per forward.
        # With init_device="meta", build_foundation_model forces
        # empty_init and skips its internal weight load; the explicit call
        # below materializes the weights on CPU instead.
        pipe.dit = build_foundation_model(
            config_path=transformer_config_path,
            weights_path=None,
            torch_dtype="bfloat16",
            init_device="meta",
            ops_implementation=ops_implementation,
            config_kwargs=transformer_config_kwargs,
        )
        from veomni.models.module_utils import load_model_weights

        load_model_weights(pipe.dit, transformer_weights_path, init_device="cpu")
        pipe.dit.dit.enable_block_offload()
        pipe.dit._self_managed = True
        pipe.dit.eval()
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = " ",
        height: int = 768,
        width: int = 1344,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        seed: int = 42,
        rand_device: str = "cpu",
        cfg_scale: float = 1.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        # Keyframe to Video
        keyframes: list[Image.Image] = None,
        keyframe_indices: list[int] = None,
        # Reference to Video
        references: list[dict] = None,
        ref_image_short_edge: int = 2048,
        ref_video_short_edge: int = 768,
        ref_video_max_pixels: int = 768 * 1344,
        progress_bar_cmd=tqdm,
    ):
        """Generate a joint video + audio sample.

        `num_frames` is snapped up to the nearest 17n+5 and that aligned value
        drives every downstream shape (video/audio latent lengths, reference video
        capping), so the returned clip may be slightly longer than requested.

        `keyframes` (FL2AV) is a list of PIL images with `keyframe_indices` in
        {0, -1}; both are resized onto the target canvas.

        `references` (Ref2AV) is a list of dicts in request order:

            {"type": "image",       "image": PIL.Image}
            {"type": "video",       "video": list[PIL.Image]}   # silent
            {"type": "audio",       "audio": Tensor[C, L], "sample_rate": int}
            {"type": "video_audio", "video": list[PIL.Image],
                                    "audio": Tensor[C, L], "sample_rate": int}

        Input contract: `video` frame lists must ALREADY. be 24fps the pipeline never resamples frame rate.
        """
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "references": references,
            "ref_image_short_edge": ref_image_short_edge,
            "ref_video_short_edge": ref_video_short_edge,
            "ref_video_max_pixels": ref_video_max_pixels,
            "imgvid_cond_noise_aug": self.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.audio_cond_noise_aug,
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, _ in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t_video = float(1.0 - self.scheduler.sigmas[progress_id])
            t_audio = float(1.0 - self.scheduler_audio.sigmas[progress_id])
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                t_video=t_video,
                t_audio=t_audio,
                device=self.device,
            )
            inputs_shared["video_latents"] = self.step(
                self.scheduler, inputs_shared["video_latents"], progress_id, noise_pred=noise_pred_video
            )
            inputs_shared["audio_latents"] = self.step(
                self.scheduler_audio, inputs_shared["audio_latents"], progress_id, noise_pred=noise_pred_audio
            )

        # 5. Decode
        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(
            inputs_shared["video_latents"],
            dtype=self.torch_dtype,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(inputs_shared["audio_latents"], dtype=self.torch_dtype)
        audio = self.output_audio_format_check(waveform)
        return video, audio


class MiniMaxH3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class MiniMaxH3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "num_frames", "height", "width", "rand_device"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, num_frames, height, width, rand_device):
        video_latent_t, latent_h, latent_w = ((num_frames - 5) // 17) * 5 + 2, height // 16, width // 16
        video_latents = pipe.generate_noise(
            (1, 24, video_latent_t, latent_h, latent_w),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=pipe.torch_dtype,
        )
        audio_latent_t = round(num_frames / 24.0 * 40.0)
        audio_latents = pipe.generate_noise(
            (2, 32, audio_latent_t), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("keyframes", "ref_blocks", "height", "width"),
            output_params=("prompt_embeds", "text_token_tags"),
            onload_model_names=("text_encoder",),
        )

    def preprocess_ref_blocks(self, pipe: MiniMaxH3Pipeline, prompt, ref_blocks):
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        counters = {"image": 0, "audio": 0, "video": 0}
        images, videos, timestamps_per_video, condition_labels = [], [], [], []
        for block in ref_blocks:
            kind = block["kind"]
            if kind == "image":
                counters["image"] += 1
                condition_labels.append(("image", counters["image"]))
                images.append(block["prepared_image"])
            elif kind == "audio":
                counters["audio"] += 1
                condition_labels.append(("audio", counters["audio"]))
            elif kind in ("video", "video_audio"):
                if int(block["ref_audio_t"]) > 0:
                    counters["audio"] += 1
                    condition_labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                condition_labels.append(("video", counters["video"]))
                sampled, timestamps = sample_qwen_video_frames(block["prepared_frames"])
                videos.append(np.stack([np.asarray(f) for f in sampled]))
                timestamps_per_video.append(timestamps)
            else:
                raise ValueError(f"unknown reference kind: {kind}")

        image_counts, video_counts, video_timestamps = [], [], []
        if len(images) > 0:
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, images)
        if len(videos) > 0:
            pixel_values_videos, video_grid_thw, video_counts, video_timestamps = video_token_counts(
                pipe.processor, videos, timestamps_per_video
            )
        input_ids, text_token_tags = presentation_ref2va(
            pipe.tokenizer, prompt, condition_labels, image_counts, video_counts, video_timestamps
        )
        return input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

    def process(self, pipe: MiniMaxH3Pipeline, prompt, keyframes=None, ref_blocks=None, height=None, width=None):
        pipe.load_models_to_device(self.onload_model_names)
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        if ref_blocks:
            input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = (
                self.preprocess_ref_blocks(pipe, prompt, ref_blocks)
            )
        elif keyframes:
            keyframes = [img.convert("RGB").resize((width, height), Image.LANCZOS) for img in keyframes]
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, keyframes)
            input_ids, text_token_tags = presentation_fl2va(pipe.tokenizer, prompt, image_counts)
        else:
            input_ids, text_token_tags = presentation_t2va(pipe.tokenizer, prompt)

        ids = input_ids.unsqueeze(0).to(pipe.device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(pipe.device, pipe.torch_dtype)
            kwargs["image_grid_thw"] = image_grid_thw.to(pipe.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(pipe.device, pipe.torch_dtype)
            kwargs["video_grid_thw"] = video_grid_thw.to(pipe.device, torch.long)
        hidden = pipe.text_encoder(**kwargs)

        return {
            "prompt_embeds": hidden.to(pipe.device, pipe.torch_dtype),
            "text_token_tags": text_token_tags.view(-1).to(pipe.device, torch.long),
        }


class MiniMaxH3Unit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video",),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae",),
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_video):
        if input_video is None or not pipe.scheduler.training:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        frames_tensor = pipe.preprocess_video(input_video, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        latents = pipe.video_vae.encode_video(frames_tensor, dtype=pipe.torch_dtype).to(pipe.torch_dtype)
        return {"video_latents": latents, "input_latents": latents}


class MiniMaxH3Unit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio",),
            output_params=("audio_latents", "audio_input_latents"),
            onload_model_names=("audio_vae",),
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_audio):
        if input_audio is None or not pipe.scheduler.training:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        waveform, sample_rate = input_audio
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        waveform = resample_waveform(
            convert_to_stereo(waveform).float(), sample_rate, pipe.audio_vae.sample_rate
        )  # [C, T]
        latents = pipe.audio_vae.encode_audio(waveform[:2].to(pipe.device), dtype=pipe.torch_dtype).to(
            pipe.torch_dtype
        )  # [C, 32, T]
        return {"audio_latents": latents, "audio_input_latents": latents}


class MiniMaxH3Unit_KeyframeEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("keyframes", "keyframe_indices", "video_latents", "rand_device", "seed", "height", "width"),
            output_params=("keyframe_cond_anchor",),
            onload_model_names=("video_vae",),
        )

    def process(
        self, pipe: MiniMaxH3Pipeline, keyframes, keyframe_indices, video_latents, rand_device, seed, height, width
    ):
        if keyframes is None:
            return {}
        assert keyframe_indices is not None and len(keyframes) == len(keyframe_indices), (
            "keyframe_indices must be provided when keyframes is not None"
        )
        assert all(idx in (0, -1) for idx in keyframe_indices), (
            "keyframe_indices must be within the range of keyframes (0 or -1)"
        )
        pipe.load_models_to_device(self.onload_model_names)
        # Encode keyframes
        all_cond_rows = []
        for img in keyframes:
            img_tensor = pipe.preprocess_image(
                img.convert("RGB").resize((width, height), Image.LANCZOS), torch_dtype=torch.float32, min_value=0
            )
            z_norm = pipe.video_vae.encode_video(
                img_tensor, dtype=pipe.torch_dtype, process_image=True
            )  # [1,24,1,H',W']
            rows = patchify_video(z_norm)
            all_cond_rows.append(rows)
        clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=pipe.device, dtype=pipe.torch_dtype)
        if pipe.imgvid_cond_noise_aug == 1.0:
            keyframe_cond_anchor = clean_cond_rows
        else:
            video_latent_t, latent_h, latent_w = (int(x) for x in video_latents.shape[2:])
            ts = torch.tensor(pipe.imgvid_cond_noise_aug, dtype=pipe.torch_dtype, device=pipe.device)
            noise = pipe.generate_noise(
                (1, 24, video_latent_t + len(keyframes), latent_h, latent_w), seed, rand_device, pipe.torch_dtype
            )[:, :, :1]
            noise_rows = patchify_video(noise).to(device=pipe.device, dtype=pipe.torch_dtype)
            frame_rows = (latent_h // 2) * (latent_w // 2)
            parts = [
                ts * clean_cond_rows[i * frame_rows : (i + 1) * frame_rows] + (1.0 - ts) * noise_rows
                for i in range(len(keyframes))
            ]
            keyframe_cond_anchor = torch.cat(parts, dim=0)
        return {"keyframe_cond_anchor": keyframe_cond_anchor.contiguous()}


class MiniMaxH3Unit_ReferenceEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "references",
                "seed",
                "video_latents",
                "num_frames",
                "ref_image_short_edge",
                "ref_video_short_edge",
                "ref_video_max_pixels",
            ),
            output_params=("ref_blocks", "ref_visual_anchor", "ref_audio_anchor"),
            onload_model_names=("video_vae", "audio_vae"),
        )

    @staticmethod
    def _nearest_multiple(value: float, multiple: int) -> int:
        return max(multiple, int(round(float(value) / multiple)) * multiple)

    def _resolve_reference_image_shape(self, pipe: MiniMaxH3Pipeline, width: int, height: int, short_edge: int):
        scale = short_edge * 1.0 / min(width, height)
        return self._nearest_multiple(width * scale, pipe.width_division_factor), self._nearest_multiple(
            height * scale, pipe.height_division_factor
        )

    def _resolve_reference_video_shape(
        self, pipe: MiniMaxH3Pipeline, width: int, height: int, short_edge: int, max_pixels: int
    ):
        scale = min(short_edge * 1.0 / min(width, height), float(np.sqrt(max_pixels / (width * height))))
        return self._nearest_multiple(width * scale, pipe.width_division_factor), self._nearest_multiple(
            height * scale, pipe.height_division_factor
        )

    def _trim_reference_video_length(self, pipe: MiniMaxH3Pipeline, frame_count: int) -> int:
        frame_count = int(frame_count)
        factor, remainder = pipe.time_division_factor, pipe.time_division_remainder
        chunks = max(1, (frame_count - remainder) // factor)
        used = chunks * factor + remainder
        assert used <= frame_count, (
            f"cannot trim {frame_count} reference video frames down to a valid "
            f"length: at least {used} frames are required"
        )
        return used

    def _encode_image_ref(self, pipe, img: Image.Image, short_edge: int):
        target_w, target_h = self._resolve_reference_image_shape(pipe, *img.size, short_edge)
        img = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        img_tensor = pipe.preprocess_image(img, torch_dtype=torch.float32, min_value=0)
        z = pipe.video_vae.encode_video(
            img_tensor.to(pipe.device),
            dtype=pipe.torch_dtype,
            process_image=True,
        )  # [1,24,1,H',W']
        return patchify_video(z), z.shape[-2], z.shape[-1], img

    def _encode_video_ref(self, pipe, frames, target_frame_count: int, short_edge: int, max_pixels: int):
        target_w, target_h = self._resolve_reference_video_shape(pipe, *frames[0].size, short_edge, max_pixels)
        frames = [f.resize((target_w, target_h), Image.LANCZOS).convert("RGB") for f in frames]
        frames = frames[: self._trim_reference_video_length(pipe, min(len(frames), target_frame_count))]
        frames_tensor = pipe.preprocess_video(frames, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        z = pipe.video_vae.encode_video(
            frames_tensor,
            dtype=pipe.torch_dtype,
            process_image=False,
        )  # [1,24,T',H',W']
        return patchify_video(z), int(z.shape[2]), int(z.shape[3]), int(z.shape[4]), frames

    def _encode_audio_ref(self, pipe, waveform, sample_rate: int):
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        waveform = resample_waveform(
            convert_to_stereo(waveform).float(), sample_rate, pipe.audio_vae.sample_rate
        )  # [C, T]
        latent = pipe.audio_vae.encode_audio(waveform[:2].to(pipe.device), dtype=pipe.torch_dtype)  # [C, 32, T]
        return pack_audio(latent), latent.shape[-1]

    @staticmethod
    def _require(ref, key, kind):
        value = ref.get(key)
        assert value is not None, f"reference type {kind!r} requires field {key!r}"
        return value

    def _build_block(
        self, pipe, ref, target_frame_count, ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels
    ):
        kind = ref["type"]
        if kind == "image":
            rows, lh, lw, prepared = self._encode_image_ref(
                pipe, self._require(ref, "image", kind), ref_image_short_edge
            )
            return {
                "kind": kind,
                "visual_clean": rows,
                "latent_t": 1,
                "latent_h": lh,
                "latent_w": lw,
                "prepared_image": prepared,
                "ref_audio_t": 0,
            }
        if kind in ("video", "video_audio"):
            if kind == "video" and ref.get("audio") is not None:
                raise ValueError("reference type 'video' is silent; use 'video_audio' to pass a soundtrack")
            frames = self._require(ref, "video", kind)
            if kind == "video_audio":
                audio_waveform = self._require(ref, "audio", kind)
                audio_sample_rate = int(self._require(ref, "sample_rate", kind))
            rows, lt, lh, lw, prepared = self._encode_video_ref(
                pipe, frames, target_frame_count, ref_video_short_edge, ref_video_max_pixels
            )
            block = {
                "kind": kind,
                "visual_clean": rows,
                "latent_t": lt,
                "latent_h": lh,
                "latent_w": lw,
                "prepared_frames": prepared,
                "ref_audio_t": 0,
            }
            if kind == "video_audio":
                audio_rows, ref_audio_t = self._encode_audio_ref(pipe, audio_waveform, audio_sample_rate)
                block["audio_clean"], block["ref_audio_t"] = audio_rows, ref_audio_t
            return block
        if kind == "audio":
            rows, ref_audio_t = self._encode_audio_ref(
                pipe, self._require(ref, "audio", kind), int(self._require(ref, "sample_rate", kind))
            )
            return {"kind": kind, "audio_clean": rows, "ref_audio_t": ref_audio_t}

    def process(
        self,
        pipe: MiniMaxH3Pipeline,
        references,
        seed,
        video_latents,
        num_frames,
        ref_image_short_edge,
        ref_video_short_edge,
        ref_video_max_pixels,
    ):
        if not references:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        seed = int(seed) if seed is not None else 42
        device = pipe.device

        ref_blocks_out = [
            self._build_block(pipe, ref, num_frames, ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels)
            for ref in references
        ]
        visual_blocks = [b for b in ref_blocks_out if "visual_clean" in b]
        imgvid_cond_num_frames = len(visual_blocks)
        visual_parts = []
        for block in visual_blocks:
            clean = block.pop("visual_clean").to(device=device, dtype=pipe.torch_dtype)
            if pipe.imgvid_cond_noise_aug == 1.0:
                anchor = clean
            else:
                full_t = video_latents.shape[2] + imgvid_cond_num_frames
                noise_shape = (1, 24, full_t, int(block["latent_h"]), int(block["latent_w"]))
                noise = pipe.generate_noise(
                    noise_shape,
                    seed=seed,
                    rand_device="cpu",
                    rand_torch_dtype=pipe.torch_dtype,
                    device="cpu",
                    torch_dtype=pipe.torch_dtype,
                )
                noise_rows = patchify_video(noise[:, :, : int(block["latent_t"])]).to(
                    device=device, dtype=pipe.torch_dtype
                )
                ts = torch.tensor(pipe.imgvid_cond_noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise_rows
            visual_parts.append(anchor)

        audio_parts = []
        for block in ref_blocks_out:
            if "audio_clean" not in block or int(block["ref_audio_t"]) <= 0:
                block.pop("audio_clean", None)
                continue
            clean = block.pop("audio_clean").to(device=device, dtype=pipe.torch_dtype)
            if pipe.audio_cond_noise_aug == 1.0:
                anchor = clean
            else:
                noise = pipe.generate_noise(
                    clean.shape,
                    seed=seed + 1,
                    rand_device="cpu",
                    rand_torch_dtype=pipe.torch_dtype,
                    device=device,
                    torch_dtype=pipe.torch_dtype,
                )
                ts = torch.tensor(pipe.audio_cond_noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise
            audio_parts.append(anchor)

        return {
            "ref_blocks": ref_blocks_out,
            "ref_visual_anchor": torch.cat(visual_parts, dim=0) if visual_parts else None,
            "ref_audio_anchor": torch.cat(audio_parts, dim=0) if audio_parts else None,
        }


class MiniMaxH3Unit_PackedSequenceBuilder(PipelineUnit):
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params_nega={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params=("video_latents", "audio_latents", "keyframe_cond_anchor", "keyframe_indices", "ref_blocks"),
            output_params=("packed",),
        )

    @staticmethod
    def _to_device(packed: dict, device) -> dict:
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in packed.items()}

    def _aligned_seq_len(self, used: int) -> int:
        return ((used + self._SEQ_ALIGN - 1) // self._SEQ_ALIGN) * self._SEQ_ALIGN

    def _axis_from_sqrt_area(self, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * self._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    def _video_t_grid(self, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor(
            [self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n)], dtype=torch.float64
        )
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    def _temporal_position_span(self, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * self._FRAME_RESCALE
        for token_index in range(self._T_GROUP):
            spans[token_index :: self._T_GROUP] *= self._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    def _video_t_span(self, n: int) -> float:
        return sum(self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n))

    def _frame_grid(self, latent_h: int, latent_w: int, sqrt_area):
        h_grid = self._axis_from_sqrt_area(latent_h, 2, sqrt_area)
        w_grid = self._axis_from_sqrt_area(latent_w, 2, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid

    def _video_grid(self, latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
        video_g = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = self._video_t_grid(latent_t, origin)[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    def _audio_w_axis(self, w_grid: torch.Tensor, audio_t: int, audio_rows: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
                torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
            ]
        )

    def _build_packed_fl2va(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_indices, audio_channel=2):
        """fl2va layout: [text | cond | audio | video | pad]."""
        frame_rows = (latent_h // 2) * (latent_w // 2)
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + cond_rows + audio_rows + video_rows
        seq_len = self._aligned_seq_len(used)

        text_sl = slice(0, text_len)
        cond_sl = slice(text_len, text_len + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        # img_pos covers both cond AND video rows, conditions first
        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_h * latent_w)
        frame, w_grid = self._frame_grid(latent_h, latent_w, sqrt_area)

        # Condition rows: temporal position depends on frame_index
        temporal_span = self._temporal_position_span(latent_t)
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = float(text_len)
            else:  # idx == -1
                cond_t = float(text_len) + temporal_span - self._FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start : cond_sl.start + sl.stop] = cond_g

        g[video_sl] = self._video_grid(latent_t, frame, float(text_len))
        g[audio_sl, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[audio_sl, 2] = self._audio_w_axis(w_grid, audio_t, audio_rows)

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0  # both cond and video rows are tagged as video (0)

        return {
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": torch.arange(0, text_len),
            "img_position_ids": g[None],
            "token_tags": token_tags,
            "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
            "seq_len": seq_len,
        }

    def _build_packed_ref2va(self, text_len, latent_t, latent_h, latent_w, audio_t, ref_blocks, audio_channel=2):
        """ref2va layout: [text | ref_0 | ref_1 | ... | target_audio | target_video | pad]"""
        ph, pw = latent_h // 2, latent_w // 2
        target_frame_rows = ph * pw
        target_video_rows = latent_t * target_frame_rows
        target_audio_rows = audio_t * audio_channel

        block_dims, total_ref_visual_rows, total_ref_audio_rows = [], 0, 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0, "ref_audio_t": int(b.get("ref_audio_t", 0))}
            if kind in ("image", "video", "video_audio"):
                lt_r, lh_r, lw_r = int(b["latent_t"]), int(b["latent_h"]), int(b["latent_w"])
                info.update(visual_rows=lt_r * (lh_r // 2) * (lw_r // 2), latent_t=lt_r, latent_h=lh_r, latent_w=lw_r)
                total_ref_visual_rows += info["visual_rows"]
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            elif kind == "audio":
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            else:
                raise ValueError(f"unknown ref kind: {kind}")
            block_dims.append(info)

        used = text_len + total_ref_visual_rows + total_ref_audio_rows + target_audio_rows + target_video_rows
        seq_len = self._aligned_seq_len(used)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[0:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[0:text_len] = 1

        # Iterate through ref blocks, placing them contiguously.
        cursor, t_cursor = text_len, float(text_len)
        ref_visual_pos_parts, ref_audio_pos_parts = [], []
        # Target w_grid used for audio W-axis (channel separation)
        target_sqrt_area = float(np.sqrt(latent_h * latent_w))
        target_w_grid = self._axis_from_sqrt_area(latent_w, 2, target_sqrt_area)

        for info in block_dims:
            kind = info["kind"]
            if kind == "image":
                v_rows, lh_r, lw_r = info["visual_rows"], info["latent_h"], info["latent_w"]
                # Own spatial grid
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)
                sl = slice(cursor, cursor + v_rows)
                g[sl, 0] = t_cursor
                g[sl, 1:] = frame
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += 1.0

            elif kind in ("video", "video_audio"):
                a_rows, v_rows, ref_at = info["audio_rows"], info["visual_rows"], info["ref_audio_t"]
                lt_r, lh_r, lw_r = info["latent_t"], info["latent_h"], info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, rv_w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)

                audio_sl = slice(cursor, cursor + a_rows)
                visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)

                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[audio_sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    # W-axis uses THIS reference video's own grid, not the target's.
                    g[audio_sl, 2] = self._audio_w_axis(rv_w_grid, ref_at, a_rows)
                    token_tags[audio_sl] = 2
                    ref_audio_pos_parts.append(torch.arange(audio_sl.start, audio_sl.stop))

                g[visual_sl] = self._video_grid(lt_r, frame, t_cursor)
                token_tags[visual_sl] = 0
                ref_visual_pos_parts.append(torch.arange(visual_sl.start, visual_sl.stop))

                cursor = visual_sl.stop
                t_cursor += max(float(ref_at), self._video_t_span(lt_r))

            elif kind == "audio":
                a_rows, ref_at = info["audio_rows"], info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    g[sl, 2] = self._audio_w_axis(target_w_grid, ref_at, a_rows)
                    token_tags[sl] = 2
                    ref_audio_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += a_rows
                t_cursor += float(ref_at)

        # Target audio + target video after ref blocks
        target_audio_sl = slice(cursor, cursor + target_audio_rows)
        target_video_sl = slice(target_audio_sl.stop, target_audio_sl.stop + target_video_rows)

        # Target spatial grid (own)
        frame_t, w_grid_t = self._frame_grid(latent_h, latent_w, target_sqrt_area)
        g[target_video_sl] = self._video_grid(latent_t, frame_t, t_cursor)

        target_audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
        g[target_audio_sl, 0] = target_audio_t_grid.repeat(audio_channel)
        g[target_audio_sl, 2] = self._audio_w_axis(w_grid_t, audio_t, target_audio_rows)

        # Build combined img_pos / audio_pos
        target_video_pos = torch.arange(target_video_sl.start, target_video_sl.stop)
        target_audio_pos = torch.arange(target_audio_sl.start, target_audio_sl.stop)

        if ref_visual_pos_parts:
            img_pos = torch.cat(ref_visual_pos_parts + [target_video_pos])
        else:
            img_pos = target_video_pos
        if ref_audio_pos_parts:
            audio_pos = torch.cat(ref_audio_pos_parts + [target_audio_pos])
        else:
            audio_pos = target_audio_pos

        token_tags[img_pos] = 0
        token_tags[target_audio_sl] = 2

        text_pos = torch.arange(0, text_len)
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

        return {
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "img_position_ids": g[None],
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "seq_len": seq_len,
        }

    def process(
        self,
        pipe: MiniMaxH3Pipeline,
        prompt_embeds,
        video_latents,
        audio_latents,
        text_token_tags=None,
        keyframe_cond_anchor=None,
        keyframe_indices=None,
        ref_blocks=None,
    ):
        text_len = prompt_embeds.shape[0]
        video_latent_t, latent_h, latent_w = video_latents.shape[2:]
        audio_latent_t = audio_latents.shape[-1]
        if ref_blocks is not None:
            packed = self._build_packed_ref2va(
                text_len, video_latent_t, latent_h, latent_w, audio_latent_t, ref_blocks
            )
        else:
            packed = self._build_packed_fl2va(
                text_len,
                video_latent_t,
                latent_h,
                latent_w,
                audio_latent_t,
                keyframe_indices if keyframe_cond_anchor is not None else [],
            )

        packed["token_tags"][packed["text_pos"]] = text_token_tags.cpu()
        if pipe.device == "mps":
            packed["img_position_ids"] = packed["img_position_ids"].to(torch.float32)
        return {"packed": self._to_device(packed, pipe.device)}


def model_fn_minimax_h3(
    dit,
    video_latents,
    audio_latents,
    packed,
    prompt_embeds,
    t_video,
    t_audio,
    keyframe_cond_anchor=None,
    ref_visual_anchor=None,
    ref_audio_anchor=None,
    imgvid_cond_noise_aug=0.999,
    audio_cond_noise_aug=1.0,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    dtype, device = video_latents.dtype, video_latents.device
    f, h, w = video_latents.shape[2:]
    audio_channel, audio_t = audio_latents.shape[0], audio_latents.shape[-1]
    video_rows = patchify_video(video_latents)
    audio_rows = pack_audio(audio_latents)

    img_pos = packed["img_pos"]
    audio_pos = packed["audio_pos"]
    text_pos = packed["text_pos"]
    cu = packed["cu_seqlens"]
    seq_len = packed["seq_len"]
    text_len = text_pos.shape[0]
    # Video Sequence
    cond_anchor = ref_visual_anchor if ref_visual_anchor is not None else keyframe_cond_anchor
    cond_rows_count = 0 if cond_anchor is None else cond_anchor.shape[0]
    x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
    x[0].index_copy_(0, img_pos[cond_rows_count:], video_rows)
    if cond_anchor is not None:
        x[0].index_copy_(0, img_pos[:cond_rows_count], cond_anchor)
    # Audio Sequence
    ref_audio_rows_count = 0 if ref_audio_anchor is None else ref_audio_anchor.shape[0]
    audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
    audio_x[0].index_copy_(0, audio_pos[ref_audio_rows_count:], audio_rows)
    if ref_audio_anchor is not None:
        audio_x[0].index_copy_(0, audio_pos[:ref_audio_rows_count], ref_audio_anchor)

    timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
    timesteps[audio_pos] = float(t_audio)
    timesteps[img_pos[:cond_rows_count]] = max(float(t_video), imgvid_cond_noise_aug)
    timesteps[audio_pos[:ref_audio_rows_count]] = max(float(t_audio), audio_cond_noise_aug)
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)

    # VeOmni DiT is the HF wrapper MiniMaxH3DiTModel: it runs the
    # same raw DiT forward (skip_mask_out_condition=True, update_mask=None),
    # slices cond rows, unpatchifies and negates, returning the same
    # (video_pred, audio_pred) pair. The wrapper
    # does NOT slice ref-audio rows, so ref2va audio references are
    # unsupported (t2va / fl2va are unaffected).
    if ref_audio_anchor is not None:
        raise NotImplementedError(
            "ref2va audio references are not supported by the VeOmni wrapper (MiniMaxH3DiTModel)."
        )
    outputs = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds,
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        skip_mask_out_condition=True,
        cond_rows=cond_rows_count,
        video_latent_shape=(f, h // 2, w // 2),
        audio_latent_shape=(audio_channel, audio_t),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    video_pred, audio_pred = outputs.predictions
    return video_pred, audio_pred
