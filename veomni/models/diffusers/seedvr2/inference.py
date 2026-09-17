"""Single-process SeedVR2 restoration using the official 3B checkpoint."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from .conditioning_seedvr2 import SeedVR2ConditionConfig, SeedVR2ConditionModel
from .configuration_seedvr2 import SeedVR2Config
from .modeling_seedvr2 import SeedVR2Model


def read_media(path):
    """Decode an RGB image/video into T,C,H,W floats and optional frame rate."""
    path = Path(path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        with Image.open(path) as source:
            frames = torch.from_numpy(np.array(source.convert("RGB"), copy=True))
        return frames.permute(2, 0, 1)[None].float() / 255, None
    import av

    with av.open(str(path)) as source:
        if not source.streams.video:
            raise ValueError(f"No video stream in {path}")
        fps = source.streams.video[0].average_rate
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1) for frame in source.decode(video=0)
        ]
    if not frames or fps is None:
        raise ValueError(f"Video must contain frames and a frame rate: {path}")
    return torch.stack(frames).float() / 255, fps


def load_official_model(weights, config=None):
    """Load all source keys strictly; prefixing dit. is the only namespace change."""
    config = SeedVR2Config() if config is None else config
    state = torch.load(Path(weights) / "seedvr2_ema_3b.pth", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected the official flat tensor state dictionary.")
    with torch.device("meta"):
        model = SeedVR2Model(config)
    mapped = {"dit." + key: value for key, value in state.items()}
    model.load_state_dict(mapped, strict=True, assign=True)
    return model


def preprocess_video(frames, target_area):
    """Official area-resize, divisible center crop, and [-1, 1] normalization."""
    if frames.ndim != 4 or frames.shape[1] != 3 or frames.shape[0] < 1:
        raise ValueError("Expected a nonempty T,C,H,W RGB video in [0, 1].")
    if not frames.is_floating_point() or not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("Frames must be finite floating-point values in [0, 1].")
    height, width = frames.shape[-2:]
    scale = math.sqrt(target_area / (height * width))
    resized = (round(height * scale), round(width * scale))
    cropped = tuple(size - size % 16 for size in resized)
    if min(cropped) < 16:
        raise ValueError("Target dimensions after divisible cropping must be at least 16.")
    frames = TVF.resize(frames, resized, InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    return (TVF.center_crop(frames, cropped) * 2 - 1).permute(1, 0, 2, 3).contiguous()


class SeedVR2Restorer:
    def __init__(self, weights, device="cpu", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = load_official_model(weights).eval()
        self.condition = SeedVR2ConditionModel(SeedVR2ConditionConfig(base_model_path=str(weights))).eval()

    @torch.inference_mode()
    def __call__(self, frames, target_area=720 * 1280, seed=666):
        """Restore T,C,H,W frames; offload VAE and DiT between stages."""
        # Seed both the sampled VAE posterior and the subsequent DiT noise.
        torch.manual_seed(seed)
        video = preprocess_video(frames, target_area)
        self.model.to("cpu")
        self.condition.to(device=self.device, dtype=self.dtype)
        latents = self.condition.encode_video(video)
        text = self.condition.text_embedding
        self.condition.to("cpu")
        self.model.to(self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            restored = self.model.restore_latents(latents, text)
        self.model.to("cpu")
        self.condition.to(device=self.device, dtype=self.dtype)
        samples = self.condition.decode_latents(restored)
        self.condition.to("cpu")
        return (samples[:, : frames.shape[0]].permute(1, 0, 2, 3).float().cpu().clamp(-1, 1) + 1) / 2
