"""Native video preprocessing and resumable flow-matching RNG."""

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def prepare_video(video, height, width, num_frames):
    """T,C,H,W [0,255] -> B,C,T,H,W [-1,1], bicubic resize and center crop."""
    if video.ndim != 4 or video.shape[1] != 3 or video.shape[0] != num_frames:
        raise ValueError(f"Expected {num_frames},3,H,W decoded frames, got {tuple(video.shape)}")
    if height % 16 or width % 16 or num_frames % 4 != 1:
        raise ValueError("Expected spatial multiples of 16 and 4n+1 frames")
    h, w = video.shape[-2:]
    scale = max(height / h, width / w)
    video = TF.resize(
        video.float(), [round(h * scale), round(w * scale)], interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    video = TF.center_crop(video, [height, width]).clamp(0, 255)
    return (video / 127.5 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


class FlowMatchingConditioner:
    """Own all condition RNG, including VAE sampling, for runtime checkpoints."""

    def __init__(self, seed, device, shift=3.0, timesteps=1000):
        if shift <= 0 or timesteps <= 0:
            raise ValueError("shift and timesteps must be positive")
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.shift, self.timesteps = shift, timesteps

    def rng_state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())

    def sample_posterior(self, parameters):
        mean, logvar = parameters.chunk(2, dim=1)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=self.generator)
        return mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * noise

    def __call__(self, latents):
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device, generator=self.generator)
        # Official LOGNORM sampling and training shift (3), not inference shift (7).
        u = torch.randn((latents.shape[0],), device=latents.device, generator=self.generator)
        sigma = torch.sigmoid(u) * (1 - 2e-5) + 1e-5
        sigma = self.shift * sigma / (1 + (self.shift - 1) * sigma)
        fraction = sigma.view(-1, 1, 1, 1, 1)
        noisy = ((1 - fraction) * latents + fraction * noise).to(latents.dtype)
        zeros = latents.new_zeros((latents.shape[0], latents.shape[1] + 1, *latents.shape[2:]))
        return {
            "hidden_states": torch.cat([noisy, zeros], dim=1),
            "timestep": sigma * self.timesteps,
            "training_target": noise - latents,
        }
