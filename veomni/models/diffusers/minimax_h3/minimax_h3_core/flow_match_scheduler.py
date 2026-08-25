"""FlowMatchScheduler — MiniMax-H3 training and inference methods."""

import torch


class FlowMatchScheduler:
    def __init__(self, shift: float = 12.0):
        self.num_train_timesteps = 1000
        self.shift = shift
        self.sigmas = None
        self.timesteps = None
        self.linear_timesteps_weights = None
        self.training = False

    @staticmethod
    def _set_timesteps_minimax_h3(num_inference_steps=50, denoising_strength=1.0, shift=12.0):
        num_train_timesteps = 1000
        base = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1]
        sigmas = shift * base / (1 + (shift - 1) * base)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def _set_training_weight(self):
        """Compute Gaussian training weights centered at timestep 500/1000."""
        steps = 1000
        x = self.sigmas * self.num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.sigmas, self.timesteps = self._set_timesteps_minimax_h3(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=self.shift,
        )
        if training:
            self._set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents: (1-sigma)*x0 + sigma*noise."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Velocity target: noise - clean.

        Together with model output negation (return -v), the prediction matches target.
        """
        target = noise - sample
        return target

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Gaussian timestep weight (not yet used in VeOmni MSE loss)."""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
