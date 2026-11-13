"""Config for :class:`Cosmos3FlowConnector`. Defaults are the ``vision_sft_nano`` recipe's."""

from typing import Optional

from ...module_configuration_base import OmniModuleConfig


class Cosmos3FlowConnectorConfig(OmniModuleConfig):
    model_type = "cosmos3_flow_connector"

    def __init__(
        self,
        hidden_size: int = 4096,
        patch_latent_dim: int = 192,
        latent_patch_size: int = 2,
        timestep_scale: float = 0.001,
        loss_scale: float = 10.0,
        num_train_timesteps: int = 1000,
        train_time_video_distribution: str = "waver",
        shift: Optional[dict] = None,
        resolution: str = "256",
        conditioning_config: Optional[dict] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_latent_dim = patch_latent_dim
        self.latent_patch_size = latent_patch_size
        self.timestep_scale = timestep_scale
        self.loss_scale = loss_scale
        # The backbone sees ``sigma * num_train_timesteps``.
        self.num_train_timesteps = num_train_timesteps
        self.train_time_video_distribution = train_time_video_distribution
        # ``{resolution bucket: shift}`` of the sigma warp ``shift * t / (1 + (shift - 1) * t)``.
        self.shift = shift if shift is not None else {"256": 3, "480": 5, "720": 10}
        self.resolution = str(resolution)
        # ``{clean prefix latent frames: weight}``: 70% T2V, 20% I2V (first latent frame), 10% V2V (first two).
        self.conditioning_config = conditioning_config if conditioning_config is not None else {0: 0.7, 1: 0.2, 2: 0.1}
        # The sampling stream is seeded ``seed + dp_rank``, so data-parallel ranks draw independently.
        self.seed = seed
        self.shift_for_resolution()
        self.conditioning_probabilities()

    # Both readers normalize on every call: a composed model's ``model_config``
    # overwrite lands through ``update``, after ``__init__`` has run.

    def conditioning_probabilities(self) -> list[tuple[int, float]]:
        """Normalized ``[(clean_prefix_latent_frames, probability), ...]`` in configured order."""
        weights = [(int(str(frames).strip()), float(weight)) for frames, weight in self.conditioning_config.items()]
        total = sum(weight for _, weight in weights)
        if not weights or total <= 0 or any(frames < 0 or weight < 0 for frames, weight in weights):
            raise ValueError(f"invalid conditioning_config {self.conditioning_config!r}.")
        return [(frames, weight / total) for frames, weight in weights]

    def shift_for_resolution(self, resolution: Optional[str] = None) -> float:
        bucket = str(resolution if resolution is not None else self.resolution)
        shifts = {str(key): float(value) for key, value in self.shift.items()}
        if bucket not in shifts:
            raise ValueError(f"resolution {bucket!r} has no entry in shift={self.shift!r}.")
        return shifts[bucket]
