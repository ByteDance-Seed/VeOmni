"""Config for :class:`Cosmos3Vae`."""

from typing import Any, Optional

from ...module_configuration_base import OmniModuleConfig


class Cosmos3VaeConfig(OmniModuleConfig):
    """The released ``vae/config.json`` plus how a clip is cut for it.

    ``vae_config`` is that file verbatim, so the module builds the same
    ``AutoencoderKLWan`` on any device (meta included) and loads the weights from
    its own subfolder. The frame fields are the recipe's: the clip lasts
    ``duration * sample_fps`` frames, aligned down to ``frame_factor * k +
    frame_factor_remainder`` (the Wan VAE turns ``4k + 1`` frames into ``k + 1``
    latent frames) within ``[min_frames, max_frames]``, read as a fixed-stride
    prefix of the source clip and padded with the last frame when it is too short.
    """

    model_type = "cosmos3_vae"

    def __init__(
        self,
        vae_config: Optional[dict[str, Any]] = None,
        latent_patch_size: int = 2,
        sample_fps: float = 5.0,
        min_frames: int = 5,
        max_frames: int = 117,
        frame_factor: int = 4,
        frame_factor_remainder: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_config = dict(vae_config or {})
        self.latent_patch_size = latent_patch_size
        self.sample_fps = sample_fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frame_factor = frame_factor
        self.frame_factor_remainder = frame_factor_remainder
        if not 0 <= frame_factor_remainder < frame_factor:
            raise ValueError(f"frame_factor_remainder must be in [0, {frame_factor}), got {frame_factor_remainder}.")

    @property
    def spatial_scale(self) -> int:
        return int(self.vae_config.get("scale_factor_spatial", 16))

    @property
    def temporal_scale(self) -> int:
        return int(self.vae_config.get("scale_factor_temporal", 4))
