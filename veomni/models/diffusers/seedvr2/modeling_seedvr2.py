"""SeedVR2 NaDiT registry model with packed prediction and supervised loss."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_seedvr2 import SeedVR2Config
from .core import na
from .core.nadit import NaDiT
from .core.runtime import require_local_sequence


@dataclass
class SeedVR2Output(ModelOutput):
    loss: dict | None = None
    predictions: list | None = None
    vid_sample: torch.Tensor | None = None


class SeedVR2Model(PreTrainedModel):
    config_class = SeedVR2Config
    supports_gradient_checkpointing = True
    _no_split_modules = ["NaMMSRTransformerBlock"]
    main_input_name = "vid"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.dit = NaDiT(**config.backbone_kwargs())
        self.post_init()

    def forward(
        self,
        vid=None,
        txt=None,
        vid_shape=None,
        txt_shape=None,
        timestep=None,
        training_target=None,
        **kwargs,
    ):
        require_local_sequence()
        if isinstance(vid, (list, tuple)):
            if not (len(vid) == len(txt) == len(timestep)):
                raise ValueError("Video, text, and timestep batch lengths must match.")
            if training_target is not None and len(training_target) != len(vid):
                raise ValueError("Target batch length must match video batch length.")
            predictions, losses = [], []
            for i, (video, text, step) in enumerate(zip(vid, txt, timestep)):
                packed_vid, vshape = na.flatten([video])
                packed_txt, tshape = na.flatten([text])
                prediction = self.dit(
                    vid=packed_vid,
                    txt=packed_txt,
                    vid_shape=vshape,
                    txt_shape=tshape,
                    timestep=step.reshape(-1),
                ).vid_sample.reshape(*video.shape[:-1], self.config.vid_out_channels)
                predictions.append(prediction)
                if training_target is not None:
                    target = training_target[i]
                    if target.shape != prediction.shape:
                        raise ValueError("Target shape must exactly match SeedVR2 prediction shape.")
                    losses.append(F.mse_loss(prediction.float(), target.float()))
            if not predictions:
                raise ValueError("SeedVR2 requires a nonempty batch.")
            return SeedVR2Output(
                loss={"mse_loss": torch.stack(losses).mean()} if losses else None,
                predictions=predictions,
            )
        if any(item is None for item in (vid, txt, vid_shape, txt_shape, timestep)):
            raise ValueError("Packed forward requires vid, txt, vid_shape, txt_shape, and timestep.")
        result = self.dit(vid=vid, txt=txt, vid_shape=vid_shape, txt_shape=txt_shape, timestep=timestep)
        loss = None
        if training_target is not None:
            if result.vid_sample.shape != training_target.shape:
                raise ValueError("Packed target shape must exactly match prediction.")
            loss = {"mse_loss": F.mse_loss(result.vid_sample.float(), training_target.float())}
        return SeedVR2Output(loss=loss, vid_sample=result.vid_sample)

    def restore_latents(self, low_quality_latents, text_embeddings, noise=None):
        """Official one-step, CFG=1 SR: z0 = noise - velocity at t=1000."""
        if noise is None:
            noise = torch.randn_like(low_quality_latents)
        if noise.shape != low_quality_latents.shape:
            raise ValueError("Noise and conditioning latents must have identical shapes.")
        conditioning = torch.cat((low_quality_latents, torch.ones_like(low_quality_latents[..., :1])), dim=-1)
        inputs = torch.cat((noise, conditioning), dim=-1)
        step = torch.tensor([1000.0], device=inputs.device)
        velocity = self(vid=[inputs], txt=[text_embeddings], timestep=[step]).predictions[0]
        return noise - velocity
