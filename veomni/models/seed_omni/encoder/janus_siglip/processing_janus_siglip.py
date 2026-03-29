import inspect
from typing import List, Optional

from transformers import BatchFeature
from transformers.image_utils import ImageInput

from ....transformers.janus.image_processing_janus import JanusImageProcessor
from ..base import BaseEncoderProcessorMixin


class JanusSiglipEncoderProcessor(BaseEncoderProcessorMixin, JanusImageProcessor):
    valid_kwargs = BaseEncoderProcessorMixin.valid_kwargs + list(
        inspect.signature(JanusImageProcessor.__init__).parameters.keys()
    )

    def __init__(
        self,
        token_size=None,
        token_num=576,
        image_size: int = 384,
        min_size: int = 14,
        image_mean: List[float] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: List[float] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        if token_size is None:
            token_size = [1, 24, 24]
        BaseEncoderProcessorMixin.__init__(self, token_num=token_num, token_size=token_size, **kwargs)
        JanusImageProcessor.__init__(
            self,
            image_size=image_size,
            min_size=min_size,
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            **kwargs,
        )

    def process(self, images: Optional[ImageInput] = None, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        features = super().preprocess(images=images, image_type="input", return_tensors=return_tensors)["pixel_values"]
        num_image_tokens = [self.token_num] * len(images)
        return BatchFeature(
            data={"features": features, "num_tokens": num_image_tokens},
            tensor_type=return_tensors,
        )
