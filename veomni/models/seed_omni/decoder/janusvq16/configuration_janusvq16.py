from ....transformers.janus.configuration_janus import JanusGenVisionConfig
from ..base import BaseDecoderConfigMixin


class JanusVQ16DecoderConfig(BaseDecoderConfigMixin, JanusGenVisionConfig):
    model_type = "janusvq16_decoder"

    def __init__(
        self,
        gen_aligner_depth: int = 2,
        gen_aligner_input_dim: int = 8,
        n_embed: int = 2048,
        gen_aligner_projector_type: str = "mlp_gelu",
        gen_head_embed: int = 2048,
        projector_train_from_scratch: bool = False,
        train_origin_projector: bool = False,
        **kwargs,
    ):
        self.gen_aligner_depth = gen_aligner_depth
        self.gen_aligner_input_dim = gen_aligner_input_dim
        self.n_embed = n_embed
        self.gen_aligner_projector_type = gen_aligner_projector_type
        self.gen_head_embed = gen_head_embed
        self.projector_train_from_scratch = projector_train_from_scratch
        self.train_origin_projector = train_origin_projector
        super().__init__(**kwargs)
