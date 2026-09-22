"""VeOmni configuration for the official SeedVR2-3B NaDiT."""

from transformers import PretrainedConfig


class SeedVR2Config(PretrainedConfig):
    model_type = "seedvr2"
    condition_model_type = "seedvr2_condition"

    def __init__(
        self,
        vid_in_channels=33,
        vid_out_channels=16,
        vid_dim=2560,
        txt_in_dim=5120,
        txt_dim=None,
        emb_dim=None,
        heads=20,
        head_dim=128,
        expand_ratio=4,
        norm="fusedrms",
        norm_eps=1e-5,
        ada="single",
        qk_bias=False,
        qk_norm="fusedrms",
        patch_size=(1, 2, 2),
        num_layers=32,
        mm_layers=10,
        mlp_type="swiglu",
        patch_type="v1",
        rope_type="mmrope3d",
        rope_dim=128,
        window=(4, 3, 3),
        window_method=None,
        txt_in_norm="fusedln",
        vid_out_norm="fusedrms",
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vid_in_channels = vid_in_channels
        self.vid_out_channels = vid_out_channels
        self.vid_dim = vid_dim
        self.txt_in_dim = txt_in_dim
        self.txt_dim = vid_dim if txt_dim is None else txt_dim
        self.emb_dim = 6 * vid_dim if emb_dim is None else emb_dim
        self.heads = heads
        self.head_dim = head_dim
        self.expand_ratio = expand_ratio
        self.norm = norm
        self.norm_eps = norm_eps
        self.ada = ada
        self.qk_bias = qk_bias
        self.qk_norm = qk_norm
        self.patch_size = list(patch_size)
        self.num_layers = num_layers
        self.mm_layers = mm_layers
        self.mlp_type = mlp_type
        self.patch_type = patch_type
        self.rope_type = rope_type
        self.rope_dim = rope_dim
        self.window = list(window)
        self.window_method = window_method or [
            "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize" for i in range(num_layers)
        ]
        self.txt_in_norm = txt_in_norm
        self.vid_out_norm = vid_out_norm
        self.hidden_size = vid_dim
        self.num_hidden_layers = num_layers
        if vid_in_channels != 2 * vid_out_channels + 1:
            raise ValueError("SeedVR2 SR requires noise, conditioning latents, and one mask channel.")
        if len(self.patch_size) != 3 or min(self.patch_size) < 1:
            raise ValueError("patch_size must contain three positive values.")

    def backbone_kwargs(self):
        fields = (
            "vid_in_channels",
            "vid_out_channels",
            "vid_dim",
            "txt_in_dim",
            "txt_dim",
            "emb_dim",
            "heads",
            "head_dim",
            "expand_ratio",
            "norm",
            "norm_eps",
            "ada",
            "qk_bias",
            "qk_norm",
            "patch_size",
            "num_layers",
            "mm_layers",
            "mlp_type",
            "patch_type",
            "rope_type",
            "rope_dim",
            "window",
            "window_method",
            "txt_in_norm",
            "vid_out_norm",
        )
        return {**{name: getattr(self, name) for name in fields}, "block_type": "mmdit_sr"}
