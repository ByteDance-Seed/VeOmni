"""VeOmni adapter for Tencent's native 480p T2V backbone."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput

from ....distributed.parallel_state import get_parallel_state
from ...modeling_base import VeOmniPreTrainedModel
from .configuration_hunyuanvideo15 import HunyuanVideo15Config
from .native.modules.embed_layers import PatchEmbed, TimestepEmbedder
from .native.modules.mlp_layers import FinalLayer
from .native.modules.modulate_layers import ModulateDiT
from .native.modules.norm_layers import RMSNorm
from .native.modules.token_refiner import IndividualTokenRefinerBlock
from .native.transformer import HunyuanVideo_1_5_DiffusionTransformer as NativeTransformer


@dataclass
class HunyuanVideo15Output(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: torch.Tensor | list[torch.Tensor] | None = None


class HunyuanVideo15Model(VeOmniPreTrainedModel):
    config_class = HunyuanVideo15Config
    supports_gradient_checkpointing = True
    _no_split_modules = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]
    _supports_sdpa = True

    def __init__(self, config, attn_implementation=None, **kwargs):
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        if config._attn_implementation not in ("eager", "sdpa", "veomni_sdpa"):
            raise ValueError(
                f"HunyuanVideo 1.5 currently supports dense SDPA only, got {config._attn_implementation!r}"
            )
        super().__init__(config)
        native = NativeTransformer(**config.to_native_dict())
        # Adopt the native hierarchy without a "backbone." prefix. State keys
        # and tensor layouts stay compatible with the official transformer.
        for name, module in native.named_children():
            self.add_module(name, module)
        for name, value in vars(native).items():
            if not name.startswith("_") and name != "training":
                setattr(self, name, value)

        # HF 5.16 may skip parameterless containers during initialization. Attach
        # native recipes to the parameter-bearing leaves that it does dispatch.
        for module in self.modules():
            if isinstance(module, PatchEmbed):
                module.proj._hunyuan_init = "patch"
            elif isinstance(module, TimestepEmbedder):
                for index in (0, 2):
                    module.mlp[index]._hunyuan_init = "timestep"
            elif isinstance(module, ModulateDiT):
                module.linear._hunyuan_init = "zero"
            elif isinstance(module, (FinalLayer, IndividualTokenRefinerBlock)):
                module.adaLN_modulation[1]._hunyuan_init = "zero"
                if isinstance(module, FinalLayer):
                    module.linear._hunyuan_init = "zero"
        if config.use_cond_type_embedding:
            self.cond_type_embedding._hunyuan_init = "zero"

        # Eager native constructors already initialized real parameters. Meta
        # constructors have no values: leave their flags unset so HF can initialize
        # any parameters missing from a subsequently loaded checkpoint.
        if not any(parameter.is_meta for parameter in self.parameters()):
            for module in self.modules():
                module._is_hf_initialized = True
        self.post_init()
        # T2V retains the official vision state keys but never consumes images.
        if self.vision_in is not None:
            self.vision_in.requires_grad_(False)

    @torch.no_grad()
    def _init_weights(self, module):
        """Restore native initialization after meta parameters are materialized."""
        # HF initialize_weights guards torch.nn.init against overwriting loaded
        # tensors. Native RMSNorm uses fill_ directly, so respect its flag here.
        if isinstance(module, RMSNorm):
            if hasattr(module, "weight") and not getattr(module.weight, "_is_hf_initialized", False):
                module.reset_parameters()
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv3d, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.reset_parameters()
        recipe = getattr(module, "_hunyuan_init", None)
        if recipe == "zero":
            torch.nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif recipe == "timestep":
            torch.nn.init.normal_(module.weight, std=0.02)
        elif recipe == "patch":
            # Views do not inherit HF's per-parameter initialization flag.
            if not getattr(module.weight, "_is_hf_initialized", False):
                channels = self.config.in_channels
                image_weights = module.weight[:, :channels]
                torch.nn.init.xavier_uniform_(image_weights.view(image_weights.shape[0], -1))
                torch.nn.init.zeros_(module.weight[:, channels:])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.img_in

    get_rotary_pos_embed = NativeTransformer.get_rotary_pos_embed
    reorder_txt_token = NativeTransformer.reorder_txt_token
    unpatchify = NativeTransformer.unpatchify
    set_attn_mode = NativeTransformer.set_attn_mode

    def forward(
        self,
        hidden_states,
        timestep,
        text_states,
        encoder_attention_mask,
        byt5_text_states=None,
        byt5_text_mask=None,
        training_target=None,
        **kwargs,
    ):
        if get_parallel_state().sp_enabled:
            raise NotImplementedError("HunyuanVideo 1.5 sequence parallelism is not implemented; use ulysses_size=1")
        is_list = isinstance(hidden_states, (list, tuple))
        values = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_states": text_states,
            "encoder_attention_mask": encoder_attention_mask,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
            "training_target": training_target,
        }
        count = len(hidden_states) if is_list else 1
        if count == 0:
            raise ValueError("Cannot train on an empty batch")
        if is_list and any(
            v is not None and (not isinstance(v, (list, tuple)) or len(v) != count) for v in values.values()
        ):
            raise ValueError("All per-sample condition lists must have the same length")
        predictions, losses = [], []
        for i in range(count):
            sample = {k: (v[i] if is_list and v is not None else v) for k, v in values.items()}
            x = sample["hidden_states"]
            if x.ndim != 5 or x.shape[1] != 2 * self.config.in_channels + 1:
                raise ValueError("Expected B,(2*C+1),F,H,W concatenated T2V inputs")
            if any(size % patch for size, patch in zip(x.shape[-3:], self.config.patch_size)):
                raise ValueError("Latent dimensions must be divisible by patch_size")
            if self.config.glyph_byT5_v2 and (sample["byt5_text_states"] is None or sample["byt5_text_mask"] is None):
                raise ValueError("ByT5 states and validity mask are required (zero/masked for non-glyph captions)")
            prediction = NativeTransformer.forward(
                self,
                x,
                sample["timestep"],
                sample["text_states"],
                None,
                sample["encoder_attention_mask"].bool(),
                extra_kwargs={
                    "byt5_text_states": sample["byt5_text_states"],
                    "byt5_text_mask": sample["byt5_text_mask"],
                },
            )[0]
            predictions.append(prediction)
            if sample["training_target"] is not None:
                target = sample["training_target"]
                if prediction.shape != target.shape:
                    raise ValueError("Prediction and flow target shapes differ")
                losses.append(F.mse_loss(prediction.float(), target.float()))
        return HunyuanVideo15Output(
            loss={"mse_loss": torch.stack(losses).mean()} if losses else None,
            predictions=predictions if is_list else predictions[0],
        )
