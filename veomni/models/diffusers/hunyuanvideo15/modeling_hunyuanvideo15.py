"""VeOmni adapter for Tencent's native 480p T2V backbone."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from ....distributed.parallel_state import get_parallel_state
from .configuration_hunyuanvideo15 import HunyuanVideo15Config
from .native.transformer import HunyuanVideo_1_5_DiffusionTransformer as NativeTransformer


@dataclass
class HunyuanVideo15Output(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: torch.Tensor | list[torch.Tensor] | None = None


class HunyuanVideo15Model(PreTrainedModel):
    config_class = HunyuanVideo15Config
    supports_gradient_checkpointing = True
    _no_split_modules = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]
    _supports_sdpa = True

    def __init__(self, config, **kwargs):
        super().__init__(config)
        native = NativeTransformer(**config.to_native_dict())
        # Adopt the native hierarchy without a "backbone." prefix. State keys
        # and tensor layouts stay compatible with the official transformer.
        for name, module in native.named_children():
            self.add_module(name, module)
        for name, value in vars(native).items():
            if not name.startswith("_") and name != "training":
                setattr(self, name, value)

        # Native constructors already applied their initialization (including
        # zero condition-type embeddings). Register HF metadata without replacing it.
        for module in self.modules():
            module._is_hf_initialized = True
        self.post_init()

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
