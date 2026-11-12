# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLaDA-V: a LLaVA-style vision-language mask predictor.

The module tree mirrors the upstream ``LlavaLLaDAModelLM`` checkpoint, so its weights
load by name: ``model.vision_tower.vision_tower.vision_model`` (SigLIP), ``model.mm_projector``
(``mlp2x_gelu``), ``model.image_newline``, and the LLaDA backbone and ``lm_head``.

Vision is the simplified single-tile path the Open-VeOmni port uses: an image is one
``image_size`` tile, its patch features go through the projector, and an
``image_newline`` embedding closes every patch row. The upstream AnyRes tiling is not
implemented. The patch features are read before SigLIP's ``post_layernorm``, which is
loaded but never used.
"""

import torch
from torch import nn
from transformers import SiglipVisionModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from ..llada.dllm_utils import dllm_lm_head_loss, ulysses_local_image_slots
from ..llada.modeling_llada import LLaDAForCausalLM, LLaDAModel, LLaDAPreTrainedModel
from .configuration_llada_v import LLaDAVConfig


class LLaDAVVisionTower(nn.Module):
    """Places SigLIP at the checkpoint's ``vision_tower.vision_tower.vision_model`` prefix.

    The checkpoint predates transformers v5, whose `SiglipVisionModel` no longer nests a
    ``vision_model`` submodule, so the extra level is rebuilt here instead of renaming
    every key.
    """

    def __init__(self, config: LLaDAVConfig):
        super().__init__()
        self.vision_tower = nn.Module()
        # Pinned to sdpa so the tower cannot inherit `veomni_flash_*_with_sp` from the
        # backbone: under Ulysses that kernel would all-to-all the patch sequence, which
        # every rank holds whole.
        self.vision_tower.vision_model = SiglipVisionModel._from_config(
            config.vision_config, attn_implementation="sdpa"
        )


class LLaDAVPreTrainedModel(LLaDAPreTrainedModel):
    config: LLaDAVConfig
    _no_split_modules = ["LLaDADecoderLayer", "SiglipEncoderLayer"]

    def _init_weights(self, module):
        """Re-apply the constructor's special initializations after meta materialization.

        ``to_empty()`` on the meta random-init path discards the ``torch.zeros`` the
        constructor wrote for ``image_newline``, and no checkpoint path restores it, so a
        first image forward would read uninitialized values. Every other module keeps HF's
        own scheme: the base class defines no hook, so each parameter arrives initialized
        by the module that created it.
        """
        newline = getattr(module, "image_newline", None)
        if newline is not None:
            newline.data.zero_()


def _needs_symmetric_collectives() -> bool:
    """Whether every rank must enter the same vision collectives (distributed world size > 1)."""
    from ....distributed.parallel_state import get_parallel_state

    return get_parallel_state().world_size > 1


class LLaDAVModel(LLaDAVPreTrainedModel, LLaDAModel):
    def __init__(self, config: LLaDAVConfig):
        LLaDAModel.__init__(self, config)
        self.vision_tower = LLaDAVVisionTower(config)
        self.mm_projector = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.image_newline = nn.Parameter(torch.zeros(config.hidden_size))
        self.post_init()

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[num_images, 3, H, W]`` pixels to ``[num_images * image_seq_len, hidden]`` embeddings."""
        tower = self.vision_tower.vision_tower.vision_model
        outputs = tower(pixel_values.to(tower.dtype), output_hidden_states=True)
        features = self.mm_projector(outputs.hidden_states[self.config.mm_vision_select_layer])
        num_images, num_patches, width = features.shape
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError(f"LLaDA-V expects a square patch grid, got {num_patches} patches per image.")
        grid = features.view(num_images, side, side, width)
        newline = self.image_newline.to(grid.dtype).view(1, 1, 1, width).expand(num_images, side, 1, width)
        return torch.cat([grid, newline], dim=2).reshape(-1, width)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            if image_mask is None:
                raise ValueError("`pixel_values` needs the `image_mask` that marks where their embeddings go.")
            image_embeds = self.get_image_features(pixel_values).to(inputs_embeds.dtype)
            slots = image_mask.bool()
            if int(slots.sum()) != image_embeds.shape[0]:
                raise ValueError(
                    f"{int(slots.sum())} image placeholder tokens for {image_embeds.shape[0]} image embeddings; "
                    f"each image takes `image_seq_len` = {self.config.image_seq_len} tokens."
                )
            slots, image_embeds = ulysses_local_image_slots(slots, image_embeds, inputs_embeds.shape[1])
            inputs_embeds = inputs_embeds.masked_scatter(slots.unsqueeze(-1), image_embeds)
        elif self.training or _needs_symmetric_collectives():
            # A rank whose micro-batch has no image still has to run the tower, or its FSDP
            # all-gathers would not match those of ranks that did. This holds for eval as much
            # as for training: an eval forward under a distributed run with images on some
            # ranks only would hang the same way. Single-process inference skips it, since
            # there is no collective to keep in step.
            size = self.config.vision_config.image_size
            dummy = inputs_embeds.new_zeros(1, self.config.vision_config.num_channels, size, size)
            inputs_embeds = inputs_embeds + 0.0 * self.get_image_features(dummy).sum()
        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class LLaDAVForConditionalGeneration(LLaDAVPreTrainedModel, LLaDAForCausalLM):
    _tied_weights_keys = {}

    def __init__(self, config: LLaDAVConfig):
        LLaDAVPreTrainedModel.__init__(self, config)
        self.model = LLaDAVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _validate_checkpoint_missing_parameters(self, missing: set[str]) -> None:
        # The loader's fallback initializes a missing name by applying `_init_weights`
        # to its owner, which for `image_newline` is the whole `LLaDAVModel`: every
        # loaded weight would be overwritten.
        if missing:
            raise ValueError(
                f"The checkpoint lacks {sorted(missing)[:8]}. LLaDA-V is trained from a converted LLaDA-V "
                "checkpoint (scripts/convert_lladav_to_veomni.py), which holds every parameter."
            )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_weight: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if (labels is None) != (loss_weight is None):
            raise ValueError(
                "LLaDA-V is trained with the masked-diffusion objective: pass `labels` and `loss_weight` together "
                "(see the `llada_v` data transform)."
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_mask=image_mask,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        loss = None
        logits = None
        if labels is not None:
            loss = dllm_lm_head_loss(hidden_states, self.lm_head, labels, loss_weight)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["LLaDAVForConditionalGeneration", "LLaDAVModel", "LLaDAVPreTrainedModel"]
