"""VeOmni-accelerated Cosmos3Mot: per-sample items in, one packed call, rows back on the items.

This hook is the only place that sees a micro-batch sample by sample: it reads
each sample's prompt embeddings (text item), generation-tower rows and latent
geometry (clip item), and hands the backbone the packed tensors plus the
per-sample lengths it builds positions and the pack layout from.
"""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state

from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ...core import build_latent_grid
from ...item_meta import FPS_KEY, LATENT_SHAPE_KEY
from ..configuration import Cosmos3MotConfig
from ..modeling import Cosmos3Mot


class TrainingMixin(TrainingModuleMixin):
    config: Cosmos3MotConfig

    @pre_forward("forward")
    def forward_pre(
        self, conversation_list: Optional[list[list[ConversationItem]]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        del kwargs
        if get_parallel_state().sp_size > 1:
            raise NotImplementedError(
                "Cosmos3 joint attention does not support sequence parallelism; set ulysses_size: 1."
            )
        patch = (1, self.config.latent_patch_size, self.config.latent_patch_size)
        hidden_und, hidden_gen, und_lengths, latent_grids, fps = [], [], [], [], []
        self._clip_items = []
        for index, sample in enumerate(conversation_list or []):
            prompts = [item for item in sample if item.type == "text" and not item.is_dummy]
            clips = [item for item in sample if item.type == "video" and not item.is_dummy]
            if len(prompts) != 1 or len(clips) != 1:
                raise ValueError(
                    f"Cosmos3 sample {index} has {len(prompts)} prompt(s) and {len(clips)} clip(s); expected one of each."
                )
            prompt, clip = prompts[0], clips[0]
            hidden_und.append(prompt.value)
            hidden_gen.append(clip.value)
            und_lengths.append(int(prompt.value.shape[0]))
            latent_grids.append(build_latent_grid(clip.meta[LATENT_SHAPE_KEY], patch))
            fps.append(clip.meta.get(FPS_KEY))
            self._clip_items.append(clip)
        self._conversation_carrier = conversation_list
        return {
            "hidden_und": torch.cat(hidden_und, dim=0),
            "hidden_gen": torch.cat(hidden_gen, dim=0),
            "und_lengths": und_lengths,
            "latent_grids": latent_grids,
            "fps": fps,
        }

    @post_forward("forward")
    def forward_post(self, hidden_gen: torch.Tensor) -> Dict[str, Any]:
        lengths = [int(item.value.shape[0]) for item in self._clip_items]
        for item, rows in zip(self._clip_items, hidden_gen.split(lengths)):
            item.value = rows
        conversation = self._conversation_carrier
        self._conversation_carrier, self._clip_items = None, []
        return {"conversation_list": conversation}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._clip_items: list[ConversationItem] = []


class VeOmniMixin(BaseMixin, TrainingMixin):
    pass


class Cosmos3MotAccelerated(VeOmniMixin, Cosmos3Mot):
    pass


__all__ = ["Cosmos3MotAccelerated"]
