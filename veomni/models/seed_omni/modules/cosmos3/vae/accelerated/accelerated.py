"""VeOmni-accelerated Cosmos3Vae: clip items in, latent items out."""

from typing import Any, Dict, Optional

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ...item_meta import LATENT_KEY
from ..configuration import Cosmos3VaeConfig
from ..modeling import Cosmos3Vae


class TrainingMixin(TrainingModuleMixin):
    config: Cosmos3VaeConfig
    device: torch.device
    dtype: torch.dtype

    @pre_forward("encode", "offline_encode")
    def encode_pre(
        self, conversation_list: Optional[list[list[ConversationItem]]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._clip_items = [
            item
            for item in iter_desired_items(conversation_list or [], types=["video"])
            if not item.meta.get(LATENT_KEY)
        ]
        return {
            "videos": [item.value.to(self.device, dtype=self.dtype, non_blocking=True) for item in self._clip_items]
        }

    @post_forward("encode", "offline_encode")
    def encode_post(self, latents: list[torch.Tensor]) -> Dict[str, Any]:
        # Written here, on the batch's own items: the endpoint body sees FSDP's input copies.
        for item, latent in zip(self._clip_items, latents):
            item.value = latent
            item.meta[LATENT_KEY] = True
        conversation = self._conversation_carrier
        self._conversation_carrier, self._clip_items = None, []
        return {"conversation_list": conversation}

    @pre_forward("online_process")
    def online_process_pre(
        self, conversation_list: Optional[list[list[ConversationItem]]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        del kwargs
        raw = sum(
            1 for item in iter_desired_items(conversation_list or [], types=["video"]) if not item.meta.get(LATENT_KEY)
        )
        if raw:
            raise ValueError(
                f"cosmos3_vae.online_process got {raw} clip(s) that were never encoded. It reads latents from the "
                "offline cache: train with `data.data_type: seedomni_cached` on a cache written by "
                "`train.training_task: offline_embedding`."
            )
        return {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._clip_items: list[ConversationItem] = []


class VeOmniMixin(BaseMixin, TrainingMixin):
    pass


class Cosmos3VaeAccelerated(VeOmniMixin, Cosmos3Vae):
    pass


__all__ = ["Cosmos3VaeAccelerated"]
