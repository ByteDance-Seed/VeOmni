"""VeOmni-accelerated Cosmos3FlowConnector.

``embed_latent`` reads each clip item's latent and leaves the item holding its
generation-tower rows (the MoT reads them from there); the per-sample targets and
masks ride on the batch to ``decode_velocity``, the graph's only loss node.
"""

from typing import Any, Dict, Optional

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ...item_meta import LATENT_KEY, LATENT_SHAPE_KEY
from ..configuration import Cosmos3FlowConnectorConfig
from ..modeling import Cosmos3FlowConnector


class TrainingMixin(TrainingModuleMixin):
    config: Cosmos3FlowConnectorConfig

    @pre_forward("embed_latent")
    def embed_latent_pre(
        self, conversation_list: Optional[list[list[ConversationItem]]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._clip_items = list(iter_desired_items(conversation_list or [], types=["video"]))
        unencoded = [item for item in self._clip_items if not item.meta.get(LATENT_KEY)]
        if unencoded:
            raise ValueError(f"cosmos3_flow_connector.embed_latent got {len(unencoded)} clip(s) with no latent yet.")
        return {"latents": [item.value for item in self._clip_items]}

    @post_forward("embed_latent")
    def embed_latent_post(
        self,
        hidden_gen: torch.Tensor,
        gen_lengths: list[int],
        latent_shapes: list[tuple[int, int, int, int]],
        training_target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
    ) -> Dict[str, Any]:
        for item, rows, shape in zip(self._clip_items, hidden_gen.split(gen_lengths), latent_shapes):
            item.value = rows
            item.meta[LATENT_SHAPE_KEY] = shape
        conversation = self._conversation_carrier
        self._conversation_carrier, self._clip_items = None, []
        return {
            "conversation_list": conversation,
            "training_target": training_target,
            "condition_mask": condition_mask,
        }

    @pre_forward("decode_velocity")
    def decode_velocity_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        training_target: Optional[list[torch.Tensor]] = None,
        condition_mask: Optional[list[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if training_target is None or condition_mask is None:
            raise ValueError("cosmos3_flow_connector.decode_velocity needs embed_latent to run earlier in the graph.")
        clip_items = list(iter_desired_items(conversation_list or [], types=["video"]))
        return {
            "hidden_gen": torch.cat([item.value for item in clip_items], dim=0),
            "latent_shapes": [item.meta[LATENT_SHAPE_KEY] for item in clip_items],
            "training_target": training_target,
            "condition_mask": condition_mask,
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._clip_items: list[ConversationItem] = []


class VeOmniMixin(BaseMixin, TrainingMixin):
    pass


class Cosmos3FlowConnectorAccelerated(VeOmniMixin, Cosmos3FlowConnector):
    pass


__all__ = ["Cosmos3FlowConnectorAccelerated"]
