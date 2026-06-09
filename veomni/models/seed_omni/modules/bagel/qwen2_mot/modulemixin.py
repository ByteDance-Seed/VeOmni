"""SeedOmni graph hooks for BAGEL's Qwen2 MoT backbone."""

from typing import Any, Dict, Optional

import torch

from veomni.utils.tensor_utils import naflatten

from ....conversation import ConversationItem, is_dummy
from ....module import ModuleMixin


class BagelQwen2MoTModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del method, conversation_list, kwargs
        # TODO(bagel-v2): build Bagel's packed training contract here.
        raise NotImplementedError("BagelQwen2MoT graph hooks are not implemented yet.")

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        del method, outputs
        # TODO(bagel-v2): scatter packed hidden states after the real backbone forward.
        raise NotImplementedError("BagelQwen2MoT graph hooks are not implemented yet.")

    def _pack_conversations(
        self,
        conversations: list[list[ConversationItem]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds_list: list[torch.Tensor] = []
        for sample in conversations:
            for item in sample:
                if is_dummy(item):
                    continue
                value = item.value
                if value.dim() == 3 and value.size(0) == 1:
                    value = value.squeeze(0)
                inputs_embeds_list.append(value.to(device=self.device, dtype=self.dtype))
        inputs_embeds, pack_shape = naflatten(inputs_embeds_list)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        return inputs_embeds, pack_shape

    def _scatter_hidden_states(
        self,
        conversation_list: list[list[ConversationItem]],
        hidden_states_list: list[torch.Tensor],
    ) -> None:
        hidden_iter = iter(hidden_states_list)
        for sample in conversation_list:
            for item in sample:
                if is_dummy(item):
                    continue
                item.value = next(hidden_iter)
        if next(hidden_iter, None) is not None:
            raise RuntimeError("BagelQwen2MoT hidden-state segment count exceeds conversation items.")


__all__ = ["BagelQwen2MoTModuleMixin"]
