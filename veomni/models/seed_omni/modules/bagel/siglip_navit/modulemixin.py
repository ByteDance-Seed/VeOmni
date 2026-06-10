"""SeedOmni graph hooks for BAGEL's SigLIP NaViT module."""

from typing import Any, Dict, Optional

import torch

from ....conversation import ConversationItem
from ....module import ModuleMixin


class BagelSiglipNavitModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del method, conversation_list, kwargs
        # TODO(bagel-v2): build NaViT packed pixels, position ids, cu_seqlens, and max_seqlen.
        raise NotImplementedError("BagelSiglipNavit graph hooks are not implemented yet.")

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        del method, outputs
        # TODO(bagel-v2): scatter packed NaViT outputs after the real vision forward.
        raise NotImplementedError("BagelSiglipNavit graph hooks are not implemented yet.")

    def dummy_inputs(self) -> Dict[str, Any]:
        patch_dim = self.config.num_channels * self.config.patch_size * self.config.patch_size
        return {
            "packed_pixel_values": torch.zeros(1, patch_dim, device=self.device, dtype=self.dtype),
            "packed_flattened_position_ids": torch.zeros(1, dtype=torch.long, device=self.device),
        }

    def generate(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BagelSiglipNavit.generate requires conversation_list.")
        pending = [item for item in conversation_list if item.type == "image" and item.role == "user"]
        if not pending:
            return {"conversation_list": conversation_list}
        raise NotImplementedError("BagelSiglipNavit.generate for image prompts is not implemented yet.")


__all__ = ["BagelSiglipNavitModuleMixin"]
