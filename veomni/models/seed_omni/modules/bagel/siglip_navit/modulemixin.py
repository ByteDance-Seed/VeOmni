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
            "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            "max_seqlen": 1,
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
        if len(pending) != 1:
            raise NotImplementedError(
                "BAGEL graph-level image understanding currently supports one image per request."
            )

        image_item = pending[0]
        packed_pixel_values = image_item.value
        if not torch.is_tensor(packed_pixel_values):
            raise TypeError(
                "BagelSiglipNavit.generate currently expects preprocessed packed image patch tokens in "
                f"ConversationItem.value, got {type(packed_pixel_values).__name__}."
            )
        position_ids = image_item.meta.get("vit_position_ids")
        vit_token_lens = image_item.meta.get("vit_token_lens")
        if not torch.is_tensor(position_ids) or not torch.is_tensor(vit_token_lens):
            raise ValueError("Image ConversationItem requires vit_position_ids and vit_token_lens metadata.")

        vit_token_lens = vit_token_lens.detach().to(device=self.device, dtype=torch.int32).reshape(-1)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_lens, dim=0), (1, 0)).to(torch.int32)
        outputs = self.forward(
            packed_pixel_values=packed_pixel_values.detach().to(device=self.device, dtype=self.dtype),
            packed_flattened_position_ids=position_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1),
            cu_seqlens=cu_seqlens,
            max_seqlen=int(vit_token_lens.max().item()),
        )
        image_embeds = outputs["image_embeds"]
        image_item.value = image_embeds
        image_item.meta["image_embeds"] = image_embeds.detach()
        image_item.meta["image_embeds_ready"] = True
        return {
            "conversation_list": conversation_list,
            "bagel_last_image_embeds": image_embeds.detach(),
        }


__all__ = ["BagelSiglipNavitModuleMixin"]
