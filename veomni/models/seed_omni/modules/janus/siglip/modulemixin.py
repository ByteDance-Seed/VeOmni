from typing import Any, Dict, List, Optional

import torch

from ....conversation import ConversationItem, collect_desired_values, iter_desired_items
from ....module import ModuleMixin


class JanusSiglipModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None

    # Training hooks

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        assert method == "forward"
        self._conversation_carrier = conversation_list
        pixel_values = self._pixels_from_raw_images(
            collect_desired_values(conversation_list, types=["image"], roles=["user"])
        )
        return {"pixel_values": pixel_values}

    def post_forward(
        self,
        method: str,
        image_embeds: torch.Tensor,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        assert method == "forward"
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        if is_dummy:
            assert image_embeds.shape[0] == 1
            image_embeds = image_embeds.squeeze(0)
            for sample in conversation:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=image_embeds,
                        role="dummy",
                        meta={"source": "janus_siglip"},
                    )
                )
        else:
            items = list(iter_desired_items(conversation, types=["image"], roles=["user"]))
            for item, emb in zip(items, image_embeds, strict=True):
                item.value = emb
        return {"conversation_list": conversation}

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> Optional[torch.Tensor]:
        if not raw_images:
            return None
        return self._processor(images=raw_images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vision_config or {}
        if isinstance(cfg, dict):
            h = cfg["image_size"]
            c = cfg["num_channels"]
        else:
            h = cfg.image_size
            c = cfg.num_channels
        return {
            "pixel_values": torch.zeros(1, c, h, h, device=self.device, dtype=self.dtype),
        }

    # Inference hooks
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        pending = [part for part in conversation_list if part.type == "image_output"]
        if not pending:
            pending = [part for part in conversation_list if part.type == "image" and part.role == "user"]

        if not pending:
            return {"conversation_list": conversation_list}

        embeds = self._encode_pixel_values(self._pixels_from_raw_images([part.value for part in pending]))
        for part, emb in zip(pending, embeds, strict=True):
            part.value = emb if emb.dim() == 2 else emb.squeeze(0)
            if part.type == "image_output":
                part.type = "image"
                assert part.role == "assistant"

        return {"conversation_list": conversation_list}


__all__ = ["JanusSiglipModuleMixin"]
