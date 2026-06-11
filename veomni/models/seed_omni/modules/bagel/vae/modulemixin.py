"""SeedOmni graph hooks for BAGEL's latent VAE module."""

import math
from typing import Any, Dict

import torch
from PIL import Image

from ....conversation import ConversationItem
from ....module import ModuleMixin


class BagelVAEModuleMixin(ModuleMixin):
    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        return kwargs

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        return outputs

    def _decode_image_graph(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        item = self._image_gen_item(conversation_list)
        packed_latents = self._packed_latents(item)
        image_height, image_width = self._raw_image_size(item)
        latent_grid = self._unpack_latents(packed_latents, image_height, image_width)
        decoded = self.decode(latents=latent_grid)["pixel_values"]
        image = self._postprocess_image(decoded)
        item.value = image
        item.meta["decoded_image_ready"] = True
        return {
            "conversation_list": conversation_list,
            "bagel_decoded_pixel_values": decoded.detach(),
            "generated": {"type": "image", "value": image, "meta": {"image_size": [image_height, image_width]}},
        }

    def _image_gen_item(self, conversation_list: list[ConversationItem] | None) -> ConversationItem:
        if conversation_list is None:
            raise ValueError("BAGEL VAE graph decode requires conversation_list.")
        for item in conversation_list:
            if item.meta.get("bagel_role") == "image_gen_latent":
                return item
        raise ValueError("BAGEL VAE graph decode found no image generation latent item.")

    def _packed_latents(self, item: ConversationItem) -> torch.Tensor:
        value = item.value
        if not torch.is_tensor(value):
            value = item.meta.get("next_latents")
        if not torch.is_tensor(value):
            raise ValueError("BAGEL VAE graph decode requires final packed latents.")
        return value.detach().to(device=self.device)

    def _raw_image_size(self, item: ConversationItem) -> tuple[int, int]:
        value = item.meta.get("raw_image_size")
        if torch.is_tensor(value):
            flat = value.detach().reshape(-1)
            if flat.numel() >= 2:
                return int(flat[0].item()), int(flat[1].item())
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        raise ValueError("BAGEL VAE graph decode requires raw_image_size metadata.")

    def _unpack_latents(self, packed_latents: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        z_channels = int(self.config.z_channels)
        patch_area = int(packed_latents.shape[-1]) // z_channels
        patch_size = math.isqrt(patch_area)
        if patch_size * patch_size * z_channels != int(packed_latents.shape[-1]):
            raise ValueError("BAGEL packed latent dimension is incompatible with VAE z_channels.")
        latent_downsample = int(self.config.downsample) * patch_size
        if image_height % latent_downsample != 0 or image_width % latent_downsample != 0:
            raise ValueError("BAGEL image size is incompatible with VAE latent downsample.")
        h = image_height // latent_downsample
        w = image_width // latent_downsample
        expected_tokens = h * w
        if int(packed_latents.shape[0]) != expected_tokens:
            raise ValueError(
                f"BAGEL packed latent token count mismatch: got {packed_latents.shape[0]}, expected {expected_tokens}."
            )
        # Official decode_image stores each token as a patch of VAE channels, so restore the grid before decode.
        latents = packed_latents.reshape(1, h, w, patch_size, patch_size, z_channels)
        latents = torch.einsum("nhwpqc->nchpwq", latents)
        return latents.reshape(1, z_channels, h * patch_size, w * patch_size).to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _postprocess_image(decoded: torch.Tensor) -> Image.Image:
        image = (decoded * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())


__all__ = ["BagelVAEModuleMixin"]
