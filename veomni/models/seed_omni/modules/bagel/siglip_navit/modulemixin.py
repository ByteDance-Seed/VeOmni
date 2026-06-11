"""SeedOmni graph hooks for BAGEL's SigLIP NaViT module."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

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
        packed_pixel_values, position_ids, vit_token_lens = self._prepare_image_item(image_item)

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

    def _prepare_image_item(self, image_item: ConversationItem) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = image_item.value
        if torch.is_tensor(value) and value.dim() == 2:
            position_ids = image_item.meta.get("vit_position_ids")
            vit_token_lens = image_item.meta.get("vit_token_lens")
            if not torch.is_tensor(position_ids) or not torch.is_tensor(vit_token_lens):
                raise ValueError("Image ConversationItem requires vit_position_ids and vit_token_lens metadata.")
            return value, position_ids, vit_token_lens

        image_tensor = self._preprocess_raw_image(value)
        packed_pixel_values = self._patchify(image_tensor, self.config.patch_size)
        position_ids = self._flattened_position_ids(
            image_tensor.shape[-2],
            image_tensor.shape[-1],
            patch_size=self.config.patch_size,
            max_num_patches_per_side=self.config.vit_max_num_patch_per_side,
        )
        vit_token_lens = torch.tensor([packed_pixel_values.shape[0]], dtype=torch.int32)
        self._ensure_raw_image_metadata(
            image_item,
            image_tensor=image_tensor,
            packed_pixel_values=packed_pixel_values,
            position_ids=position_ids,
            vit_token_lens=vit_token_lens,
        )
        return packed_pixel_values, position_ids, vit_token_lens

    def _ensure_raw_image_metadata(
        self,
        image_item: ConversationItem,
        *,
        image_tensor: torch.Tensor,
        packed_pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        vit_token_lens: torch.Tensor,
    ) -> None:
        num_img_tokens = int(vit_token_lens.reshape(-1)[0].item())
        curr_kvlen = self._scalar_meta_int(
            image_item.meta.get("key_value_lens_before", image_item.meta.get("key_value_lens")),
            default=0,
        )
        curr_rope = self._scalar_meta_int(
            image_item.meta.get("rope_before", image_item.meta.get("rope_position")),
            default=0,
        )
        query_len = num_img_tokens + 2
        local_text_indexes = torch.tensor([0, num_img_tokens + 1], dtype=torch.long)
        local_vit_indexes = torch.arange(1, num_img_tokens + 1, dtype=torch.long)
        image_item.meta.setdefault("bagel_role", "image_und")
        image_item.meta.setdefault("raw_image_size", self._raw_image_size(image_item.value))
        image_item.meta["preprocessed_image_size"] = [int(image_tensor.shape[-1]), int(image_tensor.shape[-2])]
        image_item.meta["packed_vit_tokens"] = packed_pixel_values.detach()
        image_item.meta["vit_position_ids"] = position_ids.detach().to(dtype=torch.long)
        image_item.meta["vit_token_lens"] = vit_token_lens.detach().to(dtype=torch.int32)
        image_item.meta.setdefault("image_text_indexes", local_text_indexes)
        image_item.meta.setdefault("vit_token_indexes", local_vit_indexes)
        image_item.meta.setdefault("query_lens", torch.tensor([query_len], dtype=torch.int32))
        image_item.meta.setdefault(
            "position_ids",
            torch.full((query_len,), curr_rope, dtype=torch.long),
        )
        image_item.meta.setdefault(
            "sequence_indexes",
            torch.arange(curr_kvlen, curr_kvlen + query_len, dtype=torch.long),
        )
        image_item.meta.setdefault("context_indexes", torch.arange(curr_kvlen, dtype=torch.long))
        image_item.meta.setdefault("key_value_lens_before", torch.tensor([curr_kvlen], dtype=torch.int32))
        image_item.meta.setdefault("key_value_lens_after", torch.tensor([curr_kvlen + query_len], dtype=torch.int32))
        image_item.meta.setdefault("rope_after", torch.tensor([curr_rope + 1], dtype=torch.long))
        image_item.meta.setdefault("is_causal", False)

    def _preprocess_raw_image(self, image: Any) -> torch.Tensor:
        pil_image = self._to_rgb_pil(image)
        pil_image = self._resize_image(pil_image)
        array = np.array(pil_image, copy=True)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("BAGEL raw image preprocessing expects an RGB image.")
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(self.config.image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.config.image_std, dtype=torch.float32).view(-1, 1, 1)
        return tensor.sub_(mean).div_(std)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        stride = int(self.config.patch_size)
        max_size = int(self.config.image_size)
        min_size = int(self.config.min_image_size)
        max_pixels = int(self.config.max_pixels)

        scale = min(max_size / max(width, height), 1.0)
        scale = max(scale, min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale, stride)
        if new_width * new_height > max_pixels:
            scale = max_pixels / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale, stride)
        if max(new_width, new_height) > max_size:
            scale = max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale, stride)
        return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

    @staticmethod
    def _apply_scale(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
        new_width = round(width * scale)
        new_height = round(height * scale)
        return (
            max(stride, int(round(new_width / stride) * stride)),
            max(stride, int(round(new_height / stride) * stride)),
        )

    @staticmethod
    def _to_rgb_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif torch.is_tensor(image):
            tensor = image.detach().cpu()
            if tensor.dim() != 3:
                raise TypeError(f"BAGEL raw tensor image must be 3-D, got shape {tuple(tensor.shape)}.")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            if tensor.dtype.is_floating_point:
                tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)
            pil_image = Image.fromarray(tensor.numpy())
        else:
            raise TypeError(
                f"BAGEL image understanding expects PIL, numpy, raw tensor, or packed tokens, got {type(image).__name__}."
            )

        if pil_image.mode == "RGBA" or pil_image.info.get("transparency", None) is not None:
            rgba = pil_image.convert("RGBA")
            white = Image.new(mode="RGB", size=rgba.size, color=(255, 255, 255))
            white.paste(rgba, mask=rgba.split()[3])
            return white
        return pil_image.convert("RGB")

    @staticmethod
    def _patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
        channels, height, width = image.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("BAGEL preprocessed image height and width must be divisible by patch_size.")
        image = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
        image = torch.einsum("chpwq->hwpqc", image)
        return image.reshape(-1, patch_size**2 * channels)

    @staticmethod
    def _flattened_position_ids(
        height: int,
        width: int,
        *,
        patch_size: int,
        max_num_patches_per_side: int,
    ) -> torch.Tensor:
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        coords_h = torch.arange(0, num_patches_h, dtype=torch.long)
        coords_w = torch.arange(0, num_patches_w, dtype=torch.long)
        return (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()

    @staticmethod
    def _raw_image_size(image: Any) -> list[int]:
        if isinstance(image, Image.Image):
            return [int(image.size[0]), int(image.size[1])]
        if isinstance(image, np.ndarray):
            return [int(image.shape[1]), int(image.shape[0])]
        if torch.is_tensor(image) and image.dim() == 3:
            if image.shape[0] in (1, 3, 4):
                return [int(image.shape[2]), int(image.shape[1])]
            return [int(image.shape[1]), int(image.shape[0])]
        return []

    @staticmethod
    def _scalar_meta_int(value: Any, default: int) -> int:
        if torch.is_tensor(value) and value.numel() > 0:
            return int(value.detach().reshape(-1)[0].item())
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        if isinstance(value, int):
            return value
        return default


__all__ = ["BagelSiglipNavitModuleMixin"]
