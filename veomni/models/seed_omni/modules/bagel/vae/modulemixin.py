"""SeedOmni graph hooks for BAGEL's latent VAE module."""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ....conversation import ConversationItem
from ....module import ModuleMixin


class BagelVAEModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        self._bagel_packed_batch: Optional[dict[str, Any]] = None

    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        bagel_packed_batch = kwargs.get("bagel_packed_batch")
        if bagel_packed_batch is not None:
            self._bagel_packed_batch = bagel_packed_batch
            height = width = max(16, int(self.config.downsample) * 2)
            return {
                "pixel_values": torch.zeros(
                    1,
                    int(self.config.in_channels),
                    height,
                    width,
                    device=self.device,
                    dtype=torch.float32,
                )
            }
        return kwargs

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("encode", "decode", "forward")
        batch = self._bagel_packed_batch
        self._bagel_packed_batch = None
        if batch is not None:
            return {"bagel_packed_batch": batch}
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

    def _encode_image_graph(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL VAE graph encode requires conversation_list.")
        image_item = self._vae_input_item(conversation_list, generation_kwargs)
        if image_item is None:
            return {"conversation_list": conversation_list}
        if image_item.meta.get("vae_context_ready"):
            return {"conversation_list": conversation_list}

        resized_image = self._resize_image(self._to_rgb_pil(image_item.value), generation_kwargs)
        image_tensor = self._image_to_normalized_tensor(resized_image)
        image_item.value = resized_image
        pixel_values = image_tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        latents = self.encode(pixel_values=pixel_values)["latents"]
        packed_latents = self._pack_latents(
            latents[0], image_tensor.shape[-2], image_tensor.shape[-1], generation_kwargs
        )
        context_item = self._build_vae_context_item(image_item, packed_latents, image_tensor, generation_kwargs)
        insert_idx = conversation_list.index(image_item)
        conversation_list.insert(insert_idx, context_item)

        image_item.meta["vae_context_ready"] = True
        image_item.meta["bagel_role"] = "image_und"
        image_item.meta["key_value_lens_before"] = context_item.meta["key_value_lens_after"].detach().cpu()
        image_item.meta["rope_before"] = context_item.meta["rope_after"].detach().cpu()
        return {
            "conversation_list": conversation_list,
            "bagel_last_vae_context_latents": packed_latents.detach(),
        }

    def _image_gen_item(self, conversation_list: list[ConversationItem] | None) -> ConversationItem:
        if conversation_list is None:
            raise ValueError("BAGEL VAE graph decode requires conversation_list.")
        for item in conversation_list:
            if item.meta.get("bagel_role") == "image_gen_latent":
                return item
        raise ValueError("BAGEL VAE graph decode found no image generation latent item.")

    def _vae_input_item(
        self,
        conversation_list: list[ConversationItem],
        generation_kwargs: Dict[str, Any] | None,
    ) -> ConversationItem | None:
        enable_from_kwargs = bool(generation_kwargs and generation_kwargs.get("enable_vae_context"))
        for item in conversation_list:
            if item.type != "image" or item.role != "user":
                continue
            if item.meta.get("vae_context_ready"):
                continue
            if item.meta.get("enable_vae_context") or item.meta.get("bagel_role") == "image_vae_input":
                return item
            if enable_from_kwargs:
                return item
        return None

    def _build_vae_context_item(
        self,
        image_item: ConversationItem,
        packed_latents: torch.Tensor,
        image_tensor: torch.Tensor,
        generation_kwargs: Dict[str, Any] | None,
    ) -> ConversationItem:
        kwargs = generation_kwargs or {}
        curr_kvlen = self._scalar_meta_int(
            image_item.meta.get("key_value_lens_before", image_item.meta.get("key_value_lens")),
            default=0,
        )
        curr_rope = self._scalar_meta_int(
            image_item.meta.get("rope_before", image_item.meta.get("rope_position")),
            default=0,
        )
        num_latent_tokens = int(packed_latents.shape[0])
        query_len = num_latent_tokens + 2
        local_text_indexes = torch.tensor([0, query_len - 1], device=self.device, dtype=torch.long)
        local_vae_indexes = torch.arange(1, query_len - 1, device=self.device, dtype=torch.long)
        height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        latent_downsample = int(kwargs.get("latent_downsample", 16))
        position_ids = self._flattened_position_ids(
            height,
            width,
            latent_downsample,
            int(kwargs.get("max_latent_size", 64)),
        )
        return ConversationItem(
            type="image",
            value=packed_latents.detach(),
            role="user",
            source="bagel_vae_context",
            meta={
                "bagel_role": "image_vae_context",
                "raw_image_size": [height, width],
                "preprocessed_image_size": [width, height],
                "text_indexes": local_text_indexes,
                "vae_token_indexes": local_vae_indexes,
                "vae_position_ids": position_ids.detach().to(device=self.device, dtype=torch.long),
                "query_lens": torch.tensor([query_len], device=self.device, dtype=torch.int32),
                "position_ids": torch.full((query_len,), curr_rope, device=self.device, dtype=torch.long),
                "sequence_indexes": torch.arange(
                    curr_kvlen, curr_kvlen + query_len, device=self.device, dtype=torch.long
                ),
                "context_indexes": torch.arange(curr_kvlen, device=self.device, dtype=torch.long),
                "key_value_lens_before": torch.tensor([curr_kvlen], device=self.device, dtype=torch.int32),
                "key_value_lens_after": torch.tensor([curr_kvlen + query_len], device=self.device, dtype=torch.int32),
                "rope_after": torch.tensor([curr_rope + 1], device=self.device, dtype=torch.long),
                "timestep": torch.tensor([float(kwargs.get("vae_context_timestep", 0.0))], device=self.device),
                "is_causal": False,
            },
        )

    def _packed_latents(self, item: ConversationItem) -> torch.Tensor:
        value = item.value
        if not torch.is_tensor(value):
            value = item.meta.get("next_latents")
        if not torch.is_tensor(value):
            raise ValueError("BAGEL VAE graph decode requires final packed latents.")
        return value.detach().to(device=self.device)

    def _pack_latents(
        self,
        latents: torch.Tensor,
        image_height: int,
        image_width: int,
        generation_kwargs: Dict[str, Any] | None,
    ) -> torch.Tensor:
        latent_patch_size = int((generation_kwargs or {}).get("latent_patch_size", 2))
        latent_downsample = int((generation_kwargs or {}).get("latent_downsample", 16))
        h = image_height // latent_downsample
        w = image_width // latent_downsample
        latents = latents[:, : h * latent_patch_size, : w * latent_patch_size]
        latents = latents.reshape(int(self.config.z_channels), h, latent_patch_size, w, latent_patch_size)
        latents = torch.einsum("chpwq->hwpqc", latents)
        return latents.reshape(-1, latent_patch_size * latent_patch_size * int(self.config.z_channels)).detach()

    def _preprocess_raw_image(self, image: Any, generation_kwargs: Dict[str, Any] | None) -> torch.Tensor:
        pil_image = self._resize_image(self._to_rgb_pil(image), generation_kwargs)
        return self._image_to_normalized_tensor(pil_image)

    def _image_to_normalized_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        array = np.array(pil_image, copy=True)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("BAGEL VAE raw image preprocessing expects an RGB image.")
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(self.config.image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.config.image_std, dtype=torch.float32).view(-1, 1, 1)
        return tensor.sub_(mean).div_(std)

    def _resize_image(self, image: Image.Image, generation_kwargs: Dict[str, Any] | None) -> Image.Image:
        kwargs = generation_kwargs or {}
        width, height = image.size
        stride = int(kwargs.get("vae_image_stride", self.config.image_stride))
        max_size = int(kwargs.get("vae_max_image_size", self.config.max_image_size))
        min_size = int(kwargs.get("vae_min_image_size", self.config.min_image_size))
        max_pixels = int(kwargs.get("vae_max_pixels", self.config.max_pixels))
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
                raise TypeError(f"BAGEL VAE raw tensor image must be 3-D, got shape {tuple(tensor.shape)}.")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            if tensor.dtype.is_floating_point:
                tensor = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
            else:
                tensor = tensor.to(torch.uint8)
            pil_image = Image.fromarray(tensor.numpy())
        else:
            raise TypeError(f"BAGEL VAE context expects PIL, numpy, or raw tensor input, got {type(image).__name__}.")
        if pil_image.mode == "RGBA" or pil_image.info.get("transparency", None) is not None:
            rgba = pil_image.convert("RGBA")
            white = Image.new(mode="RGB", size=rgba.size, color=(255, 255, 255))
            white.paste(rgba, mask=rgba.split()[3])
            return white
        return pil_image.convert("RGB")

    def _raw_image_size(self, item: ConversationItem) -> tuple[int, int]:
        value = item.meta.get("raw_image_size")
        if torch.is_tensor(value):
            flat = value.detach().reshape(-1)
            if flat.numel() >= 2:
                return int(flat[0].item()), int(flat[1].item())
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        raise ValueError("BAGEL VAE graph decode requires raw_image_size metadata.")

    @staticmethod
    def _flattened_position_ids(
        height: int,
        width: int,
        patch_size: int,
        max_num_patches_per_side: int,
    ) -> torch.Tensor:
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        coords_h = torch.arange(0, num_patches_h, dtype=torch.long)
        coords_w = torch.arange(0, num_patches_w, dtype=torch.long)
        return (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()

    @staticmethod
    def _scalar_meta_int(value: Any, default: int) -> int:
        if torch.is_tensor(value) and value.numel() > 0:
            return int(value.detach().reshape(-1)[0].item())
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        if isinstance(value, int):
            return value
        return default

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
