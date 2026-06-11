"""SeedOmni graph hooks for BAGEL connector call sites."""

from typing import Any, Dict, Optional

import torch

from ....conversation import ConversationItem
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import ModuleMixin


SIGNAL_IMAGE_COMPLETE = "image_complete"


class BagelFlowConnectorModuleMixin(ModuleMixin):
    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        assert method in ("embed_latent", "decode_velocity", "forward")
        return kwargs

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("embed_latent", "decode_velocity", "forward")
        return outputs

    def _embed_latent_graph(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        item = self._image_gen_item(conversation_list, require_ready_text=True)
        self._ensure_image_gen_metadata(item, generation_kwargs)
        # At the start of a denoise iteration, item.value carries either x_t0 or the previous step's x_t1.
        latents = self._current_latents(item, prefer_value=True)
        position_ids = self._meta_tensor(item, "vae_position_ids", dtype=torch.long)
        timestep = self._current_timestep(item)

        latent_embeds = self.embed_latent(
            latents=latents,
            position_ids=position_ids,
            timesteps=timestep,
        )["latent_embeds"]
        text_embeds = item.meta.get("text_embeds")
        if not torch.is_tensor(text_embeds):
            raise ValueError("BAGEL image generation item requires text_embeds metadata.")
        text_embeds = text_embeds.detach().to(device=self.device, dtype=self.dtype)
        text_indexes = self._meta_tensor(item, "text_indexes", dtype=torch.long)
        vae_token_indexes = self._meta_tensor(item, "vae_token_indexes", dtype=torch.long)
        query_lens = self._meta_tensor(item, "query_lens", dtype=torch.int32)

        packed_sequence = text_embeds.new_zeros((int(query_lens.sum().item()), self.config.hidden_size))
        packed_sequence[text_indexes] = text_embeds
        packed_sequence[vae_token_indexes] = latent_embeds.to(dtype=packed_sequence.dtype)

        item.value = packed_sequence
        item.meta["current_latents"] = latents.detach()
        item.meta["latent_embeds"] = latent_embeds.detach()
        item.meta["flow_packed_sequence_ready"] = True
        item.meta.pop("flow_hidden_ready", None)
        return {
            "conversation_list": conversation_list,
            "bagel_last_latent_embeds": latent_embeds.detach(),
            "bagel_last_packed_sequence": packed_sequence.detach(),
        }

    def _decode_velocity_graph(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        bagel_last_hidden_state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        item = self._image_gen_item(conversation_list, require_hidden=True)
        hidden_states = item.value if torch.is_tensor(item.value) else bagel_last_hidden_state
        if hidden_states is None or not torch.is_tensor(hidden_states):
            raise ValueError("BAGEL flow decode requires hidden states from qwen2_mot.")

        velocity_all = self.decode_velocity(hidden_states=hidden_states)["velocity"]
        vae_token_indexes = self._meta_tensor(item, "vae_token_indexes", dtype=torch.long)
        velocity = velocity_all[vae_token_indexes]
        current_latents = self._current_latents(item)
        dt = self._current_dt(item)
        # Keep the Euler update shape-compatible with official BAGEL; dt is intentionally a 0-dim scalar.
        next_latents = current_latents - velocity.to(current_latents.device) * dt.to(current_latents.device)

        step_index = int(item.meta.get("flow_step_index", 0)) + 1
        item.value = next_latents.detach()
        item.meta["flow_step_index"] = step_index
        item.meta["velocity"] = velocity.detach()
        item.meta["next_latents"] = next_latents.detach()
        item.meta.pop("flow_packed_sequence_ready", None)
        item.meta.pop("flow_hidden_ready", None)

        output: Dict[str, Any] = {
            "conversation_list": conversation_list,
            "bagel_last_velocity": velocity.detach(),
            "bagel_last_x_t": next_latents.detach(),
        }
        if step_index >= int(item.meta.get("max_flow_steps", 1)):
            output[FSM_SIGNAL_KEY] = SIGNAL_IMAGE_COMPLETE
        return output

    def _ensure_image_gen_metadata(
        self,
        item: ConversationItem,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        kwargs = generation_kwargs or {}
        if "max_flow_steps" in kwargs:
            item.meta["max_flow_steps"] = int(kwargs["max_flow_steps"])
        if not torch.is_tensor(item.meta.get("timesteps")) or not torch.is_tensor(item.meta.get("dts")):
            num_timesteps = int(kwargs.get("num_timesteps", 50))
            if num_timesteps < 2:
                raise ValueError("BAGEL image generation requires num_timesteps >= 2.")
            timestep_shift = float(kwargs.get("timestep_shift", 3.0))
            timesteps = torch.linspace(1, 0, num_timesteps, device=self.device, dtype=torch.float32)
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
            item.meta["timesteps"] = timesteps[:-1]
            item.meta["dts"] = timesteps[:-1] - timesteps[1:]
        if "max_flow_steps" not in item.meta:
            item.meta["max_flow_steps"] = int(item.meta["timesteps"].numel())

        if not torch.is_tensor(item.value) or item.value.numel() == 0:
            height, width = self._raw_image_size(item)
            latent_downsample = int(kwargs.get("latent_downsample", 16))
            if height % latent_downsample != 0 or width % latent_downsample != 0:
                raise ValueError("BAGEL image size must be divisible by latent_downsample.")
            latent_h = height // latent_downsample
            latent_w = width // latent_downsample
            num_latent_tokens = latent_h * latent_w
            item.value = torch.randn(
                num_latent_tokens,
                self.config.patch_latent_dim,
                device=self.device,
                dtype=torch.float32,
            )

        if not torch.is_tensor(item.meta.get("vae_position_ids")):
            height, width = self._raw_image_size(item)
            latent_downsample = int(kwargs.get("latent_downsample", 16))
            item.meta["vae_position_ids"] = self._flattened_position_ids(
                height,
                width,
                latent_downsample,
                int(self.config.max_latent_size),
            )

        num_latent_tokens = int(item.value.shape[0])
        sequence_length = num_latent_tokens + 2
        kv_len = self._scalar_tensor_meta(item, "key_value_lens", default=0, dtype=torch.int32)
        rope = self._scalar_tensor_meta(item, "rope_after_prompt", default=int(kv_len.item()), dtype=torch.long)
        item.meta.setdefault("query_lens", torch.tensor([sequence_length], device=self.device, dtype=torch.int32))
        item.meta.setdefault(
            "text_indexes", torch.tensor([0, sequence_length - 1], device=self.device, dtype=torch.long)
        )
        item.meta.setdefault(
            "vae_token_indexes",
            torch.arange(1, sequence_length - 1, device=self.device, dtype=torch.long),
        )
        item.meta.setdefault(
            "position_ids",
            torch.full((sequence_length,), int(rope.item()), device=self.device, dtype=torch.long),
        )
        item.meta.setdefault(
            "sequence_indexes",
            torch.arange(
                int(kv_len.item()), int(kv_len.item()) + sequence_length, device=self.device, dtype=torch.long
            ),
        )
        item.meta.setdefault(
            "context_indexes",
            torch.arange(int(kv_len.item()), device=self.device, dtype=torch.long),
        )
        item.meta.setdefault("flow_step_index", 0)

    def _image_gen_item(
        self,
        conversation_list: Optional[list[ConversationItem]],
        *,
        require_ready_text: bool = False,
        require_hidden: bool = False,
    ) -> ConversationItem:
        if conversation_list is None:
            raise ValueError("BAGEL flow connector graph path requires conversation_list.")
        for item in conversation_list:
            if item.meta.get("bagel_role") != "image_gen_latent":
                continue
            if require_ready_text and not item.meta.get("text_embeds_ready"):
                raise ValueError("BAGEL image generation item requires text_embeds before flow embedding.")
            if require_hidden and not item.meta.get("flow_hidden_ready"):
                raise ValueError("BAGEL image generation item requires qwen hidden states before flow decode.")
            return item
        raise ValueError("BAGEL flow connector graph path found no image generation latent item.")

    def _current_latents(self, item: ConversationItem, *, prefer_value: bool = False) -> torch.Tensor:
        # Qwen writes hidden states into item.value, so decode_velocity must read the saved latent metadata instead.
        value = item.value if prefer_value and torch.is_tensor(item.value) else item.meta.get("current_latents")
        if not torch.is_tensor(value):
            value = item.value
        if not torch.is_tensor(value):
            raise ValueError("BAGEL image generation item requires latent tensor value.")
        return value.detach().to(device=self.device)

    def _current_timestep(self, item: ConversationItem) -> torch.Tensor:
        step_index = int(item.meta.get("flow_step_index", 0))
        timesteps = item.meta.get("timesteps")
        if torch.is_tensor(timesteps):
            timesteps = timesteps.detach().reshape(-1)
            if timesteps.numel() > 0:
                return timesteps[min(step_index, timesteps.numel() - 1)].reshape(1).to(device=self.device)
        timestep = item.meta.get("timestep")
        if torch.is_tensor(timestep):
            return timestep.detach().reshape(1).to(device=self.device)
        raise ValueError("BAGEL image generation item requires timestep metadata.")

    def _current_dt(self, item: ConversationItem) -> torch.Tensor:
        step_index = int(item.meta.get("flow_step_index", 0))
        dts = item.meta.get("dts")
        if torch.is_tensor(dts):
            dts = dts.detach().reshape(-1)
            if dts.numel() > 0:
                # A shape [1] dt changes CUDA bf16 rounding versus official BAGEL's scalar update.
                return dts[min(step_index, dts.numel() - 1)].to(device=self.device)
        dt = item.meta.get("dt")
        if torch.is_tensor(dt):
            return dt.detach().reshape(-1)[0].to(device=self.device)
        raise ValueError("BAGEL image generation item requires dt metadata.")

    def _meta_tensor(
        self,
        item: ConversationItem,
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = item.meta.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"BAGEL image generation item requires {key} metadata.")
        return value.detach().to(device=self.device, dtype=dtype).reshape(-1)

    def _raw_image_size(self, item: ConversationItem) -> tuple[int, int]:
        value = item.meta.get("raw_image_size")
        if torch.is_tensor(value):
            flat = value.detach().reshape(-1)
            if flat.numel() >= 2:
                return int(flat[0].item()), int(flat[1].item())
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        raise ValueError("BAGEL image generation item requires raw_image_size metadata.")

    def _scalar_tensor_meta(
        self,
        item: ConversationItem,
        key: str,
        *,
        default: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = item.meta.get(key)
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(-1)[:1]
        if isinstance(value, (list, tuple)) and value:
            return torch.tensor([int(value[0])], device=self.device, dtype=dtype)
        if isinstance(value, int):
            return torch.tensor([value], device=self.device, dtype=dtype)
        tensor = torch.tensor([default], device=self.device, dtype=dtype)
        item.meta[key] = tensor
        return tensor

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


__all__ = ["BagelFlowConnectorModuleMixin", "SIGNAL_IMAGE_COMPLETE"]
