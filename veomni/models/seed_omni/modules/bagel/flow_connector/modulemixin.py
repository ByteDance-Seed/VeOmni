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
        bagel_packed_batch = kwargs.get("bagel_packed_batch")
        if bagel_packed_batch is None:
            return kwargs
        self._bagel_packed_batch = bagel_packed_batch
        if method in ("embed_latent", "forward"):
            patched_latents = self._patched_latents(bagel_packed_batch)
            if patched_latents is None:
                return {
                    "latents": torch.zeros(
                        1,
                        int(self.config.patch_latent_dim),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "position_ids": torch.zeros(1, device=self.device, dtype=torch.long),
                    "timesteps": torch.zeros(1, device=self.device, dtype=torch.float32),
                }
            noised_latents, mse_target = patched_latents
            bagel_packed_batch["mse_target"] = mse_target
            return {
                "latents": noised_latents,
                "position_ids": bagel_packed_batch["packed_latent_position_ids"],
                "timesteps": bagel_packed_batch["shifted_timesteps"],
            }
        if "mse_loss_indexes" not in bagel_packed_batch or "packed_hidden_states" not in bagel_packed_batch:
            return {
                "hidden_states": torch.zeros(
                    1,
                    int(self.config.hidden_size),
                    device=self.device,
                    dtype=self.dtype,
                )
            }
        return {"hidden_states": bagel_packed_batch["packed_hidden_states"][bagel_packed_batch["mse_loss_indexes"]]}

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("embed_latent", "decode_velocity", "forward")
        batch = getattr(self, "_bagel_packed_batch", None)
        self._bagel_packed_batch = None
        if batch is not None:
            result: Dict[str, Any] = {"bagel_packed_batch": batch}
            if method in ("embed_latent", "forward"):
                if "fixed_noise" in batch:
                    batch["packed_latent_embeds"] = outputs["latent_embeds"]
                return result
            if "mse_target" in batch and "mse_loss_indexes" in batch:
                velocity = outputs["velocity"]
                mse_target = batch["mse_target"].to(device=velocity.device, dtype=velocity.dtype)
                mse = (velocity - mse_target).square()
                batch["mse_tensor"] = mse
                result["_loss"] = mse.mean()
            return result
        return outputs

    def _patched_latents(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
        if "fixed_noise" not in batch:
            return None
        h, w = batch["patchified_vae_latent_shapes"][0]
        padded_latent = batch["padded_latent"].to(device=self.device)
        patch_h = padded_latent.shape[2] // h
        patch_w = padded_latent.shape[3] // w
        clean = padded_latent.reshape(
            padded_latent.shape[0],
            padded_latent.shape[1],
            h,
            patch_h,
            w,
            patch_w,
        )
        clean = clean.permute(0, 2, 4, 3, 5, 1).flatten(0, 2).flatten(1, 3)
        timesteps = batch["shifted_timesteps"].to(device=clean.device).reshape(-1, 1)
        noise = batch["fixed_noise"].to(device=clean.device, dtype=clean.dtype)
        noised = (1.0 - timesteps) * clean + timesteps * noise
        target = noise - clean
        return noised, target

    def _embed_latent_graph(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        context_item = self._vae_context_item(conversation_list)
        if context_item is not None:
            return self._embed_vae_context_graph(context_item, conversation_list)

        if kwargs.get("past_key_values") is None:
            return {"conversation_list": conversation_list}

        item = self._image_gen_item(conversation_list, require_ready_text=True, allow_missing=True)
        if item is None:
            return {"conversation_list": conversation_list}
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

    def _embed_vae_context_graph(
        self,
        item: ConversationItem,
        conversation_list: Optional[list[ConversationItem]],
    ) -> Dict[str, Any]:
        latents = item.value
        if not torch.is_tensor(latents):
            raise ValueError("BAGEL VAE context item requires packed latent tensor value.")
        latents = latents.detach().to(device=self.device, dtype=self.dtype)
        position_ids = self._meta_tensor(item, "vae_position_ids", dtype=torch.long)
        timestep = self._current_timestep(item)
        latent_embeds = self.vae2llm(latents)
        latent_embeds = latent_embeds + self.time_embedder(timestep.reshape(-1)[:1])
        latent_embeds = latent_embeds + self.latent_pos_embed(position_ids)
        latent_embeds = latent_embeds.to(dtype=self.dtype)
        text_embeds = item.meta.get("text_embeds")
        if not torch.is_tensor(text_embeds):
            raise ValueError("BAGEL VAE context item requires text_embeds metadata.")
        text_embeds = text_embeds.detach().to(device=self.device, dtype=self.dtype)
        text_indexes = self._meta_tensor(item, "text_indexes", dtype=torch.long)
        vae_token_indexes = self._meta_tensor(item, "vae_token_indexes", dtype=torch.long)
        query_lens = self._meta_tensor(item, "query_lens", dtype=torch.int32)

        packed_sequence = text_embeds.new_zeros((int(query_lens.sum().item()), self.config.hidden_size))
        packed_sequence[text_indexes] = text_embeds
        packed_sequence[vae_token_indexes] = latent_embeds.to(dtype=packed_sequence.dtype)

        item.value = packed_sequence
        item.meta["vae_context_latents"] = latents.detach()
        item.meta["vae_context_latent_embeds"] = latent_embeds.detach()
        item.meta["vae_context_packed_sequence"] = packed_sequence.detach()
        item.meta["vae_context_sequence_ready"] = True
        return {
            "conversation_list": conversation_list,
            "bagel_last_vae_context_latent_embeds": latent_embeds.detach(),
            "bagel_last_vae_context_packed_sequence": packed_sequence.detach(),
        }

    def _decode_velocity_graph(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        bagel_last_hidden_state: Optional[torch.Tensor] = None,
        bagel_last_cfg_text_hidden_state: Optional[torch.Tensor] = None,
        bagel_last_cfg_img_hidden_state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        item = self._image_gen_item(conversation_list, require_hidden=True)
        hidden_states = item.value if torch.is_tensor(item.value) else bagel_last_hidden_state
        if hidden_states is None or not torch.is_tensor(hidden_states):
            raise ValueError("BAGEL flow decode requires hidden states from qwen2_mot.")

        velocity_all = self.decode_velocity(hidden_states=hidden_states)["velocity"]
        vae_token_indexes = self._meta_tensor(item, "vae_token_indexes", dtype=torch.long)
        base_velocity = velocity_all[vae_token_indexes]
        velocity = base_velocity
        cfg_text_velocity = None
        cfg_img_velocity = None
        cfg_text_is_active = self._cfg_text_active(item)
        cfg_img_is_active = self._cfg_img_active(item)
        if cfg_img_is_active and not cfg_text_is_active:
            raise ValueError(
                "Official BAGEL applies CFG-image after CFG-text; use cfg_text_scale > 1.0 with cfg_img_scale."
            )
        if cfg_text_is_active:
            cfg_text_hidden_states = item.meta.get("cfg_text_hidden_state", bagel_last_cfg_text_hidden_state)
            if not torch.is_tensor(cfg_text_hidden_states):
                raise ValueError("BAGEL CFG text guidance requires cfg_text hidden states from qwen2_mot.")
            cfg_text_velocity_all = self.decode_velocity(hidden_states=cfg_text_hidden_states)["velocity"]
            cfg_text_velocity = cfg_text_velocity_all[vae_token_indexes]
        if cfg_img_is_active:
            cfg_img_hidden_states = item.meta.get("cfg_img_hidden_state", bagel_last_cfg_img_hidden_state)
            if not torch.is_tensor(cfg_img_hidden_states):
                raise ValueError("BAGEL CFG image guidance requires cfg_img hidden states from qwen2_mot.")
            cfg_img_velocity_all = self.decode_velocity(hidden_states=cfg_img_hidden_states)["velocity"]
            cfg_img_velocity = cfg_img_velocity_all[vae_token_indexes]
        if cfg_text_velocity is not None:
            velocity = self._combine_cfg_velocity(base_velocity, cfg_text_velocity, cfg_img_velocity, item)
        current_latents = self._current_latents(item)
        dt = self._current_dt(item)
        # Keep the Euler update shape-compatible with official BAGEL; dt is intentionally a 0-dim scalar.
        next_latents = current_latents - velocity.to(current_latents.device) * dt.to(current_latents.device)

        step_index = int(item.meta.get("flow_step_index", 0)) + 1
        item.value = next_latents.detach()
        item.meta["flow_step_index"] = step_index
        item.meta["base_velocity"] = base_velocity.detach()
        item.meta["cfg_text_velocity"] = None if cfg_text_velocity is None else cfg_text_velocity.detach()
        item.meta["cfg_img_velocity"] = None if cfg_img_velocity is None else cfg_img_velocity.detach()
        item.meta["velocity"] = velocity.detach()
        item.meta["next_latents"] = next_latents.detach()
        item.meta.pop("flow_packed_sequence_ready", None)
        item.meta.pop("flow_hidden_ready", None)

        output: Dict[str, Any] = {
            "conversation_list": conversation_list,
            "bagel_last_base_velocity": base_velocity.detach(),
            "bagel_last_cfg_text_velocity": None if cfg_text_velocity is None else cfg_text_velocity.detach(),
            "bagel_last_cfg_img_velocity": None if cfg_img_velocity is None else cfg_img_velocity.detach(),
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
        self._ensure_cfg_metadata(item, kwargs)
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
        self._ensure_cfg_text_latent_metadata(item, sequence_length)
        self._ensure_cfg_img_latent_metadata(item)

    def _image_gen_item(
        self,
        conversation_list: Optional[list[ConversationItem]],
        *,
        require_ready_text: bool = False,
        require_hidden: bool = False,
        allow_missing: bool = False,
    ) -> Optional[ConversationItem]:
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
        if allow_missing:
            return None
        raise ValueError("BAGEL flow connector graph path found no image generation latent item.")

    def _vae_context_item(
        self,
        conversation_list: Optional[list[ConversationItem]],
    ) -> Optional[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BAGEL flow connector graph path requires conversation_list.")
        for item in conversation_list:
            if item.meta.get("bagel_role") != "image_vae_context":
                continue
            if item.meta.get("vae_context_sequence_ready"):
                continue
            if not item.meta.get("text_embeds_ready"):
                continue
            return item
        return None

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

    def _cfg_text_active(self, item: ConversationItem) -> bool:
        cfg_text_scale = float(item.meta.get("cfg_text_scale", 1.0))
        if cfg_text_scale <= 1.0:
            return False
        t_value = self._current_timestep_value(item)
        interval = item.meta.get("cfg_interval", [0.0, 1.0])
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            interval = [0.0, 1.0]
        return t_value > float(interval[0]) and t_value <= float(interval[1])

    def _cfg_img_active(self, item: ConversationItem) -> bool:
        cfg_img_scale = float(item.meta.get("cfg_img_scale", 1.0))
        if cfg_img_scale <= 1.0:
            return False
        t_value = self._current_timestep_value(item)
        interval = item.meta.get("cfg_interval", [0.0, 1.0])
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            interval = [0.0, 1.0]
        return t_value > float(interval[0]) and t_value <= float(interval[1])

    def _combine_cfg_velocity(
        self,
        base_velocity: torch.Tensor,
        cfg_text_velocity: torch.Tensor,
        cfg_img_velocity: Optional[torch.Tensor],
        item: ConversationItem,
    ) -> torch.Tensor:
        cfg_img_scale = float(item.meta.get("cfg_img_scale", 1.0))
        if cfg_img_scale > 1.0 and cfg_img_velocity is None:
            raise ValueError("BAGEL CFG image guidance requires cfg_img velocity.")
        cfg_text_scale = float(item.meta.get("cfg_text_scale", 1.0))
        cfg_renorm_min = float(item.meta.get("cfg_renorm_min", 0.0))
        cfg_renorm_type = str(item.meta.get("cfg_renorm_type", "global"))

        if cfg_renorm_type == "text_channel":
            text_guided = cfg_text_velocity + cfg_text_scale * (base_velocity - cfg_text_velocity)
            norm_base = torch.norm(base_velocity, dim=-1, keepdim=True)
            norm_text_guided = torch.norm(text_guided, dim=-1, keepdim=True)
            scale = (norm_base / (norm_text_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            text_guided = text_guided * scale
            if cfg_img_scale > 1.0:
                return cfg_img_velocity + cfg_img_scale * (text_guided - cfg_img_velocity)
            return text_guided

        text_guided = cfg_text_velocity + cfg_text_scale * (base_velocity - cfg_text_velocity)
        if cfg_img_scale > 1.0:
            guided = cfg_img_velocity + cfg_img_scale * (text_guided - cfg_img_velocity)
        else:
            guided = text_guided
        if cfg_renorm_type == "global":
            norm_base = torch.norm(base_velocity)
            norm_guided = torch.norm(guided)
        elif cfg_renorm_type == "channel":
            norm_base = torch.norm(base_velocity, dim=-1, keepdim=True)
            norm_guided = torch.norm(guided, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"{cfg_renorm_type} is not supported")
        scale = (norm_base / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return guided * scale

    def _ensure_cfg_metadata(self, item: ConversationItem, kwargs: Dict[str, Any]) -> None:
        for key, default in (
            ("cfg_text_scale", 1.0),
            ("cfg_img_scale", 1.0),
            ("cfg_renorm_min", 0.0),
        ):
            if key in kwargs and key not in item.meta:
                item.meta[key] = float(kwargs[key])
            else:
                item.meta.setdefault(key, default)
        if "cfg_interval" in kwargs and "cfg_interval" not in item.meta:
            item.meta["cfg_interval"] = list(kwargs["cfg_interval"])
        else:
            item.meta.setdefault("cfg_interval", [0.0, 1.0])
        if "cfg_renorm_type" in kwargs and "cfg_renorm_type" not in item.meta:
            item.meta["cfg_renorm_type"] = str(kwargs["cfg_renorm_type"])
        else:
            item.meta.setdefault("cfg_renorm_type", "global")

    def _ensure_cfg_text_latent_metadata(self, item: ConversationItem, sequence_length: int) -> None:
        if float(item.meta.get("cfg_text_scale", 1.0)) <= 1.0:
            return
        item.meta.setdefault(
            "cfg_text_position_ids",
            torch.zeros(sequence_length, device=self.device, dtype=torch.long),
        )
        item.meta.setdefault(
            "cfg_text_sequence_indexes",
            torch.arange(sequence_length, device=self.device, dtype=torch.long),
        )
        item.meta.setdefault("cfg_text_key_value_lens", torch.tensor([0], device=self.device, dtype=torch.int32))
        item.meta.setdefault("cfg_text_context_indexes", torch.empty(0, device=self.device, dtype=torch.long))

    def _ensure_cfg_img_latent_metadata(self, item: ConversationItem) -> None:
        if float(item.meta.get("cfg_img_scale", 1.0)) <= 1.0:
            return
        for cfg_key, base_key in (
            ("cfg_img_position_ids", "position_ids"),
            ("cfg_img_sequence_indexes", "sequence_indexes"),
            ("cfg_img_key_value_lens", "key_value_lens"),
            ("cfg_img_context_indexes", "context_indexes"),
        ):
            value = item.meta.get(base_key)
            if torch.is_tensor(value):
                item.meta.setdefault(cfg_key, value.detach().to(device=self.device))

    def _current_timestep_value(self, item: ConversationItem) -> float:
        return float(self._current_timestep(item).detach().reshape(-1)[0].item())

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
