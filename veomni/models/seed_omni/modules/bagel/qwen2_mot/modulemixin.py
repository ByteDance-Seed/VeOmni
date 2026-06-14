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
        self._past_key_values: Optional[Any] = None
        self._key_values_lens: Optional[torch.Tensor] = None
        self._bagel_packed_batch: Optional[dict[str, Any]] = None

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        bagel_packed_batch: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del conversation_list, kwargs
        assert method in ("forward",)
        if bagel_packed_batch is None:
            raise NotImplementedError("BagelQwen2MoT graph hooks currently require bagel_packed_batch.")
        self._conversation_carrier = None
        self._bagel_packed_batch = bagel_packed_batch

        text_embeds = bagel_packed_batch["packed_text_embeds"].to(device=self.device, dtype=self.dtype)
        packed_sequence = text_embeds.new_zeros((int(bagel_packed_batch["sequence_length"]), text_embeds.shape[-1]))
        packed_sequence[bagel_packed_batch["packed_text_indexes"]] = text_embeds

        und_indexes = [bagel_packed_batch["packed_text_indexes"].to(device=self.device, dtype=torch.long)]
        if "packed_vit_embeds" in bagel_packed_batch:
            packed_sequence[bagel_packed_batch["packed_vit_token_indexes"]] = bagel_packed_batch[
                "packed_vit_embeds"
            ].to(device=self.device, dtype=self.dtype)
            und_indexes.append(bagel_packed_batch["packed_vit_token_indexes"].to(device=self.device, dtype=torch.long))

        gen_indexes = None
        if "packed_latent_embeds" in bagel_packed_batch:
            gen_indexes = bagel_packed_batch["packed_vae_token_indexes"].to(device=self.device, dtype=torch.long)
            packed_sequence[gen_indexes] = bagel_packed_batch["packed_latent_embeds"].to(
                device=self.device, dtype=self.dtype
            )

        return {
            "packed_sequence": packed_sequence,
            "sample_lens": bagel_packed_batch["sample_lens"],
            "attention_mask": bagel_packed_batch["nested_attention_masks"],
            "packed_position_ids": bagel_packed_batch["packed_position_ids"].to(device=self.device, dtype=torch.long),
            "packed_und_token_indexes": torch.cat(und_indexes),
            "packed_gen_token_indexes": gen_indexes,
        }

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        assert method in ("forward",)
        batch = getattr(self, "_bagel_packed_batch", None)
        self._bagel_packed_batch = None
        if batch is None:
            return outputs
        batch["packed_hidden_states"] = outputs["hidden_states"]
        return {"bagel_packed_batch": batch}

    def reset_local_inference_state(self) -> None:
        self._past_key_values = None
        self._key_values_lens = None

    @torch.no_grad()
    def generate(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BagelQwen2MoT.generate requires conversation_list.")
        image_gen_item = self._ready_image_gen_item(conversation_list)
        if image_gen_item is not None:
            return self._decode_image_flow(image_gen_item, conversation_list, generation_kwargs)

        if self._is_image_generation_request(conversation_list, generation_kwargs):
            if self._past_key_values is None:
                self._prefill_prompt(conversation_list)
            return {
                "conversation_list": conversation_list,
                "past_key_values": self._past_key_values,
                "key_values_lens": None if self._key_values_lens is None else self._key_values_lens.detach(),
            }

        if not conversation_list or conversation_list[-1].type != "output":
            raise ValueError(
                "BagelQwen2MoT.generate expects an output start/generated token at the conversation tail."
            )

        if self._past_key_values is None:
            self._prefill_prompt(conversation_list[:-1])

        tail = conversation_list[-1]
        hidden_states = self._decode_tail(tail)
        tail.value = hidden_states
        return {
            "conversation_list": conversation_list,
            "bagel_last_hidden_state": hidden_states.detach(),
            "bagel_last_hidden_state_sample": _sample_tensor(hidden_states),
            "past_key_values": self._past_key_values,
            "key_values_lens": None if self._key_values_lens is None else self._key_values_lens.detach(),
        }

    def _prefill_prompt(self, prompt_items: list[ConversationItem]) -> None:
        saw_prompt = False
        self._past_key_values = None
        self._key_values_lens = None
        for item in prompt_items:
            packed_query_sequence = self._prompt_item_value(item)
            if packed_query_sequence is None:
                continue
            saw_prompt = True
            length = int(packed_query_sequence.shape[-2])
            query_lens = self._prompt_meta_tensor(item, ("query_lens", "token_lens"), [length], dtype=torch.int32)
            position_ids = self._prompt_meta_tensor(item, ("position_ids",), None, dtype=torch.long)
            if position_ids is None:
                position_ids = torch.arange(length, device=self.device, dtype=torch.long)
            query_indexes = self._prompt_meta_tensor(item, ("sequence_indexes",), None, dtype=torch.long)
            if query_indexes is None:
                if self._key_values_lens is None:
                    query_indexes = torch.arange(length, device=self.device, dtype=torch.long)
                else:
                    start = int(self._key_values_lens.sum().item())
                    query_indexes = torch.arange(start, start + length, device=self.device, dtype=torch.long)
            key_values_lens = self._prompt_meta_tensor(item, ("key_value_lens_before",), None, dtype=torch.int32)
            if key_values_lens is None:
                key_values_lens = (
                    torch.tensor([0], device=self.device, dtype=torch.int32)
                    if self._key_values_lens is None
                    else self._key_values_lens.to(device=self.device, dtype=torch.int32)
                )
            key_value_indexes = self._prompt_meta_tensor(item, ("context_indexes",), [], dtype=torch.long)
            mode = self._prompt_inference_mode(item)
            extra_inputs: Dict[str, torch.Tensor] = {}
            if mode == "gen":
                vae_token_indexes = self._prompt_meta_tensor(item, ("vae_token_indexes",), None, dtype=torch.long)
                text_indexes = self._prompt_meta_tensor(item, ("text_indexes",), None, dtype=torch.long)
                if vae_token_indexes is None or text_indexes is None:
                    raise ValueError("BAGEL gen-mode prompt item requires VAE and text token indexes.")
                extra_inputs = {
                    "packed_vae_token_indexes": vae_token_indexes,
                    "packed_text_indexes": text_indexes,
                }
            outputs = self._forward_packed_inference(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_ids=position_ids,
                packed_query_indexes=query_indexes,
                past_key_values=self._past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=key_value_indexes,
                update_past_key_values=True,
                is_causal=bool(item.meta.get("is_causal", True)),
                mode=mode,
                **extra_inputs,
            )
            self._past_key_values = outputs.past_key_values
            self._key_values_lens = (key_values_lens + query_lens).detach().to(device=self.device, dtype=torch.int32)

        if not saw_prompt:
            raise ValueError("BAGEL qwen2_mot prompt prefill found no tensor prompt segments.")

    def _decode_image_flow(
        self,
        item: ConversationItem,
        conversation_list: list[ConversationItem],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        packed_query_sequence = item.value
        if not torch.is_tensor(packed_query_sequence):
            raise ValueError("BAGEL image generation flow item requires a packed sequence tensor.")
        if packed_query_sequence.dim() == 3 and packed_query_sequence.size(0) == 1:
            packed_query_sequence = packed_query_sequence.squeeze(0)
        packed_query_sequence = packed_query_sequence.to(device=self.device, dtype=self.dtype)

        query_lens = self._item_meta_tensor(item, "query_lens", int(packed_query_sequence.shape[0]), dtype=torch.int32)
        position_ids = self._prompt_meta_tensor(item, ("position_ids",), None, dtype=torch.long)
        if position_ids is None:
            raise ValueError("BAGEL image generation flow item requires position_ids metadata.")
        query_indexes = self._prompt_meta_tensor(item, ("sequence_indexes",), None, dtype=torch.long)
        if query_indexes is None:
            raise ValueError("BAGEL image generation flow item requires sequence_indexes metadata.")
        key_values_lens = self._prompt_meta_tensor(item, ("key_value_lens",), None, dtype=torch.int32)
        if key_values_lens is None:
            if self._key_values_lens is None:
                raise ValueError("BAGEL image generation flow requires prompt key_values_lens.")
            key_values_lens = self._key_values_lens.to(device=self.device, dtype=torch.int32)
        key_value_indexes = self._prompt_meta_tensor(item, ("context_indexes",), [], dtype=torch.long)
        vae_token_indexes = self._prompt_meta_tensor(item, ("vae_token_indexes",), None, dtype=torch.long)
        text_indexes = self._prompt_meta_tensor(item, ("text_indexes",), None, dtype=torch.long)
        if vae_token_indexes is None or text_indexes is None:
            raise ValueError("BAGEL image generation flow item requires VAE and text token indexes.")

        outputs = self._forward_packed_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=position_ids,
            packed_query_indexes=query_indexes,
            past_key_values=self._past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=vae_token_indexes,
            packed_text_indexes=text_indexes,
        )
        hidden_states = outputs.packed_query_sequence
        item.value = hidden_states
        item.meta["flow_hidden_ready"] = True
        result: Dict[str, Any] = {
            "conversation_list": conversation_list,
            "bagel_last_hidden_state": hidden_states.detach(),
            "bagel_last_hidden_state_sample": _sample_tensor(hidden_states),
            "past_key_values": self._past_key_values,
            "key_values_lens": None if self._key_values_lens is None else self._key_values_lens.detach(),
        }
        if self._cfg_text_active(item, generation_kwargs):
            cfg_text_hidden_states = self._decode_cfg_text_image_flow(
                item,
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
            )
            item.meta["cfg_text_hidden_state"] = cfg_text_hidden_states.detach()
            result["bagel_last_cfg_text_hidden_state"] = cfg_text_hidden_states.detach()
            result["bagel_last_cfg_text_hidden_state_sample"] = _sample_tensor(cfg_text_hidden_states)
        if self._cfg_img_active(item, generation_kwargs):
            cfg_img_hidden_states = self._decode_cfg_img_image_flow(
                item,
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
            )
            item.meta["cfg_img_hidden_state"] = cfg_img_hidden_states.detach()
            result["bagel_last_cfg_img_hidden_state"] = cfg_img_hidden_states.detach()
            result["bagel_last_cfg_img_hidden_state_sample"] = _sample_tensor(cfg_img_hidden_states)
        return result

    def _decode_cfg_text_image_flow(
        self,
        item: ConversationItem,
        *,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self._prompt_meta_tensor(item, ("cfg_text_position_ids",), None, dtype=torch.long)
        query_indexes = self._prompt_meta_tensor(item, ("cfg_text_sequence_indexes",), None, dtype=torch.long)
        key_values_lens = self._prompt_meta_tensor(item, ("cfg_text_key_value_lens",), [0], dtype=torch.int32)
        key_value_indexes = self._prompt_meta_tensor(item, ("cfg_text_context_indexes",), [], dtype=torch.long)
        if position_ids is None:
            position_ids = torch.zeros(int(packed_query_sequence.shape[0]), device=self.device, dtype=torch.long)
        if query_indexes is None:
            query_indexes = torch.arange(int(packed_query_sequence.shape[0]), device=self.device, dtype=torch.long)

        output = self._forward_packed_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=position_ids,
            packed_query_indexes=query_indexes,
            past_key_values=None,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=vae_token_indexes,
            packed_text_indexes=text_indexes,
        )
        return output.packed_query_sequence

    def _decode_cfg_img_image_flow(
        self,
        item: ConversationItem,
        *,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self._prompt_meta_tensor(item, ("cfg_img_position_ids",), None, dtype=torch.long)
        query_indexes = self._prompt_meta_tensor(item, ("cfg_img_sequence_indexes",), None, dtype=torch.long)
        key_values_lens = self._prompt_meta_tensor(item, ("cfg_img_key_value_lens",), None, dtype=torch.int32)
        key_value_indexes = self._prompt_meta_tensor(item, ("cfg_img_context_indexes",), None, dtype=torch.long)
        if position_ids is None:
            position_ids = self._prompt_meta_tensor(item, ("position_ids",), None, dtype=torch.long)
        if query_indexes is None:
            query_indexes = self._prompt_meta_tensor(item, ("sequence_indexes",), None, dtype=torch.long)
        if key_values_lens is None:
            key_values_lens = self._prompt_meta_tensor(item, ("key_value_lens",), None, dtype=torch.int32)
        if key_value_indexes is None:
            key_value_indexes = self._prompt_meta_tensor(item, ("context_indexes",), [], dtype=torch.long)
        if position_ids is None or query_indexes is None or key_values_lens is None:
            raise ValueError("BAGEL CFG image guidance requires cfg_img or base prompt latent metadata.")

        output = self._forward_packed_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=position_ids,
            packed_query_indexes=query_indexes,
            past_key_values=self._past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=vae_token_indexes,
            packed_text_indexes=text_indexes,
        )
        return output.packed_query_sequence

    def _decode_tail(self, tail: ConversationItem) -> torch.Tensor:
        packed_query_sequence = tail.value
        if packed_query_sequence.dim() == 3 and packed_query_sequence.size(0) == 1:
            packed_query_sequence = packed_query_sequence.squeeze(0)
        packed_query_sequence = packed_query_sequence[-1:].to(device=self.device, dtype=self.dtype)

        query_lens = self._tail_meta_tensor(tail, "query_lens", default=[1], dtype=torch.int32)
        position_ids = self._tail_meta_tensor(tail, "position_ids", default=None, dtype=torch.long)
        if position_ids is None:
            if self._key_values_lens is None:
                raise ValueError("Cannot infer BAGEL decode position without key_values_lens.")
            position_ids = self._key_values_lens.to(device=self.device, dtype=torch.long)
        key_values_lens = self._tail_meta_tensor(tail, "key_value_lens", default=None, dtype=torch.int32)
        if key_values_lens is None:
            if self._key_values_lens is None:
                raise ValueError("Cannot infer BAGEL decode KV length without prefill cache.")
            key_values_lens = self._key_values_lens.to(device=self.device, dtype=torch.int32)
        query_indexes = self._tail_meta_tensor(tail, "query_indexes", default=None, dtype=torch.long)
        if query_indexes is None:
            query_indexes = key_values_lens.to(device=self.device, dtype=torch.long)
        key_value_indexes = self._tail_meta_tensor(tail, "context_indexes", default=None, dtype=torch.long)
        if key_value_indexes is None:
            key_value_indexes = torch.arange(int(key_values_lens.sum().item()), device=self.device, dtype=torch.long)

        outputs = self._forward_packed_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=position_ids,
            packed_query_indexes=query_indexes,
            past_key_values=self._past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        self._past_key_values = outputs.past_key_values
        self._key_values_lens = key_values_lens + query_lens
        tail.meta["key_value_lens_after_qwen"] = self._key_values_lens.detach()
        return outputs.packed_query_sequence

    def _pack_prompt_items(
        self,
        prompt_items: list[ConversationItem],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        segments: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        query_indexes: list[torch.Tensor] = []
        for item in prompt_items:
            if is_dummy(item) or item.type == "output":
                continue
            value = item.value
            if not torch.is_tensor(value):
                continue
            if value.dim() == 3 and value.size(0) == 1:
                value = value.squeeze(0)
            value = value.to(device=self.device, dtype=self.dtype)
            segments.append(value)
            length = int(value.shape[-2])
            position_ids.append(self._item_meta_tensor(item, "position_ids", length, dtype=torch.long))
            query_indexes.append(self._item_meta_tensor(item, "sequence_indexes", length, dtype=torch.long))
        if not segments:
            raise ValueError("BAGEL qwen2_mot prompt prefill found no tensor prompt segments.")
        packed_query_sequence = torch.cat(segments, dim=0)
        query_lens = torch.tensor([packed_query_sequence.shape[0]], device=self.device, dtype=torch.int32)
        packed_position_ids = torch.cat(position_ids, dim=0)
        packed_query_indexes = torch.cat(query_indexes, dim=0)
        key_values_lens = torch.tensor([0], device=self.device, dtype=torch.int32)
        key_value_indexes = torch.empty(0, device=self.device, dtype=torch.long)
        return (
            packed_query_sequence,
            query_lens,
            packed_position_ids,
            packed_query_indexes,
            key_values_lens,
            key_value_indexes,
        )

    def _prompt_item_value(self, item: ConversationItem) -> Optional[torch.Tensor]:
        if is_dummy(item) or item.type == "output":
            return None
        if item.meta.get("bagel_role") == "image_gen_latent":
            return None
        value = item.value
        if not torch.is_tensor(value):
            return None
        if value.dim() == 3 and value.size(0) == 1:
            value = value.squeeze(0)
        return value.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _prompt_inference_mode(item: ConversationItem) -> str:
        if item.meta.get("bagel_role") == "image_vae_context":
            return "gen"
        return "und"

    def _prompt_meta_tensor(
        self,
        item: ConversationItem,
        keys: tuple[str, ...],
        default: Optional[list[int]],
        *,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value: Any = None
        for key in keys:
            candidate = item.meta.get(key)
            if torch.is_tensor(candidate):
                value = candidate
                break
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(-1)
        if default is None:
            return None
        return torch.tensor(default, device=self.device, dtype=dtype)

    def _is_image_generation_request(
        self,
        conversation_list: list[ConversationItem],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> bool:
        if generation_kwargs and generation_kwargs.get("infer_mode") == "gen":
            return True
        return any(
            item.meta.get("bagel_role") == "image_gen_latent" and not item.meta.get("decoded_image_ready")
            for item in conversation_list
        )

    def _cfg_text_active(
        self,
        item: ConversationItem,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> bool:
        cfg_text_scale = float(item.meta.get("cfg_text_scale", (generation_kwargs or {}).get("cfg_text_scale", 1.0)))
        if cfg_text_scale <= 1.0:
            return False
        interval = item.meta.get("cfg_interval", (generation_kwargs or {}).get("cfg_interval", [0.0, 1.0]))
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            interval = [0.0, 1.0]
        t_value = self._current_timestep_value(item)
        return t_value > float(interval[0]) and t_value <= float(interval[1])

    def _cfg_img_active(
        self,
        item: ConversationItem,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> bool:
        cfg_img_scale = float(item.meta.get("cfg_img_scale", (generation_kwargs or {}).get("cfg_img_scale", 1.0)))
        if cfg_img_scale <= 1.0:
            return False
        interval = item.meta.get("cfg_interval", (generation_kwargs or {}).get("cfg_interval", [0.0, 1.0]))
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            interval = [0.0, 1.0]
        t_value = self._current_timestep_value(item)
        return t_value > float(interval[0]) and t_value <= float(interval[1])

    @staticmethod
    def _current_timestep_value(item: ConversationItem) -> float:
        step_index = int(item.meta.get("flow_step_index", 0))
        timesteps = item.meta.get("timesteps")
        if torch.is_tensor(timesteps):
            flat = timesteps.detach().reshape(-1)
            if flat.numel() > 0:
                return float(flat[min(step_index, flat.numel() - 1)].item())
        timestep = item.meta.get("timestep")
        if torch.is_tensor(timestep):
            return float(timestep.detach().reshape(-1)[0].item())
        return 1.0

    def _ready_image_gen_item(self, conversation_list: list[ConversationItem]) -> Optional[ConversationItem]:
        for item in conversation_list:
            if (
                item.meta.get("bagel_role") == "image_gen_latent"
                and not item.meta.get("decoded_image_ready")
                and item.meta.get("flow_packed_sequence_ready")
            ):
                return item
        return None

    def _item_meta_tensor(
        self,
        item: ConversationItem,
        key: str,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = item.meta.get(key)
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(-1)
        return torch.arange(length, device=self.device, dtype=dtype)

    def _tail_meta_tensor(
        self,
        tail: ConversationItem,
        key: str,
        *,
        default: Optional[list[int]],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        value = tail.meta.get(key)
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(-1)
        if default is None:
            return None
        return torch.tensor(default, device=self.device, dtype=dtype)

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


def _sample_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 2:
        return value.detach()[:4, :4]
    return value.detach()[:16]


__all__ = ["BagelQwen2MoTModuleMixin"]
