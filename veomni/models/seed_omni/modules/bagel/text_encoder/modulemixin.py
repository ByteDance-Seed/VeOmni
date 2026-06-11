"""SeedOmni graph hooks for BAGEL text token embedding and LM head."""

from typing import Any, Dict, Optional

import torch

from ....conversation import ConversationItem, maybe_merge_outputs, seal_outputs
from ....generation_graph import FSM_SIGNAL_KEY
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin


SIGNAL_TEXT_DONE = "text_done"
SIGNAL_START_IMAGE_GEN = "start_image_gen"


class BagelTextEncoderModuleMixin(TextEncoderModuleMixin):
    """BAGEL text generation bridge for the SeedOmni V2 FSM."""

    tokenizer_class = object

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._eos_token_id: Optional[int] = None
        self._start_token_id: Optional[int] = None
        self._image_start_token_id: Optional[int] = None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = int(getattr(tokenizer, "eos_token_id", 0))
        self._start_token_id = self._resolve_token_id(tokenizer, "<|im_start|>", fallback=self._eos_token_id)
        self._image_start_token_id = self._resolve_token_id(tokenizer, "<|vision_start|>", fallback=None)

    def generate(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder.generate requires conversation_list.")
        self._materialize_image_understanding_items(conversation_list)
        tail = conversation_list[-1]
        if tail.role == "user":
            token_ids = self._prompt_token_ids(tail)
            tail.value = self.encode(token_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
            self._ensure_prompt_meta(tail, int(token_ids.numel()), conversation_list=conversation_list)
            if self._is_image_generation_request(conversation_list, generation_kwargs):
                self._ensure_image_generation_latent_item(conversation_list, tail, generation_kwargs)
                self._materialize_image_generation_items(conversation_list)
                return {"conversation_list": conversation_list}

            start_ids = self._start_token_ids(tail)
            start_embeds = self.encode(start_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=start_embeds,
                    role="assistant",
                    source="bagel_start_token",
                    meta=self._start_token_meta(tail, start_ids),
                )
            )
            return {"conversation_list": conversation_list}

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states = tail.value
            logits = self.decode(hidden_states=hidden_states)["logits"]
            output_token_id = self._select_token_id(logits, generation_kwargs, kwargs)
            self._text_token_cache.append(output_token_id)

            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
            tail.value = inputs_embeds
            tail.meta = self._next_token_meta(tail, output_token_id)
            maybe_merge_outputs(conversation_list)

            outputs["bagel_last_logits"] = logits.detach()
            outputs["bagel_last_greedy_token"] = torch.tensor([output_token_id], device=self.device, dtype=torch.long)
            image_start_token_id = self._maybe_resolve_image_start_token_id()
            if image_start_token_id is not None and output_token_id == image_start_token_id:
                self._ensure_image_generation_latent_item(
                    conversation_list,
                    tail,
                    generation_kwargs,
                    insert_before=tail,
                )
                self._materialize_image_generation_items(conversation_list)
                outputs[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
                outputs["generated"] = self._flush_text_generated(conversation_list, seal_output=False)
            elif output_token_id == self._eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(
            f"Invalid conversation tail for BAGEL text generation: type={tail.type!r}, role={tail.role!r}"
        )

    def _materialize_image_understanding_items(self, conversation_list: list[ConversationItem]) -> None:
        for item in conversation_list:
            if item.type != "image" or item.role != "user":
                continue
            if item.meta.get("image_sequence_ready"):
                continue
            if item.meta.get("bagel_role") != "image_und":
                continue

            image_embeds = item.meta.get("image_embeds", item.value)
            if not torch.is_tensor(image_embeds):
                raise TypeError(
                    "BAGEL image understanding item requires image embeddings before text prompt encoding."
                )
            image_token_ids = item.meta.get("image_token_ids")
            if not torch.is_tensor(image_token_ids):
                image_token_ids = torch.tensor(self._image_boundary_token_ids(), device=self.device, dtype=torch.long)
                item.meta["image_token_ids"] = image_token_ids
            image_text_indexes = item.meta.get("image_text_indexes")
            vit_token_indexes = item.meta.get("vit_token_indexes")
            query_lens = item.meta.get("query_lens")
            if (
                not torch.is_tensor(image_text_indexes)
                or not torch.is_tensor(vit_token_indexes)
                or not torch.is_tensor(query_lens)
            ):
                raise ValueError(
                    "BAGEL image understanding item requires image_token_ids, image_text_indexes, "
                    "vit_token_indexes, and query_lens metadata."
                )

            image_token_ids = image_token_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
            image_text_indexes = image_text_indexes.detach().to(device=self.device, dtype=torch.long).reshape(-1)
            vit_token_indexes = vit_token_indexes.detach().to(device=self.device, dtype=torch.long).reshape(-1)
            query_lens = query_lens.detach().to(device=self.device, dtype=torch.int32).reshape(-1)

            text_embeds = self.encode(image_token_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
            image_embeds = image_embeds.detach().to(device=self.device, dtype=self.dtype)
            sequence = text_embeds.new_zeros((int(query_lens.sum().item()), self.config.hidden_size))
            sequence[image_text_indexes] = text_embeds
            sequence[vit_token_indexes] = image_embeds
            item.value = sequence
            item.meta["query_lens"] = query_lens
            item.meta["image_sequence_ready"] = True

    def _is_image_generation_request(
        self,
        conversation_list: list[ConversationItem],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> bool:
        if generation_kwargs and generation_kwargs.get("infer_mode") == "gen":
            return True
        return any(item.meta.get("bagel_role") == "image_gen_latent" for item in conversation_list)

    def _materialize_image_generation_items(self, conversation_list: list[ConversationItem]) -> None:
        for item in conversation_list:
            if item.meta.get("bagel_role") != "image_gen_latent":
                continue
            if item.meta.get("text_embeds_ready"):
                continue
            text_token_ids = item.meta.get("text_token_ids")
            if not torch.is_tensor(text_token_ids):
                raise ValueError("BAGEL image generation item requires text_token_ids metadata.")
            text_token_ids = text_token_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
            item.meta["text_embeds"] = self.encode(text_token_ids)["inputs_embeds"].to(
                device=self.device, dtype=self.dtype
            )
            item.meta["text_embeds_ready"] = True

    def _ensure_image_generation_latent_item(
        self,
        conversation_list: list[ConversationItem],
        prompt_item: ConversationItem,
        generation_kwargs: Optional[Dict[str, Any]],
        *,
        insert_before: Optional[ConversationItem] = None,
    ) -> None:
        if any(
            item.meta.get("bagel_role") == "image_gen_latent" and not item.meta.get("decoded_image_ready")
            for item in conversation_list
        ):
            return
        kwargs = generation_kwargs or {}
        height = int(kwargs.get("image_height", 1024))
        width = int(kwargs.get("image_width", 1024))
        prompt_kv_len = self._scalar_meta_int(
            prompt_item.meta.get("key_value_lens_after", prompt_item.meta.get("key_value_lens"))
        )
        prompt_rope = self._scalar_meta_int(prompt_item.meta.get("rope_after"), default=prompt_kv_len)
        text_token_ids = torch.tensor(self._image_boundary_token_ids(), device=self.device, dtype=torch.long)
        latent_item = ConversationItem(
            type="image",
            value=torch.empty(0, device=self.device, dtype=torch.float32),
            role="assistant",
            source="bagel_generation_request",
            meta={
                "bagel_role": "image_gen_latent",
                "raw_image_size": [height, width],
                "text_token_ids": text_token_ids,
                "key_value_lens": torch.tensor([prompt_kv_len], device=self.device, dtype=torch.int32),
                "context_indexes": torch.arange(prompt_kv_len, device=self.device, dtype=torch.long),
                "rope_after_prompt": torch.tensor([prompt_rope], device=self.device, dtype=torch.long),
            },
        )
        if insert_before is None:
            conversation_list.append(latent_item)
            return
        for idx, item in enumerate(conversation_list):
            if item is insert_before:
                conversation_list.insert(idx, latent_item)
                return
        conversation_list.append(latent_item)

    def _prompt_token_ids(self, item: ConversationItem) -> torch.Tensor:
        value = item.value
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        if not isinstance(value, str):
            raise TypeError(f"BAGEL text prompt must be a str or token tensor, got {type(value).__name__}.")
        if self._tokenizer is None:
            raise ValueError("BAGEL text tokenizer is required for raw string prompts.")
        ids = self._tokenizer.encode(value, add_special_tokens=False)
        start = self._resolve_start_token_id()
        eos = self._resolve_eos_token_id()
        return torch.tensor([start, *ids, eos], device=self.device, dtype=torch.long)

    def _start_token_ids(self, prompt_item: ConversationItem) -> torch.Tensor:
        next_token = prompt_item.meta.get("next_token")
        if isinstance(next_token, dict) and torch.is_tensor(next_token.get("input_ids")):
            return next_token["input_ids"].detach().to(device=self.device, dtype=torch.long).reshape(-1)
        return torch.tensor([self._resolve_start_token_id()], device=self.device, dtype=torch.long)

    def _ensure_prompt_meta(
        self,
        item: ConversationItem,
        length: int,
        *,
        conversation_list: Optional[list[ConversationItem]] = None,
    ) -> None:
        base_kv_len, base_rope = self._previous_context_offsets(item, conversation_list)
        item.meta.setdefault(
            "position_ids",
            torch.arange(base_rope, base_rope + length, device=self.device, dtype=torch.long),
        )
        item.meta.setdefault(
            "sequence_indexes",
            torch.arange(base_kv_len, base_kv_len + length, device=self.device, dtype=torch.long),
        )
        item.meta.setdefault("context_indexes", torch.arange(base_kv_len, device=self.device, dtype=torch.long))
        item.meta.setdefault("token_lens", torch.tensor([length], device=self.device, dtype=torch.int32))
        item.meta.setdefault(
            "key_value_lens_before", torch.tensor([base_kv_len], device=self.device, dtype=torch.int32)
        )
        item.meta.setdefault(
            "key_value_lens_after",
            torch.tensor([base_kv_len + length], device=self.device, dtype=torch.int32),
        )
        item.meta.setdefault("rope_after", torch.tensor([base_rope + length], device=self.device, dtype=torch.long))

    def _start_token_meta(self, prompt_item: ConversationItem, input_ids: torch.Tensor) -> Dict[str, Any]:
        del input_ids
        next_token = prompt_item.meta.get("next_token")
        if isinstance(next_token, dict):
            meta = {key: self._meta_to_device(value) for key, value in next_token.items() if key != "input_ids"}
        else:
            length = self._scalar_meta_int(
                prompt_item.meta.get("key_value_lens_after"),
                default=int(prompt_item.meta["token_lens"].sum().item()),
            )
            position = self._scalar_meta_int(prompt_item.meta.get("rope_after"), default=length)
            meta = {
                "position_ids": torch.tensor([position], device=self.device, dtype=torch.long),
                "key_value_lens": torch.tensor([length], device=self.device, dtype=torch.int32),
                "context_indexes": torch.arange(length, device=self.device, dtype=torch.long),
            }
        meta.setdefault("query_lens", torch.tensor([1], device=self.device, dtype=torch.int32))
        if "query_indexes" not in meta:
            kv_len = int(meta["key_value_lens"].sum().item())
            meta["query_indexes"] = torch.tensor([kv_len], device=self.device, dtype=torch.long)
        meta["token_kind"] = "bagel_start"
        return meta

    def _previous_context_offsets(
        self,
        item: ConversationItem,
        conversation_list: Optional[list[ConversationItem]],
    ) -> tuple[int, int]:
        if conversation_list is None:
            return 0, 0
        base_kv_len = 0
        base_rope = 0
        for prior in conversation_list:
            if prior is item:
                break
            kv_after = prior.meta.get("key_value_lens_after")
            rope_after = prior.meta.get("rope_after")
            if kv_after is not None:
                base_kv_len = self._scalar_meta_int(kv_after, default=base_kv_len)
            if rope_after is not None:
                base_rope = self._scalar_meta_int(rope_after, default=base_rope)
        return base_kv_len, base_rope

    def _next_token_meta(self, tail: ConversationItem, output_token_id: int) -> Dict[str, Any]:
        old_meta = tail.meta
        prev_pos = old_meta.get("position_ids")
        if torch.is_tensor(prev_pos):
            position_ids = prev_pos.detach().to(device=self.device, dtype=torch.long).reshape(-1)[-1:] + 1
        else:
            position_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        key_value_lens = old_meta.get("key_value_lens_after_qwen")
        if not torch.is_tensor(key_value_lens):
            key_value_lens = old_meta.get("key_value_lens")
            if torch.is_tensor(key_value_lens):
                key_value_lens = key_value_lens.to(device=self.device, dtype=torch.int32) + 1
            else:
                key_value_lens = torch.tensor([int(position_ids.item())], device=self.device, dtype=torch.int32)
        else:
            key_value_lens = key_value_lens.to(device=self.device, dtype=torch.int32)
        return {
            "position_ids": position_ids,
            "key_value_lens": key_value_lens,
            "context_indexes": torch.arange(int(key_value_lens.sum().item()), device=self.device, dtype=torch.long),
            "query_lens": torch.tensor([1], device=self.device, dtype=torch.int32),
            "query_indexes": key_value_lens.to(device=self.device, dtype=torch.long),
            "token_id": int(output_token_id),
        }

    def _select_token_id(
        self,
        logits: torch.Tensor,
        generation_kwargs: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> int:
        sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, kwargs)
        scores = logits[:, -1, :] if logits.dim() == 3 else logits
        if not sampling.get("do_sample", True):
            return int(scores.argmax(dim=-1).item())
        temperature = float(sampling.get("temperature", 1.0))
        top_p = float(sampling.get("top_p", 1.0))
        if temperature != 1.0:
            scores = scores / max(temperature, 1e-6)
        if top_p < 1.0:
            scores = self._top_p_filter(scores, top_p)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _flush_text_generated(
        self,
        conversation_list: list[ConversationItem],
        *,
        seal_output: bool = True,
    ) -> Dict[str, Any]:
        token_ids = list(self._text_token_cache)
        self._text_token_cache.clear()
        if not token_ids:
            return {}
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True) if self._tokenizer is not None else ""
        if seal_output and conversation_list and conversation_list[-1].type == "output":
            seal_outputs(conversation_list, new_type="text")
        return {"type": "text", "value": text, "meta": {"token_ids": token_ids}}

    def _resolve_start_token_id(self) -> int:
        if self._start_token_id is not None:
            return int(self._start_token_id)
        if self._tokenizer is not None:
            resolved = self._resolve_token_id(self._tokenizer, "<|im_start|>", fallback=None)
            if resolved is not None:
                self._start_token_id = resolved
                return resolved
        raise ValueError("Unable to resolve BAGEL start token id.")

    def _resolve_eos_token_id(self) -> int:
        if self._eos_token_id is not None:
            return int(self._eos_token_id)
        if self._tokenizer is not None and getattr(self._tokenizer, "eos_token_id", None) is not None:
            self._eos_token_id = int(self._tokenizer.eos_token_id)
            return int(self._eos_token_id)
        raise ValueError("Unable to resolve BAGEL EOS token id.")

    def _resolve_image_start_token_id(self) -> int:
        resolved = self._maybe_resolve_image_start_token_id()
        if resolved is not None:
            return resolved
        raise ValueError("Unable to resolve BAGEL image start token id.")

    def _maybe_resolve_image_start_token_id(self) -> Optional[int]:
        if self._image_start_token_id is not None:
            return int(self._image_start_token_id)
        if self._tokenizer is None:
            return None
        resolved = self._resolve_token_id(self._tokenizer, "<|vision_start|>", fallback=None)
        if resolved is not None:
            self._image_start_token_id = resolved
            return int(resolved)
        return None

    def _image_boundary_token_ids(self) -> list[int]:
        if self._tokenizer is None:
            raise ValueError("BAGEL image generation requires tokenizer-owned image boundary tokens.")
        start = self._resolve_image_start_token_id()
        end = self._resolve_token_id(self._tokenizer, "<|vision_end|>", fallback=None)
        if end is None:
            raise ValueError("Unable to resolve BAGEL image boundary token ids.")
        return [start, end]

    @staticmethod
    def _scalar_meta_int(value: Any, default: Optional[int] = None) -> int:
        if torch.is_tensor(value):
            return int(value.detach().reshape(-1)[0].item())
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        if isinstance(value, int):
            return value
        if default is not None:
            return default
        raise ValueError("Expected scalar integer metadata.")

    @staticmethod
    def _resolve_token_id(tokenizer: Any, token: str, fallback: Optional[int]) -> Optional[int]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                return int(token_id)
        except Exception:
            pass
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                return int(ids[0])
        except Exception:
            pass
        return fallback

    def _meta_to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().to(self.device)
        if isinstance(value, dict):
            return {key: self._meta_to_device(item) for key, item in value.items()}
        return value


__all__ = ["BagelTextEncoderModuleMixin", "SIGNAL_START_IMAGE_GEN", "SIGNAL_TEXT_DONE"]
