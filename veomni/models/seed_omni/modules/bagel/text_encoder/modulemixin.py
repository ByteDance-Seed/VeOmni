"""SeedOmni graph hooks for BAGEL text token embedding and LM head."""

from typing import Any, Dict, Optional

import torch

from ....conversation import ConversationItem, maybe_merge_outputs, seal_outputs
from ....generation_graph import FSM_SIGNAL_KEY
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin


SIGNAL_TEXT_DONE = "text_done"


class BagelTextEncoderModuleMixin(TextEncoderModuleMixin):
    """BAGEL text generation bridge for the SeedOmni V2 FSM."""

    tokenizer_class = object

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._eos_token_id: Optional[int] = None
        self._start_token_id: Optional[int] = None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = int(getattr(tokenizer, "eos_token_id", 0))
        self._start_token_id = self._resolve_token_id(tokenizer, "<|im_start|>", fallback=self._eos_token_id)

    def generate(
        self,
        conversation_list: Optional[list[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder.generate requires conversation_list.")
        tail = conversation_list[-1]
        if tail.role == "user":
            token_ids = self._prompt_token_ids(tail)
            tail.value = self.encode(token_ids)["inputs_embeds"].to(device=self.device, dtype=self.dtype)
            self._ensure_prompt_meta(tail, int(token_ids.numel()))

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
            if output_token_id == self._eos_token_id:
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(
            f"Invalid conversation tail for BAGEL text generation: type={tail.type!r}, role={tail.role!r}"
        )

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

    def _ensure_prompt_meta(self, item: ConversationItem, length: int) -> None:
        item.meta.setdefault("position_ids", torch.arange(length, device=self.device, dtype=torch.long))
        item.meta.setdefault("sequence_indexes", torch.arange(length, device=self.device, dtype=torch.long))
        item.meta.setdefault("context_indexes", torch.empty(0, device=self.device, dtype=torch.long))
        item.meta.setdefault("token_lens", torch.tensor([length], device=self.device, dtype=torch.int32))
        item.meta.setdefault("key_value_lens_before", torch.tensor([0], device=self.device, dtype=torch.int32))
        item.meta.setdefault("key_value_lens_after", torch.tensor([length], device=self.device, dtype=torch.int32))

    def _start_token_meta(self, prompt_item: ConversationItem, input_ids: torch.Tensor) -> Dict[str, Any]:
        del input_ids
        next_token = prompt_item.meta.get("next_token")
        if isinstance(next_token, dict):
            meta = {key: self._meta_to_device(value) for key, value in next_token.items() if key != "input_ids"}
        else:
            length = int(prompt_item.meta["token_lens"].sum().item())
            meta = {
                "position_ids": torch.tensor([length], device=self.device, dtype=torch.long),
                "key_value_lens": torch.tensor([length], device=self.device, dtype=torch.int32),
                "context_indexes": torch.arange(length, device=self.device, dtype=torch.long),
            }
        meta.setdefault("query_lens", torch.tensor([1], device=self.device, dtype=torch.int32))
        if "query_indexes" not in meta:
            kv_len = int(meta["key_value_lens"].sum().item())
            meta["query_indexes"] = torch.tensor([kv_len], device=self.device, dtype=torch.long)
        meta["token_kind"] = "bagel_start"
        return meta

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

    def _flush_text_generated(self, conversation_list: list[ConversationItem]) -> Dict[str, Any]:
        token_ids = list(self._text_token_cache)
        self._text_token_cache.clear()
        if not token_ids:
            return {}
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True) if self._tokenizer is not None else ""
        if conversation_list and conversation_list[-1].type == "output":
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


__all__ = ["BagelTextEncoderModuleMixin", "SIGNAL_TEXT_DONE"]
