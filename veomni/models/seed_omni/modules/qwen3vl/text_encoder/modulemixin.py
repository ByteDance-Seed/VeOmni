from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import ConversationItem, is_dummy, maybe_merge_outputs
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import CPUPreprocessor, pre_forward
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from .chat_template import (
    Qwen3VLChatMarkers,
    apply_qwen3vl_chat_template,
    apply_qwen3vl_generation_prompt,
    merge_consecutive_text_parts,
    pack_text_input_ids,
    tokenize_template_parts,
)


SIGNAL_TEXT_DONE = "text_done"

# Sentinel written by Qwen3VLTextEncoderCPUPreprocessor onto every text part's
# meta so the thin ``encode_pre`` knows chat-template + tokenize already ran in
# the DataLoader worker (and otherwise falls back to the in-module path).
_OMNI_TOKENIZED = "_omni_tokenized"


class Qwen3VLTextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side chat-template + tokenize for the Qwen3-VL text encoder.

    Holds only the (picklable) tokenizer + chat markers — never the model. Builds
    CPU tensors so it can run in DataLoader workers and overlap with GPU compute.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, chat_markers: Qwen3VLChatMarkers) -> None:
        self._tokenizer = tokenizer
        self._chat_markers = chat_markers

    def __call__(self, conversation_list: list[list[ConversationItem]]) -> None:
        for sample in conversation_list or []:
            if sample and sample[0].meta.get(_OMNI_TOKENIZED):
                continue  # idempotent: already processed
            parts = apply_qwen3vl_chat_template(sample, self._chat_markers)
            tokenize_template_parts(parts, self._tokenizer, device=None)
            parts = merge_consecutive_text_parts(parts)
            for part in parts:
                if part.type == "text":
                    part.meta[_OMNI_TOKENIZED] = True
            sample.clear()
            sample.extend(parts)


class Qwen3VLTextEncoderModuleMixin(TextEncoderModuleMixin):
    """Qwen3-VL ``TextEncoder`` — ChatML templating + tokenize + wte / lm_head.

    Image items (already carrying merged vision embeds from ``qwen3vl_vision``)
    pass through ``encode`` untouched: they keep their ``(N, D)`` value, get
    wrapped by ``<|vision_start|>`` / ``<|vision_end|>`` text rows, and the
    backbone splices them in by segment order.  Only ``text`` rows are tokenized
    and embedded here.
    """

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._chat_markers: Optional[Qwen3VLChatMarkers] = None
        self._eos_token_id: Optional[int] = None
        self._im_end_token_id: Optional[int] = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = self._resolve_token_id(tokenizer, token_id=tokenizer.eos_token_id)
        self._im_end_token_id = self._resolve_token_id(tokenizer, token="<|im_end|>")
        self._chat_markers = Qwen3VLChatMarkers(
            im_start_token="<|im_start|>",
            im_end_token="<|im_end|>",
            eos_token=str(tokenizer.eos_token),
            assistant_prefix="<|im_start|>assistant\n",
            vision_start_token="<|vision_start|>",
            vision_end_token="<|vision_end|>",
        )

    # ── Training hooks ──────────────────────────────────────────────────────
    # Both ``post_forward`` call-sites are inherited from
    # ``TextEncoderModuleMixin`` (``encode`` → scatter wte embeds back onto the
    # conversation; ``decode`` → drop logits, rename ``loss`` → ``_loss``) — the
    # behaviour is identical, so only the tokenization pre-hooks differ here.
    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)
        return {"input_ids": input_ids}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side chat-template + tokenize (see :class:`Qwen3VLTextEncoderCPUPreprocessor`)."""
        if self._chat_markers is None or getattr(self, "_tokenizer", None) is None:
            return None
        return Qwen3VLTextEncoderCPUPreprocessor(self._tokenizer, self._chat_markers)

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            # Fast path: the worker-side CPU preprocessor already ran chat-template
            # + tokenize (tagged on meta). Otherwise fall back to the in-module path.
            if not (sample and sample[0].meta.get(_OMNI_TOKENIZED)):
                self._prepare_sample_training(sample)
            input_ids.extend(pack_text_input_ids(sample))
        # ``naflatten`` keeps the shape on CPU (avoids the post-forward D2H sync);
        # the flat ids may be CPU (worker path) or device (fallback) — move once.
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        return input_ids

    def _prepare_sample_training(self, sample: list[ConversationItem]) -> None:
        # Build CPU tensors (mirrors the worker preprocessor); the flat input_ids
        # are moved to device in ``_prepare_encode_inputs``, labels/attention_mask
        # at their consumers — avoids per-segment device mixing.
        parts = apply_qwen3vl_chat_template(sample, self._chat_markers)
        tokenize_template_parts(parts, self._tokenizer, device=None)
        parts = merge_consecutive_text_parts(parts)
        sample.clear()
        sample.extend(parts)

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        dtype = self.dtype
        segment_embeds_iterator = iter(segment_embeds)
        for sample in conversation_list:
            for part in sample:
                if part.type != "text":
                    continue
                part.value = next(segment_embeds_iterator).to(device=self.device, dtype=dtype)
        if next(segment_embeds_iterator, None) is not None:
            raise RuntimeError("Qwen3-VL text segment count mismatch during embed scatter.")

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for sample in conversation_list or []:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    # ``labels`` rides in meta from tokenize (CPU). Keep all label
                    # chunks on CPU; move the concatenated result to device once.
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif part.type in ("image", "video"):
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)  # CPU
        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    # ── Inference hooks ─────────────────────────────────────────────────────
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        tail = conversation_list[-1]
        if tail.role == "user":
            conversation_list = apply_qwen3vl_chat_template(conversation_list, self._chat_markers)
            conversation_list = apply_qwen3vl_generation_prompt(conversation_list, self._chat_markers)
            tokenize_template_parts(conversation_list, self._tokenizer, device=self.device)
            conversation_list = merge_consecutive_text_parts(conversation_list)
            for part in conversation_list:
                part.meta.pop("labels", None)
            input_ids = pack_text_input_ids(conversation_list)
            input_ids, encode_batch_shape = naflatten(input_ids)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]
            self._scatter_text_embeds([conversation_list], unflatten(inputs_embeds, encode_batch_shape))
            return {"conversation_list": conversation_list}

        if tail.type == "output":
            outputs: Dict[str, Any] = {"conversation_list": conversation_list}
            hidden_states: torch.Tensor = tail.value
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            sampling = self._extract_sampling_kwargs(generation_kwargs, 1.0, 1.0, {})
            output_token_id = self._sample_token(hidden_states, **sampling)
            self._text_token_cache.append(output_token_id)
            input_ids = self._token_id_tensor(output_token_id)
            inputs_embeds = self.encode(input_ids)["inputs_embeds"]

            tail.value = inputs_embeds
            maybe_merge_outputs(conversation_list)

            if output_token_id in (self._eos_token_id, self._im_end_token_id):
                outputs[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
                outputs["generated"] = self._flush_text_generated(conversation_list)
            return outputs

        raise ValueError(f"Invalid conversation tail type: {tail.type}")


__all__ = ["Qwen3VLTextEncoderModuleMixin"]
