from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import ConversationItem, is_dummy, maybe_merge_outputs
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import CPUPreprocessor, pre_forward
from ...base.text_encoder.modulemixin import TextEncoderModuleMixin
from ...qwen3vl.text_encoder.chat_template import (
    Qwen3VLChatMarkers,
    apply_qwen3vl_chat_template,
    apply_qwen3vl_generation_prompt,
)
from .chat_template import (
    Qwen3ChatMarkers,
    apply_qwen3_chat_template,
    apply_qwen3_generation_prompt,
    merge_consecutive_text_parts,
    pack_text_input_ids,
    tokenize_template_parts,
)
from .configuration import Qwen3TextEncoderConfig


SIGNAL_TEXT_DONE = "text_done"

# Sentinel written by Qwen3TextEncoderCPUPreprocessor onto every text part's meta
# so the thin ``encode_pre`` knows chat-template + tokenize already ran in the
# DataLoader worker (and otherwise falls back to the in-module path).
_OMNI_TOKENIZED = "_omni_tokenized"


class Qwen3TextEncoderCPUPreprocessor(CPUPreprocessor):
    """Worker-side chat-template + tokenize for the Qwen3 text encoder.

    Holds only the (picklable) tokenizer + chat markers + the module-level
    template function (text-only or image ChatML) — never the model. Builds CPU
    tensors so it can run in DataLoader workers and overlap with GPU compute.
    """

    def __init__(self, tokenizer: Any, chat_markers: Any, apply_template_fn: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_markers = chat_markers
        self._apply_template_fn = apply_template_fn

    def __call__(self, conversation_list: list[list[ConversationItem]]) -> None:
        for sample in conversation_list or []:
            if sample and sample[0].meta.get(_OMNI_TOKENIZED):
                continue  # idempotent: already processed
            parts = self._apply_template_fn(sample, self._chat_markers)
            tokenize_template_parts(parts, self._tokenizer, device=None)
            parts = merge_consecutive_text_parts(parts)
            for part in parts:
                if part.type == "text":
                    part.meta[_OMNI_TOKENIZED] = True
            sample.clear()
            sample.extend(parts)


class Qwen3TextEncoderModuleMixin(TextEncoderModuleMixin):
    """Qwen3 ChatML text encoder, optionally image-aware.

    With ``config.enable_image`` the module uses the Qwen3-VL image ChatML template
    (image items wrapped in ``<|vision_start|> … <|vision_end|>``; the sibling
    vision module supplies the projected patch embeds) and handles image/video
    parts in decode — enough to bootstrap a text-only Qwen3 into image
    understanding. In that mode :meth:`freeze_model` trains only the vision
    special-token embedding rows, whose ids it resolves from its own tokenizer
    (the user can't know them, but the module can). With it off the behaviour is
    the original text-only Qwen3 path.
    """

    config: Qwen3TextEncoderConfig

    # Vision special tokens whose embedding rows bootstrap image understanding;
    # ids are resolved from the tokenizer at freeze time (see :meth:`freeze_model`).
    _VISION_SPECIAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")

    def init_omni_state(self) -> None:
        super().init_omni_state()
        self._chat_markers: Optional[Qwen3ChatMarkers | Qwen3VLChatMarkers] = None
        self._eos_token_id: Optional[int] = None
        self._im_end_token_id: Optional[int] = None
        self._trainable_row_mask: Optional[torch.Tensor] = None

    @property
    def _enable_image(self) -> bool:
        return self.config.enable_image

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = int(tokenizer.eos_token_id)
        self._im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        # Only the markers differ: image mode adds the vision wrap tokens.
        if self._enable_image:
            self._chat_markers = Qwen3VLChatMarkers(
                im_start_token="<|im_start|>",
                im_end_token="<|im_end|>",
                eos_token=str(tokenizer.eos_token),
                assistant_prefix="<|im_start|>assistant\n",
                vision_start_token="<|vision_start|>",
                vision_end_token="<|vision_end|>",
            )
        else:
            self._chat_markers = Qwen3ChatMarkers(
                im_start_token="<|im_start|>",
                im_end_token="<|im_end|>",
                eos_token=str(tokenizer.eos_token),
                assistant_prefix="<|im_start|>assistant\n",
            )

    def _apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        if self._enable_image:
            return apply_qwen3vl_chat_template(sample, self._chat_markers)
        return apply_qwen3_chat_template(sample, self._chat_markers)

    def _apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        if self._enable_image:
            return apply_qwen3vl_generation_prompt(sample, self._chat_markers)
        return apply_qwen3_generation_prompt(sample, self._chat_markers)

    # ── Freeze: in image mode, train only the vision special-token rows ──────
    def freeze_model(self) -> None:
        if not self._enable_image:
            return  # fully trainable (default text-only behaviour)
        # The user can't know the vision special-token ids, but the module can:
        # resolve them from its own tokenizer so only those rows stay trainable.
        ids = [int(self._tokenizer.convert_tokens_to_ids(tok)) for tok in self._VISION_SPECIAL_TOKENS]
        weight = self.embed_tokens.weight
        weight.requires_grad_(True)
        keep = torch.zeros(weight.shape[0], dtype=torch.bool)
        keep[ids] = True
        self._trainable_row_mask = keep

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self._trainable_row_mask.to(device=grad.device)
            return grad * mask.unsqueeze(1).to(grad.dtype)

        weight.register_hook(_mask_grad)

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
        """Worker-side chat-template + tokenize (see :class:`Qwen3TextEncoderCPUPreprocessor`)."""
        if self._chat_markers is None or getattr(self, "_tokenizer", None) is None:
            return None
        apply_fn = apply_qwen3vl_chat_template if self._enable_image else apply_qwen3_chat_template
        return Qwen3TextEncoderCPUPreprocessor(self._tokenizer, self._chat_markers, apply_fn)

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
        parts = self._apply_chat_template(sample)
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
            raise RuntimeError("Qwen3 text segment count mismatch during embed scatter.")

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
                if part.type == "text":
                    hidden_states = part.value
                    if hidden_states.dim() == 3:
                        hidden_states = hidden_states.squeeze(0)
                    # ``labels`` rides in meta from tokenize (CPU). Keep all label
                    # chunks on CPU; move the concatenated result to device once.
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif self._enable_image and part.type in ("image", "video"):
                    # Vision segment carries projected patch embeds; keep one row
                    # (no label) so the sequence stays aligned, like the backbone.
                    hidden_states = part.value
                    if hidden_states.dim() == 3:
                        hidden_states = hidden_states.squeeze(0)
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)  # CPU
        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        tail = conversation_list[-1]
        if tail.role == "user":
            conversation_list = self._apply_chat_template(conversation_list)
            conversation_list = self._apply_generation_prompt(conversation_list)
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


__all__ = ["Qwen3TextEncoderModuleMixin"]
