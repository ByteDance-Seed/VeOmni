from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ....utils.conversation import ConversationItem, ItemRole, ItemType, ItemValue


@dataclass(frozen=True)
class ChatMarkers:
    """Default wire-format markers shared by every text encoder (bos / eos only).

    Per-model templates subclass-replace ``chat_markers`` with their own richer
    dataclass (role markers, vision wrap tokens, …); these two are the universal
    minimum resolved from the tokenizer in :meth:`TextEncoderChatTemplate.__init__`.
    Either may be ``None`` (e.g. Qwen has no ``bos_token``).
    """

    bos_token: str | None
    eos_token: str | None


class TextEncoderChatTemplate:
    chat_markers: ChatMarkers

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        # Default markers (bos / eos). Resolved best-effort: a model whose tokenizer
        # lacks one (e.g. Qwen has no bos) gets ``None`` rather than an error.
        self.chat_markers = ChatMarkers(
            bos_token=getattr(tokenizer, "bos_token", None),
            eos_token=getattr(tokenizer, "eos_token", None),
        )
        self.bos_token_id = self._resolve_token_id(tokenizer, "bos_token_id")
        self.eos_token_id = self._resolve_token_id(tokenizer, "eos_token_id")

    @staticmethod
    def _resolve_token_id(tokenizer: Any, attr: str) -> int | None:
        token_id = getattr(tokenizer, attr, None)
        return int(token_id) if token_id is not None else None

    @abstractmethod
    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply the chat template only — produce ``str`` rows with ``meta['loss_mask']``.

        Does NOT tokenize or merge. The full pipeline is
        ``apply_chat_template`` → :meth:`tokenize` →
        :meth:`merge_text_embeds`, bundled by :meth:`tokenize_conversation`.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix to a templated prompt (inference).

        Used by a ChatML-style module ``generate``; each per-model template
        implements it (Janus, whose inference is not ChatML, builds its assistant
        prefix inline in ``generate`` instead).
        """
        raise NotImplementedError

    def tokenize_conversation(
        self,
        sample: list[ConversationItem],
        *,
        add_generation_prompt: bool = False,
    ) -> list[ConversationItem]:
        """Full text pipeline: apply chat template → tokenize → merge tokenized text.

        ``add_generation_prompt`` (inference) inserts the assistant generation
        prefix (:meth:`apply_generation_prompt`) between templating and tokenizing,
        so the request is left ready for the model to start decoding; training
        leaves it off (the assistant turn is already in the data).
        """
        parts = self.apply_chat_template(sample)
        if add_generation_prompt:
            parts = self.apply_generation_prompt(parts)
        self.tokenize(parts)
        return self.merge_text_embeds(parts)

    def _build_conversation_item(
        self,
        type: ItemType,
        value: ItemValue,
        role: ItemRole,  # for pretrain data, role is always "assistant"
        loss_mask: int | None = None,
        meta: dict | None = None,
    ) -> ConversationItem:
        """Build one conversation row; ``text`` rows always get ``meta["loss_mask"]``.

        Only used for template-generated **text** marker rows. Media rows (image /
        video) are passed through verbatim by the per-model ``apply_chat_template``
        so they keep their ``value`` / ``source`` / ``meta`` (the per-module
        encoders filter them by ``item.source``).
        """
        part_meta = dict(meta or {})
        if type == "text":
            part_meta["loss_mask"] = int(role == "assistant") if loss_mask is None else int(loss_mask)
        return ConversationItem(type=type, value=value, role=role, meta=part_meta)

    @staticmethod
    def pack_input_ids(parts: list[ConversationItem]) -> list[torch.Tensor]:
        """Collect ``type='text'`` token-id tensors (``value``); one tensor per text row."""
        return [part.value for part in parts if part.type == "text"]

    def tokenize(self, parts: list[ConversationItem]) -> None:
        """Tokenize each ``text`` part in place: ``str`` value → token-id tensor.
        Builds tensors on CPU. Sets ``meta['labels']`` (``-100`` where ``loss_mask`` is 0) and ``meta['attention_mask']``.
        """
        for part in parts:
            if part.type == "text":
                text = part.value
                loss_mask = int(part.meta.pop("loss_mask"))
                input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                labels = input_ids if loss_mask else [-100] * len(input_ids)
                part.value = torch.tensor(input_ids, dtype=torch.long)
                part.meta["labels"] = torch.tensor(labels, dtype=torch.long)
                part.meta["attention_mask"] = torch.ones(len(input_ids), dtype=torch.long)

    @staticmethod
    def merge_text_embeds(parts: list[ConversationItem]) -> list[ConversationItem]:
        """Merge adjacent same-role ``text`` parts (concat ids / labels / mask).

        Run AFTER :meth:`tokenize`: it concatenates the per-segment token-id,
        ``labels`` and ``attention_mask`` tensors. Because labels are already
        computed per segment, distinct ``loss_mask`` spans under the same role
        (e.g. the masked assistant prefix vs. the supervised response) keep their
        own labels through the merge.
        """
        merged: list[ConversationItem] = []
        for part in parts:
            if merged and merged[-1].type == "text" and part.type == "text" and merged[-1].role == part.role:
                prev = merged[-1]
                prev.value = torch.cat([prev.value, part.value])
                prev.meta["labels"] = torch.cat([prev.meta["labels"], part.meta["labels"]])
                prev.meta["attention_mask"] = torch.cat([prev.meta["attention_mask"], part.meta["attention_mask"]])
                continue
            merged.append(part)
        return merged
