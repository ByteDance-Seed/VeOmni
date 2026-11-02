import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ...utils.conversation import _OMNI_TOKEN_LEN, ConversationItem, ItemRole, ItemType, ItemValue


# Metadata key associating a media row with its surrounding template wrapper
# rows (e.g. ``<|vision_start|>`` / ``<|vision_end|>`` in ``Qwen3VLChatTemplate``)
# so ``merge_text_embeds`` and ``truncate_to_max_seq_len`` keep them together as
# an indivisible span.
_MEDIA_WRAP_GROUP_KEY = "_omni_media_wrap_group"


def resolve_media_token_len(
    item: ConversationItem,
    *,
    spatial_merge_size: int = 2,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
) -> int:
    """Ensure a non-text ``ConversationItem`` records a positive ``_omni_token_len``.

    Precedence:
    1. Existing positive ``item.meta[_OMNI_TOKEN_LEN]`` (set by a module preprocessor).
    2. ``_omni_grid`` / ``grid_thw`` ``[t, h, w]`` → ``(t * h * w) // spatial_merge_size**2``.
    3. Tensor / PIL / ``VideoInputs`` payload dimensions using the Qwen3-VL patch/merge stride.
    4. Fallback of ``1`` for symbolic placeholders in unit tests.
    """
    existing = item.meta.get(_OMNI_TOKEN_LEN)
    if existing is not None and int(existing) > 0:
        return int(existing)

    grid = item.meta.get("_omni_grid", item.meta.get("grid_thw"))
    if grid is not None:
        if isinstance(grid, torch.Tensor):
            t, h, w = (int(x) for x in grid.reshape(-1).tolist())
        else:
            t, h, w = (int(x) for x in grid)
        token_len = max(1, (t * h * w) // (spatial_merge_size * spatial_merge_size))
        item.meta[_OMNI_TOKEN_LEN] = token_len
        return token_len

    val = getattr(item.value, "video", item.value)
    stride = patch_size * spatial_merge_size
    if isinstance(val, torch.Tensor):
        if val.dim() == 2:
            token_len = max(1, int(val.shape[0]))
        elif val.dim() == 3:
            h_tok = max(1, math.ceil(int(val.shape[-2]) / stride))
            w_tok = max(1, math.ceil(int(val.shape[-1]) / stride))
            token_len = h_tok * w_tok
        elif val.dim() == 4:
            t_tok = max(1, math.ceil(int(val.shape[0]) / temporal_patch_size))
            h_tok = max(1, math.ceil(int(val.shape[-2]) / stride))
            w_tok = max(1, math.ceil(int(val.shape[-1]) / stride))
            token_len = t_tok * h_tok * w_tok
        else:
            token_len = max(1, int(val.numel()))
    elif hasattr(val, "size") and isinstance(getattr(val, "size", None), tuple) and len(val.size) == 2:
        w, h = int(val.size[0]), int(val.size[1])
        token_len = max(1, math.ceil(h / stride)) * max(1, math.ceil(w / stride))
    else:
        token_len = 1

    item.meta[_OMNI_TOKEN_LEN] = token_len
    return token_len


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
        max_seq_len: int | None = None,
    ) -> list[ConversationItem]:
        """Full text pipeline: apply chat template → tokenize → merge tokenized text.

        ``add_generation_prompt`` (inference) inserts the assistant generation
        prefix (:meth:`apply_generation_prompt`) between templating and tokenizing,
        so the request is left ready for the model to start decoding; training
        leaves it off (the assistant turn is already in the data).

        ``max_seq_len`` (training) truncates the merged conversation to the last
        ``max_seq_len`` tokens — same suffix-truncation contract as classic
        ``ChatmlTemplate`` — so an overlong prompt does not discard the assistant
        response at the tail.
        """
        parts = self.apply_chat_template(sample)
        if add_generation_prompt:
            parts = self.apply_generation_prompt(parts)
        self.tokenize(parts)
        parts = self.merge_text_embeds(parts)
        if max_seq_len is not None:
            parts = self.truncate_to_max_seq_len(parts, max_seq_len)
        return parts

    @staticmethod
    def truncate_to_max_seq_len(parts: list[ConversationItem], max_seq_len: int) -> list[ConversationItem]:
        """Keep the last ``max_seq_len`` tokens across ``parts`` (suffix truncate).

        Walks non-dummy units from the end so assistant responses at the tail of
        long prompts are preserved (matching classic ``ChatmlTemplate``). Plain
        text rows are suffix-sliced in place (``value`` / ``labels`` /
        ``attention_mask`` / ``_omni_token_len``). Media rows and their associated
        vision wrapper rows (linked via ``_MEDIA_WRAP_GROUP_KEY``) are budgeted as
        a single indivisible group: kept together when the full group fits in the
        remaining token budget, or dropped together (stopping truncation) when it
        does not. ``is_dummy`` FSDP-anchor items stay at the tail without
        consuming sequence budget.
        """
        if max_seq_len < 0:
            raise ValueError(f"max_seq_len must be >= 0, got {max_seq_len}.")
        dummy_parts = [part for part in parts if part.is_dummy]
        non_dummy = [part for part in parts if not part.is_dummy]
        if max_seq_len == 0 or not non_dummy:
            return dummy_parts

        def _item_token_len(item: ConversationItem) -> int:
            if item.type != "text":
                return resolve_media_token_len(item)
            existing = item.meta.get(_OMNI_TOKEN_LEN)
            if existing is not None:
                return int(existing)
            if isinstance(item.value, torch.Tensor):
                return int(item.value.numel())
            return len(str(item.value))

        units: list[tuple[bool, list[ConversationItem], int]] = []
        idx = 0
        while idx < len(non_dummy):
            part = non_dummy[idx]
            wrap_group = part.meta.get(_MEDIA_WRAP_GROUP_KEY)
            if wrap_group is not None:
                end_idx = idx + 1
                while end_idx < len(non_dummy) and non_dummy[end_idx].meta.get(_MEDIA_WRAP_GROUP_KEY) == wrap_group:
                    end_idx += 1
                group_items = non_dummy[idx:end_idx]
                group_len = sum(_item_token_len(item) for item in group_items)
                units.append((True, group_items, group_len))
                idx = end_idx
                continue
            item_len = _item_token_len(part)
            units.append((part.type != "text", [part], item_len))
            idx += 1

        kept_groups_reversed: list[list[ConversationItem]] = []
        remaining = max_seq_len
        for is_atomic, group_items, group_len in reversed(units):
            if group_len <= remaining:
                kept_groups_reversed.append(group_items)
                remaining -= group_len
                if remaining == 0:
                    break
                continue
            if is_atomic or remaining == 0:
                # Cannot partially slice a media row or its vision wrappers.
                break
            (part,) = group_items
            part.value = part.value[-remaining:]
            part.meta["labels"] = part.meta["labels"][-remaining:]
            part.meta["attention_mask"] = part.meta["attention_mask"][-remaining:]
            part.meta[_OMNI_TOKEN_LEN] = remaining
            kept_groups_reversed.append([part])
            break

        kept = [item for group in reversed(kept_groups_reversed) for item in group]
        kept.extend(dummy_parts)
        return kept

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
        so they keep their ``value`` / ``role`` / ``meta`` (the per-module
        encoders select them by ``type`` / ``role`` / ``meta`` tags).
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
                part.meta[_OMNI_TOKEN_LEN] = len(input_ids)

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
            if (
                merged
                and merged[-1].type == "text"
                and part.type == "text"
                and merged[-1].role == part.role
                and merged[-1].meta.get(_MEDIA_WRAP_GROUP_KEY) == part.meta.get(_MEDIA_WRAP_GROUP_KEY)
            ):
                prev = merged[-1]
                prev.value = torch.cat([prev.value, part.value])
                prev.meta["labels"] = torch.cat([prev.meta["labels"], part.meta["labels"]])
                prev.meta["attention_mask"] = torch.cat([prev.meta["attention_mask"], part.meta["attention_mask"]])
                prev.meta[_OMNI_TOKEN_LEN] = int(prev.value.numel())
                continue
            merged.append(part)
        return merged
