# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU worker for :class:`LladaOmni`.

Tokenizing, chat-templating and the dLLM corruption all happen here, in a
DataLoader worker, before ``dyn_bsz`` measures the sample. The masked tensors
ride on the sample's ``ConversationItem`` so the GPU-side ``pre_forward`` only
has to stack and pad them.

This reuses ``process_llada_example`` rather than re-deriving the corruption:
the mask ratio is drawn from a generator keyed on ``(mask_seed, mask_epoch,
tokens)``, and a second implementation would drift from the one the non-Omni
trainer uses, which is exactly the comparison this module exists to support.
"""

from __future__ import annotations

from typing import Any

import torch

from .....data.chat_template import build_chat_template
from .....data.data_transform import process_conversation_example, process_llada_example
from .....models.auto import build_tokenizer
from .....utils.constants import IGNORE_INDEX
from ...utils.conversation import ConversationItem
from ..module_processing_base import ModulePreprocessorBase


class LladaOmniPreprocessor(ModulePreprocessorBase):
    """Chat-template + dLLM masking for one LLaDA sample, on CPU.

    Holds the chat template and the special-token ids only -- never the model --
    so it is picklable into workers.
    """

    def __init__(
        self,
        chat_template: Any,
        *,
        mask_token_id: int | None,
        max_seq_len: int,
        text_keys: str | list[str] = "messages",
        mask_eps: float = 1e-3,
        mask_seed: int = 0,
    ) -> None:
        self._chat_template = chat_template
        self._mask_token_id = mask_token_id
        self._max_seq_len = max_seq_len
        self._text_keys = text_keys
        self._mask_eps = mask_eps
        self._mask_seed = mask_seed

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LladaOmniPreprocessor":
        """Build from the checkpoint subfolder alone, mirroring HF's ``from_pretrained``."""
        overrides = dict(config_overrides or {})
        overrides.update(kwargs)
        tokenizer = build_tokenizer(module_path)
        template_name = overrides.pop("chat_template", "llada")
        chat_template = build_chat_template(template_name, tokenizer)
        # `mask_token_id` lives on the model config, not the tokenizer; read it
        # from the checkpoint's config.json so this worker needs no live model.
        mask_token_id = overrides.pop("mask_token_id", None)
        if mask_token_id is None:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(module_path)
            mask_token_id = config_dict.get("mask_token_id")
        return cls(
            chat_template,
            mask_token_id=mask_token_id,
            max_seq_len=int(overrides.pop("max_seq_len", 4096)),
            text_keys=overrides.pop("text_keys", "messages"),
            mask_eps=float(overrides.pop("mask_eps", 1e-3)),
            mask_seed=int(overrides.pop("mask_seed", 0)),
            **overrides,
        )

    def preprocess_conversations(
        self, conversation_list: list[list[Any]], inference: bool = False, **kwargs: Any
    ) -> None:
        """Replace each sample's turn list with its masked token tensors.

        The carrier arrives as one item per turn (``role`` + the turn's text), so
        the messages are rebuilt from it and handed to ``process_llada_example``,
        which chat-templates, tokenizes and draws the corruption in one call. The
        result is a single item holding the tensors, because a mask predictor is
        supervised over one concatenated sequence -- not per turn.

        ``meta[_OMNI_TOKEN_LEN]`` carries the real token count so ``dyn_bsz``
        measures physical tokens rather than turn count.

        Inference skips the corruption: generation starts from a clean prompt and
        the FSM masks what it needs.
        """
        mask_epoch = kwargs.get("mask_epoch", 0)
        for index, sample in enumerate(conversation_list):
            messages = [
                {"role": getattr(item, "role", "user"), "content": getattr(item, "value", "")}
                for item in sample
                if getattr(item, "type", None) == "text"
            ]
            if not messages:
                continue
            example = {"messages": messages}
            if inference:
                processed = self._encode_for_inference(example)
            else:
                (processed,) = process_llada_example(
                    example,
                    self._chat_template,
                    self._max_seq_len,
                    mask_token_id=self._mask_token_id,
                    text_keys=self._text_keys,
                    mask_eps=self._mask_eps,
                    mask_seed=self._mask_seed,
                    mask_epoch=mask_epoch,
                )
            item = ConversationItem(type="text", value=processed, role="assistant")
            item.meta["_omni_token_len"] = int(processed_len(processed))
            conversation_list[index] = [item]

    def _encode_for_inference(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Tokenize without corrupting; generation supplies its own masks later."""
        (tokenized,) = process_conversation_example(example, self._chat_template, self._max_seq_len, self._text_keys)
        labels = tokenized["labels"]
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.full_like(labels, IGNORE_INDEX),
            "loss_weight": torch.zeros(labels.shape, dtype=torch.float32),
            "answer_mask": torch.zeros(labels.shape, dtype=torch.long),
        }


def processed_len(value: Any) -> int:
    """Physical token span of a processed sample value."""
    if isinstance(value, dict):
        ids = value.get("input_ids")
        if ids is not None:
            return int(ids.numel())
    return 0


__all__ = ["LladaOmniPreprocessor"]
