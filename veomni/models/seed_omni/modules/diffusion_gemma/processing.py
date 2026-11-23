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

"""CPU worker for :class:`DiffusionGemmaOmni`.

Chat-templating, tokenizing and the block-diffusion canvas construction all happen
here, in a DataLoader worker. The result is one ``ConversationItem`` per sample
holding the encoder row, the drawn canvas (or all ``K`` canvases under
``compute_all_canvas_df_loss``) and the masks the decoder reads.

This reuses ``process_diffusion_gemma_example`` rather than re-deriving the
pipeline. The corruption is keyed on ``(corruption_seed, corruption_epoch,
tokens)`` and the canvas grid is cut from the answer by
``_build_diffusion_gemma_canvas``; a second implementation would drift from the one
the shared data transform draws.
"""

from __future__ import annotations

from typing import Any

from .....data.chat_template import build_chat_template
from .....data.data_transform import process_diffusion_gemma_example
from .....models.auto import build_tokenizer
from ...utils.conversation import ConversationItem
from ..module_processing_base import ModulePreprocessorBase


class DiffusionGemmaOmniPreprocessor(ModulePreprocessorBase):
    """Block-diffusion canvas construction for one DiffusionGemma sample, on CPU.

    Holds the chat template, the tokenizer and the canvas geometry only -- never the
    model -- so it is picklable into workers.
    """

    def __init__(
        self,
        chat_template: Any,
        tokenizer: Any,
        *,
        canvas_length: int,
        max_canvases: int,
        vocab_size: int,
        max_seq_len: int,
        text_keys: str | list[str] = "messages",
        noise_eps: float = 1e-3,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        canvas_start: int = 0,
        compute_all_canvas_df_loss: bool = False,
        corruption_seed: int = 0,
    ) -> None:
        self._chat_template = chat_template
        self._tokenizer = tokenizer
        self._canvas_length = canvas_length
        self._max_canvases = max_canvases
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._text_keys = text_keys
        self._noise_eps = noise_eps
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._canvas_start = canvas_start
        self._compute_all_canvas_df_loss = compute_all_canvas_df_loss
        self._corruption_seed = corruption_seed

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "DiffusionGemmaOmniPreprocessor":
        """Build from the checkpoint subfolder alone, mirroring HF's ``from_pretrained``.

        ``module_config`` is the launcher's ``model_config`` block, which is where the
        canvas geometry and the ``Eq. 13`` switches live; reading them from there keeps
        the preprocessor and the live model on one source of truth. Everything the
        checkpoint's ``config.json`` can answer is read from it instead.
        """
        overrides = dict(config_overrides or {})
        overrides.update(kwargs)
        tokenizer = build_tokenizer(module_path)
        template_name = overrides.pop("chat_template", None)
        if template_name is None:
            # The checkpoint's own template: `tokenizer_generation_prompt` drives
            # `apply_chat_template` rather than a registered spelling. Defaulting to a
            # literal name here (`gemma3`) would fail whenever this worker is built
            # through `bind_module_assets`, which binds assets without launcher
            # overrides.
            template_name = "tokenizer_generation_prompt"
        chat_template = build_chat_template(template_name, tokenizer)

        from transformers import PretrainedConfig

        config_dict, _ = PretrainedConfig.get_config_dict(module_path)
        # The 26B checkpoint nests every text field under `text_config`
        # (`vocab_size`, `eos_token_id`, ...), which is also what the transform reads.
        text_config = config_dict.get("text_config") or {}

        def _pick(key: str, default: Any) -> Any:
            value = overrides.pop(key, None)
            if value is not None:
                return value
            return text_config.get(key, config_dict.get(key, default))

        def _first_eos(value: Any) -> int:
            """Official CanvasChunker uses a single EOS id; Gemma ships a list."""
            if isinstance(value, (list, tuple)):
                value = value[0] if value else 1
            return int(value) if value is not None else 1

        return cls(
            chat_template,
            tokenizer,
            canvas_length=int(_pick("canvas_length", 256)),
            max_canvases=int(_pick("max_canvases", 8)),
            vocab_size=int(_pick("vocab_size", 262144)),
            max_seq_len=int(overrides.pop("max_seq_len", 2048)),
            text_keys=overrides.pop("text_keys", "messages"),
            noise_eps=float(overrides.pop("noise_eps", 1e-3)),
            pad_token_id=int(_pick("pad_token_id", 0)),
            eos_token_id=_first_eos(_pick("eos_token_id", 1)),
            canvas_start=int(_pick("canvas_start", 0)),
            compute_all_canvas_df_loss=bool(_pick("compute_all_canvas_df_loss", False)),
            corruption_seed=int(_pick("corruption_seed", 0)),
        )

    def preprocess_conversations(
        self, conversation_list: list[list[Any]], inference: bool = False, **kwargs: Any
    ) -> None:
        """Replace each sample's turn list with its block-diffusion tensors.

        The carrier arrives with one ``text`` item per turn, so the messages are
        rebuilt from it and handed to ``process_diffusion_gemma_example``, which
        templates, tokenizes, cuts the answer into canvases and draws the corruption.
        The result is a single item: a block-diffusion sample is one encoder row plus
        its canvas stack, not a per-turn structure.
        """
        if inference:
            self._encode_for_inference(conversation_list)
            return
        corruption_epoch = kwargs.get("mask_epoch", 0)
        for index, sample in enumerate(conversation_list):
            messages = [
                {"role": getattr(item, "role", "user"), "content": getattr(item, "value", "")}
                for item in sample
                if getattr(item, "type", None) == "text"
            ]
            if not messages:
                continue
            (processed,) = process_diffusion_gemma_example(
                {"messages": messages},
                chat_template=self._chat_template,
                tokenizer=self._tokenizer,
                max_seq_len=self._max_seq_len,
                canvas_length=self._canvas_length,
                max_canvases=self._max_canvases,
                vocab_size=self._vocab_size,
                noise_eps=self._noise_eps,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
                text_keys=self._text_keys,
                canvas_start=self._canvas_start,
                compute_all_canvas_df_loss=self._compute_all_canvas_df_loss,
                corruption_seed=self._corruption_seed,
                corruption_epoch=corruption_epoch,
            )
            item = ConversationItem(type="text", value=processed, role="assistant")
            # The encoder row is what `dyn_bsz` has to measure: the canvas stack is a
            # fixed `canvas_length` per canvas and adds no variance worth counting.
            item.meta["_omni_token_len"] = int(processed["input_ids"].numel())
            conversation_list[index] = [item]

    def _encode_for_inference(self, conversation_list: list[list[Any]]) -> None:
        """Encode the prompt for the block-diffusion sampler, with no canvas corruption.

        Block diffusion generates by denoising a fresh canvas against a prompt KV
        cache, so the request carries the prompt row alone -- no canvas stack, no
        loss tensors. The chat template is applied here through the *processor*
        path rather than through `process_diffusion_gemma_example`, because that
        transform's whole job is to cut an answer into canvases and draw their
        corruption, and generation has no answer to cut. The tokenizer and template
        are the same objects, so the prompt tokens match what training saw.
        """
        for index, sample in enumerate(conversation_list):
            content: list[dict[str, Any]] = []
            for item in sample:
                if getattr(item, "type", None) == "image":
                    content.append({"type": "image", "image": _to_pil(item.value)})
                elif getattr(item, "type", None) == "text":
                    content.append({"type": "text", "text": getattr(item, "value", "")})
            if not content:
                continue
            encoded = self._chat_template.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            processed = {k: v for k, v in encoded.items() if k != "token_type_ids"}
            item = ConversationItem(type="text", value=processed, role="assistant")
            item.meta["_omni_token_len"] = int(processed["input_ids"].numel())
            conversation_list[index] = [item]


def _to_pil(value: Any) -> Any:
    """Carrier image payload to the PIL image the processor accepts.

    The carrier holds pixels as a ``(C, H, W)`` uint8 tensor (``fetch_media``
    decodes once and keeps uint8 so the worker's IPC stays cheap) while the
    processor speaks PIL. A payload that is already acceptable passes through, so
    a carrier built by a different preprocessor still works.
    """
    import torch
    from PIL import Image

    if isinstance(value, torch.Tensor):
        return Image.fromarray(value.permute(1, 2, 0).contiguous().cpu().numpy())
    return value


__all__ = ["DiffusionGemmaOmniPreprocessor"]
