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

"""CPU worker for :class:`LladaVOmni`.

Chat-templating, tokenizing, the SigLIP tiling and the dLLM corruption all happen
here, in a DataLoader worker, before ``dyn_bsz`` measures the sample. The result is
one ``ConversationItem`` per sample holding the masked tensors plus, when the
sample has an image, its ``pixel_values`` and the ``image_mask`` saying where those
embeddings go.

This reuses ``process_llada_v_example`` rather than re-deriving the pipeline: the
mask ratio is drawn from a generator keyed on ``(mask_seed, mask_epoch, tokens)``
and the image slots come from the chat template's placeholder count, so a second
implementation would drift from the one the shared data transform draws.
"""

from __future__ import annotations

from typing import Any

from .....data.chat_template import build_chat_template
from .....data.data_transform import process_llada_v_example
from .....models.auto import build_tokenizer
from .....utils.constants import IGNORE_INDEX
from .....utils.logging import get_logger
from ...utils.conversation import ConversationItem
from ..llada.processing import _ASSISTANT_HEADER, LladaOmniPreprocessor, processed_len


logger = get_logger(__name__)


class LladaVOmniPreprocessor(LladaOmniPreprocessor):
    """LLaDA-V's CPU preprocessor: the same masking path plus the image tower's inputs."""

    def __init__(
        self,
        chat_template: Any,
        *,
        mask_token_id: int | None,
        image_processor: Any,
        image_seq_len: int,
        max_seq_len: int,
        text_keys: str | list[str] = "messages",
        mask_eps: float = 1e-3,
        mask_seed: int = 0,
    ) -> None:
        super().__init__(
            chat_template,
            mask_token_id=mask_token_id,
            max_seq_len=max_seq_len,
            text_keys=text_keys,
            mask_eps=mask_eps,
            mask_seed=mask_seed,
        )
        self._image_processor = image_processor
        self._image_seq_len = image_seq_len

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LladaVOmniPreprocessor":
        """Build from the checkpoint subfolder alone, mirroring HF's ``from_pretrained``."""
        overrides = dict(config_overrides or {})
        overrides.update(kwargs)
        tokenizer = build_tokenizer(module_path)
        template_name = overrides.pop("chat_template", "llada")
        chat_template = build_chat_template(template_name, tokenizer)

        # `mask_token_id` and `image_seq_len` live on the model config, not the
        # tokenizer; read them from the checkpoint so this worker needs no live model.
        mask_token_id = overrides.pop("mask_token_id", None)
        image_seq_len = overrides.pop("image_seq_len", None)
        if mask_token_id is None or image_seq_len is None:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(module_path)
            if mask_token_id is None:
                mask_token_id = config_dict.get("mask_token_id")
            if image_seq_len is None:
                side = config_dict["vision_config"]["image_size"] // config_dict["vision_config"]["patch_size"]
                image_seq_len = side * (side + 1)

        worker = cls(
            chat_template,
            mask_token_id=mask_token_id,
            image_processor=_load_image_processor(module_path),
            image_seq_len=int(image_seq_len),
            max_seq_len=int(overrides.pop("max_seq_len", 4096)),
            text_keys=overrides.pop("text_keys", "messages"),
            mask_eps=float(overrides.pop("mask_eps", 1e-3)),
            mask_seed=int(overrides.pop("mask_seed", 0)),
            **overrides,
        )
        # Held directly so `bind_module_assets` can reach it -- see the dense
        # preprocessor for why a tokenizer reachable only through the chat template
        # leaves the module unable to decode what it generated.
        worker._tokenizer = tokenizer
        return worker

    def preprocess_conversations(
        self, conversation_list: list[list[Any]], inference: bool = False, **kwargs: Any
    ) -> None:
        """Replace each sample's turn list with its tensors and image inputs.

        The carrier arrives as one conversation item list: one ``text`` item per turn
        plus one ``image`` item per picture, whose ``value`` is the already-decoded
        image tensor (``fetch_media`` put it there, so this worker never reads a
        file). The image placeholders are restored, then the conversation goes
        through the shared multimodal transform.

        Training draws the corruption in that transform. Inference must not -- the
        FSM lays out its own masked answer span -- so the two paths differ in
        exactly one step: whether ``apply_dllm_masking`` runs. Everything before it
        (placeholder restoration, ``image_seq_len`` slots, the image processor's
        tiles) is shared, which is what keeps the prompt layout identical to what
        the model was trained on.
        """
        mask_epoch = kwargs.get("mask_epoch", 0)
        for index, sample in enumerate(conversation_list):
            # The carrier is flat: one `text` item per turn plus one `image` item per
            # picture, with the picture's `(C, H, W)` uint8 tensor as its value. The
            # preprocessor that built it (`sharegpt4v_sft`) already lifted each
            # `<image>` out of its text turn, so the placeholder has to be put back --
            # the transform counts the placeholders to place the image slots, and a
            # conversation with an image but no placeholder is refused.
            messages: list[dict[str, Any]] = []
            images: list[Any] = []
            for item in sample:
                kind = getattr(item, "type", None)
                if kind == "image":
                    images.append(_to_pil(item.value))
                    if messages:
                        # The preprocessor lifted `<image>` out of its text turn, and
                        # ShareGPT4V puts it first in the turn, so it goes back at the
                        # front. `process_llada_v_example` counts these placeholders to
                        # place the image slots.
                        messages[-1]["content"] = f"<image>{messages[-1]['content']}"
                    else:
                        # An image turn before any text turn: the placeholder opens the
                        # sample and the next text turn carries the instruction.
                        messages.append({"role": getattr(item, "role", "user"), "content": "<image>"})
                elif kind == "text":
                    role = getattr(item, "role", "user")
                    if messages and messages[-1]["content"] == "<image>":
                        messages[-1]["content"] = f"<image>{getattr(item, 'value', '')}"
                    else:
                        messages.append({"role": role, "content": getattr(item, "value", "")})
            if not messages:
                continue
            example = {"messages": messages, "images": images}
            if inference:
                (processed,) = self._encode_for_inference(example)
            else:
                (processed,) = process_llada_v_example(
                    example,
                    self._chat_template,
                    self._max_seq_len,
                    mask_token_id=self._mask_token_id,
                    image_processor=self._image_processor,
                    image_seq_len=self._image_seq_len,
                    text_keys=self._text_keys,
                    mask_eps=self._mask_eps,
                    mask_seed=self._mask_seed,
                    mask_epoch=mask_epoch,
                )
            item = ConversationItem(type="text", value=processed, role="assistant")
            item.meta["_omni_token_len"] = int(processed_len(processed))
            conversation_list[index] = [item]

    def _encode_for_inference(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Lay out the prompt and its image slots without drawing any corruption.

        The shared transform is called with the corruption effectively disabled by
        post-processing rather than by reimplementing its placeholder and slot
        layout: the prompt tokens, ``image_mask`` and ``pixel_values`` must be
        byte-identical to training's, and a second layout implementation is exactly
        how they would stop being so. The assistant header is appended for the same
        reason as in the dense module: the template renders the supervised form and
        stops after the user turn.
        """
        import torch

        (sample,) = process_llada_v_example(
            example,
            self._chat_template,
            self._max_seq_len,
            mask_token_id=self._mask_token_id,
            image_processor=self._image_processor,
            image_seq_len=self._image_seq_len,
            text_keys=self._text_keys,
            mask_eps=self._mask_eps,
            image_key="images",
        )
        # `input_ids` already carries the image slots filled with token 0; the
        # corruption is what turned answer tokens into masks, and generation supplies
        # its own later. Nothing else in the layout is touched.
        input_ids = sample["input_ids"]
        tokenizer = self._chat_template.tokenizer
        header = tokenizer(_ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
        if header:
            input_ids = torch.cat([input_ids, torch.tensor(header, dtype=input_ids.dtype)])
            for key in ("attention_mask", "image_mask"):
                pad = torch.zeros(len(header), dtype=sample[key].dtype)
                sample[key] = torch.cat([sample[key], pad])
            sample["labels"] = torch.cat([sample["labels"], torch.full((len(header),), IGNORE_INDEX)])
            sample["loss_weight"] = torch.cat([sample["loss_weight"], torch.zeros(len(header))])
        sample["input_ids"] = input_ids
        sample["labels"] = torch.full_like(sample["labels"], IGNORE_INDEX)
        sample["loss_weight"] = torch.zeros_like(sample["loss_weight"], dtype=torch.float32)
        sample["answer_mask"] = torch.zeros_like(sample["labels"], dtype=torch.long)
        return [sample]


def _to_pil(value: Any) -> Any:
    """Carrier image payload to the PIL image ``process_llada_v_example`` expects.

    The carrier holds pixels as a ``(C, H, W)`` uint8 tensor (``fetch_images``
    decodes once and keeps uint8 so the worker's IPC stays cheap), while the shared
    transform speaks PIL. A payload that is already something the transform accepts
    -- a PIL image or a path -- is passed through untouched, so this stays correct
    for a carrier built by a different preprocessor.
    """
    import torch
    from PIL import Image

    if isinstance(value, torch.Tensor):
        # `(C, H, W)` -> `(H, W, C)`, which is what PIL reads.
        return Image.fromarray(value.permute(1, 2, 0).contiguous().cpu().numpy())
    return value


def _load_image_processor(module_path: str) -> Any:
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(module_path)


__all__ = ["LladaVOmniPreprocessor"]
