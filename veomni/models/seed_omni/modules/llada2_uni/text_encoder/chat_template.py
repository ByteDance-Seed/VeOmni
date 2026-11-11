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

"""LLaDA2.0-Uni (GlmImage) chat template and dLLM masking helpers.

Reuses the layout rules of :class:`~veomni.data.chat_template.Llada2UniTemplate`
and the masking math of :func:`~veomni.data.data_transform.apply_dllm_masking`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from veomni.data.data_transform import apply_dllm_masking
from veomni.utils.constants import IGNORE_INDEX

from ....utils.conversation import ConversationItem


_ROLE_MAP = {"system": "SYSTEM", "user": "HUMAN", "assistant": "ASSISTANT", "tool": "OBSERVATION"}
_OMNI_TOKENIZED = "_omni_tokenized"


@dataclass(frozen=True)
class LladaUniChatMarkers:
    """Wire-format strings for the GlmImage ``<role>`` understand template."""

    role_open: str = "<role>"
    role_close: str = "</role>"
    eos_token: str = "<|endoftext|>"
    understand_system: str = "You are a multimodal understanding assistant."

    def header(self, role: str) -> str:
        return self.role_open + _ROLE_MAP.get(role, role.upper()) + self.role_close


def _text_item(value: Any, role: str, *, loss_mask: int) -> ConversationItem:
    return ConversationItem(type="text", value=value, role=role, meta={"loss_mask": int(loss_mask)})


def apply_lladauni_chat_template(
    sample: list[ConversationItem],
    markers: LladaUniChatMarkers,
) -> list[ConversationItem]:
    """Apply the ``understand_image``-style ``<role>`` template (no ``<|role_end|>``)."""
    out: list[ConversationItem] = []
    items = [it for it in sample if not it.is_dummy]
    dummy_parts = [it for it in sample if it.is_dummy]

    sys_content = "".join(str(it.value) for it in items if it.role == "system" and it.type == "text")
    system = sys_content or markers.understand_system
    items = [it for it in items if it.role != "system"]

    first = True
    i = 0
    while i < len(items):
        role = items[i].role
        run = []
        while i < len(items) and items[i].role == role:
            run.append(items[i])
            i += 1

        is_answer = int(role == "assistant")
        if first:
            header = f"{markers.header('system')} {system} {markers.header(role)}"
            first = False
        else:
            header = markers.header(role)
        out.append(_text_item(header, role, loss_mask=0))

        for item in run:
            if item.type == "image":
                out.append(ConversationItem(type="image", value=item.value, role=role, meta=dict(item.meta)))
            elif item.type == "text":
                out.append(_text_item(str(item.value), role, loss_mask=is_answer))
            else:
                raise ValueError(f"LLaDA2.0-Uni text encoder: unsupported part type {item.type!r}")

        if role == "assistant":
            out.append(_text_item(markers.eos_token, role, loss_mask=1))

    out.extend(dummy_parts)
    return out


def build_image_segment_ids(item: ConversationItem, tokenizer: Any, config: Any, device: torch.device) -> torch.Tensor:
    """Wrap an image item's VQ ids as ``<|image|><|reserved_token_h|><|reserved_token_w|><boi>[VQ…]<|/image|>``."""
    vq_ids = item.meta.get("llada2_uni_vq_ids")
    if vq_ids is None and isinstance(item.value, torch.Tensor) and item.value.dtype == torch.long:
        vq_ids = item.value
    grid = item.meta.get("llada2_uni_grid")
    if vq_ids is None or grid is None:
        raise RuntimeError(
            "LladaUniTextEncoder: image item missing VQ ids — the image_tokenizer.encode "
            "node must run before the text_encoder.encode node in the graph."
        )
    h, w = int(grid[0]), int(grid[1])
    vq_ids = vq_ids.to(device=device, dtype=torch.long).reshape(-1)
    if tokenizer is not None:
        prefix_ids = (
            tokenizer("<|image|>", add_special_tokens=False)["input_ids"]
            + tokenizer(f"<|reserved_token_{h}|>", add_special_tokens=False)["input_ids"]
            + tokenizer(f"<|reserved_token_{w}|>", add_special_tokens=False)["input_ids"]
            + tokenizer("<boi>", add_special_tokens=False)["input_ids"]
        )
        suffix_ids = tokenizer("<|/image|>", add_special_tokens=False)["input_ids"]
    else:
        prefix_ids = [
            int(config.image_start_token_id),
            int(config.reserved_token_base) + h,
            int(config.reserved_token_base) + w,
            int(config.boi_token_id),
        ]
        suffix_ids = [int(config.image_end_token_id)]
    prefix = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    suffix = torch.tensor(suffix_ids, device=device, dtype=torch.long)
    return torch.cat([prefix, vq_ids, suffix], dim=0)


def prepare_llada2_uni_sample_training(
    sample: list[ConversationItem],
    *,
    tokenizer: Any,
    config: Any,
    markers: LladaUniChatMarkers,
    device: torch.device,
) -> None:
    """Apply GlmImage chat template, splice VQ image spans, merge into one sequence, and apply dLLM masking."""
    parts = apply_lladauni_chat_template(sample, markers)
    id_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    dummy_parts: list[ConversationItem] = []

    for part in parts:
        if part.is_dummy:
            dummy_parts.append(part)
            continue
        if part.type == "image":
            ids = build_image_segment_ids(part, tokenizer, config, device)
            id_chunks.append(ids)
            label_chunks.append(torch.full((ids.shape[0],), IGNORE_INDEX, dtype=torch.long, device=device))
        elif part.type == "text":
            text = str(part.value)
            loss_mask = int(part.meta.pop("loss_mask", 0))
            raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = torch.tensor(raw_ids, dtype=torch.long, device=device)
            labels = (
                ids.clone()
                if loss_mask
                else torch.full((len(raw_ids),), IGNORE_INDEX, dtype=torch.long, device=device)
            )
            id_chunks.append(ids)
            label_chunks.append(labels)

    if not id_chunks:
        ids = torch.zeros(1, dtype=torch.long, device=device)
        labels = torch.full((1,), IGNORE_INDEX, dtype=torch.long, device=device)
    else:
        ids = torch.cat(id_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)

    # Reuse `veomni.data.data_transform.apply_dllm_masking` on CPU with a deterministic
    # per-sample generator so DP/SP ranks do not depend on the ambient process RNG.
    from veomni.data.data_transform import diffusion_gemma_noise_generator, diffusion_gemma_sample_key

    ids_cpu = ids.cpu()
    generator = diffusion_gemma_noise_generator(
        diffusion_gemma_sample_key(0, 0, ids_cpu),
        0,
    )
    corrupted_ids, masked_labels, loss_weight = apply_dllm_masking(
        ids_cpu,
        labels.cpu(),
        mask_token_id=int(config.mask_token_id),
        mask_eps=float(config.mask_eps),
        generator=generator,
    )
    merged_item = ConversationItem(
        type="text",
        value=corrupted_ids.to(device=device, dtype=torch.long),
        role="assistant",
        meta={
            "labels": masked_labels.to(device=device, dtype=torch.long),
            "loss_weight": loss_weight.to(device=device, dtype=torch.float32),
            "attention_mask": torch.ones(corrupted_ids.shape[0], dtype=torch.long, device=device),
            _OMNI_TOKENIZED: True,
        },
    )
    sample.clear()
    sample.append(merged_item)
    sample.extend(dummy_parts)


__all__ = [
    "_OMNI_TOKENIZED",
    "LladaUniChatMarkers",
    "apply_lladauni_chat_template",
    "build_image_segment_ids",
    "prepare_llada2_uni_sample_training",
]
