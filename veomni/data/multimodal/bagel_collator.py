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
"""Packed data collator for the Bagel composition.

Each *sample* is described as an ordered list of segments::

    sample = {
        "segments": [
            {"kind": "text", "ids": [int, ...], "loss": bool},
            {"kind": "vit",  "image": Tensor(3, H, W)},               # understanding image
            {"kind": "vae",  "image": Tensor(3, H, W), "grid": (h, w)},  # generation image
        ]
    }

``text`` segments contribute their tokens to the packed sequence; with ``loss=True`` they
add next-token cross-entropy targets. ``vit`` segments are patchified for the SigLIP
encoder (understanding). ``vae`` segments hold the raw image the (frozen) VAE encodes and
patchifies into latent tokens trained with the rectified-flow MSE loss.

The collator concatenates all samples into a single packed 1-D sequence and emits the
tensors consumed by ``SeedOmniModel._bagel_forward``.
"""

from typing import Any, Dict, List

import torch

from ...models.seed_omni.foundation.bagel_foundation.packing_utils import (
    get_flattened_position_ids_extrapolate,
    patchify,
    prepare_attention_mask_per_sample,
)


class BagelPackedCollator:
    def __init__(
        self,
        vit_patch_size: int = 14,
        vit_max_num_patch_per_side: int = 70,
        latent_patch_size: int = 2,
        latent_downsample: int = 16,
        max_latent_size: int = 32,
        timestep_dist: str = "logit_normal",
    ):
        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.latent_patch_size = latent_patch_size
        self.latent_downsample = latent_downsample  # vae_downsample * latent_patch_size
        self.max_latent_size = max_latent_size
        self.timestep_dist = timestep_dist

    def _sample_timestep(self) -> float:
        if self.timestep_dist == "logit_normal":
            return torch.randn(1).item()
        return torch.rand(1).item()

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        packed_text_ids: List[int] = []
        packed_text_indexes: List[int] = []
        packed_label_ids: List[int] = []
        ce_indexes: List[int] = []

        packed_vit_tokens: List[torch.Tensor] = []
        packed_vit_token_indexes: List[int] = []
        packed_vit_position_ids: List[torch.Tensor] = []
        vit_token_seqlens: List[int] = []

        vae_images: List[torch.Tensor] = []
        patchified_vae_latent_shapes: List = []
        packed_latent_position_ids: List[torch.Tensor] = []
        packed_vae_token_indexes: List[int] = []
        packed_timesteps: List[torch.Tensor] = []
        mse_indexes: List[int] = []

        sample_lens: List[int] = []
        packed_position_ids: List[int] = []
        nested_attention_masks: List[torch.Tensor] = []

        curr = 0
        for sample in samples:
            sample_start = curr
            rope_pos = 0
            split_lens: List[int] = []
            attn_modes: List[str] = []

            for seg in sample["segments"]:
                kind = seg["kind"]
                if kind == "text":
                    ids = list(seg["ids"])
                    n = len(ids)
                    packed_text_ids.extend(ids)
                    packed_text_indexes.extend(range(curr, curr + n))
                    packed_position_ids.extend(range(rope_pos, rope_pos + n))
                    if seg.get("loss", False):
                        for i in range(n - 1):
                            ce_indexes.append(curr + i)
                            packed_label_ids.append(ids[i + 1])
                    rope_pos += n
                    curr += n
                    split_lens.append(n)
                    attn_modes.append("causal")

                elif kind == "vit":
                    image = seg["image"]
                    p = self.vit_patch_size
                    tokens = patchify(image, p)
                    n = tokens.shape[0]
                    vit_pos = get_flattened_position_ids_extrapolate(
                        image.shape[1], image.shape[2], p, self.vit_max_num_patch_per_side
                    )
                    packed_vit_tokens.append(tokens)
                    packed_vit_token_indexes.extend(range(curr, curr + n))
                    packed_vit_position_ids.append(vit_pos)
                    vit_token_seqlens.append(n)
                    packed_position_ids.extend([rope_pos] * n)
                    rope_pos += 1
                    curr += n
                    split_lens.append(n)
                    attn_modes.append("full")

                elif kind == "vae":
                    image = seg["image"]
                    h, w = seg["grid"]
                    n = h * w
                    vae_images.append(image)
                    patchified_vae_latent_shapes.append((h, w))
                    lat_pos = (torch.arange(h)[:, None] * self.max_latent_size + torch.arange(w)).flatten()
                    packed_latent_position_ids.append(lat_pos)
                    packed_vae_token_indexes.extend(range(curr, curr + n))
                    packed_timesteps.append(torch.full((n,), self._sample_timestep()))
                    mse_indexes.extend(range(curr, curr + n))
                    packed_position_ids.extend([rope_pos] * n)
                    rope_pos += 1
                    curr += n
                    split_lens.append(n)
                    attn_modes.append("noise")
                else:
                    raise ValueError(f"Unknown segment kind: {kind}")

            sample_lens.append(curr - sample_start)
            nested_attention_masks.append(prepare_attention_mask_per_sample(split_lens, attn_modes))

        sequence_length = curr
        batch: Dict[str, Any] = {
            "sequence_length": sequence_length,
            "sample_lens": sample_lens,
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "nested_attention_masks": nested_attention_masks,
        }

        if ce_indexes:
            ce_bool = torch.zeros(sequence_length, dtype=torch.bool)
            ce_bool[torch.tensor(ce_indexes, dtype=torch.long)] = True
            batch["ce_loss_indexes"] = ce_bool
            batch["packed_label_ids"] = torch.tensor(packed_label_ids, dtype=torch.long)

        if packed_vit_token_indexes:
            batch["packed_vit_tokens"] = torch.cat(packed_vit_tokens, dim=0)
            batch["packed_vit_token_indexes"] = torch.tensor(packed_vit_token_indexes, dtype=torch.long)
            batch["packed_vit_position_ids"] = torch.cat(packed_vit_position_ids, dim=0)
            batch["vit_token_seqlens"] = torch.tensor(vit_token_seqlens, dtype=torch.int32)

        if packed_vae_token_indexes:
            batch["vae_pixel_values"] = self._pad_and_stack(vae_images)
            batch["patchified_vae_latent_shapes"] = patchified_vae_latent_shapes
            batch["packed_latent_position_ids"] = torch.cat(packed_latent_position_ids, dim=0)
            batch["packed_vae_token_indexes"] = torch.tensor(packed_vae_token_indexes, dtype=torch.long)
            batch["packed_timesteps"] = torch.cat(packed_timesteps, dim=0)
            mse_bool = torch.zeros(sequence_length, dtype=torch.bool)
            mse_bool[torch.tensor(mse_indexes, dtype=torch.long)] = True
            batch["mse_loss_indexes"] = mse_bool

        return batch

    @staticmethod
    def _pad_and_stack(images: List[torch.Tensor]) -> torch.Tensor:
        """Zero-pad a list of (C, H, W) images (top-left aligned) to a common size and stack.

        The valid latent region is recovered downstream via ``patchified_vae_latent_shapes``,
        so padding does not affect the loss.
        """
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        out = []
        for img in images:
            c, h, w = img.shape
            if h != max_h or w != max_w:
                padded = img.new_zeros((c, max_h, max_w))
                padded[:, :h, :w] = img
                img = padded
            out.append(img)
        return torch.stack(out, dim=0)
