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
"""Sample transform + tokenizer setup for the Bagel composition.

Turns a raw multimodal sample into the ``segments`` structure consumed by
``BagelPackedCollator``. Two task types are supported out of the box:

* understanding (``image`` + ``prompt`` -> ``response``): cross-entropy on the response.
* generation (``prompt`` -> ``image``): rectified-flow MSE on the VAE latents.

This is a reference implementation: adapt :class:`BagelSampleTransform` to your dataset
schema as needed. The image preprocessing here is intentionally dependency-light (PIL +
numpy) so it works without the (WIP) submodule processors.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


def pil_img2rgb(image: "Image.Image") -> "Image.Image":
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    return image


def add_special_tokens(tokenizer):
    """Add Bagel's chat / vision special tokens and return their ids.

    Mirrors the upstream Bagel ``add_special_tokens`` so converted checkpoints and the
    training data agree on the special-token ids.
    """
    all_special_tokens = []
    for _, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = [
        t for t in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"] if t not in all_special_tokens
    ]
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return tokenizer, new_token_ids, num_new_tokens


def _to_chw_tensor(image: "Image.Image", size: tuple, mean: float, std: float) -> torch.Tensor:
    image = pil_img2rgb(image).resize((size[1], size[0]), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _round_to(value: int, multiple: int, max_value: int) -> int:
    value = max(multiple, min(value, max_value))
    return max(multiple, round(value / multiple) * multiple)


class BagelSampleTransform:
    def __init__(
        self,
        tokenizer,
        new_token_ids: Dict[str, int],
        vit_patch_size: int = 14,
        vit_image_size: int = 224,
        latent_downsample: int = 16,
        gen_image_size: int = 256,
    ):
        self.tokenizer = tokenizer
        self.ids = new_token_ids
        self.vit_patch_size = vit_patch_size
        self.vit_image_size = vit_image_size
        self.latent_downsample = latent_downsample
        self.gen_image_size = gen_image_size

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _vit_segment(self, image: "Image.Image") -> Dict[str, Any]:
        side = _round_to(self.vit_image_size, self.vit_patch_size, self.vit_image_size)
        # SigLIP normalization (mean=std=0.5).
        tensor = _to_chw_tensor(image, (side, side), 0.5, 0.5)
        return {"kind": "vit", "image": tensor}

    def _vae_segment(self, image: "Image.Image") -> Dict[str, Any]:
        side = _round_to(self.gen_image_size, self.latent_downsample, self.gen_image_size)
        tensor = _to_chw_tensor(image, (side, side), 0.5, 0.5)  # -> roughly [-1, 1]
        grid = (side // self.latent_downsample, side // self.latent_downsample)
        return {"kind": "vae", "image": tensor, "grid": grid}

    def __call__(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task = sample.get("task", "understanding")
        bos, eos = self.ids["bos_token_id"], self.ids["eos_token_id"]
        soi, eoi = self.ids["start_of_image"], self.ids["end_of_image"]

        if task == "generation":
            prompt = sample["prompt"]
            image = sample["image"]
            segments = [
                {"kind": "text", "ids": [bos] + self._encode(prompt) + [soi], "loss": False},
                self._vae_segment(image),
                {"kind": "text", "ids": [eoi, eos], "loss": False},
            ]
            return {"segments": segments}

        # understanding
        prompt = sample["prompt"]
        response = sample["response"]
        image = sample.get("image", None)
        segments = [{"kind": "text", "ids": [bos] + self._encode(prompt), "loss": False}]
        if image is not None:
            segments.append({"kind": "text", "ids": [soi], "loss": False})
            segments.append(self._vit_segment(image))
            segments.append({"kind": "text", "ids": [eoi], "loss": False})
        segments.append({"kind": "text", "ids": self._encode(response) + [eos], "loss": True})
        return {"segments": segments}
