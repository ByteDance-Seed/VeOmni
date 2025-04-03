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


import copy
import json
from collections import defaultdict
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch
from PIL import Image

from ..constants import TYPE2INDEX
from .preprocess import conv_preprocess


if TYPE_CHECKING:
    from ...models.seed_omni import SeedOmniProcessor
    from ..chat_template import ChatTemplate


def mask_before_position_id_func(input_ids: torch.Tensor):  # currently image mask only
    """Mask special multimodal tokens in input_ids to input_mm_token for position_id.
    Only supports special image tokens now. (input_image_id=-200, output_image_id=-201->-200)
    Similar to veomni.module.seed_omni.modeling_seed_omni.mask_before_text_encoder

    Args:
        input_ids (torch.Tensor)

    Returns:
        input_ids (torch.Tensor)
    """

    output_image_mask = input_ids == TYPE2INDEX["output"]["image"]
    input_image_mask = input_ids == TYPE2INDEX["input"]["image"]
    input_ids = torch.where(output_image_mask | input_image_mask, TYPE2INDEX["input"]["image"], input_ids)
    return input_ids


def mask_input_ids(modality_info: Dict, input_ids: torch.Tensor):
    """Mask special multimodal tokens in input_ids to 0 for text_encoder.word_embedding.
    And return masks including: image_input_mask, image_output_mask, etc
    For example:
        input_ids:                  torch.tensor([-200, -200,   2,  -200,   -200,   4,  5,  6,  -201,   -201])
        Returns:
            input_ids:              torch.tensor([0,    0,      2,  0,      0,      4,  5,  6,  0,      0   ])
            image_input_mask:       torch.tensor([1,    1,      0,  1,      1,      0,  0,  0,  0,      0   ])
            image_output_mask:      torch.tensor([0,    0,      0,  0,      0,      0,  0,  0,  1,      1   ])

    Args:
        input_ids (torch.Tensor)

    Returns:
        input_ids (torch.Tensor)
        mask_dict (Dict) : {modal}_[input/output]_mask.
    """
    mask_dict = {}
    for data_type in modality_info.keys():
        for modal in modality_info[data_type]:
            mask = input_ids == TYPE2INDEX[data_type][modal]
            mask_dict[f"{modal}_{data_type}_mask"] = mask
            input_ids = torch.where(mask, 0, input_ids)
    return input_ids, mask_dict


def generate_multimodal_output_mask(conversations: List[List[Tuple[str, Any]]]):
    """Generate multimodal output mask based on the input conversation. The multimodal data in
    user message is input data, while in assistant message is output data.
    For example:
        Conversations: [
            ["user", ("image", None), ("text", "Flip the input image.")],
            ["assistant", ("image", None)],
        ]
    Output Mask:
        {
            "image": torch.tensor([0, 1]).type(torch.bool)
        }
    """
    mask = defaultdict(list)
    for conversation in conversations:
        role = conversation[0]
        for message in conversation[1:]:
            if message[0] != "text":
                mask[message[0]].append(role == "assistant")
    mask = {key: torch.tensor(value).type(torch.bool) for key, value in mask.items()}
    return mask


def smart_resize(image: Image.Image, max_pixel_size: int = None, scale_factor: int = None):
    if max_pixel_size is not None:
        w, h = image.size
        scale = max_pixel_size / max(w, h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)))
    if scale_factor is not None:
        w, h = image.size
        w_bar = round(w / scale_factor) * scale_factor
        h_bar = round(h / scale_factor) * scale_factor
        image = image.resize((w_bar, h_bar))
    return image


def get_token_num_inputs(modality_input: Dict, multimodal_output_mask: Dict):
    token_num_inputs = {}
    for modal, mm_mask in multimodal_output_mask.items():
        input_token_nums = modality_input.pop(f"{modal}_input_num_tokens", torch.tensor([])).type(torch.int32)
        output_token_nums = modality_input.pop(f"{modal}_output_num_tokens", torch.tensor([])).type(torch.int32)
        token_nums = torch.zeros_like(mm_mask).type(torch.int32)
        token_nums = token_nums.masked_scatter(mm_mask, output_token_nums)
        token_nums = token_nums.masked_scatter(~mm_mask, input_token_nums)
        token_num_inputs[modal] = token_nums
    return token_num_inputs


def encode_multimodal_sample(
    sample: Dict[str, Any],
    processor: "SeedOmniProcessor",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    modality_info: Dict,
    use_special_rope=False,
    **kwargs,
) -> Dict[str, List[int]]:
    model_inputs = {}
    modality = set(modality_info["input"] + modality_info["output"])
    conversations = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data
    if isinstance(conversations, bytes):
        conversations = json.loads(conversations.decode("utf-8"))
    source = kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    conversations = conv_preprocess(source, conversations, **kwargs)
    multimodal_output_mask = generate_multimodal_output_mask(conversations)
    proceesor_input = {}

    if "image" in modality:
        if "images" in sample:
            images = sample["images"]
            images = [Image.open(BytesIO(image)).convert("RGB") for image in images]
            max_image_nums = kwargs.get("max_image_nums", len(images))
            images = images[:max_image_nums]
            max_pixel_size = kwargs.get("max_pixel_size", None)
            scale_factor = kwargs.get("scale_factor", None)
            images = [smart_resize(image, max_pixel_size, scale_factor) for image in images]

            image_mask = multimodal_output_mask.get("image", torch.tensor([]))
            image_mask_size = image_mask.shape[0]
            images = images[:image_mask_size]

            proceesor_input.update(
                {
                    "input_images": [img for img, mask in zip(images, image_mask) if not mask],
                    "output_images": [img for img, mask in zip(images, image_mask) if mask],
                }
            )
    if "video" in modality:
        raise NotImplementedError
    if "audio" in modality:
        raise NotImplementedError

    modality_input = processor(return_tensors="pt", **proceesor_input)
    token_num_inputs = get_token_num_inputs(modality_input, multimodal_output_mask)
    text_inputs = chat_template.encode_messages(conversations, copy.deepcopy(token_num_inputs))

    model_inputs.update(modality_input)
    model_inputs.update(text_inputs)

    # position_ids (dim, len)
    if position_id_func is None:  # default position_ids
        position_ids = torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)
    else:  # customized position_ids
        position_id_kwargs = {}
        input_ids = text_inputs["input_ids"].clone()
        if "image" in multimodal_output_mask:
            if use_special_rope:  # 2d rope for qwen2vl
                input_ids = mask_before_position_id_func(input_ids)

                input_image_grid_thw = model_inputs.get("image_input_grid_thw", torch.empty(0, 3, dtype=torch.int32))
                output_image_grid_thw = model_inputs.get("image_output_grid_thw", torch.empty(0, 3, dtype=torch.int32))
                image_num = input_image_grid_thw.shape[0] + output_image_grid_thw.shape[0]
                image_grid_thw = torch.zeros((image_num, 3), dtype=torch.int32)
                mask = multimodal_output_mask["image"].unsqueeze(-1).expand_as(image_grid_thw)
                image_grid_thw = image_grid_thw.masked_scatter(mask, output_image_grid_thw)
                image_grid_thw = image_grid_thw.masked_scatter(~mask, input_image_grid_thw)
                position_id_kwargs["image_grid_thw"] = image_grid_thw
            else:
                image_grid_thw = model_inputs.get("image_input_grid_thw", torch.empty(0, 3, dtype=torch.int32))
                position_id_kwargs["image_grid_thw"] = image_grid_thw

        position_ids = position_id_func(input_ids=input_ids.unsqueeze(0), **position_id_kwargs)["position_ids"]
        position_ids = position_ids.view(-1, input_ids.shape[-1])

    model_inputs["position_ids"] = position_ids

    input_ids, mask_dict = mask_input_ids(modality_info, model_inputs["input_ids"])
    model_inputs["input_ids"] = input_ids
    model_inputs.update(mask_dict)
    return [model_inputs]
