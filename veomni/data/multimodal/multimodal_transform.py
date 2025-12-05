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


import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch

from ...utils.import_utils import is_video_audio_available
from ..constants import TYPE2INDEX
from .image_utils import fetch_images
from .preprocess import conv_preprocess


if is_video_audio_available():
    from .audio_utils import fetch_audios
    from .video_utils import fetch_videos
else:

    def fetch_videos(*args, **kwargs):
        return [], []

    def fetch_audios(*args, **kwargs):
        return []


if TYPE_CHECKING:
    from ...models.seed_omni import SeedOmniProcessor
    from .multimodal_chat_template import MultimodalChatTemplate


def mask_before_position_id_func(input_ids: torch.Tensor):
    """Mask special multimodal tokens in input_ids to input_mm_token for position_id.
    Only supports special image tokens now. (input_image_id=-200, output_image_id=-201->-200)
    Similar to veomni.module.seed_omni.modeling_seed_omni.mask_before_text_encoder

    Args:
        input_ids (torch.Tensor)

    Returns:
        input_ids (torch.Tensor)
    """
    for modality in ["image", "video", "audio"]:
        output_mask = input_ids == TYPE2INDEX["output"][modality]
        input_mask = input_ids == TYPE2INDEX["input"][modality]
        input_ids = torch.where(output_mask | input_mask, TYPE2INDEX["input"][modality], input_ids)
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


def process_mm_data(
    conversations, images: List[Any], videos: List[Any], video_audios: List[Any], audio_audios: List[Any]
):
    """
    Processes multi-modal conversation data and aligns images, videos, and audio
    with a corresponding output mask indicating whether the data was produced by the assistant.

    Parameters:
    ----------
    conversations : List[List]
    images : List[Any, List of image data in order.
    videos : List[Any], List of video data in order.
    video_audios : List[Any], List of audio tracks corresponding to the videos.
    audio_audios : List[Any], List of standalone audio samples.

    Returns:
    -------
    conv_images : List[Any], List of images in the order they appeared in conversations.
    conv_videos : List[Any], List of videos in the order they appeared in conversations.
    conv_audios : List[Any], List of all audio data, including both video audio and standalone audio.

    mask : Dict[str, torch.BoolTensor]
        A dictionary with modality names as keys ("image", "video", "audio"), and boolean tensors
        indicating whether each sample was produced by the assistant (True) or the user (False).

    Example:
    --------
    Input:
        conversations = [
            ["user", ["video"], ["audio"], ["video"], ["text"]],
            ["assistant", ["audio"]]
        ]
        videos = ["video1", "video2"]
        video_audios = ["v_audio1", "v_audio2"]
        audio_audios = ["audio1", "audio2"]

    Output:
        conv_videos = ["video1", "video2"]
        conv_audios = ["v_audio1", "audio1", "v_audio2", "audio2"]
        mask["video"] = tensor([False, False])              # user videos
        mask["audio"] = tensor([False, False, False, True]) # user+assistant audios
    """
    images, videos, video_audios, audio_audios = iter(images), iter(videos), iter(video_audios), iter(audio_audios)
    conv_images, conv_videos, conv_audios = [], [], []
    mask = defaultdict(list)
    for conversation in conversations:
        role = conversation[0]
        is_output = role == "assistant"
        for message in conversation[1:]:
            data_type = message[0]
            if data_type == "text":
                continue
            elif data_type == "image":
                conv_images.append(next(images))
                mask["image"].append(is_output)
            elif data_type == "video":
                conv_videos.append(next(videos))
                conv_audios.append(next(video_audios))
                mask["video"].append(is_output)
                mask["audio"].append(is_output)
            elif data_type == "audio":
                conv_audios.append(next(audio_audios))
                mask["audio"].append(is_output)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
    mask = {key: torch.tensor(value).type(torch.bool) for key, value in mask.items()}
    return conv_images, conv_videos, conv_audios, mask


def get_multimodal_configs(modality_input: Dict, multimodal_output_mask: Dict):
    multimodal_configs, config_repr = {}, {}
    for key in modality_input.keys():
        config_key = key.split("_", 2)[-1]
        if config_key != "features":
            config_repr[config_key] = modality_input[key]
    for config_key, repr in config_repr.items():
        multimodal_configs[config_key] = {}
        for modal, mm_mask in multimodal_output_mask.items():
            if (
                f"{modal}_input_{config_key}" not in modality_input
                and f"{modal}_output_{config_key}" not in modality_input
            ):
                continue
            input_config = modality_input.get(f"{modal}_input_{config_key}", torch.empty_like(repr))
            output_config = modality_input.get(f"{modal}_output_{config_key}", torch.empty_like(repr))

            config = torch.zeros_like(repr)
            config = config.repeat_interleave(mm_mask.shape[0], dim=0)

            config[mm_mask] = output_config
            config[~mm_mask] = input_config

            multimodal_configs[config_key][modal] = config
    return multimodal_configs


def keep_input_only(multimodal_config: Dict, multimodal_output_mask: Dict):
    """Only keep the input data in multimodal_config. Used when use_special_rope=False.
    When use_special_rope=False, only do special_rope on input_multimodal_data.
    For example: 2d_rope on input_image_token, but 1d_rope on output_image_token.
    """
    for config in multimodal_config.keys():
        for modal in multimodal_config[config].keys():
            multimodal_config[config][modal] = multimodal_config[config][modal][~multimodal_output_mask[modal]]


def encode_multimodal_sample(
    sample: Dict[str, Any],
    processor: "SeedOmniProcessor",
    chat_template: "MultimodalChatTemplate",
    position_id_func: "Callable",
    modality_info: Dict,
    use_special_rope=False,  # 2d rope position id for image generation
    **kwargs,
) -> Dict[str, List[int]]:
    model_inputs = {}
    source = sample.pop("source_name") if "source_name" in sample else kwargs["source_name"]
    modality = set(modality_info["input"] + modality_info["output"])
    conversations = sample["text"] if source == "fineweb_100BT" else sample["conversations"]  # text-only data
    if isinstance(conversations, bytes):
        conversations = json.loads(conversations.decode("utf-8"))
    conversations = conv_preprocess(source, conversations, **kwargs)
    processor_input = {}

    if "image" in modality:
        images = fetch_images(sample.get("images", []), **kwargs)
    else:
        images = []
    if "video" in modality:
        videos, video_audios = fetch_videos(sample.get("videos", []), **kwargs)
        if "audio" not in modality:
            video_audios = [None] * len(videos)
    else:
        videos, video_audios = [], []
    if "audio" in modality:
        audio_audios = fetch_audios(sample.get("audios", []), **kwargs)
    else:
        audio_audios = []

    images, videos, audios, multimodal_output_mask = process_mm_data(
        conversations, images, videos, video_audios, audio_audios
    )

    if images:
        processor_input.update(
            {
                "input_images": [img for img, mask in zip(images, multimodal_output_mask["image"]) if not mask],
                "output_images": [img for img, mask in zip(images, multimodal_output_mask["image"]) if mask],
            }
        )
    if videos:
        processor_input.update(
            {
                "input_videos": [vid for vid, mask in zip(videos, multimodal_output_mask["video"]) if not mask],
                "output_videos": [img for img, mask in zip(videos, multimodal_output_mask["video"]) if mask],
            }
        )
    if audios and "audio" in modality:
        processor_input.update(
            {
                "input_audios": [aud for aud, mask in zip(audios, multimodal_output_mask["audio"]) if not mask],
                "output_audios": [aud for aud, mask in zip(audios, multimodal_output_mask["audio"]) if mask],
            }
        )

    modality_input = processor(return_tensors="pt", **processor_input)
    multimodal_config = get_multimodal_configs(modality_input, multimodal_output_mask)
    text_inputs = chat_template.encode_messages(conversations, **multimodal_config)
    model_inputs.update(modality_input)
    model_inputs.update(text_inputs)

    # position_ids (dim, len)
    if position_id_func is None:  # default position_ids
        position_ids = torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)
    else:  # customized position_ids
        input_ids = text_inputs["input_ids"].clone()
        attention_mask = text_inputs["attention_mask"].clone()
        if use_special_rope:
            input_ids = mask_before_position_id_func(input_ids)
        else:
            keep_input_only(multimodal_config, multimodal_output_mask)
        position_ids = position_id_func(
            input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), **multimodal_config
        )["position_ids"]
    model_inputs["position_ids"] = position_ids

    input_ids, mask_dict = mask_input_ids(modality_info, model_inputs["input_ids"])
    model_inputs["input_ids"] = input_ids
    model_inputs.update(mask_dict)
    return [model_inputs]


def encode_multimodal_sample_inference(
    sample: Dict[str, Any],
    processor: "SeedOmniProcessor",
    chat_template: "MultimodalChatTemplate",
    position_id_func: "Callable",
    modality_info: Dict,
    force_image_gen: bool,
    **kwargs,
):
    model_inputs = {}
    modality = set(modality_info["input"] + modality_info["output"])
    conversations = sample["conversations"]

    processor_input = {}
    if "image" in modality:
        images = fetch_images(sample.get("images", []), **kwargs)
    else:
        images = []
    if "video" in modality:
        videos, video_audios = fetch_videos(sample.get("videos", []), **kwargs)
        if "audio" not in modality:
            video_audios = [None] * len(videos)
    else:
        videos, video_audios = [], []
    if "audio" in modality:
        audio_audios = fetch_audios(sample.get("audios", []), **kwargs)
    else:
        audio_audios = []

    images, videos, audios, multimodal_output_mask = process_mm_data(
        conversations, images, videos, video_audios, audio_audios
    )

    if images:
        processor_input["input_images"] = images
    if videos:
        processor_input["input_videos"] = videos
    if audios and "audio" in modality:
        processor_input["input_audios"] = audios

    modality_input = processor(return_tensors="pt", **processor_input)
    multimodal_config = get_multimodal_configs(modality_input, multimodal_output_mask)
    text_inputs = chat_template.encode_messages(conversations, **multimodal_config)

    if force_image_gen:
        text_inputs["input_ids"] = torch.cat(
            [text_inputs["input_ids"], torch.tensor([chat_template.image_start_id])],
            dim=-1,
        )
        text_inputs["attention_mask"] = torch.cat([text_inputs["attention_mask"], torch.tensor([1])], dim=-1)

    model_inputs.update(modality_input)
    model_inputs.update(text_inputs)

    # position_ids (dim, len)
    if position_id_func is None:  # default position_ids
        position_id_returns = {"position_ids": torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)}
    else:  # customized position_ids
        input_ids = text_inputs["input_ids"].clone()
        attention_mask = text_inputs["attention_mask"].clone()
        position_id_returns = position_id_func(
            input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), **multimodal_config
        )

    model_inputs.update(position_id_returns)

    input_ids, mask_dict = mask_input_ids(modality_info, model_inputs["input_ids"])
    model_inputs["input_ids"] = input_ids
    model_inputs.update(mask_dict)
    return [model_inputs]
