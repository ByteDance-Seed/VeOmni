from typing import TYPE_CHECKING, Any, Callable

import torch

from veomni.data.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.multimodal.audio_utils import fetch_audios
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.data.multimodal.video_utils import fetch_videos


if TYPE_CHECKING:
    from transformers import ProcessorMixin


SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def process_sample_qwen_omni(
    sample: dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example with qwen2_5_vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
    )  # source_name if use multisource_dataset
    conversations = (
        sample["conversations"] if ("conversations" in sample and len(sample["conversations"]) > 0) else sample
    )
    conversations = conv_preprocess(source, conversations, **kwargs)
    input_conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_MESSAGE,
                },
            ],
        },
    ]
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        tmp_conv = {
            "role": conversation[0],
            "content": contents,
        }
        input_conversations.append(tmp_conv)
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    images = sample.get("images", [])
    if images:
        images = fetch_images(images, **kwargs)
    else:
        images = []

    videos = sample.get("videos", [])
    if videos:
        videos, video_audios = fetch_videos(videos, **kwargs)
    else:
        videos, video_audios = [], []

    audios = sample.get("audios", [])
    if audios:
        audio_audios = fetch_audios(audios, **kwargs)
    else:
        audio_audios = []

    audios = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(video_audios.pop(0))
            elif content["type"] == "audio":
                audios.append(audio_audios.pop(0))

    model_inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data  # batch_feature to dict
    # process audio inputs:
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0  # filter videos without audios
        input_features = input_features[valid_mask].permute(0, 2, 1)[
            feature_attention_mask[valid_mask].bool()
        ]  # l, dim

    if feature_attention_mask is None or input_features.shape[0] == 0:
        audio_feature_lengths = None
    else:
        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths

    input_ids = model_inputs["input_ids"].squeeze(0)
    image_mask = input_ids == 151655
    video_mask = input_ids == 151656
    audio_mask = input_ids == 151646
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    model_inputs["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
        second_per_grids=model_inputs.pop("video_second_per_grid", None),
    )["position_ids"]  # (dim, l)

    model_inputs["position_ids"] = model_inputs["position_ids"].clone()
    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    user_start_index = torch.where(input_ids == 872)[0].tolist()  # "user" 872
    assistant_start_index = torch.where(input_ids == 77091)[0].tolist()  # "assistant" 77091
    user_start_index.append(len(input_ids) + 1)
    user_i = 0
    for assis_i in assistant_start_index:
        while user_start_index[user_i] < assis_i:
            user_i += 1
        labels[assis_i + 2 : user_start_index[user_i] - 1] = input_ids[assis_i + 2 : user_start_index[user_i] - 1]
    model_inputs["labels"] = labels
    return [model_inputs]
