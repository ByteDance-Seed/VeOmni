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

"""
Dataset Preprocessors

This module contains both built-in and custom dataset preprocessors.
All preprocessors are registered using the @register_preprocessor decorator.

To add custom preprocessors, simply define a function and decorate it with @register_preprocessor.
"""

import random
import re

from ...utils.registry import Registry


PREPROCESSOR_REGISTRY = Registry("preprocessor")


def conv_preprocess(source: str, conversations, **kwargs):
    return PREPROCESSOR_REGISTRY[source](conversations, **kwargs)


# ============================================================================
# Built-in Dataset Preprocessors
# ============================================================================


@PREPROCESSOR_REGISTRY.register("sharegpt4v_pretrain")
@PREPROCESSOR_REGISTRY.register("sharegpt4v_captioner")
def sharegpt4v_pretrain_preprocess(conversations, generation_ratio=0.0, **kwargs):
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        role = message["from"]
        value = message["value"]
        if role == "human":
            value = value.replace("<image>", "")
            constructed_conversation.append(["user", ("image", None)])
        else:
            constructed_conversation.append(["assistant", ("text", value)])
    generate_sample = random.random() < generation_ratio
    if generate_sample:
        instruction = f"Generate an image based on the following caption: {constructed_conversation[-1][0][1]}"
        constructed_conversation = [["user", ("text", instruction)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sharegpt4v_captioner_sft")
@PREPROCESSOR_REGISTRY.register("sharegpt4v_sft")
def sharegpt4v_sft_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value:
            value = value.replace("<image>", "")
            constructed_conversation.append([role, ("image", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("doom")
def doom_preprocess(conversations, max_image_nums=None, **kwargs):
    """
    merge the assistant output in a single message
    """
    constructed_conversation = []
    image_count = 0
    role_mapping = {"human": "user", "gpt": "assistant"}
    prev_conversation = []
    prev_role = "user"
    for i, message in enumerate(conversations):
        role = role_mapping[message["from"]]
        value = message["value"]
        if i == 0:
            value = value.strip()
        if value == "<image>":
            cur_message = [("image", None)]
            image_count += 1
        else:
            cur_message = [("text", value)]
        if role == prev_role == "assistant":
            cur_message = [("text", "\n\n")] + cur_message
            prev_conversation += cur_message
        elif role == prev_role:
            prev_conversation += cur_message
        else:
            constructed_conversation.append([prev_role] + prev_conversation)
            prev_role = role
            prev_conversation = cur_message
        if max_image_nums is not None and image_count >= max_image_nums:
            break
    if len(prev_conversation) != 0:
        constructed_conversation.append([prev_role] + prev_conversation)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("seed_edit")
def seed_edit_preprocess(conversations, **kwargs):
    constructed_conversation = []
    for message in conversations:
        value = message["value"]
        parts = value.split("<image>")
        if parts == ["", ""]:  # "<image>"
            cur_message = ["assistant", ("image", None)]
        else:
            cur_message = ["user"]
            for part in parts:
                if part == "":
                    cur_message += [("image", None)]
                else:
                    cur_message += [("text", part), ("image", None)]
            cur_message = cur_message[:-1]
        constructed_conversation.append(cur_message)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("imagenet1k")
def imagenet1k_preprocess(conversations, **kwargs):
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed_conversation = [
        ["user", ("text", class_label)],
        ["assistant", ("image", None)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("imagenet1k_caption")
def imagenet1k_caption_preprocess(conversations, **kwargs):
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image.")],
        ["assistant", ("text", class_label)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("fineweb_100BT")
def fineweb_preprocess(conversations, **kwargs):
    constructed_conversation = [
        ["assistant", ("text", conversations)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("wikihow_ct_0904")
def wikihow_preprocess(conversations, stage="pretrain", **kwargs):
    constructed_conversation = []
    role_mapping = {"human": "user", "gpt": "assistant"}
    for conv in conversations:
        role = role_mapping[conv["from"]]
        value = conv["value"]
        cur_message = [role]
        if "<image>" in value:
            value = value.replace("<image>", "").strip()
            cur_message.append(("image", None))
            if value != "":
                cur_message.append(("text", value))
        else:
            cur_message.append(("text", value))
        constructed_conversation.append(cur_message)
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("Detailed_Caption")
def detailed_caption_preprocess(conversations, **kwargs):
    constructed_conversation = []
    assert conversations[-1]["from"] == "gpt"
    caption = conversations[-1]["value"][8:].strip()  # skip Answer:
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("ArxivQA")
def arxivqa_preprocess(conversations, **kwargs):
    question = conversations[0]["value"].replace("<image>\n", "").strip()
    answer = conversations[1]["value"].strip()
    constructed_conversation = [["user", ("image", None), ("text", question)], ["assistant", ("text", answer)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("pixelprose")
def pixelprose_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("DenseFusion-1M")
@PREPROCESSOR_REGISTRY.register("DenseFusion-4V-100k")
def densefusion_preprocess(conversations, **kwargs):
    caption = conversations[0]["value"]
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sam")
def sam_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("sam_gen")
def sam_gen_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("pixelprose_gen")
def pixelprose_gen_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("chart_to_table")
def chart_to_table_preprocess(conversations, **kwargs):
    caption = conversations
    constructed_conversation = [
        ["user", ("image", None), ("text", "Convert the image to a table.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("CHartQA")
def chartqa_preprocess(conversations, **kwargs):
    question = conversations[0]["value"].replace("<image>\n", "").strip()
    answer = conversations[1]["value"].strip()
    constructed_conversation = [["user", ("image", None), ("text", question)], ["assistant", ("text", answer)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("megalith")
def megalith_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("journeydb")
def journeydb_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("dalle3_1m")
def dalle3_1m_preprocess(conversations, short_description_ratio=0.25, **kwargs):
    caption = conversations
    if random.random() < short_description_ratio:
        caption = caption.split(".")[0]
    constructed_conversation = [["user", ("text", caption)], ["assistant", ("image", None)]]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("wit")
def wit_preprocess(conversations, **kwargs):
    text_content_1, text_content_2, text_content_3 = "", "", ""
    if conversations["page_title"]:
        text_content_1 += conversations["page_title"] + "\n"
    if conversations["context_page_description"]:
        text_content_2 += conversations["context_page_description"] + "\n"
    if conversations["caption_reference_description"]:
        text_content_3 += conversations["caption_reference_description"]

    constructed_conversation = [
        ["user", ("text", text_content_1)],
        ["assistant", ("text", text_content_2)],
        ["user", ("image", None)],
        ["assistant", ("text", text_content_3)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("mmsci")
def mmsci_preprocess(conversations, **kwargs):
    caption = conversations[0]["value"]

    def replace_figure_number(text):
        return re.sub(r"^(Figure|Fig\.) \d+[:]*", "", text)

    caption = replace_figure_number(caption).strip()
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe the image in detail.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("LLaVA-Video-178K")
def llava_video_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value:
            value = value.replace("<image>\n", "")
            constructed_conversation.append([role, ("video", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@PREPROCESSOR_REGISTRY.register("VoiceAssistant")
def voice_assistant_preprocess(conversations, **kwargs):
    constructed_conversation = [
        ["user", ("audio", None)],
        ["assistant", ("text", conversations[1]["value"])],
    ]
    return constructed_conversation


# @PREPROCESSOR_REGISTRY.register("your_dataset_name")
# def your_dataset_preprocess(conversations, **kwargs):
#     ...
