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
Custom Dataset Preprocessors

Add your custom dataset preprocessors here using the @register_dataset decorator.
This allows you to extend VeOmni's dataset support without modifying preprocess.py.
"""

from .dataset_registry import register_dataset


# Example 1: Internal VQA dataset
@register_dataset("internal_vqa")
def internal_vqa_preprocess(conversations, **kwargs):
    """
    Preprocessor for internal VQA dataset

    Expected format:
        conversations = [
            {"from": "human", "value": "<image> What is shown in the image?"},
            {"from": "gpt", "value": "A detailed answer..."}
        ]
    """
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []

    if not conversations or conversations[0]["from"] != "human":
        conversations = conversations[1:]

    for message in conversations:
        role = role_mapping[message["from"]]
        value = message["value"]

        if "<image>" in value:
            value = value.replace("<image>", "").strip()
            constructed_conversation.append([role, ("image", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])

    return constructed_conversation


# # Example 2: Custom OCR dataset
# @register_dataset("custom_ocr")
# def custom_ocr_preprocess(conversations, **kwargs):
#     """
#     Preprocessor for OCR dataset

#     Expected format:
#         conversations = "Extracted text from the image"
#     """
#     if isinstance(conversations, str):
#         text = conversations
#     else:
#         # If it's a list, take the last assistant message
#         text = conversations[-1]["value"]

#     constructed_conversation = [
#         ["user", ("image", None), ("text", "Extract all text from this image.")],
#         ["assistant", ("text", text)],
#     ]
#     return constructed_conversation


# # Example 3: Custom chart understanding dataset
# @register_dataset("custom_chart_understanding")
# def custom_chart_understanding_preprocess(conversations, **kwargs):
#     """
#     Preprocessor for chart understanding tasks

#     Expected format:
#         conversations = [
#             {"from": "human", "value": "<image> Analyze this chart"},
#             {"from": "gpt", "value": "This chart shows..."}
#         ]
#     """
#     role_mapping = {"human": "user", "gpt": "assistant"}
#     constructed_conversation = []

#     for message in conversations:
#         role = role_mapping[message["from"]]
#         value = message["value"]

#         if "<image>" in value:
#             value = value.replace("<image>", "").strip()
#             if value:
#                 constructed_conversation.append([role, ("image", None), ("text", value)])
#             else:
#                 constructed_conversation.append([role, ("image", None)])
#         else:
#             constructed_conversation.append([role, ("text", value)])

#     return constructed_conversation


# Add more custom datasets below as needed
# @register_dataset("your_dataset_name")
# def your_dataset_preprocess(conversations, **kwargs):
#     ...
