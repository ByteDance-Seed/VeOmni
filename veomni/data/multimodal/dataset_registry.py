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
Dataset Registry Extension for VeOmni

This module provides a plugin-style dataset registration system that allows
users to register custom dataset preprocessors without modifying preprocess.py.

Usage:
    1. Register custom datasets using the @register_dataset decorator:

        from veomni.data.multimodal.dataset_registry import register_dataset

        @register_dataset("my_custom_dataset")
        def my_custom_preprocess(conversations, **kwargs):
            # Your preprocessing logic
            return constructed_conversation

    2. Call register_all_datasets() before creating your dataset/dataloader:

        from veomni.data.multimodal.dataset_registry import register_all_datasets

        # This adds custom datasets to DATASETS dict
        register_all_datasets()

        # Now you can use "my_custom_dataset" in your config
"""

from typing import Callable, Dict


# Registry for custom dataset preprocessors
_CUSTOM_DATASETS: Dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    """
    Decorator to register a custom dataset preprocessor.

    Args:
        name: Dataset name (used as 'source_name' in your data config)

    Returns:
        Decorator function

    Example:
        @register_dataset("internal_vqa")
        def internal_vqa_preprocess(conversations, **kwargs):
            constructed_conversation = []
            # ... your preprocessing logic
            return constructed_conversation
    """

    def decorator(func: Callable) -> Callable:
        if name in _CUSTOM_DATASETS:
            raise ValueError(f"Dataset '{name}' is already registered")
        _CUSTOM_DATASETS[name] = func
        return func

    return decorator


def get_custom_datasets() -> Dict[str, Callable]:
    """
    Get all registered custom datasets.

    Returns:
        Dictionary mapping dataset names to preprocessor functions
    """
    return _CUSTOM_DATASETS.copy()


def register_all_datasets() -> None:
    """
    Register all custom datasets with VeOmni's DATASETS dict.

    This function patches veomni.data.multimodal.preprocess.DATASETS
    with all custom datasets registered via @register_dataset decorator.

    Should be called ONCE before creating datasets/dataloaders.

    Example:
        from veomni.data.multimodal.dataset_registry import register_all_datasets
        from veomni.data.multimodal.multimodal_transform import encode_multimodal_sample

        # Register custom datasets
        register_all_datasets()

        # Now VeOmni can use your custom datasets
        dataset = MultimodalDataset(...)
    """
    from .preprocess import DATASETS

    # Extend DATASETS with custom datasets
    registered_count = 0
    for name, func in _CUSTOM_DATASETS.items():
        if name in DATASETS:
            print(f"Warning: Overriding existing dataset '{name}' in VeOmni's DATASETS")
        DATASETS[name] = func
        registered_count += 1

    if registered_count > 0:
        print(f"✓ Registered {registered_count} custom dataset(s) with VeOmni")


# Example custom datasets for reference
@register_dataset("example_simple_caption")
def example_simple_caption_preprocess(conversations, **kwargs):
    """
    Example: Simple image captioning dataset

    Expected input format:
        conversations = [
            {"from": "human", "value": "<image>"},
            {"from": "gpt", "value": "A cat sitting on a table."}
        ]
    """
    assert len(conversations) == 2
    assert conversations[0]["from"] == "human"
    assert conversations[1]["from"] == "gpt"

    caption = conversations[1]["value"]
    constructed_conversation = [
        ["user", ("image", None), ("text", "Describe this image.")],
        ["assistant", ("text", caption)],
    ]
    return constructed_conversation


@register_dataset("example_multi_turn")
def example_multi_turn_preprocess(conversations, **kwargs):
    """
    Example: Multi-turn conversation dataset

    Expected input format:
        conversations = [
            {"from": "human", "value": "<image> What is in the image?"},
            {"from": "gpt", "value": "A dog."},
            {"from": "human", "value": "What color is it?"},
            {"from": "gpt", "value": "Brown."}
        ]
    """
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    for message in conversations:
        role = role_mapping.get(message["from"])
        if role is None:
            raise ValueError(f"Unknown role: {message['from']}")

        value = message["value"]
        cur_message = [role]

        if "<image>" in value:
            value = value.replace("<image>", "").strip()
            cur_message.append(("image", None))
            if value:
                cur_message.append(("text", value))
        else:
            cur_message.append(("text", value))

        constructed_conversation.append(cur_message)

    return constructed_conversation
