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
Dataset Preprocessor Registry for VeOmni

This module provides a registration system for dataset-specific preprocessor functions.

Terminology:
- Dataset: Classes that handle data loading (MultiSourceDataset, MappingDataset, etc.)
- Preprocessor: Functions that convert raw data samples into model-ready format

This registry manages preprocessor functions, not dataset classes.

Usage:
    1. Register preprocessors using the @register_preprocessor decorator:

        from veomni.data.multimodal.preprocessor_registry import register_preprocessor

        @register_preprocessor("my_custom_dataset")
        def my_custom_preprocess(conversations, **kwargs):
            # Your preprocessing logic
            return constructed_conversation

    2. The registry automatically collects all preprocessors from:
       - Built-in preprocessors in preprocess.py (via @register_preprocessor decorators)
       - User-defined preprocessors anywhere using @register_preprocessor

    3. Use get_preprocessor() to access registered preprocessors:

        from veomni.data.multimodal.preprocessor_registry import get_preprocessor

        # Get the registered preprocessor for "my_custom_dataset"
        preprocessor = get_preprocessor("my_custom_dataset")
        result = preprocessor(conversations)
"""

from typing import Callable, Dict


# Global registry for all dataset preprocessors
_PREPROCESSOR_REGISTRY: Dict[str, Callable] = {}


def register_preprocessor(name: str) -> Callable:
    """
    Decorator to register a dataset-specific preprocessor function.

    Args:
        name: Dataset source name (used as 'source_name' in your data config)

    Returns:
        Decorator function

    Example:
        @register_preprocessor("internal_vqa")
        def internal_vqa_preprocess(conversations, **kwargs):
            constructed_conversation = []
            # ... your preprocessing logic
            return constructed_conversation

        # Can register multiple names for the same function:
        @register_preprocessor("sharegpt4v_pretrain")
        @register_preprocessor("sharegpt4v_captioner")
        def sharegpt4v_pretrain_preprocess(conversations, **kwargs):
            # ... preprocessing logic
            return constructed_conversation
    """

    def decorator(func: Callable) -> Callable:
        if name in _PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Preprocessor for '{name}' is already registered. Cannot register duplicate preprocessor."
            )
        _PREPROCESSOR_REGISTRY[name] = func
        return func

    return decorator


def get_all_preprocessors() -> Dict[str, Callable]:
    """
    Get all registered preprocessors.

    This function ensures that all built-in preprocessors from preprocess.py
    are loaded into the registry.

    Returns:
        Dictionary mapping dataset source names to preprocessor functions
    """
    # Import to trigger registration of all built-in preprocessors
    # This import causes all @register_preprocessor decorators to execute
    from . import preprocess  # noqa: F401

    return _PREPROCESSOR_REGISTRY.copy()


def get_preprocessor(name: str) -> Callable:
    """
    Get a specific preprocessor by dataset source name.

    Args:
        name: Dataset source name

    Returns:
        Preprocessor function

    Raises:
        ValueError: If preprocessor for the dataset name is not registered
    """
    # Ensure all preprocessors are loaded
    get_all_preprocessors()

    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(f"Unknown dataset name: {name}. No preprocessor registered for this source.")

    return _PREPROCESSOR_REGISTRY[name]


def list_preprocessors() -> list:
    """
    List all registered dataset source names.

    Returns:
        List of dataset source names (sorted)
    """
    get_all_preprocessors()
    return sorted(_PREPROCESSOR_REGISTRY.keys())


def is_preprocessor_registered(name: str) -> bool:
    """
    Check if a preprocessor is registered for the given dataset source name.

    Args:
        name: Dataset source name

    Returns:
        True if registered, False otherwise
    """
    get_all_preprocessors()
    return name in _PREPROCESSOR_REGISTRY


def conv_preprocess(source: str, conversations, **kwargs):
    """
    Convenience function to preprocess conversations for a specific dataset.

    This is a wrapper around the dataset registry that provides backward compatibility
    with the original API.

    Args:
        source: Dataset name (e.g., "sharegpt4v_pretrain", "doom", etc.)
        conversations: Conversation data in the format expected by the dataset preprocessor
        **kwargs: Additional arguments passed to the preprocessor

    Returns:
        Preprocessed conversation in the format expected by the model

    Raises:
        ValueError: If dataset name is not registered

    Example:
        >>> conversations = [
        ...     {"from": "human", "value": "<image> What is in the image?"},
        ...     {"from": "gpt", "value": "A cat"}
        ... ]
        >>> result = conv_preprocess("sharegpt4v_sft", conversations)
    """
    preprocessor = get_preprocessor(source)
    return preprocessor(conversations, **kwargs)
