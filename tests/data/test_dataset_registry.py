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

"""Tests for the custom dataset registry system"""

import pytest


def test_dataset_registration():
    """Test that custom datasets can be registered"""
    from veomni.data.multimodal.dataset_registry import get_custom_datasets, register_dataset

    # Define a test dataset
    @register_dataset("test_dataset")
    def test_preprocess(conversations, **kwargs):
        return [["user", ("text", "test")]]

    datasets = get_custom_datasets()
    assert "test_dataset" in datasets
    assert datasets["test_dataset"] == test_preprocess


def test_register_all_datasets():
    """Test that registered datasets are added to VeOmni's DATASETS dict"""
    from veomni.data.multimodal.dataset_registry import get_custom_datasets, register_all_datasets

    # First verify datasets are in the custom registry
    custom_datasets = get_custom_datasets()
    assert "example_simple_caption" in custom_datasets
    assert "example_multi_turn" in custom_datasets
    assert "internal_vqa" in custom_datasets
    assert "custom_ocr" in custom_datasets
    assert "custom_chart_understanding" in custom_datasets

    # Register with VeOmni
    register_all_datasets()
    from veomni.data.multimodal.preprocess import DATASETS

    # Check example datasets are registered
    assert "example_simple_caption" in DATASETS
    assert "example_multi_turn" in DATASETS

    # Check custom datasets are registered
    assert "internal_vqa" in DATASETS
    assert "custom_ocr" in DATASETS
    assert "custom_chart_understanding" in DATASETS


def test_custom_dataset_preprocess():
    """Test that custom dataset preprocessors work correctly"""
    from veomni.data.multimodal.dataset_registry import get_custom_datasets

    # Get custom datasets directly from registry
    custom_datasets = get_custom_datasets()

    # Test example_simple_caption
    test_conversations = [
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "A beautiful sunset over the ocean."},
    ]
    result = custom_datasets["example_simple_caption"](test_conversations)
    assert result == [
        ["user", ("image", None), ("text", "Describe this image.")],
        ["assistant", ("text", "A beautiful sunset over the ocean.")],
    ]

    # Test example_multi_turn
    test_conversations = [
        {"from": "human", "value": "<image> What is in the image?"},
        {"from": "gpt", "value": "A dog."},
        {"from": "human", "value": "What color is it?"},
        {"from": "gpt", "value": "Brown."},
    ]
    result = custom_datasets["example_multi_turn"](test_conversations)
    expected = [
        ["user", ("image", None), ("text", "What is in the image?")],
        ["assistant", ("text", "A dog.")],
        ["user", ("text", "What color is it?")],
        ["assistant", ("text", "Brown.")],
    ]
    assert result == expected


def test_duplicate_registration():
    """Test that duplicate registration raises an error"""
    from veomni.data.multimodal.dataset_registry import register_dataset

    with pytest.raises(ValueError, match="already registered"):

        @register_dataset("test_dup_dataset")
        def test_preprocess1(conversations, **kwargs):
            return []

        @register_dataset("test_dup_dataset")  # Duplicate name
        def test_preprocess2(conversations, **kwargs):
            return []


if __name__ == "__main__":
    # Run tests
    test_dataset_registration()
    test_register_all_datasets()
    test_custom_dataset_preprocess()
    print("✓ All tests passed!")
