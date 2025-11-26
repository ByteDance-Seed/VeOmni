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

"""Tests for the dataset preprocessor registry system"""

import pytest


def test_preprocessor_registration():
    """Test that custom preprocessors can be registered"""
    from veomni.data.multimodal.preprocessor_registry import get_all_preprocessors, register_preprocessor

    # Define a test preprocessor
    @register_preprocessor("test_preprocessor_new")
    def test_preprocess(conversations, **kwargs):
        return [["user", ("text", "test")]]

    preprocessors = get_all_preprocessors()
    assert "test_preprocessor_new" in preprocessors
    assert preprocessors["test_preprocessor_new"] == test_preprocess


def test_builtin_preprocessors_registered():
    """Test that built-in preprocessors from preprocess.py are automatically registered"""
    from veomni.data.multimodal.preprocessor_registry import get_all_preprocessors

    preprocessors = get_all_preprocessors()

    # Check that some built-in preprocessors are present
    assert "sharegpt4v_pretrain" in preprocessors
    assert "sharegpt4v_sft" in preprocessors
    assert "doom" in preprocessors
    assert "seed_edit" in preprocessors
    assert "imagenet1k" in preprocessors
    assert "fineweb_100BT" in preprocessors


def test_custom_preprocessors_registered():
    """Test that custom preprocessors from preprocess.py are registered"""
    from veomni.data.multimodal.preprocessor_registry import get_all_preprocessors

    preprocessors = get_all_preprocessors()

    # Check custom preprocessors from preprocess.py
    assert "internal_vqa" in preprocessors


def test_list_preprocessors():
    """Test listing all registered preprocessors"""
    from veomni.data.multimodal.preprocessor_registry import list_preprocessors

    preprocessor_names = list_preprocessors()
    assert isinstance(preprocessor_names, list)
    assert len(preprocessor_names) > 0
    assert "sharegpt4v_pretrain" in preprocessor_names
    assert "doom" in preprocessor_names


def test_is_preprocessor_registered():
    """Test checking if a preprocessor is registered"""
    from veomni.data.multimodal.preprocessor_registry import is_preprocessor_registered

    assert is_preprocessor_registered("sharegpt4v_pretrain") is True
    assert is_preprocessor_registered("nonexistent_preprocessor_xyz") is False


def test_get_preprocessor():
    """Test getting a preprocessor by name"""
    from veomni.data.multimodal.preprocessor_registry import get_preprocessor

    # Get a built-in preprocessor
    preprocessor = get_preprocessor("sharegpt4v_pretrain")
    assert callable(preprocessor)

    # Test with unknown preprocessor - should raise ValueError
    with pytest.raises(ValueError, match="Unknown dataset name"):
        get_preprocessor("nonexistent_preprocessor_xyz")


def test_custom_preprocessor_execution():
    """Test that custom preprocessors work correctly"""
    from veomni.data.multimodal.preprocessor_registry import get_preprocessor

    # Test internal_vqa from preprocess.py
    test_conversations = [
        {"from": "human", "value": "<image> What is shown in the image?"},
        {"from": "gpt", "value": "A detailed answer..."},
    ]
    preprocessor = get_preprocessor("internal_vqa")
    result = preprocessor(test_conversations)
    assert isinstance(result, list)
    assert len(result) == 2


def test_multiple_names_same_function():
    """Test that the same preprocessor can be registered under multiple names"""
    from veomni.data.multimodal.preprocessor_registry import get_all_preprocessors

    preprocessors = get_all_preprocessors()

    # sharegpt4v_pretrain_preprocess is registered as both "sharegpt4v_pretrain" and "sharegpt4v_captioner"
    assert "sharegpt4v_pretrain" in preprocessors
    assert "sharegpt4v_captioner" in preprocessors
    assert preprocessors["sharegpt4v_pretrain"] == preprocessors["sharegpt4v_captioner"]


def test_duplicate_registration_warning():
    """Test that duplicate registration issues a warning"""
    import warnings

    from veomni.data.multimodal.preprocessor_registry import register_preprocessor

    @register_preprocessor("test_dup_preprocessor_1")
    def test_preprocess1(conversations, **kwargs):
        return [["first"]]

    # Second registration should warn, not error
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @register_preprocessor("test_dup_preprocessor_1")
        def test_preprocess2(conversations, **kwargs):
            return [["second"]]

        assert len(w) == 1
        assert "already registered" in str(w[0].message)


def test_preprocessor_execution():
    """Test that preprocessors execute correctly"""
    from veomni.data.multimodal.preprocessor_registry import get_preprocessor

    # Test with a built-in preprocessor
    test_conversations = [
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "A beautiful sunset."},
    ]
    preprocessor = get_preprocessor("sharegpt4v_pretrain")
    result = preprocessor(test_conversations)
    assert isinstance(result, list)
    assert len(result) > 0


def test_conv_preprocess():
    """Test that conv_preprocess convenience function works"""
    from veomni.data.multimodal.preprocess import conv_preprocess

    # Test with a built-in preprocessor
    test_conversations = [
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "A beautiful sunset."},
    ]
    result = conv_preprocess("sharegpt4v_pretrain", test_conversations)
    assert isinstance(result, list)
    assert len(result) > 0

    # Test that it raises ValueError for unknown preprocessor
    with pytest.raises(ValueError, match="Unknown dataset name"):
        conv_preprocess("nonexistent_preprocessor_xyz", test_conversations)


if __name__ == "__main__":
    # Run tests
    test_preprocessor_registration()
    test_builtin_preprocessors_registered()
    test_custom_preprocessors_registered()
    test_list_preprocessors()
    test_is_preprocessor_registered()
    test_get_preprocessor()
    test_custom_preprocessor_execution()
    test_multiple_names_same_function()
    test_duplicate_registration_warning()
    test_preprocessor_execution()
    test_conv_preprocess()
    print("✓ All tests passed!")
