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
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    # Define a test preprocessor
    @PREPROCESSOR_REGISTRY.register("test_preprocessor_new")
    def test_preprocess(conversations, **kwargs):
        return [["user", ("text", "test")]]

    assert "test_preprocessor_new" in PREPROCESSOR_REGISTRY.valid_keys()
    assert PREPROCESSOR_REGISTRY["test_preprocessor_new"] == test_preprocess


def test_builtin_preprocessors_registered():
    """Test that built-in preprocessors from preprocess.py are automatically registered"""
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    # Check that some built-in preprocessors are present
    assert "sharegpt4v_pretrain" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "sharegpt4v_sft" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "doom" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "seed_edit" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "imagenet1k" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "fineweb_100BT" in PREPROCESSOR_REGISTRY.valid_keys()


def test_get_preprocessor():
    """Test getting a preprocessor by name"""
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    # Get a built-in preprocessor
    preprocessor = PREPROCESSOR_REGISTRY["sharegpt4v_pretrain"]
    assert callable(preprocessor)

    # Test with unknown preprocessor - should raise ValueError
    with pytest.raises(ValueError, match="Unknown preprocessor name: nonexistent_preprocessor_xyz"):
        PREPROCESSOR_REGISTRY["nonexistent_preprocessor_xyz"]


def test_multiple_names_same_function():
    """Test that the same preprocessor can be registered under multiple names"""
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    # sharegpt4v_pretrain_preprocess is registered as both "sharegpt4v_pretrain" and "sharegpt4v_captioner"
    assert "sharegpt4v_pretrain" in PREPROCESSOR_REGISTRY.valid_keys()
    assert "sharegpt4v_captioner" in PREPROCESSOR_REGISTRY.valid_keys()
    assert PREPROCESSOR_REGISTRY["sharegpt4v_pretrain"] == PREPROCESSOR_REGISTRY["sharegpt4v_captioner"]


def test_duplicate_registration_error():
    """Test that duplicate registration raises an exception"""
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY

    @PREPROCESSOR_REGISTRY.register("test_dup_preprocessor_1")
    def test_preprocess1(conversations, **kwargs):
        return [["first"]]

    # Second registration should raise ValueError
    with pytest.raises(ValueError, match="already registered"):

        @PREPROCESSOR_REGISTRY.register("test_dup_preprocessor_1")
        def test_preprocess2(conversations, **kwargs):
            return [["second"]]


def test_conv_preprocess():
    """Test that conv_preprocess convenience function works"""
    from veomni.data.multimodal import conv_preprocess

    # Test with a built-in preprocessor
    test_conversations = [
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "A beautiful sunset."},
    ]
    result = conv_preprocess("sharegpt4v_pretrain", test_conversations)
    assert isinstance(result, list)
    assert len(result) > 0

    # Test that it raises ValueError for unknown preprocessor
    with pytest.raises(ValueError, match="Unknown preprocessor name: nonexistent_preprocessor_xyz"):
        conv_preprocess("nonexistent_preprocessor_xyz", test_conversations)


def test_e2e_custom_and_builtin_preprocessor_flow():
    """
    E2E test: Register a custom preprocessor, process sample data with both
    builtin and custom preprocessors, and verify interleave structure.
    """
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY, conv_preprocess

    # Step 1: Register a customized preprocessor
    @PREPROCESSOR_REGISTRY.register("custom_vqa_e2e")
    def custom_vqa_preprocess(conversations, **kwargs):
        """
        Custom preprocessor that converts VQA format to interleaved structure.
        Expected input: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        Expected output: [["user", ("text", "Q: ..."), ("image", None)], ["assistant", ("text", "A: ...")]]
        """
        constructed_conversation = []
        role_mapping = {"human": "user", "gpt": "assistant"}

        for message in conversations:
            role = role_mapping[message["from"]]
            value = message["value"]

            if role == "user":
                # Add prefix and include image placeholder
                text_with_prefix = f"Q: {value.replace('<image>', '').strip()}"
                constructed_conversation.append([role, ("text", text_with_prefix), ("image", None)])
            else:
                # Add prefix for assistant response
                text_with_prefix = f"A: {value}"
                constructed_conversation.append([role, ("text", text_with_prefix)])

        return constructed_conversation

    # Step 2: Process sample data with existing builtin preprocessor
    builtin_test_data = [
        {"from": "human", "value": "<image> What do you see in this image?"},
        {"from": "gpt", "value": "I see a beautiful landscape with mountains."},
    ]

    builtin_result = conv_preprocess("sharegpt4v_sft", builtin_test_data)

    # Step 3: Assert builtin preprocessor output follows interleave structure
    assert isinstance(builtin_result, list), "Result should be a list"
    assert len(builtin_result) == 2, "Should have 2 conversation turns"

    # Verify first turn (user message)
    assert builtin_result[0][0] == "user", "First turn should be from user"
    assert len(builtin_result[0]) >= 2, "User turn should have role and at least one content item"
    # Check for interleaved image and text
    assert builtin_result[0][1][0] == "image", "First content item should be image"
    assert builtin_result[0][1][1] is None, "Image placeholder should be None"
    assert builtin_result[0][2][0] == "text", "Second content item should be text"
    assert isinstance(builtin_result[0][2][1], str), "Text content should be a string"

    # Verify second turn (assistant message)
    assert builtin_result[1][0] == "assistant", "Second turn should be from assistant"
    assert builtin_result[1][1][0] == "text", "Assistant response should be text"
    assert isinstance(builtin_result[1][1][1], str), "Text content should be a string"

    # Step 4: Process sample data with the custom preprocessor
    custom_test_data = [
        {"from": "human", "value": "<image> Describe this scene"},
        {"from": "gpt", "value": "A serene beach at sunset"},
    ]

    custom_result = conv_preprocess("custom_vqa_e2e", custom_test_data)

    # Step 5: Assert custom preprocessor output follows interleave structure
    assert isinstance(custom_result, list), "Custom result should be a list"
    assert len(custom_result) == 2, "Should have 2 conversation turns"

    # Verify first turn (user message with custom prefix)
    assert custom_result[0][0] == "user", "First turn should be from user"
    assert len(custom_result[0]) == 3, "User turn should have role, text, and image"
    assert custom_result[0][1][0] == "text", "First content item should be text"
    assert custom_result[0][1][1].startswith("Q: "), "Text should have Q: prefix"
    assert custom_result[0][2][0] == "image", "Second content item should be image"
    assert custom_result[0][2][1] is None, "Image placeholder should be None"

    # Verify second turn (assistant message with custom prefix)
    assert custom_result[1][0] == "assistant", "Second turn should be from assistant"
    assert custom_result[1][1][0] == "text", "Assistant response should be text"
    assert custom_result[1][1][1].startswith("A: "), "Text should have A: prefix"
    assert "serene beach" in custom_result[1][1][1].lower(), "Should contain expected content"

    # Step 6: Verify the interleave structure is consistent across both preprocessors
    # Both should return format: [[role, (modality, content), ...], ...]
    for result in [builtin_result, custom_result]:
        for turn in result:
            assert isinstance(turn, list), "Each turn should be a list"
            assert len(turn) >= 2, "Each turn should have at least role and one content item"
            assert isinstance(turn[0], str), "First element should be role (string)"
            assert turn[0] in ["user", "assistant"], "Role should be user or assistant"

            # Check all content items are tuples of (modality, content)
            for item in turn[1:]:
                assert isinstance(item, tuple), "Content items should be tuples"
                assert len(item) == 2, "Content tuple should have (modality, content)"
                assert item[0] in ["text", "image"], "Modality should be text or image"
