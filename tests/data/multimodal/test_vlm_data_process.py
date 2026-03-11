# Test module for Vision-Language Model data processing validation
# This module compares outputs when using HF-loaded processor and position_id_func versus VeOmni-loaded sample transform function
#
# The test validates both ways produce the same input_ids, attention_mask, image_grid_thw, video_grid_thw, pixel_values, pixel_values_videos, position_ids, mm_token_type_ids
#
# Supported models:
# - Qwen3-VL: Qwen/Qwen3-VL-2B-Instruct (uses process_sample_qwen3_vl)
# - Qwen3.5: Qwen/Qwen3.5-0.8B (uses process_sample_qwen3_vl_transformers_v5)

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from veomni.data import build_multimodal_chat_template
from veomni.data.data_transform import (
    process_sample_qwen_vl,
)
from veomni.models import build_foundation_model, build_processor
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")


# Mapping from function name to actual function
PROCESS_SAMPLE_FUNCTION_MAP = {
    "process_sample_qwen_vl": process_sample_qwen_vl,
}


device = get_device_type()

# Test data directory (relative to this file)
TEST_DATA_DIR = Path(__file__).resolve().parents[2] / "testdata"


@dataclass
class ModelTestConfig:
    """Configuration for testing a specific VLM model.

    Attributes:
        model_id: HuggingFace model identifier (for loading pretrained processor)
        config_path: Path to model config file (for building model without downloading weights)
        process_sample_func_name: Name of the process_sample function in data_transform module
        chat_template_name: Name of the chat template to use
        test_image: Image for testing
        test_text: Text prompt to use with the image

    """

    model_id: str
    config_path: str
    process_sample_func_name: str
    chat_template_name: str
    test_image: str = str(TEST_DATA_DIR / "qwen-vl-demo.jpeg")
    test_text: str = "Describe this image."


def load_hf_processor(model_path):
    """Load HuggingFace processor from model path."""
    print(f"\n[Setup] Loading HF processor from path: {model_path}")
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def load_hf_model(model_path):
    """Load HuggingFace model from model path."""
    print(f"\n[Setup] Loading HF model from path: {model_path}")
    return AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True)


def hf_process_sample(config: ModelTestConfig, hf_processor, hf_model):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": config.test_image},
                {"type": "text", "text": config.test_text},
            ],
        },
    ]

    if _is_transformers_v5:
        inputs = hf_processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        position_ids, rope_deltas = hf_model.model.get_rope_index(
            input_ids=inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
    else:
        inputs = hf_processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        position_ids, rope_deltas = hf_model.model.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )

    output = inputs
    output["position_ids"] = position_ids
    output["rope_deltas"] = rope_deltas

    # Move tensors to device
    output = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in output.items()}

    return output


def load_veomni_processor(model_path):
    """Load VeOmni processor from model path."""
    print(f"\n[Setup] Loading VeOmni processor from path: {model_path}")
    return build_processor(model_path)


def load_veomni_model(config_path, device):
    """Build and return the veomni model for testing."""
    print(f"\n[Setup] Building veomni model on device: {device}")
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        moe_implementation="eager",
        init_device=device,
    )
    model.eval()
    return model


def veomni_process_sample(
    config: ModelTestConfig,
    veomni_processor,
    veomni_model,
) -> dict[str, torch.Tensor]:
    # Get processing function from mapping
    process_sample_func = PROCESS_SAMPLE_FUNCTION_MAP.get(config.process_sample_func_name)
    if process_sample_func is None:
        raise ValueError(
            f"Unknown process_sample function: {config.process_sample_func_name}. "
            f"Available functions: {list(PROCESS_SAMPLE_FUNCTION_MAP.keys())}"
        )

    # Prepare sample in VeOmni format
    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [
            {"from": "human", "value": f"<image>{config.test_text}"},
        ],
        "images": [config.test_image],
    }

    # Initialize chat template
    chat_template = build_multimodal_chat_template(
        config.chat_template_name,
        veomni_processor.tokenizer,
    )

    position_id_func = veomni_model.get_position_id_func()

    # Process with the function
    outputs = process_sample_func(
        sample=sample,
        processor=veomni_processor,
        chat_template=chat_template,
        position_id_func=position_id_func,
    )
    output = outputs[0]

    # Move tensors to device
    output = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in output.items()}

    return output


@pytest.mark.parametrize(
    "config, check_mm_token_type_ids",
    [
        pytest.param(
            ModelTestConfig(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                config_path="./tests/toy_config/qwen25vl_toy/config.json",
                process_sample_func_name="process_sample_qwen_vl",
                chat_template_name="qwen2_5vl",
            ),
            False,  # check_mm_token_type_ids
            marks=pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0"),
            id="qwen2_5_vl",
        ),
        pytest.param(
            ModelTestConfig(
                model_id="Qwen/Qwen3-VL-2B-Instruct",
                config_path="./tests/toy_config/qwen3vl_toy/config.json",
                process_sample_func_name="process_sample_qwen_vl",
                chat_template_name="qwen3vl",
            ),
            False,  # check_mm_token_type_ids
            marks=pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0"),
            id="qwen3_vl",
        ),
        pytest.param(
            ModelTestConfig(
                model_id="Qwen/Qwen3.5-0.8B",
                config_path="./tests/toy_config/qwen3_5_toy/config.json",
                process_sample_func_name="process_sample_qwen_vl",
                chat_template_name="qwen3vl",
            ),
            True,  # check_mm_token_type_ids
            marks=pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0"),
            id="qwen3_5",
        ),
    ],
)
def test_vlm_processor_comparison(config, check_mm_token_type_ids):
    """Test that HF and VeOmni processors produce identical outputs."""
    # Process data with both processors
    hf_processor = load_hf_processor(config.model_id)
    hf_model = load_hf_model(config.model_id)
    hf_output = hf_process_sample(config, hf_processor, hf_model)

    veomni_processor = load_veomni_processor(config.model_id)
    veomni_model = load_veomni_model(config.config_path, device)
    veomni_output = veomni_process_sample(config, veomni_processor, veomni_model)

    # Compare core tensors - these should be IDENTICAL
    hf_image_mask = hf_output["input_ids"] == hf_processor.image_token_id
    hf_output["input_ids"][hf_image_mask] = 0
    hf_video_mask = hf_output["input_ids"] == hf_processor.video_token_id
    hf_output["input_ids"][hf_video_mask] = 0

    torch.testing.assert_close(
        hf_output["input_ids"][0],
        veomni_output["input_ids"],
        atol=0.0,
        rtol=0.0,
        msg="Veomni input_ids mismatch vs HF generate!",
    )

    for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
        if hf_output.get(key) is not None:
            assert veomni_output.get(key) is not None, f"HF has {key} but VeOmni does not"
            torch.testing.assert_close(
                hf_output[key],
                veomni_output[key],
            )
            print(f"VeOmni {key} Output matches HF!")
        else:
            assert veomni_output.get(key) is None, f"VeOmni has {key} but HF does not"

    torch.testing.assert_close(
        hf_output["position_ids"].squeeze(1),
        veomni_output["position_ids"],
    )

    torch.testing.assert_close(
        hf_output["attention_mask"].squeeze(0),
        veomni_output["attention_mask"],
    )

    if check_mm_token_type_ids:
        torch.testing.assert_close(
            hf_output["mm_token_type_ids"].squeeze(0),
            veomni_output["mm_token_type_ids"],
        )
        print("VeOmni mm_token_type_ids Output matches HF!")
