"""Smoke tests for Qwen3-VL SeedOmni V2 modules (registry + M-RoPE helper)."""

from pathlib import Path

import torch

from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY
from veomni.models.seed_omni.modules.qwen3vl.llm.modulemixin import Qwen3VLLlmModuleMixin


def test_registry_resolves_qwen3vl_modules():
    for key, cfg_name, model_name in [
        ("qwen3vl_vision", "Qwen3VLVisionEncoderConfig", "Qwen3VLVisionEncoder"),
        ("qwen3vl_text_encoder", "Qwen3VLTextEncoderConfig", "Qwen3VLTextEncoder"),
        ("qwen3vl_llm", "Qwen3VLLlmConfig", "Qwen3VLLlm"),
    ]:
        assert OMNI_CONFIG_REGISTRY[key]().__name__ == cfg_name
        assert OMNI_MODEL_REGISTRY[key]().__name__ == model_name
    # Qwen3-VL reuses the Qwen2-VL image processor; video has its own processor.
    # The vision module needs both, so the registry returns an {image, video} dict.
    procs = OMNI_PROCESSOR_REGISTRY["qwen3vl_vision"]()
    assert procs["image"].__name__ == "Qwen2VLImageProcessor"
    assert procs["video"].__name__ == "Qwen3VLVideoProcessor"


def test_vision_position_ids_layout():
    # grid (t=1, h=4, w=6), merge=2 -> llm grid (1, 2, 3); start offset 5.
    pos = Qwen3VLLlmModuleMixin._vision_position_ids(5, torch.tensor([1, 4, 6]), merge=2)
    assert pos.shape == (3, 6)
    # temporal: single frame -> all equal to start
    assert pos[0].tolist() == [5, 5, 5, 5, 5, 5]
    # height: 0,0,0,1,1,1 + start
    assert pos[1].tolist() == [5, 5, 5, 6, 6, 6]
    # width: 0,1,2,0,1,2 + start
    assert pos[2].tolist() == [5, 6, 7, 5, 6, 7]


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    TextEncoder = OMNI_MODEL_REGISTRY["qwen3vl_text_encoder"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()

    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=True))
    te.save_pretrained(tmp_path)

    rcfg = TextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "qwen3vl_text_encoder"

    te2 = TextEncoder.from_pretrained(tmp_path)
    assert isinstance(te2, TextEncoder)
    assert te2.config.vocab_size == 64
    assert te2.config.hidden_size == 16
