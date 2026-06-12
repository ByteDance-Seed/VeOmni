"""Smoke tests for the self-contained BAGEL transformers test reference."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from tests.seed_omni.bagel.transformers import (
    AutoEncoderParams,
    Bagel,
    BagelConfig,
    BagelOfficialReferenceWrapper,
    BagelReferenceConfig,
    BagelReferenceForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    SiglipVisionConfig,
    SiglipVisionModel,
    register_for_auto_classes,
)


def test_bagel_transformers_reference_loads_and_forwards_dummy_data(tmp_path):
    register_for_auto_classes()
    config = BagelReferenceConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1)
    model = BagelReferenceForCausalLM(config)
    model.save_pretrained(tmp_path)

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    outputs = loaded(input_ids=input_ids, labels=labels, output_hidden_states=True)

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 4, 32)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == 2


def test_bagel_transformers_reference_exposes_official_symbols():
    register_for_auto_classes()

    assert Bagel.config_class is BagelConfig
    assert issubclass(Qwen2Config, Qwen2Model.config_class)
    assert issubclass(Qwen2Config, Qwen2ForCausalLM.config_class)
    assert SiglipVisionModel.config_class is SiglipVisionConfig
    assert AutoEncoderParams.__name__ == "AutoEncoderParams"


def test_bagel_official_reference_wrapper_assembles_text_only_model():
    llm_config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_module="Qwen2MoTDecoderLayer",
        qk_norm=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    wrapper = BagelOfficialReferenceWrapper.from_configs(
        llm_config=llm_config,
        visual_gen=False,
        visual_und=False,
        init_on_meta=True,
    )

    assert isinstance(wrapper.model, Bagel)
    assert wrapper.config.llm_config is llm_config
    assert wrapper.config.visual_gen is False
    assert wrapper.config.visual_und is False
