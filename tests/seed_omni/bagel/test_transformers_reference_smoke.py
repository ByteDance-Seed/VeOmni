"""Smoke tests for the self-contained BAGEL transformers test reference."""

from __future__ import annotations

from tests.seed_omni.bagel.transformers import (
    AutoEncoderParams,
    Bagel,
    BagelConfig,
    BagelOfficialReference,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    SiglipVisionConfig,
    SiglipVisionModel,
    load_vendored_model,
)


def test_bagel_transformers_reference_exposes_official_symbols():
    assert Bagel.config_class is BagelConfig
    assert issubclass(Qwen2Config, Qwen2Model.config_class)
    assert issubclass(Qwen2Config, Qwen2ForCausalLM.config_class)
    assert SiglipVisionModel.config_class is SiglipVisionConfig
    assert AutoEncoderParams.__name__ == "AutoEncoderParams"
    assert callable(load_vendored_model)


def test_bagel_official_reference_assembles_text_only_model():
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
    reference = BagelOfficialReference.from_configs(
        llm_config=llm_config,
        visual_gen=False,
        visual_und=False,
        init_on_meta=True,
    )

    assert isinstance(reference.model, Bagel)
    assert reference.config.llm_config is llm_config
    assert reference.config.visual_gen is False
    assert reference.config.visual_und is False
