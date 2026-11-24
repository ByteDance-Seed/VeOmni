import copy
from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.diffusion_gemma.gemma4_config import convert_gemma4_config_dict
from veomni.models.transformers.diffusion_gemma.gemma4_initialization import (
    SELF_CONDITIONING_PARAMETER_NAMES,
    Gemma4ToDiffusionGemmaCheckpointTensorConverter,
    convert_gemma4_fqn_to_index_mapping,
    validate_gemma4_initialization_missing_parameters,
)


class _TinyDiffusionGemma:
    def __init__(self):
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))

    def state_dict(self):
        names = {
            "model.decoder.embed_tokens.weight",
            "model.decoder.layers.0.layer_scalar",
            "model.decoder.layers.1.layer_scalar",
            "model.decoder.norm.weight",
            "model.encoder.language_model.embed_tokens.weight",
            "model.encoder.language_model.layers.0.layer_scalar",
            "model.encoder.language_model.layers.1.layer_scalar",
            "model.encoder.language_model.norm.weight",
            "model.encoder.vision_tower.patch_embedder.input_proj.weight",
            "model.encoder.embed_vision.embedding_projection.weight",
            "lm_head.weight",
            *SELF_CONDITIONING_PARAMETER_NAMES,
        }
        return {name: torch.empty(0) for name in names}

    def named_parameters(self, remove_duplicate=True):
        del remove_duplicate
        return iter((name, tensor) for name, tensor in self.state_dict().items() if not name.endswith(".layer_scalar"))


def _raw_tiny_tensors():
    return {
        "model.language_model.embed_tokens.weight": torch.tensor([1.0]),
        "model.language_model.layers.0.layer_scalar": torch.tensor([2.0]),
        "model.language_model.layers.1.layer_scalar": torch.tensor([3.0]),
        "model.language_model.norm.weight": torch.tensor([4.0]),
        "model.vision_tower.patch_embedder.input_proj.weight": torch.tensor([5.0]),
        "model.embed_vision.embedding_projection.weight": torch.tensor([6.0]),
    }


def test_raw_gemma4_converter_maps_every_key_and_duplicates_layer_scalars():
    model = _TinyDiffusionGemma()
    converter = Gemma4ToDiffusionGemmaCheckpointTensorConverter(model)
    emitted = {}
    for name, tensor in _raw_tiny_tensors().items():
        result = converter.convert(name, tensor)
        emitted[result.name] = result.tensor
    emitted.update({item.name: item.tensor for item in converter.finalize()})

    assert set(emitted) == {
        "model.decoder.embed_tokens.weight",
        "model.decoder.layers.0.layer_scalar",
        "model.decoder.layers.1.layer_scalar",
        "model.decoder.norm.weight",
        "model.encoder.language_model.layers.0.layer_scalar",
        "model.encoder.language_model.layers.1.layer_scalar",
        "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        "model.encoder.embed_vision.embedding_projection.weight",
    }
    assert torch.equal(
        emitted["model.decoder.layers.1.layer_scalar"],
        emitted["model.encoder.language_model.layers.1.layer_scalar"],
    )
    assert model._veomni_checkpoint_source_format == "raw_gemma4"


def test_raw_gemma4_converter_fails_closed_on_incomplete_layout():
    converter = Gemma4ToDiffusionGemmaCheckpointTensorConverter(_TinyDiffusionGemma())
    name, tensor = next(iter(_raw_tiny_tensors().items()))
    converter.convert(name, tensor)
    with pytest.raises(RuntimeError, match="does not exactly match"):
        converter.finalize()


def test_raw_gemma4_missing_parameter_gate_allows_only_self_conditioning():
    model = _TinyDiffusionGemma()
    model._veomni_checkpoint_source_format = "raw_gemma4"
    missing = {
        *SELF_CONDITIONING_PARAMETER_NAMES,
        "lm_head.weight",
        "model.encoder.language_model.embed_tokens.weight",
        "model.encoder.language_model.norm.weight",
    }
    validate_gemma4_initialization_missing_parameters(model, missing)
    assert missing == SELF_CONDITIONING_PARAMETER_NAMES

    receiver_missing = {
        *SELF_CONDITIONING_PARAMETER_NAMES,
        "lm_head.weight",
        "model.encoder.language_model.embed_tokens.weight",
        "model.encoder.language_model.norm.weight",
    }
    del model._veomni_checkpoint_source_format
    validate_gemma4_initialization_missing_parameters(model, receiver_missing)
    assert receiver_missing == SELF_CONDITIONING_PARAMETER_NAMES

    # FSDP transforms tied parameters before loading and can deduplicate all of
    # their alias names. The genuinely new parameters remain the same.
    fsdp_missing = set(SELF_CONDITIONING_PARAMETER_NAMES)
    validate_gemma4_initialization_missing_parameters(model, fsdp_missing)
    assert fsdp_missing == SELF_CONDITIONING_PARAMETER_NAMES

    partial_alias_missing = {*SELF_CONDITIONING_PARAMETER_NAMES, "lm_head.weight"}
    validate_gemma4_initialization_missing_parameters(model, partial_alias_missing)
    assert partial_alias_missing == SELF_CONDITIONING_PARAMETER_NAMES

    unrelated = {"model.decoder.norm.weight"}
    validate_gemma4_initialization_missing_parameters(model, unrelated)
    assert unrelated == {"model.decoder.norm.weight"}

    model._veomni_checkpoint_source_format = "raw_gemma4"
    with pytest.raises(RuntimeError, match="unexpected parameter set"):
        validate_gemma4_initialization_missing_parameters(model, {"model.decoder.norm.weight"})


def test_raw_gemma4_index_conversion_is_complete_and_assigns_new_weights():
    raw = {name: index % 2 + 1 for index, name in enumerate(_raw_tiny_tensors())}
    converted = convert_gemma4_fqn_to_index_mapping(raw)

    assert len(converted) == len(raw) + 2 + len(SELF_CONDITIONING_PARAMETER_NAMES)
    assert (
        converted["model.encoder.language_model.layers.0.layer_scalar"]
        == converted["model.decoder.layers.0.layer_scalar"]
    )
    for name in SELF_CONDITIONING_PARAMETER_NAMES:
        assert converted[name] == converted["model.decoder.norm.weight"]


def test_gemma4_config_conversion_is_explicit_and_does_not_mutate_source():
    source = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "audio_config": None,
        "audio_token_id": 3,
        "boi_token_id": 4,
        "eoi_token_id": 5,
        "image_token_id": 6,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "attention_k_eq_v": True,
            "enable_moe_block": True,
            "hidden_size": 8,
            "num_hidden_layers": 2,
        },
        "vision_config": {"model_type": "gemma4_vision"},
    }
    before = copy.deepcopy(source)
    converted = convert_gemma4_config_dict(source)

    assert source == before
    assert converted["architectures"] == ["DiffusionGemmaForBlockDiffusion"]
    assert converted["model_type"] == "diffusion_gemma"
    assert converted["text_config"]["model_type"] == "diffusion_gemma_text"
    assert "audio_config" not in converted
    assert "attention_k_eq_v" not in converted["text_config"]
    assert converted["canvas_length"] == 256
    assert converted["self_conditioning_prob"] == 0.5
