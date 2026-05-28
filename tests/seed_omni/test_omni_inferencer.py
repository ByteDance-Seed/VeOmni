"""Smoke tests for :mod:`veomni.trainer.omni_inferencer`.

Scope
-----
* :class:`InferenceRequest` carries the expected default shape.
* Public module surface: :class:`OmniInferencer` + :class:`InferenceRequest`
  are exported and constructible-without-weights raises a clear error.
* :func:`_read_model_type` (validation gate before ``from_pretrained``)
  rejects unknown / missing ``model_type`` with a useful message —
  exercised via the public ``OmniInferencer`` path with a hand-built
  config.json on disk.

Full end-to-end (real Janus weights) inference is covered by
``tasks/infer/infer_omni.py`` and exercised in the broader integration
suite; this file pins the boundary error contracts and stays runnable
without GPU / weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veomni.models.seed_omni import OmniConfig
from veomni.trainer.omni_inferencer import (
    InferenceRequest,
    OmniInferencer,
    _read_model_type,
)


# ── InferenceRequest ─────────────────────────────────────────────────────────


def test_inference_request_defaults_are_empty_or_zero():
    req = InferenceRequest(prompt="hi")
    assert req.prompt == "hi"
    assert req.images == []
    assert req.force_image_gen is False
    assert req.generation_kwargs == {}
    assert req.max_new_tokens == 2048


def test_inference_request_preserves_generation_kwargs():
    req = InferenceRequest(
        prompt="x",
        generation_kwargs={"temperature": 0.5, "top_p": 0.9, "do_sample": False},
    )
    assert req.generation_kwargs == {"temperature": 0.5, "top_p": 0.9, "do_sample": False}


def test_inference_request_is_a_plain_dataclass():
    # ``OmniInferencer.generate_from_request`` accepts an InferenceRequest;
    # we want it to be JSON-loggable for traceability so the dataclass
    # MUST NOT carry private state (only the documented fields).
    fields = InferenceRequest.__dataclass_fields__
    assert set(fields) == {
        "prompt",
        "images",
        "force_image_gen",
        "generation_kwargs",
        "max_new_tokens",
    }


# ── OmniInferencer construction error contracts ──────────────────────────────


def test_omni_inferencer_rejects_non_omni_config():
    with pytest.raises(TypeError, match=r"OmniConfig"):
        OmniInferencer("not-a-config")  # type: ignore[arg-type]


def test_omni_inferencer_requires_tokenizer_path(tmp_path: Path):
    cfg = OmniConfig(
        modules={"x": {"weights_path": str(tmp_path)}},
        generation_graph={"initial": "s", "states": {"s": {}}},
    )
    with pytest.raises(ValueError, match=r"tokenizer_path"):
        OmniInferencer(cfg)


def test_omni_inferencer_requires_generation_graph(tmp_path: Path):
    cfg = OmniConfig(
        modules={"x": {"weights_path": str(tmp_path)}},
        tokenizer_path=str(tmp_path),
    )
    with pytest.raises(ValueError, match=r"generation_graph"):
        OmniInferencer(cfg)


# ── _read_model_type ────────────────────────────────────────────────────────


def test_read_model_type_rejects_unregistered_family(tmp_path: Path):
    """A model_type that HF transformers DOES recognise but the omni
    registry doesn't — surfaces as a clear ``KeyError`` from our guard,
    not a buried ``AttributeError`` deep inside ``from_pretrained``.

    We pick ``llama`` here because HF Llama is part of every transformers
    install we support, while no omni mixin claims that family.
    """
    cfg = {"model_type": "llama", "hidden_size": 8, "num_attention_heads": 1, "num_hidden_layers": 1}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(KeyError, match="not registered in OMNI_MODEL_REGISTRY"):
        _read_model_type(str(tmp_path))


def test_read_model_type_rejects_completely_unknown_family(tmp_path: Path):
    """A model_type unknown to both HF and the omni registry — HF raises
    first; that's still loud / actionable so we accept it."""
    cfg = {"model_type": "definitely_not_registered_xyz"}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises((ValueError, KeyError)):
        _read_model_type(str(tmp_path))


# ── Public surface ──────────────────────────────────────────────────────────


def test_module_exports_inferencer_and_request():
    import veomni.trainer.omni_inferencer as module

    assert "OmniInferencer" in module.__all__
    assert "InferenceRequest" in module.__all__
