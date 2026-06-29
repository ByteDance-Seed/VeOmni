"""Smoke tests for :mod:`veomni.trainer.omni.omni_inferencer` and the
:func:`veomni.models.seed_omni.read_model_type` registry-dispatch helper.

Scope
-----
* :class:`InferenceRequest` carries the expected default shape and the
  inferencer module's public surface (``__all__``) exports it.
* :func:`read_model_type` (validation gate before ``from_pretrained``)
  rejects unknown / missing ``model_type`` with a useful message.

Full end-to-end (real Janus weights) inference is covered by
``tasks/infer/infer_omni.py`` and the broader integration suite; this file
stays runnable without GPU / weights and assumes the caller passes a
well-formed :class:`~veomni.arguments.arguments_types_omni.OmniArguments`
(no construction-time defensive checks to pin).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from veomni.models.seed_omni import read_model_type
from veomni.trainer.omni.omni_inferencer import InferenceRequest, OmniInferencer


# ── InferenceRequest ─────────────────────────────────────────────────────────


def test_inference_request_defaults_are_empty_or_zero():
    req = InferenceRequest(prompt="hi")
    assert req.prompt == "hi"
    assert req.images == []
    assert req.generation_kwargs == {}


def test_inference_request_preserves_generation_kwargs():
    req = InferenceRequest(
        prompt="x",
        generation_kwargs={"temperature": 0.5, "top_p": 0.9, "do_sample": False},
    )
    assert req.generation_kwargs == {"temperature": 0.5, "top_p": 0.9, "do_sample": False}


def test_inference_request_is_a_plain_dataclass():
    # ``OmniInferencer.run_request`` accepts an InferenceRequest; we want it
    # to be JSON-loggable for traceability so the dataclass MUST NOT carry
    # private state (only the documented fields).
    fields = InferenceRequest.__dataclass_fields__
    assert set(fields) == {
        "prompt",
        "images",
        "generation_kwargs",
    }


def test_inferencer_injects_resolved_infer_type_into_runtime_kwargs():
    inferencer = OmniInferencer.__new__(OmniInferencer)
    inferencer.base = SimpleNamespace(
        args=SimpleNamespace(
            infer=SimpleNamespace(
                infer_type="infer_gen",
                generation_kwargs={"temperature": 0.5},
            )
        )
    )

    kwargs = inferencer._runtime_generation_kwargs()

    assert kwargs == {"temperature": 0.5, "infer_type": "infer_gen"}
    assert inferencer.args.infer.generation_kwargs == {"temperature": 0.5}


def test_inferencer_rejects_conflicting_runtime_infer_type():
    inferencer = OmniInferencer.__new__(OmniInferencer)
    inferencer.base = SimpleNamespace(
        args=SimpleNamespace(
            infer=SimpleNamespace(
                infer_type="infer_gen",
                generation_kwargs={"infer_type": "infer_und"},
            )
        )
    )

    with pytest.raises(ValueError, match="conflicts"):
        inferencer._runtime_generation_kwargs()


# ── read_model_type ─────────────────────────────────────────────────────────


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
        read_model_type(str(tmp_path))


def test_read_model_type_rejects_completely_unknown_family(tmp_path: Path):
    """A model_type unknown to both HF and the omni registry — HF raises
    first; that's still loud / actionable so we accept it."""
    cfg = {"model_type": "definitely_not_registered_xyz"}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises((ValueError, KeyError)):
        read_model_type(str(tmp_path))


# ── Public surface ──────────────────────────────────────────────────────────


def test_module_exports_inferencer_and_request():
    import veomni.trainer.omni.omni_inferencer as module

    assert "OmniInferencer" in module.__all__
    assert "OmniModuleInferencer" in module.__all__
    assert "InferenceRequest" in module.__all__


def test_module_needs_distributed_only_when_declared_non_eager():
    from veomni.trainer.omni.omni_inferencer import _module_needs_distributed

    # No accelerator block → eager default → single-process.
    assert not _module_needs_distributed({"model": {"weights_path": "janus_siglip"}})
    # fsdp2 / ddp need a distributed (torchrun) launch + own ParallelState.
    assert _module_needs_distributed(
        {
            "model": {"weights_path": "janus_siglip"},
            "train": {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}},
        }
    )
    assert _module_needs_distributed(
        {
            "model": {"weights_path": "janus_llama"},
            "train": {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}}},
        }
    )
    # eager is the only single-process load (the inference default). fsdp_mode is
    # now Literal["ddp", "fsdp2", "eager"] — FSDP1 / "none" were removed.
    assert not _module_needs_distributed(
        {
            "model": {"weights_path": "janus_siglip"},
            "train": {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}},
        }
    )


def test_omni_config_module_config_merges_model_and_train_blocks():
    from veomni.models.seed_omni.configuration_omni import OmniConfig

    omni_config = OmniConfig.from_dict(
        {
            "modules": {
                "janus_siglip": {
                    "model": {
                        "model_path": "/tmp/global/janus_siglip",
                        "model_config": {"freeze": True},
                    },
                    "data": {"train_path": "/tmp/unused"},
                    "train": {"init_device": "meta"},
                }
            }
        }
    )
    module_args = omni_config.module_config("janus_siglip")
    assert module_args.model.model_path == "/tmp/global/janus_siglip"
    assert module_args.model.model_config == {"freeze": True}
    assert module_args.train.init_device == "meta"
