"""Composed-model FSDP wrap scope: ``{child}.{ClassName}`` targets.

These exercise :meth:`OmniModelRuntime._parallelize_composed_model` and
``_is_fsdp_wrap_target`` directly. They live next to the Janus modules only
because the ambiguity they guard against was found there (text-encoder
``Embedding`` vs the VQVAE codebook, which reads ``self.embedding.weight``
without going through the embedding's own forward); nothing below imports a
Janus module.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from veomni.distributed.torch_compile import CompileConfig


def _defer_runtime(name: str, model: torch.nn.Module, *, ulysses_size: int = 1):
    """A ``ModuleRuntime`` stand-in that defers its own parallelize pass."""
    from contextlib import nullcontext

    rt = MagicMock()
    rt._defer_parallelize = True
    rt.reads_offline_cache = False
    rt.skip_hf_weight_load = False
    rt.weights_path = f"/tmp/{name}"
    rt.model = model
    rt.args = SimpleNamespace(
        model_path=f"/tmp/{name}",
        lora_config=None,
        basic_modules=[],
        broadcast_model_weights_from_rank0=False,
        ep_sharded_stream_load=False,
        optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
        accelerator=SimpleNamespace(
            init_device="meta",
            ulysses_size=ulysses_size,
            enable_async=False,
            cp_size=1,
            tp_size=1,
            pp_size=1,
            torch_compile=CompileConfig(),
            gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False, early_stop=True),
            fsdp_config=SimpleNamespace(
                fsdp_scope="model",
                fsdp_mode="fsdp2",
                reshard_after_forward=True,
                mixed_precision=SimpleNamespace(enable=False),
                forward_prefetch=False,
                offload=False,
                offload_pin_memory=False,
                max_load_broadcast_size=20.0,
                expert_weight_fp8_all_gather=False,
                expert_fsdp_shard_dim=None,
            ),
        ),
    )
    rt._scoped.return_value = nullcontext()
    return rt


def _omni_runtime_args(*, accelerator=None, optimizer=None):
    composer = _defer_runtime("composer", torch.nn.Linear(2, 2)).args
    return SimpleNamespace(
        accelerator=accelerator or composer.accelerator,
        optimizer=optimizer if optimizer is not None else composer.optimizer,
        broadcast_model_weights_from_rank0=composer.broadcast_model_weights_from_rank0,
        ep_sharded_stream_load=composer.ep_sharded_stream_load,
    )


@pytest.fixture
def base_parallel_state():
    """The composed wrap runs under ``base``, which ``OmniTrainer._setup`` registers."""
    from veomni.distributed import parallel_state

    parallel_state._PARALLEL_STATE_REGISTRY["base"] = parallel_state.ParallelState()
    try:
        yield
    finally:
        parallel_state._PARALLEL_STATE_REGISTRY.pop("base", None)
        parallel_state.set_parallel_state(None)


@pytest.mark.usefixtures("base_parallel_state")
def test_composed_wrap_scopes_child_no_split_modules(monkeypatch: pytest.MonkeyPatch):
    """Each child's ``_no_split_modules`` is prefixed with that child's name."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    class _ChildA(torch.nn.Module):
        _no_split_modules = ["LayerA"]

    class _ChildB(torch.nn.Module):
        _no_split_modules = []

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = ["LayerA"]
    omni.module_runtimes = {
        "a": _defer_runtime("a", _ChildA()),
        "b": _defer_runtime("b", _ChildB()),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    omni._parallelize_composed_model()

    assert captured["basic_modules"] == ["a.LayerA"]
    assert omni.model._no_split_modules == ["a.LayerA"]
    assert captured["weights_path"] == {"a": "/tmp/a", "b": "/tmp/b"}
    assert "_ChildA" not in captured["basic_modules"]
    assert "_ChildB" not in captured["basic_modules"]


@pytest.mark.usefixtures("base_parallel_state")
def test_composed_wrap_scopes_embedding_to_owning_child(monkeypatch: pytest.MonkeyPatch):
    """TextEncoder ``Embedding`` must not also match a sibling VQ codebook."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    class _TextEnc(torch.nn.Module):
        _no_split_modules = ["Embedding"]

    class _Vqvae(torch.nn.Module):
        _no_split_modules = ["VqvaeVectorQuantizer"]

    class _Llama(torch.nn.Module):
        _no_split_modules = ["LlamaDecoderLayer"]

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = ["Embedding", "VqvaeVectorQuantizer"]
    omni.module_runtimes = {
        "text": _defer_runtime("text", _TextEnc()),
        "vqvae": _defer_runtime("vqvae", _Vqvae()),
        "llama": _defer_runtime("llama", _Llama()),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    omni._parallelize_composed_model()

    assert captured["basic_modules"] == [
        "text.Embedding",
        "vqvae.VqvaeVectorQuantizer",
        "llama.LlamaDecoderLayer",
    ]
    assert omni.model._no_split_modules == captured["basic_modules"]
    assert "Embedding" not in captured["basic_modules"]
    assert "_TextEnc" not in captured["basic_modules"]
    assert "_Vqvae" not in captured["basic_modules"]


@pytest.mark.usefixtures("base_parallel_state")
def test_composed_wrap_does_not_inspect_module_level_sp():
    """Wrap uses the already-built runtimes; module SP overlays are not compared."""
    from unittest.mock import patch

    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {
        "a": _defer_runtime("a", torch.nn.Linear(2, 2), ulysses_size=1),
        "b": _defer_runtime("b", torch.nn.Linear(2, 2), ulysses_size=4),
    }
    omni.omni_model_runtime_args = _omni_runtime_args()
    with patch(
        "veomni.distributed.torch_parallelize.build_parallelize_model", side_effect=lambda model, **kwargs: model
    ):
        omni._parallelize_composed_model()


@pytest.mark.usefixtures("base_parallel_state")
def test_composed_wrap_uses_composer_accelerator_not_module_overlay(monkeypatch: pytest.MonkeyPatch):
    """Wrap knobs come from the OmniModel accelerator / optimizer, not a module overlay."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    captured: dict = {}

    def _fake_build(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr("veomni.distributed.torch_parallelize.build_parallelize_model", _fake_build)

    module = _defer_runtime("a", torch.nn.Linear(2, 2))
    module.args.accelerator.init_device = "cpu"
    module.args.optimizer.type = "adamw"
    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {"a": module}
    omni.omni_model_runtime_args = _omni_runtime_args(
        optimizer=SimpleNamespace(type="muon", muon_expert_zero_comm=True),
    )
    omni._parallelize_composed_model()

    assert captured["init_device"] == "meta"
    assert captured["muon_expert_zero_comm"] is True


def test_fsdp_wrap_target_scopes_class_to_child_prefix():
    """A scoped target wraps its class under that child only; bare stays global."""
    from veomni.distributed.torch_parallelize import _is_fsdp_wrap_target

    targets = {"text.Embedding", "llama.LlamaDecoderLayer"}
    assert _is_fsdp_wrap_target("text.embed_tokens", "Embedding", targets)
    assert _is_fsdp_wrap_target("llama.language_model.layers.0", "LlamaDecoderLayer", targets)
    # The VQ codebook is the same class under a different child.
    assert not _is_fsdp_wrap_target("vqvae.vqmodel.quantize.embedding", "Embedding", targets)
    # A prefix must match on a path boundary, not a string prefix.
    assert not _is_fsdp_wrap_target("text_encoder.embed_tokens", "Embedding", targets)
    assert _is_fsdp_wrap_target("layers.0", "LlamaDecoderLayer", {"LlamaDecoderLayer"})


def test_composed_wrap_skips_when_fsdp_scope_is_module(monkeypatch: pytest.MonkeyPatch):
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    called = []
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize.build_parallelize_model",
        lambda model, **kwargs: called.append(kwargs) or model,
    )

    omni = OmniModelRuntime.__new__(OmniModelRuntime)
    omni.model = MagicMock()
    omni.model._no_split_modules = []
    omni.module_runtimes = {"a": _defer_runtime("a", torch.nn.Linear(2, 2))}
    composer = _defer_runtime("composer", torch.nn.Linear(2, 2)).args.accelerator
    composer.fsdp_config.fsdp_scope = "module"
    omni.omni_model_runtime_args = _omni_runtime_args(accelerator=composer)
    omni._parallelize_composed_model()

    assert called == []
