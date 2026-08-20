"""Tests for the per-module accelerator training knobs moved off `OmniTrainingArguments`.

Covers: `AcceleratorConfig` gaining the six SeedOmni-V2-only fields, `ChunkMBSConfig`
validation, per-module `accelerator.*` override survival through
`build_module_runtime_args`, and `_validate_omni_accelerator` being invoked both for
the top-level default (`OmniArguments.__post_init__`) and per resolved module
(`resolve_omni_model`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from veomni.arguments import OmniDataArguments, OmniInferArguments
from veomni.arguments.arguments_types import (
    AcceleratorConfig,
    ChunkMBSConfig,
    FSDPConfig,
    GradientCheckpointingConfig,
    TorchCompileConfig,
)
from veomni.omni_arguments.arguments_types import (
    OmniArguments,
    OmniModelRuntimeArguments,
    OmniModuleRuntimeArguments,
    _validate_omni_accelerator,
    build_module_runtime_args,
)


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Janus" / "janus_1.3b"


def _janus_args(*, modules_override: dict | None = None) -> OmniArguments:
    cfg_dir = _janus_cfg_dir()
    model_config = {
        "modules": str(cfg_dir / "modules_train.yaml"),
        "train_graph": str(cfg_dir / "graph_train.yaml"),
        "infer_graph": {"infer_gen": str(cfg_dir / "graph_infer_gen.yaml")},
    }
    if modules_override is not None:
        from veomni.omni_arguments.parser import load_yaml_with_inherit

        loaded = load_yaml_with_inherit(str(cfg_dir / "modules_train.yaml"))
        for name, override in modules_override.items():
            loaded.setdefault(name, {})
            loaded[name] = {**loaded[name], **override}
        model_config["modules"] = loaded
    return OmniArguments(
        model=OmniModelRuntimeArguments(model_path="/tmp/janus", model_config=model_config),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )


def test_accelerator_config_has_seed_omni_v2_fields_with_expected_defaults():
    acc = AcceleratorConfig()
    assert acc.init_device == "meta"
    assert acc.broadcast_model_weights_from_rank0 is True
    assert acc.ep_sharded_stream_load is False
    assert isinstance(acc.gradient_checkpointing, GradientCheckpointingConfig)
    assert isinstance(acc.torch_compile, TorchCompileConfig)
    assert isinstance(acc.chunk_mbs_config, ChunkMBSConfig)


def test_chunk_mbs_config_rejects_non_positive_chunk_mbs():
    ChunkMBSConfig(chunk_mbs=1)  # boundary value is valid
    with pytest.raises(ValueError, match="chunk_mbs"):
        ChunkMBSConfig(chunk_mbs=0)


def test_validate_omni_accelerator_accepts_defaults():
    _validate_omni_accelerator(AcceleratorConfig())


def test_meta_init_for_fsdp2_is_enforced_at_construction():
    """The rule moved onto ``AcceleratorConfig`` itself, so a module override cannot dodge it.

    ``_validate_omni_accelerator`` no longer repeats it: it could only ever see a
    config that already passed.
    """
    with pytest.raises(AssertionError, match="init_device: meta"):
        AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="fsdp2"), init_device="cpu")


def test_ddp_cpu_init_is_enforced_at_construction():
    with pytest.raises(AssertionError, match="init_device: cpu is not supported"):
        AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="ddp"), init_device="cpu")


def test_ep_sharded_stream_load_with_broadcast_is_enforced_at_construction():
    with pytest.raises(AssertionError, match="ep_sharded_stream_load"):
        AcceleratorConfig(ep_sharded_stream_load=True, broadcast_model_weights_from_rank0=True)


def test_validate_omni_accelerator_rejects_chunk_mbs_with_pad_to_length():
    acc = AcceleratorConfig(chunk_mbs_config=ChunkMBSConfig(enable=True))
    with pytest.raises(ValueError, match="pad_to_length"):
        _validate_omni_accelerator(acc, pad_to_length=128)


def test_validate_omni_accelerator_rejects_chunk_mbs_with_reentrant_checkpointing():
    acc = AcceleratorConfig(
        chunk_mbs_config=ChunkMBSConfig(enable=True),
        gradient_checkpointing=GradientCheckpointingConfig(enable=True, enable_reentrant=True),
    )
    with pytest.raises(ValueError, match="non-reentrant"):
        _validate_omni_accelerator(acc)


def test_validate_omni_accelerator_bans_torch_compile():
    acc = AcceleratorConfig(torch_compile=TorchCompileConfig(enable=True))
    with pytest.raises(ValueError, match="torch_compile"):
        _validate_omni_accelerator(acc)


def test_omni_arguments_post_init_validates_the_top_level_default():
    with pytest.raises(ValueError, match="torch_compile"):
        OmniArguments(
            model=OmniModelRuntimeArguments(
                accelerator=AcceleratorConfig(torch_compile=TorchCompileConfig(enable=True))
            ),
            data=OmniDataArguments(train_path=""),
            infer=OmniInferArguments(),
        )


def test_build_module_runtime_args_merges_per_module_gradient_checkpointing_override():
    """Global `model.accelerator.gradient_checkpointing` is the base; per-module YAML can override."""
    global_args = OmniModuleRuntimeArguments(
        accelerator=AcceleratorConfig(gradient_checkpointing=GradientCheckpointingConfig(enable=True)),
    )
    modules = build_module_runtime_args(
        global_args,
        "/tmp/model",
        {
            "module_a": {"accelerator": {"gradient_checkpointing": {"enable": False}}},
            "module_b": {"model_path": "module_b"},
        },
    )
    assert modules["module_a"].accelerator.gradient_checkpointing.enable is False
    assert modules["module_b"].accelerator.gradient_checkpointing.enable is True


def test_resolve_omni_model_accepts_valid_per_module_accelerator_override():
    """A per-module `accelerator.*` override that passes validation resolves cleanly."""
    args = _janus_args(
        modules_override={"janus_llama": {"accelerator": {"gradient_checkpointing": {"enable": False}}}}
    )
    modules = args.resolve_model().modules
    assert modules["janus_llama"].accelerator.gradient_checkpointing.enable is False
    # Untouched modules keep the top-level default.
    assert modules["janus_siglip"].accelerator.gradient_checkpointing.enable is True


@pytest.fixture
def veomni_caplog(caplog):
    """`caplog`, but also attached directly to the ``veomni`` logger.

    The library's root logger (``veomni``) sets ``propagate = False`` (see
    ``veomni/utils/logging.py``) so records never reach the root logger caplog normally
    listens on; attach its handler here so `caplog.text` works for `veomni.*` loggers too.
    """
    logger = logging.getLogger("veomni")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_resolve_omni_model_for_inference_forces_eager_without_broadcast_warning(veomni_caplog):
    """`for_inference=True` forces `fsdp_mode=eager` per module (for modules that don't already
    pin their own `fsdp_mode`, e.g. `janus_text_encoder`); `broadcast_model_weights_from_rank0`
    must be forced off alongside it so `_validate_omni_accelerator` does not warn for every module
    in the common single-process eager-inference default (see `_resolve_default_accelerator`).
    """
    args = _janus_args()
    with veomni_caplog.at_level("WARNING"):
        modules = args.resolve_model(for_inference=True).modules
    assert modules["janus_text_encoder"].accelerator.fsdp_config.fsdp_mode == "eager"
    assert modules["janus_text_encoder"].accelerator.broadcast_model_weights_from_rank0 is False
    assert "broadcast_model_weights_from_rank0" not in veomni_caplog.text


def test_resolve_omni_model_validates_each_module_accelerator():
    """A per-module override that fails `_validate_omni_accelerator` must raise at resolve time.

    The top-level `model.accelerator` default passes validation on its own (no
    `torch_compile.enable`); only the `janus_llama` module override sets it, so this
    only fails because `resolve_omni_model` validates every resolved module's own
    `accelerator`, not just the top-level default.
    """
    args = _janus_args(modules_override={"janus_llama": {"accelerator": {"torch_compile": {"enable": True}}}})
    with pytest.raises(ValueError, match="torch_compile"):
        args.resolve_model()
