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

"""Tests for the self-resolving, model-scoped ``AcceleratorConfig``.

Two coupled changes are pinned here. ``AcceleratorConfig`` used to depend on
``TrainingArguments._validate_accelerator()`` to fill in
``dp_size``/``dp_replicate_size``/``dp_shard_size`` and to check the init-device
rules, so any instance built elsewhere was half-initialized; that resolution now
lives in ``AcceleratorConfig.__post_init__``. And ``accelerator``/``optimizer``
now hang off ``model``, not ``train``, because both are per-model decisions — an
omni model gives each module its own pair.

Run:
    pytest -v tests/arguments/test_accelerator_config.py
"""

import dataclasses
import json

import pytest

from veomni.arguments import arguments_types
from veomni.arguments.arguments_types import (
    AcceleratorConfig,
    BaseModelArguments,
    ChunkMBSConfig,
    DataArguments,
    FSDPConfig,
    ModelArguments,
    ModelRuntimeArguments,
    OptimizerConfig,
    TrainingArguments,
    VeOmniArguments,
)
from veomni.arguments.parser import _instantiate_recursive


@pytest.fixture
def world_size(monkeypatch):
    """Set WORLD_SIZE for the duration of one test."""

    def _set(size):
        monkeypatch.setenv("WORLD_SIZE", str(size))

    return _set


def make_model_args(**kwargs) -> ModelArguments:
    """``ModelArguments`` needs a config_path or model_path; nothing here reads it."""
    return ModelArguments(config_path="dummy", **kwargs)


def make_root_args(**model_kwargs) -> VeOmniArguments:
    """Root config with the required model/data identifiers stubbed out."""
    return VeOmniArguments(
        model=make_model_args(**model_kwargs),
        data=DataArguments(train_path="dummy"),
    )


def test_accelerator_resolves_dp_sizes_on_its_own(world_size):
    world_size(8)
    acc = AcceleratorConfig()

    assert acc.world_size == 8
    assert acc.dp_size == 8
    assert acc.dp_replicate_size == 1
    assert acc.dp_shard_size == 8


def test_ulysses_size_is_carved_out_of_dp(world_size):
    world_size(8)
    acc = AcceleratorConfig(ulysses_size=2)

    assert acc.dp_size == 4
    assert acc.dp_shard_size == 4


def test_hsdp_shard_size_derives_the_replicate_size(world_size):
    world_size(8)
    acc = AcceleratorConfig(dp_shard_size=4)

    assert acc.dp_replicate_size == 2
    assert acc.dp_shard_size == 4


def test_hsdp_replicate_size_derives_the_shard_size(world_size):
    world_size(8)
    acc = AcceleratorConfig(dp_replicate_size=2)

    assert acc.dp_replicate_size == 2
    assert acc.dp_shard_size == 4


def test_hsdp_accepts_a_consistent_explicit_pair(world_size):
    world_size(8)
    acc = AcceleratorConfig(dp_replicate_size=2, dp_shard_size=4)

    assert (acc.dp_replicate_size, acc.dp_shard_size) == (2, 4)


def test_hsdp_rejects_a_pair_that_does_not_multiply_to_dp_size(world_size):
    world_size(8)
    with pytest.raises(AssertionError, match="dp_size should be equal to dp_replicate_size"):
        AcceleratorConfig(dp_replicate_size=2, dp_shard_size=8)


def test_hsdp_rejects_a_shard_size_that_does_not_divide_dp_size(world_size):
    world_size(8)
    with pytest.raises(ValueError, match="dp_size should be a multiple of dp_shard_size"):
        AcceleratorConfig(dp_shard_size=3)


def test_world_size_must_divide_the_non_dp_product(world_size):
    world_size(6)
    with pytest.raises(ValueError, match="World size should be a multiple of"):
        AcceleratorConfig(ulysses_size=4)


def test_two_accelerators_resolve_independently(world_size):
    """The omni case: one config per module, each with its own topology."""
    world_size(8)
    encoder = AcceleratorConfig(ulysses_size=1)
    decoder = AcceleratorConfig(ulysses_size=2)

    assert encoder.dp_size == 8
    assert decoder.dp_size == 4


def test_fsdp2_requires_meta_init(world_size):
    world_size(1)
    with pytest.raises(AssertionError, match="init_device: meta"):
        AcceleratorConfig(init_device="cpu")


def test_ep_rejects_cpu_init(world_size):
    world_size(1)
    with pytest.raises(AssertionError, match="cpu init is not supported when enable ep"):
        AcceleratorConfig(ep_size=2, init_device="cpu", fsdp_config=FSDPConfig(fsdp_mode="ddp"))


def test_ep_sharded_stream_load_conflicts_with_broadcast(world_size):
    world_size(1)
    with pytest.raises(AssertionError, match="ep_sharded_stream_load requires"):
        AcceleratorConfig(ep_sharded_stream_load=True, broadcast_model_weights_from_rank0=True)


def test_ddp_warns_that_broadcast_is_ignored(world_size, monkeypatch):
    world_size(1)
    warnings = []
    monkeypatch.setattr(arguments_types.logger, "warning_rank0", lambda msg, *a, **k: warnings.append(msg))

    AcceleratorConfig(
        init_device="cuda",
        broadcast_model_weights_from_rank0=True,
        fsdp_config=FSDPConfig(fsdp_mode="ddp"),
    )

    assert any("broadcast_model_weights_from_rank0=True" in msg for msg in warnings)


@pytest.mark.parametrize(
    "name",
    [
        "init_device",
        "broadcast_model_weights_from_rank0",
        "ep_sharded_stream_load",
        "gradient_checkpointing",
        "torch_compile",
        "chunk_mbs_config",
    ],
)
def test_moved_knobs_live_on_the_accelerator_and_not_on_training_arguments(name, world_size):
    world_size(1)

    assert hasattr(make_model_args().accelerator, name)
    assert not hasattr(TrainingArguments(), name)


def test_accelerator_and_optimizer_hang_off_model_not_train(world_size):
    world_size(1)
    args = make_root_args()

    assert isinstance(args.model.accelerator, AcceleratorConfig)
    assert isinstance(args.model.optimizer, OptimizerConfig)
    assert not hasattr(args.train, "accelerator")
    assert not hasattr(args.train, "optimizer")


def test_batch_config_is_derived_from_the_models_accelerator(world_size):
    """``train`` no longer owns a topology, so only the root can pair the two."""
    world_size(8)
    args = make_root_args()

    assert args.model.accelerator.dp_size == 8
    assert args.train.global_batch_size == args.train.micro_batch_size * 8


def test_batch_config_follows_a_model_level_ulysses_override(world_size):
    world_size(8)
    args = make_root_args(accelerator=AcceleratorConfig(ulysses_size=2))

    assert args.model.accelerator.dp_size == 4
    assert args.train.global_batch_size == args.train.micro_batch_size * 4


def test_model_runtime_arguments_is_a_standalone_training_unit():
    """What an omni module inherits: model fields + its own accelerator/optimizer.

    Notably without ``config_path``/``tokenizer_path``/``safetensor_idx_path`` — a
    module is addressed by its subfolder in a composed checkpoint, so inheriting
    those would hand every module a tokenizer it has no use for.
    """
    names = {f.name for f in dataclasses.fields(ModelRuntimeArguments)}

    assert {"model_path", "model_config", "basic_modules", "lora_config", "ops_implementation"} <= names
    assert {"accelerator", "optimizer"} <= names
    assert names.isdisjoint({"config_path", "tokenizer_path", "safetensor_idx_path"})


def test_base_localizes_model_path_so_every_subclass_inherits_it(monkeypatch):
    """Without this on the base, a composed model has to redo it for each module."""
    seen = []

    def fake_copy_to_local(path, **kwargs):
        seen.append(path)
        return f"/local/cache/{path.rsplit('/', 1)[-1]}"

    monkeypatch.setattr("veomni.utils.fs.copy_to_local", fake_copy_to_local)
    monkeypatch.setattr("veomni.utils.fs.is_non_local", lambda p: str(p).startswith("hdfs://"))

    runtime = ModelRuntimeArguments(model_path="hdfs://ns/ckpt/vision")

    assert seen == ["hdfs://ns/ckpt/vision"]
    assert runtime.model_path == "/local/cache/vision"


def test_model_arguments_still_localizes_through_super(monkeypatch):
    monkeypatch.setattr("veomni.utils.fs.copy_to_local", lambda path, **kw: f"/local/cache/{path.rsplit('/', 1)[-1]}")
    monkeypatch.setattr("veomni.utils.fs.is_non_local", lambda p: str(p).startswith("hdfs://"))

    args = ModelArguments(model_path="hdfs://ns/ckpt/qwen3")

    assert args.model_path == "/local/cache/qwen3"


def test_a_config_missing_both_paths_fails_before_any_download(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not try to download for a config that cannot be valid")

    monkeypatch.setattr("veomni.utils.fs.copy_to_local", explode)

    with pytest.raises(ValueError, match="`config_path` must be specified"):
        ModelArguments()


@pytest.fixture
def index_cache(monkeypatch):
    """Isolate the process-wide index cache so tests cannot leak into each other."""
    monkeypatch.setattr(BaseModelArguments, "_fqn_to_index_mapping_cache", {})


def _write_index(tmp_path, weight_map):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path


def test_index_mapping_is_derived_from_model_path_without_an_explicit_path(tmp_path, index_cache):
    _write_index(tmp_path, {"layer.weight": "model-00001-of-00002.safetensors"})

    runtime = ModelRuntimeArguments(model_path=str(tmp_path))

    assert runtime.fqn_to_index_mapping == {"layer.weight": 1}


def test_index_mapping_is_parsed_once_across_modules_sharing_a_checkpoint(tmp_path, index_cache, monkeypatch):
    """Sibling omni modules routinely point at the same checkpoint."""
    _write_index(tmp_path, {"layer.weight": "model-00001-of-00002.safetensors"})
    parses = []
    import veomni.models.checkpoint_tensor_loading as ctl

    real = ctl.parse_fqn_to_index_mapping_from_json
    monkeypatch.setattr(ctl, "parse_fqn_to_index_mapping_from_json", lambda p: (parses.append(p), real(p))[1])

    first = ModelRuntimeArguments(model_path=str(tmp_path))
    second = ModelRuntimeArguments(model_path=str(tmp_path))

    assert first.fqn_to_index_mapping == second.fqn_to_index_mapping
    assert len(parses) == 1


def test_index_mapping_is_not_read_until_asked_for(tmp_path, index_cache, monkeypatch):
    import veomni.models.checkpoint_tensor_loading as ctl

    monkeypatch.setattr(
        ctl,
        "parse_fqn_to_index_mapping_from_json",
        lambda p: pytest.fail("index should not be parsed at construction time"),
    )
    _write_index(tmp_path, {"layer.weight": "model-00001-of-00002.safetensors"})

    ModelRuntimeArguments(model_path=str(tmp_path))


def test_model_arguments_honours_an_explicit_index_path(tmp_path, index_cache):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_index(elsewhere, {"layer.weight": "model-00003-of-00004.safetensors"})
    _write_index(tmp_path, {"layer.weight": "model-00001-of-00002.safetensors"})

    args = ModelArguments(
        model_path=str(tmp_path),
        safetensor_idx_path=str(elsewhere / "model.safetensors.index.json"),
    )

    assert args.fqn_to_index_mapping == {"layer.weight": 3}


def test_a_checkpoint_without_an_index_still_warns(tmp_path, index_cache, monkeypatch):
    warnings = []
    monkeypatch.setattr(arguments_types.logger, "warning_once", lambda msg, *a, **k: warnings.append(msg))

    args = ModelArguments(model_path=str(tmp_path))

    assert args.fqn_to_index_mapping is None
    assert any("single file instead of sharded" in msg for msg in warnings)


def test_model_arguments_extends_the_runtime_shape():
    assert issubclass(ModelArguments, ModelRuntimeArguments)
    assert issubclass(ModelRuntimeArguments, BaseModelArguments)


def test_the_runtime_pair_is_declared_once_for_subclasses_to_inherit():
    """An omni module/model args class should not have to re-declare either field."""
    assert {"accelerator", "optimizer"}.isdisjoint({f.name for f in dataclasses.fields(BaseModelArguments)})

    own = ModelRuntimeArguments.__dataclass_fields__
    assert own["accelerator"].name == "accelerator"
    assert own["optimizer"].name == "optimizer"


def test_eager_is_reserved_but_refuses_to_run():
    """Accepting it silently would hand the model to DDP, the wrapper it promises to skip."""
    with pytest.raises(NotImplementedError, match="not wired up yet"):
        FSDPConfig(fsdp_mode="eager")


def test_unknown_fsdp_mode_still_names_the_supported_ones():
    with pytest.raises(ValueError, match="fsdp2.*ddp.*eager"):
        FSDPConfig(fsdp_mode="fsdp1")


def test_offload_pin_memory_defaults_to_torch_behaviour():
    assert FSDPConfig().offload_pin_memory is True


def test_grad_clip_scope_defaults_to_per_module():
    assert OptimizerConfig().grad_clip_scope == "per_module"


def test_global_grad_clip_scope_refuses_rather_than_silently_clipping_per_module():
    with pytest.raises(NotImplementedError, match="grad_clip_scope='global'"):
        OptimizerConfig(grad_clip_scope="global")


def test_chunk_mbs_validates_itself():
    with pytest.raises(ValueError, match="chunk_mbs must be >= 1"):
        ChunkMBSConfig(chunk_mbs=0)


@pytest.mark.parametrize(
    "key",
    [
        "init_device",
        "broadcast_model_weights_from_rank0",
        "ep_sharded_stream_load",
        "gradient_checkpointing",
        "torch_compile",
        "chunk_mbs_config",
    ],
)
def test_parser_points_a_relocated_key_at_its_new_home(key, world_size):
    world_size(1)
    value = {"enable": True} if key in ("gradient_checkpointing", "torch_compile", "chunk_mbs_config") else "meta"

    with pytest.raises(ValueError, match=rf"train\.{key} has moved to model\.accelerator\.{key}"):
        _instantiate_recursive(TrainingArguments, {key: value}, path="train")


@pytest.mark.parametrize("block", ["accelerator", "optimizer"])
def test_parser_points_a_relocated_block_at_model(block, world_size):
    """The whole block moved, so a config that still nests it under train must say so."""
    world_size(1)
    with pytest.raises(ValueError, match=rf"train\.{block} has moved to model\.{block}"):
        _instantiate_recursive(TrainingArguments, {block: {}}, path="train")


def test_parser_rejects_a_key_no_dataclass_declares(world_size):
    world_size(1)
    with pytest.raises(ValueError, match="model.accelerator.offload is not a field of AcceleratorConfig"):
        _instantiate_recursive(
            ModelArguments,
            {"accelerator": {"offload": {"enable_activation": True}}},
            path="model",
        )


def test_parser_names_the_valid_keys_when_it_rejects_one(world_size):
    world_size(1)
    with pytest.raises(ValueError, match="offload_config"):
        _instantiate_recursive(
            ModelArguments,
            {"accelerator": {"offload": {"enable_activation": True}}},
            path="model",
        )


def test_parser_still_accepts_the_new_paths(world_size):
    world_size(1)
    args = _instantiate_recursive(
        ModelArguments,
        {
            "config_path": "dummy",
            "accelerator": {
                "init_device": "meta",
                "gradient_checkpointing": {"enable": False},
                "chunk_mbs_config": {"chunk_mbs": 4},
            },
            "optimizer": {"lr": 3.0e-4},
        },
        path="model",
    )

    assert args.accelerator.init_device == "meta"
    assert args.accelerator.gradient_checkpointing.enable is False
    assert args.accelerator.chunk_mbs_config.chunk_mbs == 4
    assert args.optimizer.lr == pytest.approx(3.0e-4)
