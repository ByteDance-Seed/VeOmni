import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.frozen_ple import freeze_ple_embeddings, is_ple_table, prepare_ple_storage


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _qwen4_model():
    """Build a small model with one PLE table and a trainable projection."""
    model = nn.Module()
    model.config = type("Config", (), {"model_type": "qwen4_exp"})()
    model.ple = nn.Module()
    model.ple.ple_embedding = nn.Module()
    model.ple.ple_embedding.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(4, 2)})
    model.other = nn.Linear(2, 2)
    return model


def test_is_ple_table_matches_shard_weights_only():
    """Match PLE shard weights while excluding biases and unrelated parameters."""
    assert is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight")
    assert not is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.bias")
    assert not is_ple_table("other.weight")


def test_freeze_rejects_other_model_types():
    """Reject PLE freezing for model types outside the supported scope."""
    model = nn.Linear(2, 2)
    model.config = type("Config", (), {"model_type": "llama"})()
    with pytest.raises(ValueError, match="qwen4_exp"):
        freeze_ple_embeddings(model)


@pytest.mark.parametrize("freeze", [False, True])
def test_invalid_frozen_ple_dtype(freeze):
    """Reject a dtype without freezing or an unsupported dtype."""
    from veomni.trainer.vlm_trainer import VLMTrainingArguments

    dtype = "float16" if freeze else "bfloat16"
    with pytest.raises(ValueError, match="frozen_ple_dtype"):
        VLMTrainingArguments(freeze_ple_embeddings=freeze, frozen_ple_dtype=dtype)


def _build_runtime(dtype):
    """Build a CPU VLM runtime and verify dtype finalization after an upcast."""
    from veomni.trainer.vlm_trainer import VLMModelRuntime

    runtime = VLMModelRuntime.__new__(VLMModelRuntime)
    runtime.model = _qwen4_model()
    runtime.model_config = runtime.model.config
    runtime.train_args = SimpleNamespace(
        freeze_ple_embeddings=True, frozen_ple_dtype=dtype, freeze_vit=False, freeze_audio_tower=False
    )
    from veomni.arguments.arguments_types import OptimizerConfig

    runtime.args = SimpleNamespace(optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0), lora_config=None)
    runtime._freeze_model_module()
    table = runtime.model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert not table.requires_grad
    # Reproduce the upcast performed by the parallelization entry point.
    runtime.model.float()
    before = table.detach().clone()
    runtime._build_optimizer()
    expected_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    assert table is runtime.model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert table.dtype == expected_dtype
    torch.testing.assert_close(table, before.to(expected_dtype), rtol=0, atol=0)
    assert runtime.model.other.weight.dtype == torch.float32
    return runtime


def _train_step(runtime):
    """Update the projection and verify that the PLE table remains frozen."""
    model = runtime.model
    embedding = model.ple.ple_embedding.ngram_embedding["shard_0"]
    table = embedding.weight
    before_table = table.detach().clone()
    before_weight = model.other.weight.detach().clone()

    runtime.optimizer.zero_grad()
    model.other(embedding(torch.tensor([0, 1])).float()).square().mean().backward()
    runtime.optimizer.step()

    assert not table.requires_grad
    assert table.grad is None
    assert table not in runtime.optimizer.state
    assert all(table is not param for group in runtime.optimizer.param_groups for param in group["params"])
    torch.testing.assert_close(table, before_table, rtol=0, atol=0)
    assert not torch.equal(model.other.weight, before_weight)


@pytest.mark.parametrize("dtype", [None, "float32", "bfloat16"])
def test_runtime_trains_and_resumes_with_frozen_ple(monkeypatch, tmp_path, dtype):
    """Compare resumed and continuous updates for each supported table dtype."""
    monkeypatch.setattr("veomni.utils.device.get_device_type", lambda: "cpu")
    monkeypatch.setattr("veomni.utils.helper.get_device_type", lambda: "cpu")
    monkeypatch.setattr("veomni.optim.optimizer.get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))

    runtime = _build_runtime(dtype)
    _train_step(runtime)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": runtime.model.state_dict(), "optimizer": runtime.optimizer.state_dict()}, checkpoint_path)

    resumed = _build_runtime(dtype)

    def load_checkpoint():
        """Restore the saved model and optimizer through the runtime load entry."""
        state = torch.load(checkpoint_path, weights_only=True)
        resumed.model.load_state_dict(state["model"])
        resumed.optimizer.load_state_dict(state["optimizer"])

    resumed.checkpoint = SimpleNamespace(load=load_checkpoint)
    resumed.load()
    _train_step(runtime)
    _train_step(resumed)
    for name, weight in resumed.model.state_dict().items():
        torch.testing.assert_close(weight, runtime.model.state_dict()[name], rtol=0, atol=0)


def test_runtime_leaves_ple_trainable_when_freeze_is_disabled():
    """Preserve table trainability when the VLM freeze option is disabled."""
    from veomni.trainer.vlm_trainer import VLMModelRuntime

    model = _qwen4_model()
    runtime = VLMModelRuntime.__new__(VLMModelRuntime)
    runtime.model = model
    runtime.model_config = model.config
    runtime.args = SimpleNamespace(lora_config=None)
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=False, freeze_vit=False, freeze_audio_tower=False)
    runtime._freeze_model_module()
    assert model.ple.ple_embedding.ngram_embedding["shard_0"].weight.requires_grad


def test_freeze_ple_on_meta_preserves_dtype():
    """Freeze meta-initialized tables without materializing or casting them."""
    model = _qwen4_model().to("meta")
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert freeze_ple_embeddings(model) == 1
    assert not table.requires_grad
    assert table.dtype == torch.float32


@pytest.mark.parametrize("value", ['"false"', '"true"', "0", "null", "true", "false"])
def test_yaml_freeze_requires_boolean(value, monkeypatch, tmp_path):
    """Validate quoted and unquoted freeze settings through the YAML parser."""
    from veomni.arguments import parse_args
    from veomni.trainer.vlm_trainer import VLMTrainingArguments

    config = tmp_path / "freeze.yaml"
    config.write_text(
        f"freeze_ple_embeddings: {value}\n" + ("frozen_ple_dtype: bfloat16\n" if value == "true" else "")
    )
    monkeypatch.setattr(sys, "argv", ["test", str(config)])
    if value in ("true", "false"):
        assert parse_args(VLMTrainingArguments).freeze_ple_embeddings is (value == "true")
    else:
        with pytest.raises(ValueError, match="must be a boolean"):
            parse_args(VLMTrainingArguments)


@pytest.mark.parametrize("setting", ["freeze_ple_embeddings: true", "frozen_ple_dtype: bfloat16"])
def test_text_training_rejects_ple_options(setting, monkeypatch, tmp_path):
    """Reject VLM-only PLE options through the plain training YAML parser."""
    from veomni.arguments import parse_args
    from veomni.arguments.arguments_types import TrainingArguments

    config = tmp_path / "text.yaml"
    config.write_text(setting + "\n")
    monkeypatch.setattr(sys, "argv", ["test", str(config)])
    with pytest.raises(ValueError, match="Invalid configuration keys"):
        parse_args(TrainingArguments)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_dtensor_cast_preserves_local_shard_and_metadata(dtype, tmp_path):
    """Preserve DTensor layout, local values, identity, and ExtraParallel metadata."""
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        with torch.device("cpu"):
            model = _qwen4_model()
            mesh = init_device_mesh("cpu", (1,))
            embedding = model.ple.ple_embedding.ngram_embedding["shard_0"]
            source_dtype = torch.float32 if dtype == "bfloat16" else torch.bfloat16
            values = torch.arange(8, dtype=source_dtype).reshape(4, 2) / 8
            param = nn.Parameter(distribute_tensor(values, mesh, [Shard(0)]), requires_grad=False)
            param.spec_info = SimpleNamespace(para_name="ple")
            metadata = param.spec_info
            embedding.weight = param
            placements = param.placements

            prepare_ple_storage(model, dtype, expected_device="cpu")

            assert embedding.weight is param
            assert not param.requires_grad
            assert param.dtype == getattr(torch, dtype)
            assert param.to_local().dtype == getattr(torch, dtype)
            assert param.placements == placements
            assert param.device_mesh == mesh
            assert param.spec_info is metadata
            torch.testing.assert_close(param.to_local(), values.to(getattr(torch, dtype)), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
