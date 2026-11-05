from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.frozen_ple import freeze_ple_embeddings, is_ple_table, prepare_ple_storage


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _qwen4_model():
    model = nn.Module()
    model.config = type("Config", (), {"model_type": "qwen4_exp"})()
    model.ple = nn.Module()
    model.ple.ple_embedding = nn.Module()
    model.ple.ple_embedding.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(4, 2)})
    model.other = nn.Linear(2, 2)
    return model


def test_is_ple_table_matches_shard_weights_only():
    assert is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight")
    assert not is_ple_table("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.bias")
    assert not is_ple_table("other.weight")


def test_prepare_ple_storage_changes_only_the_table():
    model = _qwen4_model()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    other = model.other.weight
    table.data = table.data.to(torch.float32)
    other.data = other.data.to(torch.float32)
    prepare_ple_storage(model, "bfloat16")
    assert model.ple.ple_embedding.ngram_embedding["shard_0"].weight.dtype == torch.bfloat16
    assert model.other.weight.dtype == torch.float32


def test_freeze_rejects_other_model_types():
    model = nn.Linear(2, 2)
    model.config = type("Config", (), {"model_type": "llama"})()
    try:
        freeze_ple_embeddings(model)
    except ValueError as exc:
        assert "qwen4_exp" in str(exc)
    else:
        raise AssertionError("non-qwen4_exp models should be rejected")


def test_training_arguments_validate_frozen_ple_dtype():
    from veomni.arguments.arguments_types import TrainingArguments
    from veomni.trainer.vlm_trainer import VLMTrainingArguments

    enabled = TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="bfloat16")
    assert enabled.freeze_ple_embeddings is True
    assert enabled.frozen_ple_dtype == "bfloat16"
    assert VLMTrainingArguments(freeze_ple_embeddings=True).freeze_ple_embeddings is True

    try:
        TrainingArguments(frozen_ple_dtype="bfloat16")
    except ValueError as exc:
        assert "freeze_ple_embeddings" in str(exc)
    else:
        raise AssertionError("frozen_ple_dtype without freeze_ple_embeddings should be rejected")

    try:
        TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="float16")
    except ValueError as exc:
        assert "bfloat16" in str(exc)
    else:
        raise AssertionError("float16 is not a supported frozen PLE dtype")


def _build_runtime():
    from veomni.models.model_runtime import VeOmniModelRuntime

    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = _qwen4_model()
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=True, frozen_ple_dtype="bfloat16")
    from veomni.arguments.arguments_types import OptimizerConfig

    runtime.args = SimpleNamespace(optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0))
    runtime._apply_frozen_ple()
    runtime._build_optimizer()
    return runtime


def _train_step(runtime):
    model = runtime.model
    embedding = model.ple.ple_embedding.ngram_embedding["shard_0"]
    table = embedding.weight
    before_table = table.detach().clone()
    before_weight = model.other.weight.detach().clone()

    runtime.optimizer.zero_grad()
    model.other(embedding(torch.tensor([0, 1])).float()).square().mean().backward()
    runtime.optimizer.step()

    assert table.dtype == torch.bfloat16
    assert not table.requires_grad
    assert table.grad is None
    assert table not in runtime.optimizer.state
    assert all(table is not param for group in runtime.optimizer.param_groups for param in group["params"])
    torch.testing.assert_close(table, before_table, rtol=0, atol=0)
    assert not torch.equal(model.other.weight, before_weight)


def test_runtime_trains_and_resumes_with_frozen_ple(monkeypatch, tmp_path):
    monkeypatch.setattr("veomni.utils.device.get_device_type", lambda: "cpu")
    monkeypatch.setattr("veomni.optim.optimizer.get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))

    runtime = _build_runtime()
    _train_step(runtime)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": runtime.model.state_dict(), "optimizer": runtime.optimizer.state_dict()}, checkpoint_path)

    resumed = _build_runtime()

    def load_checkpoint():
        state = torch.load(checkpoint_path, weights_only=True)
        resumed.model.load_state_dict(state["model"])
        resumed.optimizer.load_state_dict(state["optimizer"])

    resumed.checkpoint = SimpleNamespace(load=load_checkpoint)
    resumed.load()
    for name, weight in resumed.model.state_dict().items():
        torch.testing.assert_close(weight, runtime.model.state_dict()[name], rtol=0, atol=0)

    _train_step(runtime)
    _train_step(resumed)
    for name, weight in resumed.model.state_dict().items():
        torch.testing.assert_close(weight, runtime.model.state_dict()[name], rtol=0, atol=0)


def test_runtime_leaves_ple_trainable_when_freeze_is_disabled():
    from veomni.models.model_runtime import VeOmniModelRuntime

    model = _qwen4_model()
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = model
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=False, frozen_ple_dtype=None)
    runtime._apply_frozen_ple()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert table.requires_grad
