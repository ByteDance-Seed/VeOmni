from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.frozen_ple import freeze_ple_embeddings, is_ple_table


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


def test_training_arguments_validate_frozen_ple_dtype():
    """Accept supported freeze settings and reject invalid dtype combinations."""
    from veomni.arguments.arguments_types import TrainingArguments
    from veomni.trainer.vlm_trainer import VLMTrainingArguments

    enabled = TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="bfloat16")
    assert enabled.freeze_ple_embeddings is True
    assert enabled.frozen_ple_dtype == "bfloat16"
    assert VLMTrainingArguments(freeze_ple_embeddings=True).freeze_ple_embeddings is True

    with pytest.raises(ValueError, match="freeze_ple_embeddings"):
        TrainingArguments(frozen_ple_dtype="bfloat16")
    with pytest.raises(ValueError, match="bfloat16"):
        TrainingArguments(freeze_ple_embeddings=True, frozen_ple_dtype="float16")


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
    runtime._setup_lora = lambda: None
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
    """Preserve table trainability when the freeze option is disabled."""
    from veomni.models.model_runtime import VeOmniModelRuntime

    model = _qwen4_model()
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = model
    runtime.train_args = SimpleNamespace(freeze_ple_embeddings=False, frozen_ple_dtype=None)
    runtime._setup_lora = lambda: None
    runtime._freeze_model_module()
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert table.requires_grad


def test_freeze_ple_on_meta_preserves_dtype():
    """Freeze meta-initialized tables without materializing or casting them."""
    model = _qwen4_model().to("meta")
    table = model.ple.ple_embedding.ngram_embedding["shard_0"].weight
    assert freeze_ple_embeddings(model) == 1
    assert not table.requires_grad
    assert table.dtype == torch.float32
