"""CPU contracts for the DiT remove-padding API, without a built-in model opt-in."""

import importlib
import sys
from contextlib import nullcontext
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from torch import nn

import veomni.trainer.dit_trainer as dit_module
from veomni.arguments import ModelArguments, parse_args
from veomni.data.data_collator import MakeMicroBatchCollator
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTModelArguments,
    DiTTrainer,
    DiTTrainingArguments,
    VeOmniDiTArguments,
)


def _api():
    return importlib.import_module("veomni.models.diffusers.packing")


def _args(enabled=True, micro_batch_size=2):
    return VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused", use_remove_padding=enabled),
        data=DiTDataArguments(train_path="unused"),
        train=DiTTrainingArguments(
            training_task="offline_training", dyn_bsz=False, micro_batch_size=micro_batch_size, global_batch_size=4
        ),
    )


def test_dit_only_default_false():
    assert DiTModelArguments(config_path="unused").use_remove_padding is False
    assert "use_remove_padding" not in {f.name for f in fields(ModelArguments)}
    args = _args()
    assert args.model.use_remove_padding is True
    assert args.train.dyn_bsz is False
    assert args.train.gradient_accumulation_steps == 2


def test_yaml_cli_and_saved_config_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "dit.yaml"
    path.write_text(yaml.safe_dump(asdict(_args(enabled=False))))
    monkeypatch.setattr(sys, "argv", ["train_dit", str(path), "--model.use_remove_padding", "true"])
    parsed = parse_args(VeOmniDiTArguments)
    assert parsed.model.use_remove_padding is True
    assert parsed.train.micro_batch_size == 2
    assert parsed.train.dyn_bsz is False
    path.write_text(yaml.safe_dump(asdict(parsed)))
    monkeypatch.setattr(sys, "argv", ["train_dit", str(path)])
    assert parse_args(VeOmniDiTArguments).model.use_remove_padding is True
    config = yaml.safe_load(path.read_text())
    config["train"]["dyn_bsz"] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="dyn_bsz"):
        parse_args(VeOmniDiTArguments)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="use_remove_padding.*bool"):
        DiTModelArguments(config_path="unused", use_remove_padding=value)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("train.training_task", "offline_embedding", "offline_training"),
        ("train.training_task", "online_training", "offline_training"),
        ("train.dyn_bsz", True, "dyn_bsz"),
        ("train.bsz_warmup_ratio", 0.1, "warmup"),
        ("train.micro_batch_size", 0, "micro_batch_size"),
        ("train.global_batch_size", 3, "multiple"),
        ("data.dataloader.drop_last", False, "drop_last"),
        ("model.accelerator.ulysses_size", 2, "ulysses_size"),
        ("model.accelerator.cp_size", 2, "cp_size"),
        ("model.accelerator.tp_size", 2, "tp_size"),
        ("model.accelerator.pp_size", 2, "pp_size"),
        ("model.accelerator.extra_parallel_sizes", [2], "extra_parallel"),
        ("model.accelerator.fsdp_config.fsdp_mode", "ddp", "fsdp2"),
        ("model.accelerator.fsdp_config.offload", True, "offload"),
        ("model.accelerator.offload_config.enable_activation", True, "offload"),
        ("model.accelerator.offload_config.enable_async_activation", True, "offload"),
        ("model.accelerator.torch_compile.enable", True, "compile"),
        ("model.lora_config", {"r": 4}, "LoRA"),
    ],
)
def test_unsupported_execution_rejected_before_distributed_setup(monkeypatch, path, value, message):
    args = _args()
    owner = args
    for part in path.split(".")[:-1]:
        owner = getattr(owner, part)
    setattr(owner, path.split(".")[-1], value)
    setup = Mock(side_effect=AssertionError("must fail before distributed setup"))
    monkeypatch.setattr(dit_module.BaseTrainer, "_setup", setup)
    with pytest.raises(ValueError, match=message):
        DiTTrainer(args)
    setup.assert_not_called()
    # Unsupported combinations are irrelevant when the new option is off.
    args.model.use_remove_padding = False
    _api().validate_diffusion_remove_padding_config(args)


@pytest.mark.parametrize("field_name", ["micro_batch_size", "global_batch_size"])
@pytest.mark.parametrize("value", [0, -2, 2.0, 1.5, True, False])
@pytest.mark.parametrize("entrypoint", ["parse", "setup"])
def test_batch_sizes_require_positive_integers(tmp_path, monkeypatch, field_name, value, entrypoint):
    args = _args()
    setup = Mock(side_effect=AssertionError("must fail before distributed setup"))
    monkeypatch.setattr(dit_module.BaseTrainer, "_setup", setup)
    if entrypoint == "parse":
        config = asdict(args)
        config["train"][field_name] = value
        path = tmp_path / "dit.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys, "argv", ["train_dit", str(path)])
        with pytest.raises(ValueError, match="positive integer"):
            parse_args(VeOmniDiTArguments)
    else:
        setattr(args.train, field_name, value)
        with pytest.raises(ValueError, match="positive integer"):
            DiTTrainer(args)
    setup.assert_not_called()


def test_unspecified_global_batch_size_is_derived():
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused", use_remove_padding=True),
        data=DiTDataArguments(train_path="unused"),
        train=DiTTrainingArguments(training_task="offline_training", dyn_bsz=False, micro_batch_size=2),
    )
    assert args.train.global_batch_size == 2 * args.model.accelerator.dp_size
    assert args.train.gradient_accumulation_steps == 1


def test_setup_preserves_enabled_microbatch_and_disabled_behavior(monkeypatch):
    monkeypatch.setattr(dit_module.BaseTrainer, "_setup", lambda self: None)
    monkeypatch.setattr(dit_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=1))
    for enabled, expected in [(True, 2), (False, 1)]:
        args = _args(enabled)
        trainer = DiTTrainer.__new__(DiTTrainer)
        trainer.base = SimpleNamespace(args=args, _setup=lambda: None)
        trainer._setup()
        assert args.train.micro_batch_size == expected
        assert args.train.dataloader_batch_size == 4
        assert args.train.dyn_bsz is False
        if enabled:
            assert args.train.gradient_accumulation_steps == 2


class _Condition:
    supports_sample_inputs = True

    def prepare_samples(self, *, values):
        return [
            {"model_inputs": {"x": x}, "targets": {"x": torch.zeros_like(x)}, "metadata": {"index": i}}
            for i, x in enumerate(values)
        ]


class _Model(nn.Module):
    supports_remove_padding = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.configured = []

    def configure_remove_padding(self, *, attn_implementation):
        self.configured.append(attn_implementation)

    def forward(self, *, sample_inputs):
        predictions = [{"x": sample["model_inputs"]["x"] * self.weight} for sample in sample_inputs]
        losses = torch.stack(
            [(pred["x"] - sample["targets"]["x"]).square().mean() for pred, sample in zip(predictions, sample_inputs)]
        )
        return _api().DiffusionBatchOutput(sample_predictions=predictions, sample_losses={"mse": losses})


def _loader_trainer(monkeypatch, model_cls=_Model, condition_cls=_Condition, enabled=True):
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(args=_args(enabled))
    trainer.training_task = "offline_training"
    config = SimpleNamespace(model_type="test_dit", condition_model_type="test_condition", architectures=["TestDiT"])
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setattr(dit_module, "build_config", lambda *a, **kw: config)
    monkeypatch.setitem(dit_module.MODELING_REGISTRY._local_mapping, "test_dit", lambda arch: model_cls)
    monkeypatch.setitem(dit_module.MODELING_REGISTRY._local_mapping, "test_condition", lambda: condition_cls)
    monkeypatch.setattr(dit_module, "apply_ops_config", lambda cfg: None)
    condition_build = Mock(side_effect=lambda **kw: setattr(trainer, "condition_model", condition_cls()))
    trainer._build_condition_model = condition_build
    model_build = Mock(side_effect=model_cls)
    monkeypatch.setattr(dit_module, "build_foundation_model", lambda **kw: model_build())
    return trainer, model_build, condition_build


@pytest.mark.parametrize("which", ["model", "condition"])
def test_unsupported_pair_rejected_before_weights(monkeypatch, which):
    trainer, model_build, condition_build = _loader_trainer(
        monkeypatch,
        model_cls=nn.Linear if which == "model" else _Model,
        condition_cls=object if which == "condition" else _Condition,
    )
    with pytest.raises(ValueError, match="remove_padding|sample_inputs"):
        trainer._build_model()
    model_build.assert_not_called()
    condition_build.assert_not_called()


def test_startup_configures_model_once_and_disabled_is_unchanged(monkeypatch):
    for enabled in (False, True):
        trainer, _, _ = _loader_trainer(monkeypatch, enabled=enabled)
        trainer._build_model()
        backend = trainer.base.args.model.ops_implementation.attn_implementation
        assert trainer.base.model.configured == ([backend] if enabled else [])


def test_backend_error_from_model_hook_is_not_swallowed(monkeypatch):
    trainer, _, _ = _loader_trainer(monkeypatch)
    monkeypatch.setattr(_Model, "configure_remove_padding", Mock(side_effect=ValueError("unsupported backend")))
    with pytest.raises(ValueError, match="unsupported backend"):
        trainer._build_model()


def test_capability_flag_without_hook_is_rejected():
    incomplete = SimpleNamespace(supports_remove_padding=True)
    with pytest.raises(ValueError, match="configure_remove_padding"):
        _api().configure_diffusion_remove_padding(incomplete, enabled=True, attn_implementation="eager")
    _api().configure_diffusion_remove_padding(object(), enabled=False, attn_implementation="eager")


def test_unadapted_qwen_image_does_not_opt_in():
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    with pytest.raises(ValueError, match="remove_padding"):
        _api().validate_diffusion_remove_padding_support(QwenImageTransformer2DModel, _Condition)


def _forward_trainer(monkeypatch):
    monkeypatch.setattr(dit_module, "use_parallel_state", lambda name: nullcontext())
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.training_task = "offline_training"
    trainer.condition_model = _Condition()
    trainer.base = SimpleNamespace(
        args=_args(),
        model=_Model(),
        device="cpu",
        LOG_SAMPLE=False,
        num_micro_batches=2,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
    )
    return trainer


def test_trainer_forward_preserves_sample_weighting_gradients_and_root_hooks(monkeypatch):
    trainer = _forward_trainer(monkeypatch)
    calls = []
    trainer.base.model.register_forward_pre_hook(lambda *args: calls.append("forward"))
    batches = [
        [torch.tensor([1.0]), torch.tensor([2.0, 3.0, 4.0])],
        [torch.tensor([5.0, 6.0]), torch.tensor([7.0])],
    ]
    collator = MakeMicroBatchCollator(num_micro_batch=2, internal_data_collator=dit_module.DiTDataCollator())
    micro_batches = collator([({"values": x},) for batch in batches for x in batch])
    actual = sum(trainer.forward_backward_step(micro_batch)[0] for micro_batch in micro_batches)
    weight = torch.tensor(2.0, requires_grad=True)
    expected = torch.stack([(x * weight).square().mean() for batch in batches for x in batch]).mean()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(trainer.base.model.weight.grad, weight.grad)
    assert calls == ["forward", "forward"]
    assert trainer.base.model.configured == []  # Forward does not configure or mutate model execution.


def test_disabled_forward_uses_legacy_condition_and_loss(monkeypatch):
    trainer = _forward_trainer(monkeypatch)
    trainer.base.args.model.use_remove_padding = False
    model = nn.Linear(1, 1, bias=False)
    calls = []
    trainer.condition_model = SimpleNamespace(process_condition=lambda **kw: {"input": kw["values"][0]})

    class Legacy(nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(loss={"mse": model(**kwargs).square().mean()})

    trainer.base.model = Legacy()
    loss, _ = trainer.forward_backward_step({"values": [torch.tensor([[3.0]])]})
    assert set(calls[0]) == {"input"}
    assert loss.ndim == 0
    assert model.weight.grad is not None


@pytest.mark.parametrize("size", [0, 1, 3])
def test_trainer_rejects_partial_or_oversized_prepared_microbatch(monkeypatch, size):
    trainer = _forward_trainer(monkeypatch)
    with pytest.raises(ValueError, match="sample_inputs"):
        trainer.forward_backward_step({"values": [torch.ones(1)] * size})


def test_trainer_rejects_legacy_output_on_enabled_path(monkeypatch):
    trainer = _forward_trainer(monkeypatch)
    with pytest.raises(TypeError, match="DiffusionBatchOutput"):
        trainer.postforward(SimpleNamespace(loss={"mse": torch.tensor(1.0)}), {"sample_inputs": [{}, {}]})


@pytest.mark.parametrize("losses", [None, {}, [], {"mse": torch.tensor(1.0)}, {"mse": torch.ones(3)}])
def test_output_requires_one_loss_per_sample(losses):
    output = _api().DiffusionBatchOutput(sample_predictions=[{}, {}], sample_losses=losses)
    with pytest.raises(ValueError, match="sample_losses"):
        output.mean_losses(batch_size=2)


def test_output_checks_prediction_count_and_keeps_autograd():
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    output = _api().DiffusionBatchOutput(sample_predictions=[{}, {}], sample_losses={"mse": values})
    output.mean_losses(batch_size=2)["mse"].backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="sample_predictions"):
        output.mean_losses(batch_size=1)


@pytest.mark.parametrize("predictions", [None, [], [torch.ones(1), torch.ones(1)], torch.ones(2)])
def test_output_rejects_malformed_predictions(predictions):
    output = _api().DiffusionBatchOutput(sample_predictions=predictions, sample_losses={"mse": torch.ones(2)})
    with pytest.raises(ValueError, match="sample_predictions"):
        output.mean_losses(batch_size=2)


def test_sample_contract_uses_native_pytrees():
    samples = _Condition().prepare_samples(values=[torch.ones(2)])
    _api().validate_diffusion_samples(samples, batch_size=1)
    leaves, _ = torch.utils._pytree.tree_flatten(samples)
    assert sum(isinstance(x, torch.Tensor) for x in leaves) == 2
    del samples[0]["targets"]
    with pytest.raises(ValueError, match="targets"):
        _api().validate_diffusion_samples(samples, batch_size=1)
