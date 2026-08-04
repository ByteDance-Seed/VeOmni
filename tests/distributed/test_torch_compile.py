from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from veomni.arguments.arguments_types import (
    ChunkMBSConfig,
    DataArguments,
    GradientCheckpointingConfig,
    ModelArguments,
    OpsImplementationConfig,
    TrainingArguments,
    VeOmniArguments,
)
from veomni.arguments.arguments_types import (
    TorchCompileConfig as ArgumentsTorchCompileConfig,
)
from veomni.distributed.torch_compile import (
    CompileConfig,
    compile_decoder_blocks,
    mark_compile_step_begin,
    validate_compile_config_for_fsdp2,
)


def _model_args() -> ModelArguments:
    return ModelArguments(
        config_path="dummy_config.json",
        ops_implementation=OpsImplementationConfig(load_balancing_loss_implementation="eager"),
    )


class ToyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyModel(nn.Module):
    _no_split_modules = ["ToyDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([ToyDecoderLayer(), ToyDecoderLayer()])
        self.vision = ToyVisionBlock()
        self.lm_head = nn.Linear(4, 8)


def test_compile_decoder_blocks_compiles_only_decoder_layers(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)

        def wrapped(*args, **inner_kwargs):
            return fn(*args, **inner_kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = ToyModel()
    compiled = compile_decoder_blocks(
        model,
        CompileConfig(backend="inductor", mode="reduce-overhead", fullgraph=True, dynamic=False),
    )

    assert compiled == 2
    for layer in model.layers:
        assert layer._veomni_forward_compiled is True
        assert layer._veomni_original_forward is ToyDecoderLayer.forward
    assert not getattr(model.vision, "_veomni_forward_compiled", False)
    assert not getattr(model.lm_head, "_veomni_forward_compiled", False)
    assert not getattr(model.embed_tokens, "_veomni_forward_compiled", False)
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "inductor", "mode": "reduce-overhead"}] * 2
    assert model.layers[0](torch.ones(2, 4)).shape == (2, 4)


def test_compile_decoder_blocks_uses_no_split_modules(monkeypatch):
    calls = []

    class ToyOtherDecoderLayer(ToyDecoderLayer):
        pass

    class MixedDecoderModel(nn.Module):
        _no_split_modules = ["ToyDecoderLayer"]

        def __init__(self):
            super().__init__()
            self.selected = ToyDecoderLayer()
            self.unselected = ToyOtherDecoderLayer()

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = MixedDecoderModel()
    compiled = compile_decoder_blocks(model, CompileConfig())

    assert compiled == 1
    assert getattr(model.selected, "_veomni_forward_compiled", False)
    assert not getattr(model.unselected, "_veomni_forward_compiled", False)
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "inductor"}]


def test_compile_decoder_blocks_rejects_mode_with_cudagraphs_backend():
    with pytest.raises(ValueError, match="'cudagraphs' backend"):
        compile_decoder_blocks(
            ToyModel(),
            CompileConfig(backend="cudagraphs", mode="reduce-overhead"),
        )


def test_compile_decoder_blocks_accepts_cudagraphs_backend_without_mode(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    compile_decoder_blocks(
        ToyModel(),
        CompileConfig(backend="cudagraphs", mode=None, fullgraph=True, dynamic=False),
    )
    assert calls == [{"fullgraph": True, "dynamic": False, "backend": "cudagraphs"}] * 2


def test_compile_decoder_blocks_skips_already_compiled(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    model = ToyModel()
    first = compile_decoder_blocks(model, CompileConfig())
    second = compile_decoder_blocks(model, CompileConfig())

    assert first == 2
    assert second == 0


def test_compile_decoder_blocks_no_decoder_layers_returns_zero(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    class NoDecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = ToyVisionBlock()

    assert compile_decoder_blocks(NoDecoderModel(), CompileConfig()) == 0


def test_mark_compile_step_begin_calls_torch_compiler_api(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.distributed.torch_compile.IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)
    mark_compile_step_begin(enable_compile=False)

    assert calls == ["mark"]


def test_mark_compile_step_begin_skips_non_cuda(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.distributed.torch_compile.IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)

    assert calls == []


def test_mark_compile_step_begin_skips_without_torch_compiler(monkeypatch):
    monkeypatch.setattr("veomni.distributed.torch_compile.IS_CUDA_AVAILABLE", True)
    monkeypatch.delattr(torch, "compiler", raising=False)

    mark_compile_step_begin(enable_compile=True)


def test_compile_config_detects_cuda_graphs():
    assert CompileConfig(backend="inductor", mode=None).uses_cuda_graphs() is False
    assert CompileConfig(backend="inductor", mode="reduce-overhead").uses_cuda_graphs() is True
    assert CompileConfig(backend="cudagraphs", mode=None).uses_cuda_graphs() is True


def test_validate_compile_config_rejects_cuda_graphs_with_forward_reshard():
    with pytest.raises(RuntimeError, match="reshard_after_forward=False"):
        validate_compile_config_for_fsdp2(
            CompileConfig(enable=True, backend="inductor", mode="reduce-overhead"),
            enable_reshard_after_forward=True,
        )


def test_validate_compile_config_accepts_inductor_default_mode_with_forward_reshard():
    validate_compile_config_for_fsdp2(
        CompileConfig(enable=True, backend="inductor", mode=None),
        enable_reshard_after_forward=True,
    )


def test_validate_compile_config_accepts_cuda_graphs_without_forward_reshard():
    validate_compile_config_for_fsdp2(
        CompileConfig(enable=True, backend="inductor", mode="reduce-overhead"),
        enable_reshard_after_forward=False,
    )


def test_torch_compile_config_defaults():
    cfg = ArgumentsTorchCompileConfig()
    assert cfg.enable is False
    assert cfg.backend == "inductor"
    assert cfg.mode is None
    assert cfg.fullgraph is True
    assert cfg.dynamic is False


def test_enable_compile_requires_dynamic_batching():
    with pytest.raises(ValueError, match="train.torch_compile.enable requires train.dyn_bsz=True"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                torch_compile=ArgumentsTorchCompileConfig(enable=True), dyn_bsz=False, pad_to_length=False
            ),
        )


def test_enable_compile_rejects_chunk_mbs():
    with pytest.raises(ValueError, match="train.chunk_mbs_config.enable is not supported"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                torch_compile=ArgumentsTorchCompileConfig(enable=True),
                chunk_mbs_config=ChunkMBSConfig(enable=True),
                dyn_bsz=True,
                pad_to_length=False,
            ),
        )


def test_chunk_mbs_rejects_static_padding():
    with pytest.raises(ValueError, match="not supported with train.pad_to_length"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                chunk_mbs_config=ChunkMBSConfig(enable=True),
                dyn_bsz=True,
                pad_to_length=True,
                micro_batch_size=2,
            ),
        )


def test_chunk_mbs_rejects_reentrant_gradient_checkpointing():
    with pytest.raises(ValueError, match="requires non-reentrant gradient checkpointing"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                chunk_mbs_config=ChunkMBSConfig(enable=True),
                gradient_checkpointing=GradientCheckpointingConfig(enable_reentrant=True),
            ),
        )


def test_chunk_mbs_rejects_dpo_trainer():
    with pytest.raises(ValueError, match="not supported by the DPO trainer"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8, data_type="dpo"),
            train=TrainingArguments(chunk_mbs_config=ChunkMBSConfig(enable=True)),
        )


def test_dpo_trainer_rejects_chunk_mbs_regardless_of_data_type():
    from veomni.trainer.text_dpo_trainer import TextDPOTrainer

    args = SimpleNamespace(train=SimpleNamespace(chunk_mbs_config=ChunkMBSConfig(enable=True)))
    with pytest.raises(ValueError, match="not supported by the DPO trainer"):
        TextDPOTrainer(args)


def test_rl_trainer_rejects_chunk_mbs():
    from veomni.trainer.base_rl_trainer import BaseRLTrainer

    args = SimpleNamespace(train=SimpleNamespace(chunk_mbs_config=ChunkMBSConfig(enable=True)))
    trainer = BaseRLTrainer.__new__(BaseRLTrainer)
    trainer.args = args
    with pytest.raises(ValueError, match="not supported by RL trainers"):
        trainer._setup()


def test_enable_compile_requires_padding_for_dynamic_batching():
    with pytest.raises(
        ValueError, match="train.torch_compile.enable requires train.dyn_bsz=True and train.pad_to_length=True"
    ):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                torch_compile=ArgumentsTorchCompileConfig(enable=True), dyn_bsz=True, pad_to_length=False
            ),
        )


def test_enable_compile_accepts_static_padded_dynamic_batching():
    args = VeOmniArguments(
        model=_model_args(),
        data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            torch_compile=ArgumentsTorchCompileConfig(enable=True),
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16


@dataclass
class ToyMultimodalDataArguments(DataArguments):
    supports_torch_compile = False

    mm_configs: dict = field(default_factory=dict)


def test_enable_compile_rejects_multimodal_data_arguments():
    with pytest.raises(ValueError, match="text trainers only"):
        VeOmniArguments(
            model=_model_args(),
            data=ToyMultimodalDataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                torch_compile=ArgumentsTorchCompileConfig(enable=True),
                dyn_bsz=True,
                pad_to_length=True,
                micro_batch_size=2,
            ),
        )


@dataclass
class ToyTextDataArguments(DataArguments):
    extra_text_config: str = "text"


def test_enable_compile_accepts_text_data_argument_subclass():
    args = VeOmniArguments(
        model=_model_args(),
        data=ToyTextDataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            torch_compile=ArgumentsTorchCompileConfig(enable=True),
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16
