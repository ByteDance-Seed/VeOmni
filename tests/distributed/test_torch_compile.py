from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from veomni.arguments.arguments_types import (
    DataArguments,
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
    compile_module_forwards,
    mark_compile_step_begin,
    select_leaf_compile_modules,
)


def _model_args() -> ModelArguments:
    return ModelArguments(
        config_path="dummy_config.json",
        ops_implementation=OpsImplementationConfig(load_balancing_loss_implementation="eager"),
    )


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


def test_compile_module_forwards_replaces_forward(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)

        def wrapped(*args, **inner_kwargs):
            return fn(*args, **inner_kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    block = ToyBlock()
    compiled = compile_module_forwards(
        [("layers.0", block)],
        CompileConfig(backend="inductor", mode="reduce-overhead", fullgraph=False, dynamic=False),
    )

    assert compiled == 1
    assert block._veomni_forward_compiled is True
    assert block._veomni_original_forward is ToyBlock.forward
    assert calls == [{"backend": "inductor", "mode": "reduce-overhead", "fullgraph": False, "dynamic": False}]
    assert block(torch.ones(2, 4)).shape == (2, 4)


def test_compile_module_forwards_skips_duplicate_module(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    block = ToyBlock()
    compiled = compile_module_forwards([("a", block), ("b", block)], CompileConfig())

    assert compiled == 1


def test_mark_compile_step_begin_calls_torch_compiler_api(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.distributed.torch_compile.get_device_type", lambda: "cuda")
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)
    mark_compile_step_begin(enable_compile=False)

    assert calls == ["mark"]


def test_mark_compile_step_begin_skips_non_cuda(monkeypatch):
    calls = []

    monkeypatch.setattr("veomni.distributed.torch_compile.get_device_type", lambda: "npu")
    monkeypatch.setattr(torch, "compiler", SimpleNamespace(cudagraph_mark_step_begin=lambda: calls.append("mark")))

    mark_compile_step_begin(enable_compile=True)

    assert calls == []


def test_mark_compile_step_begin_skips_without_torch_compiler(monkeypatch):
    monkeypatch.setattr("veomni.distributed.torch_compile.get_device_type", lambda: "cuda")
    monkeypatch.delattr(torch, "compiler", raising=False)

    mark_compile_step_begin(enable_compile=True)


def test_select_leaf_compile_modules_skips_parent_targets():
    parent = nn.Sequential(ToyBlock())
    sibling = ToyBlock()

    selected = select_leaf_compile_modules(
        [
            ("", nn.Sequential(parent, sibling)),
            ("layers.0", parent),
            ("layers.0.mlp", parent[0]),
            ("layers.1", sibling),
        ]
    )

    assert selected == [("layers.0.mlp", parent[0]), ("layers.1", sibling)]


def test_select_leaf_compile_modules_keeps_root_when_it_is_only_target():
    root = nn.Sequential(ToyBlock())

    assert select_leaf_compile_modules([("", root)]) == [("", root)]


def test_enable_compile_requires_dynamic_batching():
    with pytest.raises(ValueError, match="train.torch_compile.enable requires train.dyn_bsz=True"):
        VeOmniArguments(
            model=_model_args(),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(
                torch_compile=ArgumentsTorchCompileConfig(enable=True), dyn_bsz=False, pad_to_length=False
            ),
        )


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


def test_enable_compile_legacy_alias_sets_compile_enable():
    args = VeOmniArguments(
        model=_model_args(),
        data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(enable_compile=True, dyn_bsz=True, pad_to_length=True, micro_batch_size=2),
    )

    assert args.train.torch_compile.enable is True
    assert args.train.pad_to_length == 16


def test_enable_compile_legacy_option_aliases_set_torch_compile_config():
    args = VeOmniArguments(
        model=_model_args(),
        data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            enable_compile=True,
            compile_backend="eager",
            compile_mode="default",
            compile_fullgraph=True,
            compile_dynamic=True,
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.torch_compile.enable is True
    assert args.train.torch_compile.backend == "eager"
    assert args.train.torch_compile.mode == "default"
    assert args.train.torch_compile.fullgraph is True
    assert args.train.torch_compile.dynamic is True
