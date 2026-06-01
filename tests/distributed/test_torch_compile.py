from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from veomni.arguments.arguments_types import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from veomni.distributed.torch_compile import (
    compile_module_forwards,
    mark_compile_step_begin,
    select_leaf_compile_modules,
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
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )

    assert compiled == 1
    assert block._veomni_forward_compiled is True
    assert block._veomni_original_forward is ToyBlock.forward
    assert calls == [{"backend": "inductor", "mode": "reduce-overhead", "fullgraph": False, "dynamic": False}]
    assert block(torch.ones(2, 4)).shape == (2, 4)


def test_compile_module_forwards_skips_duplicate_module(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_: fn)

    block = ToyBlock()
    compiled = compile_module_forwards([("a", block), ("b", block)])

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
    with pytest.raises(ValueError, match="train.enable_compile requires train.dyn_bsz=True"):
        VeOmniArguments(
            model=ModelArguments(config_path="dummy_config.json"),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(enable_compile=True, dyn_bsz=False, pad_to_length=False),
        )


def test_enable_compile_requires_padding_for_dynamic_batching():
    with pytest.raises(
        ValueError, match="train.enable_compile requires train.dyn_bsz=True and train.pad_to_length=True"
    ):
        VeOmniArguments(
            model=ModelArguments(config_path="dummy_config.json"),
            data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
            train=TrainingArguments(enable_compile=True, dyn_bsz=True, pad_to_length=False),
        )


def test_enable_compile_accepts_static_padded_dynamic_batching():
    args = VeOmniArguments(
        model=ModelArguments(config_path="dummy_config.json"),
        data=DataArguments(train_path="dummy.jsonl", max_seq_len=8),
        train=TrainingArguments(
            enable_compile=True,
            dyn_bsz=True,
            pad_to_length=True,
            micro_batch_size=2,
        ),
    )

    assert args.train.pad_to_length == 16
