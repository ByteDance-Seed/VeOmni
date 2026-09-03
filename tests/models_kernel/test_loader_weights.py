# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""models_kernel weight I/O: empty init and save/load."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from veomni.models_kernel.checkpoint.weights import init_empty_weights, load_model_weights, save_model_weights


def test_init_empty_weights_puts_params_on_meta():
    with init_empty_weights():
        module = torch.nn.Linear(3, 3)
    assert module.weight.is_meta
    assert module.bias.is_meta


def test_save_and_load_roundtrip(tmp_path):
    src = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        src.weight.fill_(1.5)
        src.bias.fill_(0.25)

    save_model_weights(str(tmp_path), src.state_dict(), global_rank=0, save_dtype="float32")

    dst = torch.nn.Linear(4, 4, bias=True)
    dst.config = SimpleNamespace()
    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.zero_()
    load_model_weights(dst, str(tmp_path), init_device="cpu")

    assert torch.equal(dst.weight, src.weight)
    assert torch.equal(dst.bias, src.bias)


def test_weights_does_not_import_ops_or_models():
    path = Path(__file__).resolve().parents[2] / "veomni" / "models_kernel" / "checkpoint" / "weights.py"
    tree = ast.parse(path.read_text())
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if name == "veomni.ops" or name.startswith("veomni.ops."):
                forbidden.append(name)
            if name == "veomni.models" or (
                name.startswith("veomni.models.") and not name.startswith("veomni.models_kernel")
            ):
                forbidden.append(name)
    assert forbidden == []
