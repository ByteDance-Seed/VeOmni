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

"""models_kernel auto / registry construct helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from tests.models_kernel.compare import eager_kernels_config
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.auto import build_foundation_model
from veomni.models_kernel.registry import MODELING_REGISTRY, get_model_class


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        architectures=["Qwen3ForCausalLM"],
        attn_implementation="eager",
    )


def test_get_model_class_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown Modeling"):
        get_model_class(_tiny_qwen3_config())


def test_get_model_class_hf_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    from transformers import AutoModelForCausalLM

    assert get_model_class(_tiny_qwen3_config()) is AutoModelForCausalLM


def test_build_foundation_model_requires_kernels_config():
    previous = get_kernels_config()
    try:
        set_kernels_config(None)
        with pytest.raises(ValueError, match="kernels_implementation"):
            build_foundation_model(_tiny_qwen3_config())
    finally:
        set_kernels_config(previous)


def test_build_foundation_model_installs_kernels_config():
    previous = get_kernels_config()
    cfg = eager_kernels_config()
    try:
        set_kernels_config(None)
        with pytest.raises(ValueError, match="Unknown Modeling"):
            build_foundation_model(_tiny_qwen3_config(), kernels_implementation=cfg)
        assert get_kernels_config() is cfg
    finally:
        set_kernels_config(previous)


def test_modeling_registry_starts_without_qwen3():
    assert "qwen3" not in MODELING_REGISTRY.valid_keys()


def test_auto_and_registry_do_not_import_ops_or_models():
    root = Path(__file__).resolve().parents[2] / "veomni" / "models_kernel"
    forbidden = []
    for path in (root / "auto.py", root / "registry.py", root / "loader.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name == "veomni.ops" or name.startswith("veomni.ops."):
                    forbidden.append((str(path), name))
                if name == "veomni.models" or (
                    name.startswith("veomni.models.") and not name.startswith("veomni.models_kernel")
                ):
                    forbidden.append((str(path), name))
    assert forbidden == []
