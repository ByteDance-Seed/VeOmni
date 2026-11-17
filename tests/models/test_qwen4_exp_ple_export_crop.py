# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HF export crops only the Qwen4-Exp PLE runtime pad."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.row_crop import checkpoint_row_counts, crop_dense_state_dict
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import (
    create_qwen4_exp_checkpoint_tensor_converter,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


_PLE = "layers.0.ngram_embedding.shard_0.weight"


class _Ple(nn.Module):
    def __init__(self, logical_rows, padded_rows):
        super().__init__()
        self.rows_per_checkpoint_shard = logical_rows
        self.padded_rows_per_shard = padded_rows
        self.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(padded_rows, 4)})


class _Model(nn.Module):
    def __init__(self, logical_rows=10, padded_rows=16, layer_rows=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [_Ple(*rows) for rows in ([(logical_rows, padded_rows)] if layer_rows is None else layer_rows)]
        )
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                split_ngram_parts=4,
                make_ngram_vocab_size_divisible_by=8,
            )
        )
        self._create_checkpoint_tensor_converter = create_qwen4_exp_checkpoint_tensor_converter


def _padded_weight(logical_rows=10, padded_rows=16, cols=4):
    """Create a deterministic PLE weight with a zero-filled padding tail."""
    weight = torch.arange(padded_rows * cols, dtype=torch.float32).reshape(padded_rows, cols)
    weight[logical_rows:] = 0
    return weight


def test_crop_leaves_unpadded_and_non_ple_tensors():
    converter = create_qwen4_exp_checkpoint_tensor_converter(_Model())
    logical = _padded_weight()[:10]
    assert converter.crop_hf_export_tensor(_PLE, logical) is logical

    other = torch.zeros(16, 4)
    assert converter.crop_hf_export_tensor("model.embed_tokens.weight", other) is other


def test_crop_rejects_a_short_ple_shard():
    """Reject weights that cannot contain all logical PLE rows."""
    converter = create_qwen4_exp_checkpoint_tensor_converter(_Model())
    with pytest.raises(RuntimeError, match="shorter than logical rows"):
        converter.crop_hf_export_tensor(_PLE, torch.zeros(9, 4))


@pytest.mark.parametrize("ple_disabled", [False, True])
def test_models_without_ple_skip_cropping(ple_disabled):
    """Opt out for ordinary models and Qwen4-Exp with PLE disabled."""
    model = _Model(layer_rows=[]) if ple_disabled else nn.Linear(2, 2)
    weight = torch.zeros(3, 2)
    state = {"model.embed_tokens.weight": weight}
    assert crop_dense_state_dict(state, checkpoint_row_counts(model)) is state


def test_multiple_layers_keep_their_own_rows():
    """Crop per-table prefixes while preserving dtype, source weights, and other keys."""
    model = _Model(layer_rows=[(10, 16), (19, 24), (8, 8)])
    state = {
        f"layers.{index}.ngram_embedding.shard_0.weight": _padded_weight(logical, padded)
        for index, (logical, padded) in enumerate([(10, 16), (19, 24), (8, 8)])
    }
    original = {name: weight.clone() for name, weight in state.items()}
    other = torch.ones(2, 4)
    state["other.weight"] = other
    counts = checkpoint_row_counts(model)
    cropped = crop_dense_state_dict(state, counts)
    assert cropped["other.weight"] is other
    for name, (logical, _padded) in counts.items():
        weight = state[name]
        assert cropped[name].dtype == weight.dtype
        torch.testing.assert_close(weight, original[name], rtol=0, atol=0)
        assert cropped[name].shape == (logical, 4)
        torch.testing.assert_close(cropped[name], weight[:logical], rtol=0, atol=0)


def test_missing_ple_row_metadata_is_rejected():
    """Reject recognized PLE weights whose module omits the shape contract."""
    model = _Model()
    del model.layers[0].rows_per_checkpoint_shard
    with pytest.raises(RuntimeError, match="no checkpoint row metadata"):
        checkpoint_row_counts(model)
