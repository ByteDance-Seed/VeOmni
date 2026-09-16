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

"""CPU tests for models op-construction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.models.utils.op_utils import attention_op, linear_bias, resolve_op_impl
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


@pytest.mark.parametrize("initially_configured", [False, True], ids=["none", "existing-config"])
@pytest.mark.parametrize("fail", [False, True], ids=["normal-exit", "exception"])
def test_ops_config_scope_restores_nested_bindings(initially_configured, fail):
    previous = get_ops_config()
    initial = eager_ops_config() if initially_configured else None
    inner = eager_ops_config()
    inner.cross_entropy_loss_implementation = "chunk_loss"
    with ops_config_scope(initial):
        assert get_ops_config() is initial

        def nested_scope():
            with ops_config_scope(inner):
                assert get_ops_config() is inner
                with ops_config_scope(None):
                    assert get_ops_config() is None
                assert get_ops_config() is inner
                # The scoped code may itself install another config (as model builders do).
                set_ops_config(eager_ops_config())
                if fail:
                    raise RuntimeError("construction failed")

        if fail:
            with pytest.raises(RuntimeError, match="construction failed"):
                nested_scope()
        else:
            nested_scope()
        assert get_ops_config() is initial
    assert get_ops_config() is previous


def test_resolve_op_impl_defaults_to_eager():
    set_ops_config(None)
    assert resolve_op_impl("rms_norm_implementation") == "eager"


def test_resolve_op_impl_reads_ops_config():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"))
    assert resolve_op_impl("cross_entropy_loss_implementation") == "chunk_loss"


def test_resolve_op_impl_remaps_npu_ce_alias():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="npu"))
    assert resolve_op_impl("cross_entropy_loss_implementation", npu_as="chunk_loss") == "chunk_loss"
    assert resolve_op_impl("cross_entropy_loss_implementation") == "npu"


def test_linear_bias_empty_sentinel():
    linear = torch.nn.Linear(4, 4, bias=False)
    bias = linear_bias(linear)
    assert bias.numel() == 0
    assert bias.device == linear.weight.device
    assert bias.dtype == linear.weight.dtype


def test_attention_op_defaults_to_eager():
    set_ops_config(None)
    op = attention_op()
    assert isinstance(op, VeomniOp)
    assert op.op == "attention"
    assert op.variant == "standard"
    assert op.impl == "eager"


def test_attention_op_reads_ops_config(available_nvidia_ops):
    set_ops_config(SimpleNamespace(attn_implementation="veomni_flash_attention_2"))
    op = attention_op()
    assert op.impl == "veomni_flash_attention_2"
    assert attention_op() is op


def test_resolve_moe_impl_reads_ops_config():
    from veomni.models.utils.op_utils import resolve_moe_impl

    set_ops_config(SimpleNamespace(moe_implementation="fused_triton"))
    assert resolve_moe_impl() == "fused_triton"
    set_ops_config(SimpleNamespace(moe_implementation="eager"))
    assert resolve_moe_impl() == "eager"


@pytest.mark.parametrize("impl", ["eager", "sdpa", "veomni_sdpa"])
def test_drop_packed_attention_metadata_strips_gdn_keys_for_sdpa_and_eager(impl):
    from veomni.models.utils.op_utils import drop_packed_attention_metadata

    kwargs = {
        "cu_seq_lens_q": torch.tensor([0, 32], dtype=torch.int32),
        "max_length_q": 32,
        "keep": 1,
    }
    filtered = drop_packed_attention_metadata(kwargs, impl=impl)
    assert "cu_seq_lens_q" not in filtered
    assert "max_length_q" not in filtered
    assert filtered["keep"] == 1
    assert "cu_seq_lens_q" in kwargs


def test_prepare_dense_attention_inputs_builds_packed_mask_for_multi_segment():
    from veomni.models.utils.op_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32), "keep": 1}
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    filtered, packed_mask = prepare_dense_attention_inputs(
        kwargs, impl="veomni_sdpa", attention_mask=attention_mask, hidden_states=hidden
    )
    assert "cu_seq_lens_q" not in filtered
    assert filtered["keep"] == 1
    assert packed_mask is not None
    assert packed_mask.shape[-2:] == (8, 8)
    # Token 4 (start of sample 1) must not see token 0 (sample 0).
    q4_k0 = packed_mask[0, 0, 4, 0] if packed_mask.dtype == torch.bool else packed_mask[0, 0, 4, 0]
    if packed_mask.dtype == torch.bool:
        assert not bool(q4_k0)
    else:
        assert q4_k0 < 0


def test_prepare_dense_attention_inputs_strips_single_segment_without_replacing_mask():
    from veomni.models.utils.op_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 8], dtype=torch.int32)}
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    filtered, mask = prepare_dense_attention_inputs(
        kwargs, impl="sdpa", attention_mask=attention_mask, hidden_states=hidden
    )
    assert "cu_seq_lens_q" not in filtered
    assert mask is attention_mask


def test_drop_packed_attention_metadata_keeps_keys_for_flash():
    from veomni.models.utils.op_utils import drop_packed_attention_metadata

    kwargs = {"cu_seq_lens_q": torch.tensor([0, 32], dtype=torch.int32)}
    assert drop_packed_attention_metadata(kwargs, impl="flash_attention_2") is kwargs
