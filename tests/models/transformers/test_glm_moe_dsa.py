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

"""GLM-MoE-DSA registry, parallel-plan, and Hugging Face parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM as HFGlmMoeDsaForCausalLM
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel as HFGlmMoeDsaModel

from tests.models.compare import (
    assert_eager_matches_hf,
    assert_outputs_and_grads_match,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_glm_moe_dsa_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_RTOL
from veomni.utils.device import IS_CUDA_AVAILABLE


def _glm_cls(architecture: str):
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.glm_moe_dsa.generated import patched_modeling_glm_moe_dsa_npu as gen
    else:
        from veomni.models.transformers.glm_moe_dsa.generated import patched_modeling_glm_moe_dsa_gpu as gen
    return getattr(gen, architecture)


def _build_ours(
    config: GlmMoeDsaConfig,
    ops: SimpleNamespace | None = None,
    architecture: str = "GlmMoeDsaForCausalLM",
):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _glm_cls(architecture)(config)


def _assert_dsa_wiring(attn) -> None:
    """GPU patches bind DSA ops; NPU generated modeling keeps the HF attention path."""
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        assert not hasattr(attn, "veomni_dsa_attention")
        assert attn.indexer is not None
        assert not hasattr(attn.indexer, "veomni_dsa_indexer")
        return
    assert attn.veomni_dsa_attention.variant == "glm"
    assert attn.indexer.veomni_dsa_indexer.variant == "glm"


def test_glm_moe_dsa_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config)
    ours = _build_ours(config)
    _assert_dsa_wiring(ours.model.layers[0].self_attn)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    # DSA indexer/sparse attention ULP (~1e-7); not bitwise vs Hugging Face.
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_glm_moe_dsa_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config("GlmMoeDsaModel")
    hf = HFGlmMoeDsaModel(config)
    ours = _build_ours(config, architecture="GlmMoeDsaModel")
    _assert_dsa_wiring(ours.layers[0].self_attn)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    gradient_weights = torch.randn(2, 8, config.hidden_size)
    assert_outputs_and_grads_match(
        hf,
        ours,
        lambda model: model(input_ids=input_ids, use_cache=False).last_hidden_state * gradient_weights,
    )


def test_glm_moe_dsa_registry_installs_checkpoint_hooks_and_ep_plan():
    from veomni.models import get_model_class

    for architecture in ("GlmMoeDsaForCausalLM", "GlmMoeDsaModel"):
        model_cls = get_model_class(_tiny_config(architecture))
        assert callable(model_cls._create_checkpoint_tensor_converter)
        assert callable(model_cls._convert_fqn_to_index_mapping)

    causal_cls = get_model_class(_tiny_config())
    ep_plan = causal_cls.get_parallel_plan(None).extra_parallel_plan["ep"]
    assert set(ep_plan) == {
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    }
    assert all(placement.dim == 0 for placement in ep_plan.values())


def test_glm_moe_dsa_chunked_prefill_and_independent_decode_matches_hf():
    """Compressed DSA KV lives on past_key_values, not module buffers."""
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config).eval()
    ours = _build_ours(config).eval()
    _assert_dsa_wiring(ours.model.layers[0].self_attn)
    ours.load_state_dict(hf.state_dict())
    assert not hasattr(ours.model.layers[0].self_attn, "_cached_k_pe")
    assert not hasattr(ours.model.layers[0].self_attn, "_cached_kv")

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    prefix = input_ids[:, :4]
    suffix = input_ids[:, 4:]
    independent_ids = torch.randint(3, config.vocab_size, (2, 1))

    with torch.no_grad():
        hf_prefix = hf(input_ids=prefix, use_cache=True)
        ours_prefix = ours(input_ids=prefix, use_cache=True)
        hf_chunk = hf(input_ids=suffix, past_key_values=hf_prefix.past_key_values, use_cache=True)
        ours_chunk = ours(input_ids=suffix, past_key_values=ours_prefix.past_key_values, use_cache=True)
        hf_full = hf(input_ids=input_ids, use_cache=False)
        ours_full = ours(input_ids=input_ids, use_cache=False)
        hf_independent = hf(input_ids=independent_ids, use_cache=True)
        ours_independent = ours(input_ids=independent_ids, use_cache=True)

    torch.testing.assert_close(ours_chunk.logits, hf_chunk.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(ours_chunk.logits, ours_full.logits[:, -2:], atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(hf_chunk.logits, hf_full.logits[:, -2:], atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(ours_independent.logits, hf_independent.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_glm_moe_dsa_cache_reorder_matches_hf():
    torch.manual_seed(1)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config).eval()
    ours = _build_ours(config).eval()
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 4))
    decode_ids = torch.randint(3, config.vocab_size, (2, 1))
    beam_idx = torch.tensor([1, 0])

    with torch.no_grad():
        hf_prefix = hf(input_ids=input_ids, use_cache=True)
        ours_prefix = ours(input_ids=input_ids, use_cache=True)
        hf_prefix.past_key_values.reorder_cache(beam_idx)
        ours_prefix.past_key_values.reorder_cache(beam_idx)
        reordered_decode_ids = decode_ids[beam_idx]
        hf_decode = hf(
            input_ids=reordered_decode_ids,
            past_key_values=hf_prefix.past_key_values,
            use_cache=True,
        )
        ours_decode = ours(
            input_ids=reordered_decode_ids,
            past_key_values=ours_prefix.past_key_values,
            use_cache=True,
        )
        hf_swapped = hf(input_ids=input_ids[beam_idx], use_cache=True)
        hf_swapped_decode = hf(
            input_ids=reordered_decode_ids,
            past_key_values=hf_swapped.past_key_values,
            use_cache=True,
        )

    torch.testing.assert_close(ours_decode.logits, hf_decode.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(ours_decode.logits, hf_swapped_decode.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)


class _FusedAttentionSpy:
    impl = "flashmla_cudnn"

    def __init__(self):
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return torch.zeros_like(args[3])


class _FusedIndexerSpy:
    impl = "cudnn"

    def __init__(self, topk: int):
        self.topk = topk
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        query = args[0]
        return torch.zeros(query.shape[0], query.shape[1], self.topk, dtype=torch.int32, device=query.device)


def test_glm_moe_dsa_fused_drops_standard_causal_mask_and_rejects_padding():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")
    config = _tiny_config()
    model = _build_ours(config)
    model.eval()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    attn_spy = _FusedAttentionSpy()
    indexer_spy = _FusedIndexerSpy(config.index_topk)
    model.model.layers[0].self_attn.veomni_dsa_attention = attn_spy
    model.model.layers[0].self_attn.indexer.veomni_dsa_indexer = indexer_spy
    model(input_ids=input_ids, use_cache=False)
    assert attn_spy.kwargs is not None and attn_spy.kwargs["attention_mask"] is None
    assert indexer_spy.kwargs is not None and indexer_spy.kwargs["attention_mask"] is None

    padded = torch.ones(2, 8, dtype=torch.long)
    padded[:, -2:] = 0
    with pytest.raises(ValueError, match="eager implementation"):
        model(input_ids=input_ids, attention_mask=padded, use_cache=False)


def test_glm_moe_dsa_fused_forward_uses_marked_causal_mask(monkeypatch):
    """The generated GLM path must drop a no-padding causal mask without scanning."""
    from veomni.ops.kernels.dsa import mask as mask_mod
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")

    def unexpected_scan(*args, **kwargs):
        pytest.fail("GLM fused forward must drop a marked standard causal mask without scanning")

    monkeypatch.setattr(mask_mod, "is_standard_causal_mask", unexpected_scan)
    config = _tiny_config()
    model = _build_ours(config)
    model.eval()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    attn_spy = _FusedAttentionSpy()
    indexer_spy = _FusedIndexerSpy(config.index_topk)
    model.model.layers[0].self_attn.veomni_dsa_attention = attn_spy
    model.model.layers[0].self_attn.indexer.veomni_dsa_indexer = indexer_spy
    model(input_ids=input_ids, use_cache=False)
    assert attn_spy.kwargs is not None and attn_spy.kwargs["attention_mask"] is None
    assert indexer_spy.kwargs is not None and indexer_spy.kwargs["attention_mask"] is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA GLM fused forward provenance")
def test_glm_moe_dsa_fused_cuda_forward_does_not_scan_standard_causal(monkeypatch):
    """GPU modeling must drop the generated causal mask without scanning it.

    HuggingFace ``create_causal_mask`` still does a packed-sequence ``.all()``
    host check. That is a separate HF-verbatim sync, not the DSA mask scan.
    """
    from veomni.ops.kernels.dsa import mask as mask_mod
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")

    def unexpected_scan(*args, **kwargs):
        pytest.fail("GLM fused CUDA forward must drop a marked standard causal mask without scanning")

    monkeypatch.setattr(mask_mod, "is_standard_causal_mask", unexpected_scan)
    config = _tiny_config()
    model = _build_ours(config).cuda()
    model.eval()
    input_ids = torch.randint(3, config.vocab_size, (2, 8), device="cuda")
    attn_spy = _FusedAttentionSpy()
    indexer_spy = _FusedIndexerSpy(config.index_topk)
    model.model.layers[0].self_attn.veomni_dsa_attention = attn_spy
    model.model.layers[0].self_attn.indexer.veomni_dsa_indexer = indexer_spy
    model(input_ids=input_ids, use_cache=False)
    assert attn_spy.kwargs is not None and attn_spy.kwargs["attention_mask"] is None
    assert indexer_spy.kwargs is not None and indexer_spy.kwargs["attention_mask"] is None


def test_glm_moe_dsa_eager_dropout_train_eval_seed_and_grads():
    config = _tiny_config()
    config.attention_dropout = 0.5
    model = _build_ours(config)
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    model.train()
    torch.manual_seed(1)
    train_a = model(input_ids=input_ids, use_cache=False).logits
    torch.manual_seed(2)
    train_b = model(input_ids=input_ids, use_cache=False).logits
    assert not torch.allclose(train_a, train_b)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        eval_a = model(input_ids=input_ids, use_cache=False).logits
        torch.manual_seed(2)
        eval_b = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(eval_a, eval_b)
    model.train()
    torch.manual_seed(3)
    logits = model(input_ids=input_ids, use_cache=False).logits
    grads = torch.autograd.grad(
        logits.sum(),
        [param for param in model.parameters() if param.requires_grad],
        allow_unused=True,
    )
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


def test_glm_moe_dsa_eager_output_attentions_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False, output_attentions=True)
    ours_out = ours(input_ids=input_ids, use_cache=False, output_attentions=True)
    assert ours_out.attentions is not None and ours_out.attentions != ()
    assert len(ours_out.attentions) == config.num_hidden_layers
    assert len(hf_out.attentions) == config.num_hidden_layers
    for ours_weights, hf_weights in zip(ours_out.attentions, hf_out.attentions, strict=True):
        assert ours_weights.shape == hf_weights.shape
        torch.testing.assert_close(ours_weights, hf_weights, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_glm_moe_dsa_fused_rejects_output_attentions():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")
    config = _tiny_config()
    model = _build_ours(config)
    model.model.layers[0].self_attn.veomni_dsa_attention = _FusedAttentionSpy()
    with pytest.raises(ValueError, match="output_attentions=True"):
        model(input_ids=torch.randint(3, config.vocab_size, (2, 8)), use_cache=False, output_attentions=True)


def test_glm_moe_dsa_fused_rejects_packed_position_ids():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")
    config = _tiny_config()
    model = _build_ours(config).eval()
    attn_spy = _FusedAttentionSpy()
    indexer_spy = _FusedIndexerSpy(config.index_topk)
    model.model.layers[0].self_attn.veomni_dsa_attention = attn_spy
    model.model.layers[0].self_attn.indexer.veomni_dsa_indexer = indexer_spy
    input_ids = torch.randint(3, config.vocab_size, (1, 8))
    position_ids = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    with pytest.raises(ValueError, match="eager implementation"):
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert attn_spy.kwargs is None
    assert indexer_spy.kwargs is None


def test_glm_moe_dsa_fused_rejects_gapped_position_ids():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        pytest.skip("NPU GLM keeps the Hugging Face attention path")
    config = _tiny_config()
    model = _build_ours(config).eval()
    attn_spy = _FusedAttentionSpy()
    indexer_spy = _FusedIndexerSpy(config.index_topk)
    model.model.layers[0].self_attn.veomni_dsa_attention = attn_spy
    model.model.layers[0].self_attn.indexer.veomni_dsa_indexer = indexer_spy
    input_ids = torch.randint(3, config.vocab_size, (1, 4))
    position_ids = torch.tensor([[0, 1, 4, 5]])
    with pytest.raises(ValueError, match="eager implementation"):
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert attn_spy.kwargs is None
    assert indexer_spy.kwargs is None
