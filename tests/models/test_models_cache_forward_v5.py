"""``use_cache=True`` forward smoke for transformers v5 generated modeling.

The sibling ``test_models_logits_equal_v5`` runs every forward with
``use_cache=False``, so the *cache* code paths in the patchgen-generated
modeling are never exercised by CI. That blind spot is exactly how a
transformers-version bump can silently break inference / generation: the
``DynamicCache`` API can change shape between releases (e.g. the 5.x migration
to the layered ``DynamicCache`` — ``has_previous_state(layer_idx)``,
``cache_params.layers[idx]``, ``update_conv_state`` / ``update_recurrent_state``
for linear attention) while the bitwise-equal forward test stays green.

This test closes the gap generically: for every v5 model it runs a single
``use_cache=True`` prefill and asserts the forward does not crash and the cache
is populated. It is a crash / cache-population smoke (VeOmni-only, no HF bitwise
compare) — which is precisely what catches cache-API drift on the next bump.

Cache families covered:

- Standard attention KV cache (``DynamicCache.get_seq_length()`` advances):
  qwen2, qwen3, qwen3_moe, deepseek_v3, glm_moe_dsa, and the VLM text towers.
- Linear-attention layered cache (gated DeltaNet ``conv_states`` /
  ``recurrent_states``): qwen3_5, qwen3_5_moe. Unlike the bitwise test we keep
  the real ``linear_attention`` layers (no full-attention override) so the
  ``update_conv_state`` / ``update_recurrent_state`` / ``_update_linear_attn_mask``
  paths actually fire — they need fla + GPU, so those cases skip otherwise.
"""

import copy
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from veomni.ops import apply_ops_config
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)
from veomni.utils.import_utils import is_package_available

# Reuse the sibling harness: toy-config path, input construction, dtype map.
from .test_models_logits_equal_v5 import (
    _DTYPE_MAP,
    _make_inputs,
    _release,
    _toy,
)


# Required by ``dist.init_process_group`` in the module-scoped fixture.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12357")

_FLA_AVAILABLE = is_package_available("fla")
_FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


@dataclass(frozen=True)
class CacheCase:
    case_id: str
    toy_config_dir: str
    arch: str
    kind: str  # "causal_lm" | "qwen3_5_text" | "vlm_full" | "omni_thinker"
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    # "standard" KV cache vs "linear" (gated DeltaNet conv/recurrent) cache.
    cache_family: str = "standard"
    needs_fla: bool = False
    forward_attr: Optional[str] = None  # e.g. "thinker" for Omni
    config_overrides: dict = field(default_factory=dict)


_EAGER_EXPERTS = {"_experts_implementation": "eager"}

CASES = [
    # ── standard attention KV cache ──────────────────────────────────────
    CacheCase("qwen2", _toy("qwen2_toy"), "Qwen2ForCausalLM", "causal_lm"),
    CacheCase("qwen3", _toy("qwen3_toy"), "Qwen3ForCausalLM", "causal_lm"),
    CacheCase("qwen3_moe", _toy("qwen3_moe_toy"), "Qwen3MoeForCausalLM", "causal_lm", config_overrides=_EAGER_EXPERTS),
    CacheCase(
        "deepseek_v3", _toy("deepseek_v3_toy"), "DeepseekV3ForCausalLM", "causal_lm", attn_implementation="eager"
    ),
    CacheCase(
        "glm_moe_dsa", _toy("glm_moe_dsa_toy"), "GlmMoeDsaForCausalLM", "causal_lm", config_overrides=_EAGER_EXPERTS
    ),
    # ── VLM full forward (text tower drives the KV cache) ────────────────
    CacheCase("qwen2_vl", _toy("qwen2vl_toy"), "Qwen2VLForConditionalGeneration", "vlm_full"),
    CacheCase("qwen2_5_vl", _toy("qwen25vl_toy"), "Qwen2_5_VLForConditionalGeneration", "vlm_full"),
    CacheCase("qwen3_vl", _toy("qwen3vl_toy"), "Qwen3VLForConditionalGeneration", "vlm_full"),
    CacheCase(
        "qwen3_vl_moe",
        _toy("qwen3vlmoe_toy"),
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        config_overrides=_EAGER_EXPERTS,
    ),
    # ── Omni (forward on ``model.thinker``; talker/token2wav stay dark) ───
    CacheCase(
        "qwen2_5_omni",
        _toy("qwen25omni_toy"),
        "Qwen2_5OmniForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        forward_attr="thinker",
    ),
    CacheCase(
        "qwen3_omni_moe",
        _toy("qwen3omni_toy"),
        "Qwen3OmniMoeForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        forward_attr="thinker",
        config_overrides=_EAGER_EXPERTS,
    ),
    # ── linear-attention layered cache (gated DeltaNet) ──────────────────
    # Keep the real linear_attention layers (no full-attention override): the
    # whole point is to drive update_conv_state / update_recurrent_state /
    # _update_linear_attn_mask, which need fla + GPU.
    # No ``_experts_implementation`` override: it is a no-op on the VeOmni path
    # (see _build_veomni_model); MoE stays eager via the all-eager ops config.
    CacheCase(
        "qwen3_5",
        _toy("qwen3_5_toy"),
        "Qwen3_5ForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        cache_family="linear",
        needs_fla=True,
    ),
    CacheCase(
        "qwen3_5_moe",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        cache_family="linear",
        needs_fla=True,
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """1-rank NCCL group for VeOmni's SP-aware attention wrappers."""
    if not IS_CUDA_AVAILABLE:
        yield
        return
    import torch.distributed as dist

    we_initialised = False
    if not dist.is_initialized():
        get_torch_device().set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend=get_dist_comm_backend(), rank=0, world_size=1)
        we_initialised = True
    try:
        yield
    finally:
        if we_initialised and dist.is_initialized():
            dist.destroy_process_group()


def _make_config(case: CacheCase):
    from transformers import AutoConfig

    full = AutoConfig.from_pretrained(case.toy_config_dir)
    if case.kind == "qwen3_5_text":
        # Text sub-config, but — unlike the bitwise test — keep the real
        # linear_attention layer_types so the gated-DeltaNet cache path runs.
        cfg = copy.deepcopy(full.text_config)
    else:
        cfg = full
    cfg.architectures = [case.arch]
    # MoE knobs (e.g. ``_experts_implementation``) live on the language
    # sub-config: ``cfg.text_config`` for VLMs, ``cfg.thinker_config.text_config``
    # for Omni, ``cfg`` itself for text-only models.
    override_target = cfg
    if case.kind == "vlm_full" and hasattr(cfg, "text_config"):
        override_target = cfg.text_config
    elif case.kind == "omni_thinker" and hasattr(cfg, "thinker_config"):
        override_target = getattr(cfg.thinker_config, "text_config", cfg.thinker_config)
    for k, v in case.config_overrides.items():
        setattr(override_target, k, v)
    return cfg


def _build_veomni_model(case: CacheCase, config):
    """Random-init VeOmni model (a smoke test needs no real weights)."""
    from veomni.models.auto import build_foundation_model

    from ..tools.training_utils import make_eager_ops_config

    if case.cache_family == "linear":
        # Everything eager EXCEPT the three gated-DeltaNet OpSlots, which have no
        # eager impl (the varlen path raises) and need fla. Starting from the
        # all-eager config keeps the rest of the model — crucially the MoE
        # experts (``moe_experts`` OpSlot, default ``fused_triton``) — off fused
        # kernels, so this case depends only on fla + flash_attn (the skip gates)
        # and stays a pure cache-path smoke. ``_experts_implementation`` would NOT
        # achieve this: the patched ``*MoeExperts`` dropped the HF decorator and
        # dispatches solely on the OpSlot.
        apply_ops_config(
            make_eager_ops_config(
                attn_implementation=case.attn_implementation,
                causal_conv1d_implementation="fla",
                chunk_gated_delta_rule_implementation="fla",
                rms_norm_gated_implementation="fla",
            )
        )
    else:
        apply_ops_config(make_eager_ops_config(attn_implementation=case.attn_implementation))

    model = build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    )
    return model.eval()


def _forward_target(model, case: CacheCase):
    # qwen3_5 text sub-config builds the *ForCausalLM directly; VLMs forward on
    # the top-level conditional-generation model; Omni forwards on ``.thinker``.
    if case.forward_attr is None:
        return model
    target = model
    for part in case.forward_attr.split("."):
        target = getattr(target, part)
    return target


def _assert_cache_populated(case: CacheCase, config, past_key_values, seq_len: int):
    assert past_key_values is not None, f"[{case.case_id}] use_cache=True returned no past_key_values"

    if case.cache_family == "linear":
        text_cfg = config if case.kind == "qwen3_5_text" else config.text_config
        n_linear = 0
        for i, lt in enumerate(text_cfg.layer_types):
            if lt != "linear_attention":
                continue
            layer = past_key_values.layers[i]
            assert past_key_values.has_previous_state(i), f"[{case.case_id}] layer {i}: has_previous_state False"
            assert getattr(layer, "conv_states", None) is not None, f"[{case.case_id}] layer {i}: conv_states None"
            assert getattr(layer, "recurrent_states", None) is not None, f"[{case.case_id}] layer {i}: recurrent None"
            n_linear += 1
        assert n_linear > 0, f"[{case.case_id}] no linear-attention layers exercised"
    else:
        seq = past_key_values.get_seq_length()
        assert seq == seq_len, f"[{case.case_id}] cache seq_length {seq} != prefill {seq_len}"


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_cache_forward_v5(case: CacheCase):
    """use_cache=True prefill: forward must not crash and the cache must fill."""
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    if case.needs_fla and not _FLA_AVAILABLE:
        pytest.skip("flash-linear-attention (fla) required for the linear-attention cache path.")
    if case.attn_implementation == "flash_attention_2" and not _FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn package not installed.")

    device = get_device_type()
    dtype = _DTYPE_MAP[case.dtype]
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)
    seq_len = input_ids.shape[-1]

    model = _build_veomni_model(case, config)
    try:
        with torch.no_grad():
            out = _forward_target(model, case)(input_ids=input_ids.clone(), use_cache=True, **fwd_kwargs)
        _assert_cache_populated(case, config, out.past_key_values, seq_len)
    finally:
        del model
        _release()
