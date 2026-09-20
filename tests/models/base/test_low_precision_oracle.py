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

"""BF16 FA2/SDPA HF oracles and disk-backed ``weights_path`` loading.

Family tests cover eager FP32 via in-memory ``load_state_dict``. This file
restores the model-level coverage those tests do not replace:

- An independent Hugging Face model in BF16 with FA2 or SDPA.
- ``build_foundation_model(weights_path=...)`` through safetensors for dense,
  merged-MoE, VL, and Omni checkpoints. A single Linear roundtrip is not enough.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
import torch
from transformers import PretrainedConfig

from tests.models.compare import qwen_image_inputs
from tests.models.tiny_configs import (
    tiny_glm_moe_dsa_config,
    tiny_qwen2_5_omni_thinker_config,
    tiny_qwen3_config,
    tiny_qwen3_moe_config,
    tiny_qwen3_vl_config,
)
from tests.tools.training_utils import make_eager_ops_config
from veomni.models import build_foundation_model
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    empty_cache,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12357")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Tiny BF16 FA2/SDPA vs Hugging Face. Dense full-attention is bitwise.
# MoE/DSA/vision accumulate BF16 ULP through expert loops or vision scatter.
_BF16_ATOL = 5e-2
_BF16_RTOL = 5e-2


@dataclass(frozen=True)
class Case:
    case_id: str
    arch: str
    kind: str
    config_factory: Callable[[], PretrainedConfig]
    hf_cls_factory: Callable[[], type]
    attn_implementation: str = "flash_attention_2"
    logits_equal: bool = False
    atol: float = _BF16_ATOL
    rtol: float = _BF16_RTOL
    config_overrides: dict = field(default_factory=dict)


def _hf_qwen3():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    return Qwen3ForCausalLM


def _hf_qwen3_moe():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM


def _hf_glm():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    return GlmMoeDsaForCausalLM


def _hf_qwen3_vl():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration


def _hf_qwen2_5_omni_thinker():
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    return Qwen2_5OmniThinkerForConditionalGeneration


_ORACLE_CASES = [
    Case(
        "qwen3-fa2",
        "Qwen3ForCausalLM",
        "causal_lm",
        tiny_qwen3_config,
        _hf_qwen3,
        logits_equal=True,
    ),
    Case(
        "qwen3_moe-fa2",
        "Qwen3MoeForCausalLM",
        "causal_lm",
        tiny_qwen3_moe_config,
        _hf_qwen3_moe,
        # Expert loops accumulate BF16 ULP; not the dense FA2 bitwise path.
        atol=5e-2,
        rtol=5e-2,
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "glm_moe_dsa-sdpa",
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        tiny_glm_moe_dsa_config,
        _hf_glm,
        attn_implementation="sdpa",
        # DSA + SDPA in BF16 is not the dense FA2 bitwise path; ~0.16 max abs.
        atol=0.2,
        rtol=0.2,
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "qwen3_vl-fa2",
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen3_vl_config,
        _hf_qwen3_vl,
        # Vision scatter mixes BF16 reductions; keep the measured family budget.
        atol=5e-2,
        rtol=5e-2,
    ),
    Case(
        "qwen2_5_omni-fa2",
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen2_5_omni_thinker_config,
        _hf_qwen2_5_omni_thinker,
        # Thinker + expert/vision mix; same measured family budget as Qwen3-VL.
        atol=5e-2,
        rtol=5e-2,
    ),
]

_LOADER_CASES = [
    Case("qwen3-fa2-loader", "Qwen3ForCausalLM", "causal_lm", tiny_qwen3_config, _hf_qwen3, logits_equal=True),
    Case(
        "qwen3_moe-fa2-loader",
        "Qwen3MoeForCausalLM",
        "causal_lm",
        tiny_qwen3_moe_config,
        _hf_qwen3_moe,
        atol=5e-2,
        rtol=5e-2,
        config_overrides={"_experts_implementation": "eager"},
    ),
    Case(
        "qwen3_vl-fa2-loader",
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen3_vl_config,
        _hf_qwen3_vl,
        atol=5e-2,
        rtol=5e-2,
    ),
    Case(
        "qwen2_5_omni-fa2-loader",
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen2_5_omni_thinker_config,
        _hf_qwen2_5_omni_thinker,
        atol=5e-2,
        rtol=5e-2,
    ),
]


@pytest.fixture(autouse=True)
def _deterministic_backend_flags():
    """Scope cuDNN flags so a frozen collection cannot be assigned into."""
    if not IS_CUDA_AVAILABLE:
        yield
        return

    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=False,
        benchmark_limit=torch.backends.cudnn.benchmark_limit,
        deterministic=True,
        allow_tf32=False,
    ):
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev_deterministic, warn_only=True)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
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


def _release() -> None:
    gc.collect()
    if IS_CUDA_AVAILABLE:
        empty_cache()


def _skip_if_unavailable(case: Case) -> None:
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")


def _make_config(case: Case) -> PretrainedConfig:
    config = case.config_factory()
    config.architectures = [case.arch]
    for key, value in case.config_overrides.items():
        setattr(config, key, value)
    return config


def _make_inputs(case: Case, config, device, dtype) -> tuple[torch.Tensor, dict]:
    vocab = min(getattr(config, "vocab_size", 128), getattr(getattr(config, "text_config", None), "vocab_size", 128))
    if case.kind == "causal_lm":
        input_ids = torch.randint(3, vocab, (1, 16), device=device)
        return input_ids, {}
    input_ids = torch.randint(3, 100, (2, 20), device=device)
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids").to(device)
    image.pop("labels")
    fwd = {}
    for key, value in image.items():
        if torch.is_tensor(value) and value.is_floating_point():
            fwd[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            fwd[key] = value.to(device)
        else:
            fwd[key] = value
    return ids, fwd


def _ops_config(case: Case):
    return make_eager_ops_config(attn_implementation=case.attn_implementation)


def _build_hf_model(case: Case, config, dtype: torch.dtype):
    cls = case.hf_cls_factory()
    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    with torch.device(get_device_type()):
        model = cls._from_config(config, torch_dtype=dtype, attn_implementation=case.attn_implementation)
    return model.eval()


def _build_veomni_model(case: Case, config, hf_state_dict):
    model = build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype="bfloat16",
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
        ops_implementation=_ops_config(case),
    )
    model.load_state_dict(hf_state_dict)
    return model.eval()


def _save_hf_checkpoint(state_dict: dict, config, dst_dir: str) -> None:
    from safetensors.torch import save_file

    config.save_pretrained(dst_dir)
    save_file(
        {key: value.detach().contiguous().cpu() for key, value in state_dict.items()},
        os.path.join(dst_dir, "model.safetensors"),
    )


def _build_veomni_model_from_disk(case: Case, config, hf_state_dict, hf_buffers, weights_dir: str):
    """Load through ``weights_path`` and restore non-persistent HF buffers.

    Rotary ``inv_freq`` is ``persistent=False``. Meta-init rematerializes it on
    CPU, which can differ from on-device init by one ULP. Restoring HF buffers
    keeps this a parameter-path check.
    """
    _save_hf_checkpoint(hf_state_dict, config, weights_dir)
    model = build_foundation_model(
        config_path=weights_dir,
        weights_path=weights_dir,
        config_kwargs=dict(case.config_overrides),
        torch_dtype="bfloat16",
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
        ops_implementation=_ops_config(case),
    )
    persistent_keys = set(hf_state_dict.keys())
    for name, ve_buf in model.named_buffers():
        if name in persistent_keys:
            continue
        src = hf_buffers.get(name)
        if src is None:
            continue
        ve_buf.copy_(src.to(ve_buf.device, dtype=ve_buf.dtype))
    return model.eval()


def _assert_logits(case: Case, logits_hf: torch.Tensor, logits_ve: torch.Tensor) -> None:
    assert logits_hf.shape == logits_ve.shape, (
        f"[{case.case_id}] shape mismatch: hf={tuple(logits_hf.shape)} ve={tuple(logits_ve.shape)}"
    )
    if case.logits_equal:
        if torch.equal(logits_hf, logits_ve):
            return
        diff = (logits_hf.float() - logits_ve.float()).abs()
        mismatched = logits_hf != logits_ve
        raise AssertionError(
            f"[{case.case_id}] logits not bitwise equal: "
            f"{int(mismatched.sum().item())}/{logits_hf.numel()} mismatched, "
            f"max_abs_diff={float(diff.max().item()):.3e}"
        )
    torch.testing.assert_close(logits_ve.float(), logits_hf.float(), atol=case.atol, rtol=case.rtol)


def _hf_logits(case: Case, config, input_ids, fwd_kwargs, dtype):
    model_hf = _build_hf_model(case, config, dtype)
    with torch.no_grad():
        logits_hf = model_hf(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
    state_dict = copy.deepcopy(model_hf.state_dict())
    buffers = {name: buf.detach().clone() for name, buf in model_hf.named_buffers()}
    del model_hf
    _release()
    return logits_hf, state_dict, buffers


@pytest.mark.parametrize("case", _ORACLE_CASES, ids=[case.case_id for case in _ORACLE_CASES])
def test_bf16_fa2_sdpa_logits_match_independent_hf(case: Case):
    _skip_if_unavailable(case)
    device = get_device_type()
    dtype = torch.bfloat16
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)
    logits_hf, state_dict, _buffers = _hf_logits(case, config, input_ids, fwd_kwargs, dtype)
    model_ve = _build_veomni_model(case, config, state_dict)
    with torch.no_grad():
        logits_ve = model_ve(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
    del model_ve, state_dict
    _release()
    _assert_logits(case, logits_hf, logits_ve)


@pytest.mark.parametrize("case", _LOADER_CASES, ids=[case.case_id for case in _LOADER_CASES])
def test_weights_path_loader_logits_match_independent_hf(case: Case):
    _skip_if_unavailable(case)
    device = get_device_type()
    dtype = torch.bfloat16
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)
    logits_hf, state_dict, buffers = _hf_logits(case, config, input_ids, fwd_kwargs, dtype)
    tmp_dir = tempfile.mkdtemp(prefix="veomni_loader_oracle_")
    try:
        model_ve = _build_veomni_model_from_disk(case, config, state_dict, buffers, tmp_dir)
        with torch.no_grad():
            logits_ve = model_ve(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs).logits.detach().clone()
        del model_ve
        _release()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del state_dict
        _release()
    _assert_logits(case, logits_hf, logits_ve)


_BACKWARD_CASES = [case for case in _ORACLE_CASES if case.kind == "causal_lm"]


def test_low_precision_oracle_scoped_flags_survive_a_frozen_cudnn_context():
    """Direct cuDNN assignment fails after a freeze; scoped flags must not."""
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=False,
        deterministic=True,
        allow_tf32=False,
    ):
        try:
            torch.backends.cudnn.deterministic = True
        except RuntimeError:
            assigned = False
        else:
            assigned = True
        if assigned:
            pytest.skip("this PyTorch build does not freeze cuDNN flags")
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=False,
            deterministic=True,
            allow_tf32=False,
        ):
            assert torch.backends.cudnn.deterministic is True


@pytest.mark.parametrize("case", _BACKWARD_CASES, ids=[case.case_id for case in _BACKWARD_CASES])
def test_bf16_causal_lm_backward_matches_independent_hf(case: Case):
    _skip_if_unavailable(case)
    device = get_device_type()
    dtype = torch.bfloat16
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)
    labels = input_ids.clone()
    model_hf = _build_hf_model(case, config, dtype).train()
    state_dict = copy.deepcopy(model_hf.state_dict())
    loss_hf = model_hf(input_ids=input_ids.clone(), labels=labels.clone(), use_cache=False, **fwd_kwargs).loss
    loss_hf.backward()
    hf_grads = {
        name: param.grad.detach().clone() for name, param in model_hf.named_parameters() if param.grad is not None
    }
    del model_hf
    _release()
    model_ve = _build_veomni_model(case, config, state_dict).train()
    loss_ve = model_ve(input_ids=input_ids.clone(), labels=labels.clone(), use_cache=False, **fwd_kwargs).loss
    loss_ve.backward()
    torch.testing.assert_close(loss_ve.float(), loss_hf.float(), atol=case.atol, rtol=case.rtol)
    shared = [
        (name, grad, model_ve.get_parameter(name).grad)
        for name, grad in hf_grads.items()
        if name in dict(model_ve.named_parameters())
    ]
    assert shared, f"[{case.case_id}] no shared parameter gradients"
    name, hf_grad, ve_grad = next(item for item in shared if item[2] is not None)
    torch.testing.assert_close(ve_grad.float(), hf_grad.float(), atol=case.atol, rtol=case.rtol, msg=name)
    del model_ve, state_dict
    _release()
