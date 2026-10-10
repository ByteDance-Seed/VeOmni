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
# See the License for the specific language governing permissions and
# limitations under the License.

"""BF16 FA2/SDPA HF oracles and disk-backed ``weights_path`` loading.

Family tests cover eager FP32 via in-memory ``load_state_dict``. This file
restores the model-level coverage those tests do not replace:

- An independent Hugging Face model in BF16 with FA2 or SDPA.
- ``build_foundation_model(weights_path=...)`` through safetensors for dense,
  merged-MoE, DSA, VL, and Omni checkpoints. A single Linear roundtrip is not
  enough.

Cases are bitwise by default. A case that is not bitwise names the measured
source of drift next to its budget.
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
    tiny_deepseek_v3_config,
    tiny_glm_moe_dsa_config,
    tiny_gpt_oss_config,
    tiny_qwen2_5_omni_thinker_config,
    tiny_qwen2_5_vl_config,
    tiny_qwen2_config,
    tiny_qwen2_vl_config,
    tiny_qwen3_5_config,
    tiny_qwen3_5_moe_config,
    tiny_qwen3_5_moe_text_config,
    tiny_qwen3_5_text_config,
    tiny_qwen3_config,
    tiny_qwen3_moe_config,
    tiny_qwen3_omni_moe_thinker_config,
    tiny_qwen3_vl_config,
    tiny_qwen3_vl_moe_config,
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

# Kinds: "causal_lm" (text ids), "qwen3_5_text" (text ids + cu_seq_lens_q),
# "vlm_full" / "omni_thinker" (packed images), "qwen3_5_vlm_full" (images +
# VeOmni-only empty cu_seq_lens_q).
_TEXT_KINDS = ("causal_lm", "qwen3_5_text")


@dataclass(frozen=True)
class Case:
    case_id: str
    arch: str
    kind: str
    config_factory: Callable[[], PretrainedConfig]
    hf_cls_factory: Callable[[], type]
    attn_implementation: str = "flash_attention_2"
    logits_equal: bool = True
    atol: float = 0.0
    rtol: float = 0.0
    grads_equal: bool = True
    grad_atol: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    config_overrides: dict = field(default_factory=dict)


def _hf(module: str, name: str) -> Callable[[], type]:
    def factory() -> type:
        return getattr(importlib.import_module(f"transformers.models.{module}"), name)

    return factory


def _qwen3_5_full_attention(factory: Callable[..., PretrainedConfig]) -> Callable[[], PretrainedConfig]:
    # GatedDeltaNet binds different fla / causal_conv1d kernels in HF and VeOmni.
    return lambda: factory(layer_types=["full_attention", "full_attention"])


def _qwen3_vl_moe_loader_config() -> PretrainedConfig:
    config = tiny_qwen3_vl_moe_config()
    # A square (E, 2I, H) gate_up is ambiguous between the HF and v5 layouts.
    config.text_config.moe_intermediate_size = 16
    return config


_EAGER_EXPERTS = {"_experts_implementation": "eager"}

# VeOmni's tensorized ``fast_pos_embed_interpolate`` patches (Qwen3-VL and
# Qwen3.5) cast the bilinear weights to BF16 before the four-way sum. Hugging
# Face 5.16.1 keeps them in FP32 (``get_vision_interpolation_indices_and_weights``).
# Measured max abs at |logits| <= 0.71 is 4.9e-3 on H20 (about one BF16 ULP)
# and 1.27e-2 on L20 (qwen3_vl_moe, 3/5120 elements, about three ULPs).
# Qwen3-Omni keeps the HF method and is bitwise.
_QWEN3_VL_POS_EMBED = {"logits_equal": False, "atol": 2e-2}

# ``veomni_sdpa`` pins EFFICIENT_ATTENTION for masked calls. HF's default
# dispatch picks cuDNN attention on H20. Forward is bitwise, but backward
# differs by up to 3.9e-3 (embed_tokens) on H20.
_SDPA_MASKED_BACKWARD = {"grads_equal": False, "grad_atol": 1e-2}

_ORACLE_CASES = [
    Case("qwen2-fa2", "Qwen2ForCausalLM", "causal_lm", tiny_qwen2_config, _hf("qwen2", "Qwen2ForCausalLM")),
    Case("qwen3-fa2", "Qwen3ForCausalLM", "causal_lm", tiny_qwen3_config, _hf("qwen3", "Qwen3ForCausalLM")),
    Case(
        "qwen3_moe-fa2",
        "Qwen3MoeForCausalLM",
        "causal_lm",
        tiny_qwen3_moe_config,
        _hf("qwen3_moe", "Qwen3MoeForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
    ),
    Case(
        "glm_moe_dsa-sdpa",
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        tiny_glm_moe_dsa_config,
        _hf("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        attn_implementation="sdpa",
        config_overrides=_EAGER_EXPERTS,
        **_SDPA_MASKED_BACKWARD,
    ),
    Case(
        "qwen3_5-text-fa2",
        "Qwen3_5ForCausalLM",
        "qwen3_5_text",
        _qwen3_5_full_attention(tiny_qwen3_5_text_config),
        _hf("qwen3_5", "Qwen3_5ForCausalLM"),
    ),
    Case(
        "qwen3_5_moe-text-sdpa",
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        _qwen3_5_full_attention(tiny_qwen3_5_moe_text_config),
        _hf("qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
        attn_implementation="sdpa",
        **_SDPA_MASKED_BACKWARD,
    ),
    Case(
        "qwen2_vl-fa2",
        "Qwen2VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen2_vl_config,
        _hf("qwen2_vl", "Qwen2VLForConditionalGeneration"),
    ),
    Case(
        "qwen2_5_vl-fa2",
        "Qwen2_5_VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen2_5_vl_config,
        _hf("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    ),
    Case(
        "qwen3_vl-fa2",
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen3_vl_config,
        _hf("qwen3_vl", "Qwen3VLForConditionalGeneration"),
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen3_vl_moe-fa2",
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        tiny_qwen3_vl_moe_config,
        _hf("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
        config_overrides=_EAGER_EXPERTS,
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen3_5_vl-sdpa",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_vlm_full",
        _qwen3_5_full_attention(tiny_qwen3_5_config),
        _hf("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        attn_implementation="sdpa",
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen3_5_moe_vl-sdpa",
        "Qwen3_5MoeForConditionalGeneration",
        "qwen3_5_vlm_full",
        _qwen3_5_full_attention(tiny_qwen3_5_moe_config),
        _hf("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        attn_implementation="sdpa",
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen2_5_omni-fa2",
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen2_5_omni_thinker_config,
        _hf("qwen2_5_omni", "Qwen2_5OmniThinkerForConditionalGeneration"),
    ),
    Case(
        "qwen3_omni_moe-fa2",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen3_omni_moe_thinker_config,
        _hf("qwen3_omni_moe", "Qwen3OmniMoeThinkerForConditionalGeneration"),
        config_overrides=_EAGER_EXPERTS,
    ),
]

_FP32_EAGER = {"attn_implementation": "eager", "dtype": torch.float32}

_LOADER_CASES = [
    Case(
        "qwen3_moe-eager-loader",
        "Qwen3MoeForCausalLM",
        "causal_lm",
        tiny_qwen3_moe_config,
        _hf("qwen3_moe", "Qwen3MoeForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
        **_FP32_EAGER,
    ),
    Case(
        "deepseek_v3-eager-loader",
        "DeepseekV3ForCausalLM",
        "causal_lm",
        tiny_deepseek_v3_config,
        _hf("deepseek_v3", "DeepseekV3ForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
        **_FP32_EAGER,
    ),
    Case(
        "gpt_oss-eager-loader",
        "GptOssForCausalLM",
        "causal_lm",
        tiny_gpt_oss_config,
        _hf("gpt_oss", "GptOssForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
        **_FP32_EAGER,
    ),
    Case(
        "glm_moe_dsa-eager-loader",
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        tiny_glm_moe_dsa_config,
        _hf("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
        **_FP32_EAGER,
    ),
    Case("qwen3-fa2-loader", "Qwen3ForCausalLM", "causal_lm", tiny_qwen3_config, _hf("qwen3", "Qwen3ForCausalLM")),
    Case(
        "qwen3_moe-fa2-loader",
        "Qwen3MoeForCausalLM",
        "causal_lm",
        tiny_qwen3_moe_config,
        _hf("qwen3_moe", "Qwen3MoeForCausalLM"),
        config_overrides=_EAGER_EXPERTS,
    ),
    Case(
        "glm_moe_dsa-sdpa-loader",
        "GlmMoeDsaForCausalLM",
        "causal_lm",
        tiny_glm_moe_dsa_config,
        _hf("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        attn_implementation="sdpa",
        config_overrides=_EAGER_EXPERTS,
    ),
    Case(
        "qwen3_vl-fa2-loader",
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        tiny_qwen3_vl_config,
        _hf("qwen3_vl", "Qwen3VLForConditionalGeneration"),
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen3_vl_moe-fa2-loader",
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        _qwen3_vl_moe_loader_config,
        _hf("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
        config_overrides=_EAGER_EXPERTS,
        **_QWEN3_VL_POS_EMBED,
    ),
    Case(
        "qwen2_5_omni-fa2-loader",
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen2_5_omni_thinker_config,
        _hf("qwen2_5_omni", "Qwen2_5OmniThinkerForConditionalGeneration"),
    ),
    Case(
        "qwen3_omni_moe-fa2-loader",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "omni_thinker",
        tiny_qwen3_omni_moe_thinker_config,
        _hf("qwen3_omni_moe", "Qwen3OmniMoeThinkerForConditionalGeneration"),
        config_overrides=_EAGER_EXPERTS,
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


def _make_inputs(case: Case, config, device, dtype) -> tuple[torch.Tensor, dict, dict]:
    """Return ``(input_ids, shared_kwargs, veomni_only_kwargs)``.

    VeOmni's Qwen3.5 decoder requires ``cu_seq_lens_q``. A single text row
    passes ``[0, seq_len]`` to both sides. Padded image batches pass an empty
    tensor to VeOmni only, since HF FA2 would treat it as packed varlen.
    """
    vocab = min(getattr(config, "vocab_size", 128), getattr(getattr(config, "text_config", None), "vocab_size", 128))
    if case.kind in _TEXT_KINDS:
        seq_len = 16
        input_ids = torch.randint(3, vocab, (1, seq_len), device=device)
        shared = {}
        if case.kind == "qwen3_5_text":
            shared["cu_seq_lens_q"] = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        return input_ids, shared, {}
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
    veomni_only = {}
    if case.kind == "qwen3_5_vlm_full":
        veomni_only["cu_seq_lens_q"] = torch.empty(0, dtype=torch.int32, device=device)
    return ids, fwd, veomni_only


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
        torch_dtype=str(case.dtype).removeprefix("torch."),
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
        torch_dtype=str(case.dtype).removeprefix("torch."),
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
    dtype = case.dtype
    config = _make_config(case)
    input_ids, fwd_kwargs, ve_kwargs = _make_inputs(case, config, device, dtype)
    logits_hf, state_dict, _buffers = _hf_logits(case, config, input_ids, fwd_kwargs, dtype)
    model_ve = _build_veomni_model(case, config, state_dict)
    with torch.no_grad():
        logits_ve = (
            model_ve(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs, **ve_kwargs).logits.detach().clone()
        )
    del model_ve, state_dict
    _release()
    _assert_logits(case, logits_hf, logits_ve)


@pytest.mark.parametrize("case", _LOADER_CASES, ids=[case.case_id for case in _LOADER_CASES])
def test_weights_path_loader_logits_match_independent_hf(case: Case):
    _skip_if_unavailable(case)
    device = get_device_type()
    dtype = case.dtype
    config = _make_config(case)
    input_ids, fwd_kwargs, ve_kwargs = _make_inputs(case, config, device, dtype)
    logits_hf, state_dict, buffers = _hf_logits(case, config, input_ids, fwd_kwargs, dtype)
    tmp_dir = tempfile.mkdtemp(prefix="veomni_loader_oracle_")
    try:
        model_ve = _build_veomni_model_from_disk(case, config, state_dict, buffers, tmp_dir)
        with torch.no_grad():
            logits_ve = (
                model_ve(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs, **ve_kwargs)
                .logits.detach()
                .clone()
            )
        del model_ve
        _release()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del state_dict
        _release()
    _assert_logits(case, logits_hf, logits_ve)


_BACKWARD_CASES = [case for case in _ORACLE_CASES if case.kind in _TEXT_KINDS]


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
    dtype = case.dtype
    config = _make_config(case)
    input_ids, fwd_kwargs, ve_kwargs = _make_inputs(case, config, device, dtype)
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
    loss_ve = model_ve(
        input_ids=input_ids.clone(), labels=labels.clone(), use_cache=False, **fwd_kwargs, **ve_kwargs
    ).loss
    loss_ve.backward()
    assert torch.equal(loss_ve, loss_hf), f"[{case.case_id}] loss {loss_ve.item()} != {loss_hf.item()}"
    ve_grads = {name: param.grad for name, param in model_ve.named_parameters() if param.grad is not None}
    assert hf_grads.keys() == ve_grads.keys(), (
        f"[{case.case_id}] gradient sets differ: "
        f"hf_only={sorted(hf_grads.keys() - ve_grads.keys())} ve_only={sorted(ve_grads.keys() - hf_grads.keys())}"
    )
    for name, hf_grad in hf_grads.items():
        ve_grad = ve_grads[name]
        if case.grads_equal:
            assert torch.equal(ve_grad, hf_grad), (
                f"[{case.case_id}] {name} grad not bitwise equal: "
                f"max_abs_diff={float((ve_grad.float() - hf_grad.float()).abs().max().item()):.3e}"
            )
        else:
            torch.testing.assert_close(ve_grad.float(), hf_grad.float(), atol=case.grad_atol, rtol=0, msg=name)
    del model_ve, state_dict
    _release()
