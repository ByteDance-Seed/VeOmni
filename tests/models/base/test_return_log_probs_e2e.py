# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Model-forward integration tests for fused log-probability outputs.

Construct Qwen3 and Qwen3-VL generated classes directly from toy configs;
both families use text-only inputs here. With labels and
``return_log_probs=True``, the model's loss helper returns log-probabilities
and entropy in ``output.fused_linear_aux``, with ``loss`` and ``logits`` set
to ``None``. Outputs match the label shape after next-token shifting, ignore-label
masking, and trailing zero padding.

The bitwise check compares against a full-logits forward using the same
token-level log-probability and entropy helpers, under deterministic settings
and with one chunk covering the packed batch. It tests model/loss integration
and projection boundaries, not independent kernel numerics. Additional cases
check the verl-style consumer fields, top-k distillation outputs, and gradients
to ``lm_head.weight``. Public registry/build coverage lives in
``tests/models/base/test_auto_registry.py``.
"""

import gc
import os

import pytest
import torch
import torch.nn.functional as F

from veomni.models.configuration_base import as_veomni_config
from veomni.utils.device import IS_CUDA_AVAILABLE, empty_cache, get_device_type


# Same env-var contract as the existing logits-equality test: keep GPU
# kernel patches gated off so the comparison hits the canonical path.
os.environ.setdefault("VEOMNI_USE_LIGER_KERNEL", "0")
os.environ.setdefault("VEOMNI_USE_FUSED_KERNELS", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12356")
# Required by torch.use_deterministic_algorithms for cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TOY_QWEN3 = os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3_toy")
TOY_QWEN3_VL = os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3vl_toy")
IGNORE_INDEX = -100


def _release():
    gc.collect()
    if IS_CUDA_AVAILABLE:
        empty_cache()


def _apply_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(autouse=True)
def _deterministic_backend_flags():
    """Scope mutable CUDA backend flags so test order cannot freeze or leak them."""
    if not IS_CUDA_AVAILABLE:
        yield
        return

    prev_deterministic = torch.are_deterministic_algorithms_enabled()
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


def _have_python_dev_headers() -> bool:
    """Triton JIT needs Python development headers to build its helper."""
    import sysconfig

    include = sysconfig.get_path("include")
    return include is not None and os.path.isfile(os.path.join(include, "Python.h"))


def _reference_log_probs_and_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a full-logits reference with the shared token-level helpers.

    Shift labels to next-token targets, evaluate FP32 logits, zero ignored
    positions, and pad the last output position. Sharing log-probability and
    entropy math with the fused path isolates model integration and chunked
    projection behavior; this is not an independent reference for those ops.
    """
    from veomni.models.loss_utils.chunk_logprobs import (
        _per_token_entropy_from_logits,
        _per_token_log_probs_from_logits,
    )

    shifted = labels[..., 1:].contiguous()
    sliced = logits[..., :-1, :].contiguous()
    flat = sliced.reshape(-1, sliced.size(-1)).float()
    target = shifted.reshape(-1)
    log_probs_flat = _per_token_log_probs_from_logits(flat, target, ignore_index)
    entropy_flat = _per_token_entropy_from_logits(flat)
    mask = target != ignore_index
    entropy_flat = torch.where(mask, entropy_flat, torch.zeros_like(entropy_flat))

    log_probs = log_probs_flat.view_as(shifted)
    entropy = entropy_flat.view_as(shifted)
    return F.pad(log_probs, (0, 1), value=0.0), F.pad(entropy, (0, 1), value=0.0)


def _build_model(toy_path: str, ce_impl: str = "chunk_loss"):
    """Construct a generated Qwen3 or Qwen3-VL class for forward-path tests.

    Install eager ops with the requested CE implementation during construction,
    restore the previous selection, and move the model to the current device
    in FP32. Direct construction isolates the generated forward/loss contract;
    public model registration and loading are tested separately.
    """
    from transformers import AutoConfig

    from tests.models.compare import eager_ops_config, pin_eager_attn_implementation

    config = AutoConfig.from_pretrained(toy_path)
    if config.model_type == "qwen3":
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM as ModelClass,
        )
    elif config.model_type == "qwen3_vl":
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLForConditionalGeneration as ModelClass,
        )
    else:
        raise ValueError(f"Unsupported toy model type: {config.model_type}")

    ops = eager_ops_config()
    ops.cross_entropy_loss_implementation = ce_impl
    as_veomni_config(config, validate=False)._set_ops_implementation(ops)
    model = ModelClass(config)
    pin_eager_attn_implementation(model)
    return model.to(device=get_device_type(), dtype=torch.float32)


def _skip_unless_cuda(toy_path: str):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(toy_path):
        pytest.skip(f"Path not found: {toy_path}")


_MODELS = [
    pytest.param(TOY_QWEN3, "qwen3", id="qwen3-text"),
    pytest.param(TOY_QWEN3_VL, "qwen3_vl", id="qwen3_vl-vlm"),
]


def _check_bitwise_matches_logits_reference(model: torch.nn.Module, family: str) -> None:
    bi_active = _have_python_dev_headers()
    bi_ctx = None
    if bi_active:
        from veomni.ops.batch_invariant import set_batch_invariant_mode

        bi_ctx = set_batch_invariant_mode(True)
        bi_ctx.__enter__()

    try:
        model.eval()
        torch.manual_seed(0)
        B, L = 2, 16
        input_ids = torch.randint(0, 1024, (B, L), device=model.device, dtype=torch.long)
        labels = input_ids.clone()
        labels[0, 0] = IGNORE_INDEX
        labels[1, ::5] = IGNORE_INDEX

        with torch.no_grad():
            ref_logits = model(input_ids=input_ids, use_cache=False).logits
            ref_log_probs, ref_entropy = _reference_log_probs_and_entropy_from_logits(ref_logits, labels)

            out = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                return_log_probs=True,
                chunk_size=B * L + 1,
            )
    finally:
        if bi_ctx is not None:
            bi_ctx.__exit__(None, None, None)

    assert out.loss is None, "loss must be None when return_log_probs=True"
    assert out.logits is None, "logits must be None when return_log_probs=True"
    assert out.fused_linear_aux.log_probs is not None, "log_probs must be populated when return_log_probs=True"
    assert out.fused_linear_aux.log_probs.shape == labels.shape, (
        f"shape mismatch: got {tuple(out.fused_linear_aux.log_probs.shape)} expected {tuple(labels.shape)}"
    )
    assert out.fused_linear_aux.log_probs.dtype == ref_log_probs.dtype, (
        f"dtype mismatch: got {out.fused_linear_aux.log_probs.dtype} expected {ref_log_probs.dtype}"
    )

    if not torch.equal(out.fused_linear_aux.log_probs, ref_log_probs):
        diff = (out.fused_linear_aux.log_probs - ref_log_probs).abs()
        ne = out.fused_linear_aux.log_probs != ref_log_probs
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"[{family}] per-token log_probs not bitwise equal: "
            f"{int(ne.sum().item())}/{out.fused_linear_aux.log_probs.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )

    assert out.fused_linear_aux.entropy is not None, f"[{family}] entropy must be populated when return_log_probs=True"
    assert out.fused_linear_aux.entropy.shape == labels.shape, (
        f"[{family}] entropy shape {tuple(out.fused_linear_aux.entropy.shape)} != labels shape {tuple(labels.shape)}"
    )
    if not torch.equal(out.fused_linear_aux.entropy, ref_entropy):
        diff = (out.fused_linear_aux.entropy - ref_entropy).abs()
        ne = out.fused_linear_aux.entropy != ref_entropy
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"[{family}] per-token entropy not bitwise equal: "
            f"{int(ne.sum().item())}/{out.fused_linear_aux.entropy.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )

    shifted_target_is_ign = F.pad(labels[..., 1:] == IGNORE_INDEX, (0, 1), value=True)
    masked_lp = out.fused_linear_aux.log_probs[shifted_target_is_ign]
    valid_lp = out.fused_linear_aux.log_probs[~shifted_target_is_ign]
    masked_ent = out.fused_linear_aux.entropy[shifted_target_is_ign]
    valid_ent = out.fused_linear_aux.entropy[~shifted_target_is_ign]
    assert torch.all(masked_lp == 0.0), (
        f"[{family}] IGN-target positions must emit 0.0 log_probs, got max_abs={masked_lp.abs().max().item():.3e}"
    )
    assert torch.all(masked_ent == 0.0), (
        f"[{family}] IGN-target positions must emit 0.0 entropy, got max_abs={masked_ent.abs().max().item():.3e}"
    )
    assert torch.all(valid_lp < 0), (
        f"[{family}] valid-target positions must emit negative log_probs, got max={valid_lp.max().item():.3e}"
    )
    assert torch.all(valid_ent > 0), (
        f"[{family}] valid-target positions must emit positive entropy, got min={valid_ent.min().item():.3e}"
    )


def _check_plain_verl_consumer_contract(model: torch.nn.Module, family: str) -> None:
    model.eval()
    torch.manual_seed(0)
    B, L = 2, 16
    input_ids = torch.randint(0, 1024, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX
    labels[1, ::5] = IGNORE_INDEX

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, use_cache=False, return_log_probs=True)

    assert out.loss is None, f"[{family}] loss must be None when return_log_probs=True"
    assert out.logits is None, f"[{family}] logits must be None when return_log_probs=True"
    assert out.fused_linear_aux.log_probs is not None, f"[{family}] fused_linear_aux.log_probs must be populated"
    assert out.fused_linear_aux.entropy is not None, f"[{family}] fused_linear_aux.entropy must be populated"
    assert out.fused_linear_aux.log_probs.shape == labels.shape, (
        f"[{family}] log_probs shape {tuple(out.fused_linear_aux.log_probs.shape)} != labels shape {tuple(labels.shape)}"
    )
    assert out.fused_linear_aux.entropy.shape == labels.shape, (
        f"[{family}] entropy shape {tuple(out.fused_linear_aux.entropy.shape)} != labels shape {tuple(labels.shape)}"
    )
    assert torch.isfinite(out.fused_linear_aux.log_probs).all(), f"[{family}] log_probs has non-finite values"
    assert torch.isfinite(out.fused_linear_aux.entropy).all(), f"[{family}] entropy has non-finite values"
    assert (out.fused_linear_aux.log_probs <= 0).all(), (
        f"[{family}] log_probs must be <= 0; got max={out.fused_linear_aux.log_probs.max().item():.3e}"
    )
    assert (out.fused_linear_aux.entropy >= 0).all(), (
        f"[{family}] entropy must be >= 0; got min={out.fused_linear_aux.entropy.min().item():.3e}"
    )


def _check_log_probs_backward(model: torch.nn.Module, family: str) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)

    B, L = 1, 8
    input_ids = torch.randint(0, 1024, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX

    out = model(input_ids=input_ids, labels=labels, use_cache=False, return_log_probs=True)
    log_probs = out.fused_linear_aux.log_probs
    mask = (labels != IGNORE_INDEX).float()
    scalar = (-log_probs * mask).sum() / mask.sum().clamp_min(1)
    scalar.backward()

    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None, f"[{family}] lm_head.weight.grad must be populated by backward"
    assert torch.isfinite(lm_head_grad).all(), f"[{family}] lm_head.weight.grad has non-finite values"
    assert lm_head_grad.abs().max().item() > 0, f"[{family}] lm_head.weight.grad is all zero"
    model.zero_grad(set_to_none=True)


def _check_topk_distill_forward_and_backward(model: torch.nn.Module, family: str) -> None:
    from veomni.models.loss_utils import chunk_topk_distill_function

    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)

    B, L, K = 2, 16, 4
    V = model.lm_head.weight.shape[0]
    input_ids = torch.randint(0, 1024, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX
    labels[1, ::5] = IGNORE_INDEX

    teacher_logits = torch.randn(B, L, V, device=model.device, dtype=torch.float32)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = teacher_log_probs.topk(K, dim=-1)
    teacher_topk_log_probs = teacher_topk_log_probs.contiguous()
    teacher_topk_ids = teacher_topk_ids.contiguous()

    out = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        return_log_probs=True,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )

    assert out.fused_linear_aux.distillation_losses is not None, f"[{family}] distillation_losses must be populated"
    assert out.fused_linear_aux.student_mass is not None, f"[{family}] student_mass must be populated"
    assert out.fused_linear_aux.teacher_mass is not None, f"[{family}] teacher_mass must be populated"
    for name, t in [
        ("distillation_losses", out.fused_linear_aux.distillation_losses),
        ("student_mass", out.fused_linear_aux.student_mass),
        ("teacher_mass", out.fused_linear_aux.teacher_mass),
    ]:
        assert t.shape == labels.shape, f"[{family}] {name} shape {tuple(t.shape)} != labels {tuple(labels.shape)}"
        assert torch.isfinite(t).all(), f"[{family}] {name} has non-finite values"

    assert (out.fused_linear_aux.distillation_losses >= 0).all(), (
        f"[{family}] distillation_losses must be >= 0; got min={out.fused_linear_aux.distillation_losses.min().item():.3e}"
    )
    assert (out.fused_linear_aux.student_mass >= 0).all() and (out.fused_linear_aux.student_mass <= 1 + 1e-5).all(), (
        f"[{family}] student_mass out of [0, 1]; "
        f"got [{out.fused_linear_aux.student_mass.min().item():.3e}, {out.fused_linear_aux.student_mass.max().item():.3e}]"
    )
    assert (out.fused_linear_aux.teacher_mass >= 0).all() and (out.fused_linear_aux.teacher_mass <= 1 + 1e-5).all(), (
        f"[{family}] teacher_mass out of [0, 1]; "
        f"got [{out.fused_linear_aux.teacher_mass.min().item():.3e}, {out.fused_linear_aux.teacher_mass.max().item():.3e}]"
    )
    assert not out.fused_linear_aux.student_mass.requires_grad, f"[{family}] student_mass must be detached"
    assert not out.fused_linear_aux.teacher_mass.requires_grad, f"[{family}] teacher_mass must be detached"

    with torch.no_grad():
        hidden_out = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    last_hidden = getattr(hidden_out, "hidden_states", None)
    if last_hidden is not None and len(last_hidden) >= 1:
        last_hidden_t = last_hidden[-1]
        _ref_lp, _ref_ent, ref_dist, ref_smass, ref_tmass = chunk_topk_distill_function(
            last_hidden_t,
            model.lm_head.weight,
            labels,
            teacher_topk_ids,
            teacher_topk_log_probs,
        )
        torch.testing.assert_close(
            out.fused_linear_aux.distillation_losses,
            ref_dist,
            rtol=0,
            atol=0,
            msg=lambda m: f"[{family}] distillation_losses != kernel direct: {m}",
        )
        torch.testing.assert_close(
            out.fused_linear_aux.student_mass,
            ref_smass,
            rtol=0,
            atol=0,
            msg=lambda m: f"[{family}] student_mass != kernel direct: {m}",
        )
        torch.testing.assert_close(
            out.fused_linear_aux.teacher_mass,
            ref_tmass,
            rtol=0,
            atol=0,
            msg=lambda m: f"[{family}] teacher_mass != kernel direct: {m}",
        )

    mask = (labels != IGNORE_INDEX).float()
    scalar = (out.fused_linear_aux.distillation_losses * mask).sum() / mask.sum().clamp_min(1)
    scalar.backward()
    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None, f"[{family}] lm_head.weight.grad must be populated by distillation backward"
    assert torch.isfinite(lm_head_grad).all(), f"[{family}] lm_head.weight.grad has non-finite values"
    assert lm_head_grad.abs().max().item() > 0, f"[{family}] lm_head.weight.grad is all zero"
    model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_return_log_probs_bitwise_matches_logits_reference(toy_path, family):
    """Build each generated model family once and run all four return_log_probs contracts."""
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    torch.manual_seed(0)
    model = _build_model(toy_path, ce_impl="chunk_loss")
    try:
        _check_bitwise_matches_logits_reference(model, family)
        _check_plain_verl_consumer_contract(model, family)
        _check_log_probs_backward(model, family)
        _check_topk_distill_forward_and_backward(model, family)
    finally:
        del model
        _release()
