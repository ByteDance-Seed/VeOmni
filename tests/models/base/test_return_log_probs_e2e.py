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
    from veomni.ops.config import get_ops_config, set_ops_config

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
    previous = get_ops_config()
    set_ops_config(ops)
    try:
        model = ModelClass(config)
    finally:
        set_ops_config(previous)
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


@pytest.mark.parametrize("toy_path,family", _MODELS)
@pytest.mark.parametrize(
    "ce_impl",
    [
        pytest.param("chunk_loss", id="chunk_loss"),
        pytest.param("eager", id="eager"),
    ],
)
def test_return_log_probs_bitwise_matches_logits_reference(ce_impl, toy_path, family):
    """Compare fused auxiliary outputs against a full-logits forward bitwise.

    Exercise eager and chunk-loss selections on text-only Qwen3 and Qwen3-VL
    inputs. The reference shares the token-level log-probability and entropy
    helpers with the model's loss path. Deterministic backend settings and
    optional batch-invariant mode control rounding; a chunk covering the whole
    packed batch avoids a different projection boundary. Also check next-token
    label alignment, ignore-label masking, and trailing zero padding.
    """
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    # Batch-invariant mode patches mm/addmm/log_softmax via Triton, which
    # requires the Python development headers (``Python.h``) to JIT its
    # CUDA introspection helper. Toggle it on iff those headers are
    # present; otherwise fp32 + deterministic algorithms is sufficient
    # for bitwise equality here, since both paths call into the same
    # aten op with identical inputs.
    bi_active = _have_python_dev_headers()
    bi_ctx = None
    if bi_active:
        from veomni.ops.batch_invariant import set_batch_invariant_mode

        bi_ctx = set_batch_invariant_mode(True)
        bi_ctx.__enter__()

    try:
        torch.manual_seed(0)
        model = _build_model(toy_path, ce_impl=ce_impl).eval()

        B, L = 2, 16
        # Keeping token IDs below 32000 avoids the multimodal placeholder ids
        # (image_token_id, video_token_id, ...) used by VLM configs;
        # this keeps the forward on the text-only path so we can compare
        # bitwise against the lm_head reference.
        input_ids = torch.randint(0, 32000, (B, L), device=model.device, dtype=torch.long)
        labels = input_ids.clone()
        labels[0, 0] = IGNORE_INDEX
        labels[1, ::5] = IGNORE_INDEX

        with torch.no_grad():
            # Reference path: full-logits forward + reference log-probs / entropy.
            ref_logits = model(input_ids=input_ids, use_cache=False).logits
            ref_log_probs, ref_entropy = _reference_log_probs_and_entropy_from_logits(ref_logits, labels)

            # The generated forward returns the loss helper's log-probabilities
            # and entropy in fused_linear_aux, leaving loss and logits None.
            # ``chunk_size=B*L+1`` forces a single chunk over the whole packed
            # batch so the matmul
            # boundary matches the reference forward exactly — using
            # ``L+1`` alone would split B=2 into 2 chunks and surface
            # fp32 epsilon drift from cuBLAS algorithm selection at the
            # chunk boundary (qwen3_vl-vlm hits this; text doesn't).
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
            f"[{family}/{ce_impl}] per-token log_probs not bitwise equal: "
            f"{int(ne.sum().item())}/{out.fused_linear_aux.log_probs.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )

    # Entropy contract: same shape as log_probs, populated, bitwise equal
    # to the reference (same ``_per_token_entropy_from_logits`` helper on
    # the same fp32 logits).
    assert out.fused_linear_aux.entropy is not None, (
        f"[{family}/{ce_impl}] entropy must be populated when return_log_probs=True"
    )
    assert out.fused_linear_aux.entropy.shape == labels.shape, (
        f"[{family}/{ce_impl}] entropy shape {tuple(out.fused_linear_aux.entropy.shape)} != labels shape {tuple(labels.shape)}"
    )
    if not torch.equal(out.fused_linear_aux.entropy, ref_entropy):
        diff = (out.fused_linear_aux.entropy - ref_entropy).abs()
        ne = out.fused_linear_aux.entropy != ref_entropy
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"[{family}/{ce_impl}] per-token entropy not bitwise equal: "
            f"{int(ne.sum().item())}/{out.fused_linear_aux.entropy.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )

    # IGNORE_INDEX masking contract: the kernel emits exactly 0 wherever
    # the shifted target is IGN (and at the trailing pad position). The
    # kernel predicts ``labels[t+1]`` from ``hidden[t]``, so an IGN at
    # ``labels[k]`` zeros output position ``k-1``. Both log_probs and
    # entropy follow the same masking contract.
    shifted_target_is_ign = F.pad(labels[..., 1:] == IGNORE_INDEX, (0, 1), value=True)
    masked_lp = out.fused_linear_aux.log_probs[shifted_target_is_ign]
    valid_lp = out.fused_linear_aux.log_probs[~shifted_target_is_ign]
    masked_ent = out.fused_linear_aux.entropy[shifted_target_is_ign]
    valid_ent = out.fused_linear_aux.entropy[~shifted_target_is_ign]
    assert torch.all(masked_lp == 0.0), (
        f"[{family}/{ce_impl}] IGN-target positions must emit 0.0 log_probs, got max_abs={masked_lp.abs().max().item():.3e}"
    )
    assert torch.all(masked_ent == 0.0), (
        f"[{family}/{ce_impl}] IGN-target positions must emit 0.0 entropy, got max_abs={masked_ent.abs().max().item():.3e}"
    )
    # log p(.) < 0 strictly for any non-degenerate distribution at random
    # init (probability < 1).
    assert torch.all(valid_lp < 0), (
        f"[{family}/{ce_impl}] valid-target positions must emit negative log_probs, got max={valid_lp.max().item():.3e}"
    )
    # H[p] > 0 strictly for any non-degenerate distribution.
    assert torch.all(valid_ent > 0), (
        f"[{family}/{ce_impl}] valid-target positions must emit positive entropy, got min={valid_ent.min().item():.3e}"
    )

    del model, ref_logits, ref_log_probs, ref_entropy, out
    _release()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_plain_forward_matches_verl_consumer_contract(toy_path, family):
    """Check the fused-output fields read by verl-style consumers.

    Call the generated model directly with labels and ``return_log_probs=True``,
    without a loader adapter or engine override. The model/loss path must return
    finite, label-shaped ``fused_linear_aux.log_probs`` and
    ``fused_linear_aux.entropy``, with non-positive log-probabilities,
    non-negative entropy, and both ``loss`` and ``logits`` set to ``None``.
    """
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    torch.manual_seed(0)
    model = _build_model(toy_path, ce_impl="chunk_loss").eval()

    B, L = 2, 16
    input_ids = torch.randint(0, 32000, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX
    labels[1, ::5] = IGNORE_INDEX

    with torch.no_grad():
        # The path verl takes: plain model forward — no helper import.
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

    del model, out
    _release()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_return_log_probs_backward_flows_gradients(toy_path, family):
    """A loss built from fused log-probabilities must train the LM head.

    Exercise ``output.fused_linear_aux.log_probs`` on both text and VLM classes
    and require finite, nonzero ``lm_head.weight.grad``.
    """
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    torch.manual_seed(1)
    model = _build_model(toy_path, ce_impl="chunk_loss").train()
    model.zero_grad(set_to_none=True)

    B, L = 1, 8
    input_ids = torch.randint(0, 32000, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX

    out = model(input_ids=input_ids, labels=labels, use_cache=False, return_log_probs=True)
    log_probs = out.fused_linear_aux.log_probs  # [B, L], non-positive
    mask = (labels != IGNORE_INDEX).float()
    # Surrogate scalar: a PPO-style per-token-weighted sum of NLL
    # (== -log_probs * mask, mean over valid tokens).
    scalar = (-log_probs * mask).sum() / mask.sum().clamp_min(1)
    scalar.backward()

    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None, f"[{family}] lm_head.weight.grad must be populated by backward"
    assert torch.isfinite(lm_head_grad).all(), f"[{family}] lm_head.weight.grad has non-finite values"
    assert lm_head_grad.abs().max().item() > 0, f"[{family}] lm_head.weight.grad is all zero"

    del model, out, log_probs, scalar
    _release()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_return_log_probs_with_topk_distill_populates_three_fields(toy_path, family):
    """Check top-k distillation outputs on the plain model forward path.

    Teacher top-k IDs and log-probabilities reach the model's loss helper via
    forward kwargs. The helper returns distillation losses, student mass, and
    teacher mass inside ``output.fused_linear_aux``, alongside log-probabilities
    and entropy.

    Check finite, label-shaped outputs and detached probability masses in
    [0, 1]. The loss-sign assertion is specific to this synthetic teacher/student
    setup; truncated-support KL is not generally non-negative. When hidden
    states are exposed, compare bitwise with the same distillation helper called
    directly. Backward through the distillation loss must train the LM head.
    """
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    from veomni.models.loss_utils import chunk_topk_distill_function

    torch.manual_seed(0)
    model = _build_model(toy_path, ce_impl="chunk_loss").train()
    model.zero_grad(set_to_none=True)

    B, L, K = 2, 16, 4
    # VLM configs nest vocab_size under config.text_config; pull it from
    # the lm_head weight to stay model-family-agnostic.
    V = model.lm_head.weight.shape[0]
    input_ids = torch.randint(0, 32000, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX
    labels[1, ::5] = IGNORE_INDEX

    # Teacher tensors: derive from a fresh logits draw + log_softmax + topk
    # so the per-position teacher_mass sums to <= 1 (matches verl's source
    # which gets them from a real teacher forward).
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

    # 1) Distillation fields populated, finite, correct shape.
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

    # 2) Forward KL is non-negative *in this synthetic setting*. Top-k
    #    forward KL on truncated support is only guaranteed >= 0 in the
    #    full-support limit; here we condition on (peaky teacher
    #    log_softmax topk) vs (random-init student near log(1/V)) so
    #    log p_t,k > log q_s,k holds per top-k slot. This both pins the
    #    expected sign and surfaces a regression if the kernel ever
    #    flips the (log p_t - log q_s) sign convention.
    assert (out.fused_linear_aux.distillation_losses >= 0).all(), (
        f"[{family}] distillation_losses must be >= 0; got min={out.fused_linear_aux.distillation_losses.min().item():.3e}"
    )

    # 3) Mass values live on [0, 1].
    assert (out.fused_linear_aux.student_mass >= 0).all() and (out.fused_linear_aux.student_mass <= 1 + 1e-5).all(), (
        f"[{family}] student_mass out of [0, 1]; "
        f"got [{out.fused_linear_aux.student_mass.min().item():.3e}, {out.fused_linear_aux.student_mass.max().item():.3e}]"
    )
    assert (out.fused_linear_aux.teacher_mass >= 0).all() and (out.fused_linear_aux.teacher_mass <= 1 + 1e-5).all(), (
        f"[{family}] teacher_mass out of [0, 1]; "
        f"got [{out.fused_linear_aux.teacher_mass.min().item():.3e}, {out.fused_linear_aux.teacher_mass.max().item():.3e}]"
    )

    # 4) Mass tensors detached.
    assert not out.fused_linear_aux.student_mass.requires_grad, f"[{family}] student_mass must be detached"
    assert not out.fused_linear_aux.teacher_mass.requires_grad, f"[{family}] teacher_mass must be detached"

    # 5) Bitwise equivalence vs the same kernel called directly on the
    #    model's penultimate hidden state. Best-effort hidden-state extraction
    #    via output_hidden_states=True; if a model family hides the lm_head
    #    input we just skip this check (the forward-population coverage from
    #    steps 1-4 already proves the dispatch wiring).
    with torch.no_grad():
        hidden_out = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    last_hidden = getattr(hidden_out, "hidden_states", None)
    if last_hidden is not None and len(last_hidden) >= 1:
        last_hidden_t = last_hidden[-1]  # [B, L, H]
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

    # 6) Backward through distillation_losses reaches lm_head.weight.
    mask = (labels != IGNORE_INDEX).float()
    scalar = (out.fused_linear_aux.distillation_losses * mask).sum() / mask.sum().clamp_min(1)
    scalar.backward()
    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None, f"[{family}] lm_head.weight.grad must be populated by distillation backward"
    assert torch.isfinite(lm_head_grad).all(), f"[{family}] lm_head.weight.grad has non-finite values"
    assert lm_head_grad.abs().max().item() > 0, f"[{family}] lm_head.weight.grad is all zero"

    del model, out
    _release()
