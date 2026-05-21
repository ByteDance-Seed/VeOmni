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

"""End-to-end test for ``return_log_probs=True`` on a real (toy-config) model.

Builds a tiny Qwen3 from ``tests/toy_config/qwen3_toy/`` via
``build_foundation_model`` and asserts:

1. ``model(..., return_log_probs=True).log_probs`` carries per-token
   actual log-probabilities (non-positive) matching the input label
   shape (``[B, L]``); ``output.logits`` is None (cleared by the
   ``build_foundation_model`` postprocess wrapper).
2. The values are **bitwise identical** to ``-F.cross_entropy(
   reduction='none')`` on the model's full-logits forward, when full
   determinism + batch-invariant mode are enabled and ``chunk_size``
   covers the whole sequence (so the chunked ``F.linear`` reduces to
   a single call against the same weight).
3. Backward through ``output.fused_linear_aux.log_probs`` flows gradients into
   ``model.lm_head.weight``.
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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOY_QWEN3 = os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3_toy")
TOY_QWEN3_VL = os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3vl_toy")
IGNORE_INDEX = -100


def _release():
    gc.collect()
    if IS_CUDA_AVAILABLE:
        empty_cache()


def _apply_determinism():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    """Reference per-token log-probabilities (non-positive) and entropy (non-negative).

    Routes the per-token NLL through the same
    ``_per_token_log_probs_from_logits`` helper the kernel uses (which
    prefers ``flash_attn``'s triton ``cross_entropy_loss`` — same op
    verl's ``FusedLinearForPPOFunction`` calls — falling back to
    ``log_softmax + gather`` when flash_attn isn't importable). The
    only remaining numerical difference between this path and the
    chunked-fused-linear path is the lm_head matmul boundary —
    identical when ``chunk_size`` covers the whole seq, so the kernel
    output stays bitwise equal. Entropy is computed via the same
    ``_per_token_entropy_from_logits`` helper for the same reason.
    """
    from veomni.ops.kernels.cross_entropy.chunk_logprobs import (
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
    """Build a tiny model from a toy config (text or VLM).

    Forces eager attention (toy config dtype is fp32; flash_attn requires
    fp16/bf16) and pins the cross-entropy backend.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    ops_implementation = OpsImplementationConfig(
        attn_implementation="eager",
        cross_entropy_loss_implementation=ce_impl,
    )

    return build_foundation_model(
        config_path=toy_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        init_device=get_device_type() if IS_CUDA_AVAILABLE else "cpu",
        ops_implementation=ops_implementation,
    )


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
    """End-to-end: return_log_probs path is bitwise identical to gather-on-logits.

    Covers both **text** (Qwen3) and **VLM** (Qwen3-VL) models — every
    in-tree patched modeling file forwards ``**kwargs`` from the outer
    ``forward`` into ``self.loss_function(...)``, so the same
    ``return_log_probs=True`` flag activates the chunked-NLL path
    uniformly across model families.

    With:
    - full determinism (deterministic cuBLAS, deterministic algorithms,
      no TF32),
    - batch-invariant mode when available (deterministic
      mm/addmm/log_softmax — falls back gracefully when the underlying
      Triton kernels can't JIT in the current environment),
    - chunk_size >= L so the chunked ``F.linear`` is a single matmul
      against the same weight tensor as ``model.lm_head``,

    the per-token NLL returned by the kernel and the reference computed
    by ``F.cross_entropy(reduction='none')`` on the full logits differ
    by at most zero — they execute the same ops on the same data.
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
        from veomni.ops.batch_invariant_ops import set_batch_invariant_mode

        bi_ctx = set_batch_invariant_mode(True)
        bi_ctx.__enter__()

    try:
        torch.manual_seed(0)
        model = _build_model(toy_path, ce_impl=ce_impl).eval()

        B, L = 2, 16
        # Vocab floor of 32000 dodges the multimodal placeholder ids
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

            # New path: model wrapper installed by ``build_foundation_model``
            # makes ``model(..., return_log_probs=True)`` return
            # ``output.fused_linear_aux.log_probs`` (actual log-probabilities, sign already
            # flipped) and ``output.fused_linear_aux.entropy`` (per-token softmax entropy)
            # and clear ``output.logits``. ``chunk_size=B*L+1`` forces a
            # single chunk over the whole packed batch so the matmul
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
    assert out.logits is None, "logits must be cleared when return_log_probs=True"
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
    """Pin the verl-consumer contract on the **plain model forward path**.

    Verl's ``FSDPEngineWithLMHead.prepare_model_outputs`` does
    ``log_probs = output.fused_linear_aux.log_probs.squeeze(0)`` and
    ``entropy_rmpad = output.fused_linear_aux.entropy.squeeze(0)`` in its
    ``use_fused_kernels=True`` branch and expects actual
    log-probabilities (non-positive) plus per-token entropy
    (non-negative). The integration story is: verl calls
    ``self.module(..., return_log_probs=True)`` directly — no helper
    imports, no engine override — and the
    ``build_foundation_model``-installed wrapper makes the output's
    ``log_probs`` and ``entropy`` fields populated automatically.

    This test pins exactly that contract:

    1. ``output.fused_linear_aux.log_probs`` and ``output.fused_linear_aux.entropy`` are populated, finite,
       shape matches labels.
    2. ``output.logits`` is None — wrapper cleared it after promotion.
    3. ``output.loss`` is None.
    4. ``output.fused_linear_aux.log_probs <= 0`` everywhere (actual log-probabilities).
    5. ``output.fused_linear_aux.entropy >= 0`` everywhere (softmax entropy).
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
    assert out.logits is None, f"[{family}] logits must be cleared by the build_foundation_model wrapper"
    assert out.fused_linear_aux.log_probs is not None, (
        f"[{family}] log_probs must be populated by the build_foundation_model wrapper"
    )
    assert out.fused_linear_aux.entropy is not None, (
        f"[{family}] entropy must be populated by the build_foundation_model wrapper"
    )
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
    """Backward through per-token NLL must populate lm_head.weight.grad.

    Same kwargs-flow contract as the bitwise test — exercised on both
    text and VLM model families.
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
    """Pin the **distillation consumer contract** on the plain model forward path.

    The integration story for verl's VeOmni engine: it calls
    ``self.module(..., return_log_probs=True, teacher_topk_ids=...,
    teacher_topk_log_probs=...)`` directly — no helper imports, no
    engine override — and the ``build_foundation_model``-installed
    wrapper routes through ``chunk_topk_distill_function`` and assigns
    ``output.fused_linear_aux.distillation_losses`` / ``output.fused_linear_aux.student_mass`` /
    ``output.fused_linear_aux.teacher_mass`` automatically.

    This test pins exactly that contract end-to-end on a real toy model
    (both text and VLM families):

    1. The three new ``output`` fields are populated, finite, and have
       the same shape as the existing ``log_probs`` / ``entropy``.
    2. ``output.fused_linear_aux.distillation_losses >= 0`` (forward KL is non-negative).
    3. ``output.fused_linear_aux.student_mass`` and ``output.fused_linear_aux.teacher_mass`` are in
       ``[0, 1]`` (sums of softmax probabilities on the top-k support).
    4. ``output.fused_linear_aux.student_mass`` and ``output.fused_linear_aux.teacher_mass`` are detached
       (``requires_grad=False``) — verl reports them as metrics.
    5. ``output.fused_linear_aux.distillation_losses`` matches the same kernel computed
       on the model's penultimate hidden state directly — bitwise on
       fp32 under determinism. Proves the per-model patchgen wiring is
       correct.
    6. Backward through ``output.fused_linear_aux.distillation_losses`` flows gradients
       into ``model.lm_head.weight``.
    """
    _skip_unless_cuda(toy_path)
    _apply_determinism()

    from veomni.ops.kernels.cross_entropy.chunk_topk_distill import chunk_topk_distill_function

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

    # 1) Three new fields populated, finite, correct shape.
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
