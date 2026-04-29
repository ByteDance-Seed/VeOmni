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
3. Backward through ``output.log_probs`` flows gradients into
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


def _reference_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Reference per-token actual log-probabilities (non-positive).

    Matches the kernel's arithmetic path exactly: ``log_softmax + gather``
    on the model's full-logits forward, with the kernel's
    ``labels[..., 1:]`` / ``hidden[..., :-1]`` shift and a trailing pad
    of 0.0 to match input shape. The only remaining numerical
    difference between this path and the chunked-fused-linear path is
    the lm_head matmul boundary — identical when chunk_size covers the
    whole seq.
    """
    shifted = labels[..., 1:].contiguous()
    sliced = logits[..., :-1, :].contiguous()
    flat = sliced.reshape(-1, sliced.size(-1)).float()
    log_softmax = flat.log_softmax(dim=-1)
    safe = shifted.reshape(-1).clamp(min=0).unsqueeze(-1)
    log_probs_flat = log_softmax.gather(-1, safe).squeeze(-1)
    mask = shifted.reshape(-1) != ignore_index
    log_probs_flat = torch.where(mask, log_probs_flat, torch.zeros_like(log_probs_flat))
    log_probs = log_probs_flat.view_as(shifted)
    return F.pad(log_probs, (0, 1), value=0.0)  # non-positive: actual log p(y_t)


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


def _skip_unless_cuda_v4(toy_path: str):
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(toy_path):
        pytest.skip(f"Path not found: {toy_path}")
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if is_transformers_version_greater_or_equal_to("5.0.0"):
        pytest.skip("Toy config targets transformers v4 model definition.")


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
    _skip_unless_cuda_v4(toy_path)
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
            # Reference path: full-logits forward + reference log-probs.
            ref_logits = model(input_ids=input_ids, use_cache=False).logits
            ref_log_probs = _reference_log_probs_from_logits(ref_logits, labels)

            # New path: model wrapper installed by ``build_foundation_model``
            # makes ``model(..., return_log_probs=True)`` return
            # ``output.log_probs`` (actual log-probabilities, sign already
            # flipped) and clear ``output.logits``. ``chunk_size=L+1``
            # forces a single chunk so the matmul boundary matches the
            # reference forward exactly.
            out = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                return_log_probs=True,
                chunk_size=L + 1,
            )
    finally:
        if bi_ctx is not None:
            bi_ctx.__exit__(None, None, None)

    assert out.loss is None, "loss must be None when return_log_probs=True"
    assert out.logits is None, "logits must be cleared when return_log_probs=True"
    assert out.log_probs is not None, "log_probs must be populated when return_log_probs=True"
    assert out.log_probs.shape == labels.shape, (
        f"shape mismatch: got {tuple(out.log_probs.shape)} expected {tuple(labels.shape)}"
    )
    assert out.log_probs.dtype == ref_log_probs.dtype, (
        f"dtype mismatch: got {out.log_probs.dtype} expected {ref_log_probs.dtype}"
    )

    if not torch.equal(out.log_probs, ref_log_probs):
        diff = (out.log_probs - ref_log_probs).abs()
        ne = out.log_probs != ref_log_probs
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"[{family}/{ce_impl}] per-token log_probs not bitwise equal: "
            f"{int(ne.sum().item())}/{out.log_probs.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )

    # IGNORE_INDEX masking contract: the kernel emits exactly 0 wherever
    # the shifted target is IGN (and at the trailing pad position). The
    # kernel predicts ``labels[t+1]`` from ``hidden[t]``, so an IGN at
    # ``labels[k]`` zeros output position ``k-1``.
    shifted_target_is_ign = F.pad(labels[..., 1:] == IGNORE_INDEX, (0, 1), value=True)
    masked_vals = out.log_probs[shifted_target_is_ign]
    valid_vals = out.log_probs[~shifted_target_is_ign]
    assert torch.all(masked_vals == 0.0), (
        f"[{family}/{ce_impl}] IGN-target positions must emit 0.0, got max_abs={masked_vals.abs().max().item():.3e}"
    )
    # log p(.) < 0 strictly for any non-degenerate distribution at random
    # init (probability < 1).
    assert torch.all(valid_vals < 0), (
        f"[{family}/{ce_impl}] valid-target positions must emit negative log_probs, got max={valid_vals.max().item():.3e}"
    )

    del model, ref_logits, ref_log_probs, out
    _release()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_plain_forward_matches_verl_consumer_contract(toy_path, family):
    """Pin the verl-consumer contract on the **plain model forward path**.

    Verl's ``FSDPEngineWithLMHead.prepare_model_outputs`` does
    ``log_probs = output.log_probs.squeeze(0)`` in its
    ``use_fused_kernels=True`` branch and expects actual
    log-probabilities (non-positive). The integration story is:
    verl calls ``self.module(..., return_log_probs=True)`` directly —
    no helper imports, no engine override — and the
    ``build_foundation_model``-installed wrapper makes the output's
    ``log_probs`` field populated automatically.

    This test pins exactly that contract:

    1. ``output.log_probs`` is populated, finite, shape matches labels.
    2. ``output.logits`` is None — wrapper cleared it after promotion.
    3. ``output.loss`` is None.
    4. ``output.log_probs <= 0`` everywhere (actual log-probabilities).
    """
    _skip_unless_cuda_v4(toy_path)
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
    assert out.log_probs is not None, f"[{family}] log_probs must be populated by the build_foundation_model wrapper"
    assert out.log_probs.shape == labels.shape, (
        f"[{family}] log_probs shape {tuple(out.log_probs.shape)} != labels shape {tuple(labels.shape)}"
    )
    assert torch.isfinite(out.log_probs).all(), f"[{family}] log_probs has non-finite values"
    assert (out.log_probs <= 0).all(), f"[{family}] log_probs must be <= 0; got max={out.log_probs.max().item():.3e}"

    del model, out
    _release()


@pytest.mark.parametrize("toy_path,family", _MODELS)
def test_return_log_probs_backward_flows_gradients(toy_path, family):
    """Backward through per-token NLL must populate lm_head.weight.grad.

    Same kwargs-flow contract as the bitwise test — exercised on both
    text and VLM model families.
    """
    _skip_unless_cuda_v4(toy_path)
    _apply_determinism()

    torch.manual_seed(1)
    model = _build_model(toy_path, ce_impl="chunk_loss").train()
    model.zero_grad(set_to_none=True)

    B, L = 1, 8
    input_ids = torch.randint(0, 32000, (B, L), device=model.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 0] = IGNORE_INDEX

    out = model(input_ids=input_ids, labels=labels, use_cache=False, return_log_probs=True)
    log_probs = out.log_probs  # [B, L], non-positive
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
