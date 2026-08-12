# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

import contextlib
import warnings
from pathlib import Path

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()

# The generated modeling module by import path, for ``mock.patch`` targets.
_PATCHED_MODULE = "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu"


# Warnings this module tolerates, matched as substrings of the warning message.
# Everything else fails test_target_kernel_emits_no_unexpected_warnings, so a new
# warning cannot slip into a run unnoticed. Only add an entry for a warning that
# provably originates outside this repository; a warning raised by our own code is
# a bug to fix, not an entry here.
_TOLERATED_WARNING_SUBSTRINGS = (
    # tilelang deprecating one of its own pass-config keys. Raised from
    # tilelang/transform/pass_config.py::normalize_pass_configs on every
    # JITKernel construction, for the `TL_DISABLE_TMA_LOWER` config that every
    # DeepSeek-V4 tilelang kernel in this tree sets -- including the attention
    # forward these tests call to produce the LSE.
    "`tl.disable_tma_lower` is deprecated",
)

# Importing ``veomni.arguments`` or the generated modeling module (which pulls in
# ``veomni.distributed``) imports ``bytedance.hdfs_stdenv``, and that package calls the
# deprecated ``Logger.warn`` during its own env setup. Third-party and not fixable from
# here, so it is filtered rather than left to accumulate in pytest's summary. The
# filter is pinned to that package's module path, so the same warning raised from our
# own code would still be reported.
_HDFS_STDENV_DEPRECATION_FILTER = pytest.mark.filterwarnings(
    r"ignore:The 'warn' method is deprecated:DeprecationWarning:bytedance\.hdfs_stdenv\..*"
)


def reference_compressed_target(q, kv, attn_sink, topk_idxs, compressed_start, sm_scale):
    """Paper-correct teacher for DeepSeek-V3.2 eq. (4), in fp32.

    Deliberately written without reference to any LSE: the per-head denominator
    comes from one softmax over ``[selected slots ‖ sink]``, so the window and
    sink contributions are structurally present. This is the property Megatron's
    teacher violates (NVIDIA/Megatron-LM#5776) and the reason this reference must
    never be replaced by a call into the implementation.

    Args:
        q:               [B, S, H, D] any float dtype
        kv:              [B, S_kv, D] any float dtype
        attn_sink:       [H]
        topk_idxs:       [B, S, W + C] int, -1 for misses
        compressed_start: W, the index where the compressed slice begins
        sm_scale:        float

    Returns:
        [B, S, C] fp32, L1-normalised over the compressed slice.
    """
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    batch_index = torch.arange(b, device=kv.device).view(b, 1, 1)
    gathered = kv[batch_index, topk_idxs.clamp_min(0)]  # [B, S, W + C, D]
    logits = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
    compressed = probs[..., compressed_start:].sum(dim=2)  # head sum -> [B, S, C]
    return compressed / compressed.sum(-1, keepdim=True).clamp_min(1e-20)


def test_reference_target_responds_to_sink_and_window():
    """The two perturbations Megatron's compressed-only teacher is blind to.

    A per-head softmax over the compressed entries alone would make both of
    these no-ops, so this test is what distinguishes a correct teacher from a
    plausible one. Two heads with opposing preferences are required: with one
    head the outer normalisation cancels the denominator entirely.
    """
    torch.manual_seed(0)
    b, s, h, d, w, c = 1, 3, 2, 16, 2, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, w + c, d)
    topk = torch.arange(w + c).view(1, 1, -1).expand(b, s, -1).contiguous().to(torch.int32)
    sink = torch.zeros(h)
    scale = d**-0.5

    base = reference_compressed_target(q, kv, sink, topk, w, scale)

    bumped_sink = sink.clone()
    bumped_sink[0] = 5.0
    assert not torch.allclose(base, reference_compressed_target(q, kv, bumped_sink, topk, w, scale), atol=1e-4)

    bumped_kv = kv.clone()
    bumped_kv[:, 0] *= 4.0  # a window row, not a compressed row
    assert not torch.allclose(base, reference_compressed_target(q, bumped_kv, sink, topk, w, scale), atol=1e-4)


def test_reference_target_is_normalised():
    torch.manual_seed(1)
    q = torch.randn(2, 5, 4, 16)
    kv = torch.randn(2, 12, 16)
    topk = torch.randint(0, 12, (2, 5, 8), dtype=torch.int32)
    topk[0, 0, 0] = -1
    target = reference_compressed_target(q, kv, torch.zeros(4), topk, 4, 16**-0.5)
    assert torch.allclose(target.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert (target >= 0).all()


def test_reference_target_all_invalid_compressed_row_is_zero():
    """A query whose *compressed* slots are all misses is not a distribution.

    ``test_reference_target_is_normalised`` only masks a slot in the window slice,
    where the compressed mass is unaffected. When the whole compressed slice
    misses, the mass being normalised is exactly zero and the ``clamp_min(1e-20)``
    divides it by a floor instead of by itself, so the row comes out as zeros
    rather than as a uniform distribution or a NaN. A later task takes a KL
    against this row and has to tolerate it, so the contract is pinned here and
    in ``test_target_kernel_all_invalid_compressed_row_is_zero`` for the kernel.
    """
    torch.manual_seed(3)
    b, s, h, d, w, c = 1, 2, 2, 16, 4, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, w + c, d)
    topk = torch.arange(w + c).view(1, 1, -1).expand(b, s, -1).contiguous().to(torch.int32)
    topk[0, 0, w:] = -1  # query 0 keeps a valid window and loses every compressed slot

    target = reference_compressed_target(q, kv, torch.zeros(h), topk, w, d**-0.5)
    assert (target[0, 0] == 0).all()
    assert not torch.isnan(target).any()
    # The zero row is local: the untouched query is still normalised.
    assert torch.allclose(target[0, 1].sum(), torch.ones(()), atol=1e-5)


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("c", [64, 128, 100])
def test_target_kernel_matches_reference(heads, c):
    """``heads=8`` pads to 16 inside the kernel; the padded heads must contribute
    nothing to the head sum. That is not automatic — a padded head has zero Q and
    would contribute ``exp2(0 - lse_pad)`` unless its LSE is padded to +inf.

    The ``c`` axis is the number of ``block_I``-sized slot tiles the kernel's
    ``T.Pipelined(NI)`` loop runs, which is the axis production actually stresses:
    ``index_topk = 512`` at ``block_I = 64`` is ``NI = 8``, never the ``NI = 1``
    that ``c = 64`` alone exercises. ``c = 128`` gives ``NI = 2``, so the loop body
    runs more than once against a reused ``tgt`` fragment and a moving output
    slice. ``c = 100`` gives ``NI = 2`` *and* is not a multiple of ``block_I``, so
    it is the only case that runs the interface's ``-1``-sentinel padding and the
    final ``[:, :, :topk]`` slice that has to hide it again.
    """
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(0)
    b, s, d, w = 2, 8, 64, 64
    device = DEVICE
    q = torch.randn(b, s, heads, d, device=device, dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device=device, dtype=torch.bfloat16)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device=device, dtype=torch.int32)
    topk[0, 0, w] = -1  # a compressed miss in the first slot tile
    topk[1, 0, w + c - 1] = -1  # and one in the last, which only NI > 1 reaches
    scale = d**-0.5

    _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
    actual = sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)
    # The sentinel padding a non-multiple ``c`` needs must not survive into the
    # returned shape.
    assert actual.shape == (b, s, c)
    actual = actual / actual.sum(-1, keepdim=True).clamp_min(1e-20)

    expected = reference_compressed_target(q, kv, sink, topk, w, scale)
    # A normalised entry is ~1/C, i.e. 0.008 to 0.016 across this
    # parametrisation, so a tolerance of the order of
    # 1e-2 would exceed the values being compared and accept anything. It is not
    # a hypothetical: summing the wrong axis of the score tile yields a
    # per-head row sum whose normalised form sits 1.5e-2 from the truth, i.e.
    # inside a 2e-2 tolerance. The kernel and this reference consume the same
    # bf16 inputs and both accumulate in fp32, so they agree to 2e-8 absolute
    # and 5e-7 relative as measured on GB200; the bounds below leave three
    # orders of magnitude of headroom for a different GEMM summation order or a
    # TF32 einsum, while still rejecting the wrong-axis sum by a factor of 60.
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-2)


def test_target_kernel_zeroes_invalid_slots():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(2)
    b, s, heads, d, w, c = 1, 4, 16, 64, 64, 64
    q = torch.randn(b, s, heads, d, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device=DEVICE, dtype=torch.bfloat16)
    sink = torch.zeros(heads, device=DEVICE, dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device=DEVICE, dtype=torch.int32)
    topk[:, :, w + 3] = -1
    scale = d**-0.5

    _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
    target = sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)
    assert (target[:, :, 3] == 0).all()


def test_target_kernel_all_invalid_compressed_row_is_zero():
    """The kernel's half of the contract pinned in
    ``test_reference_target_all_invalid_compressed_row_is_zero``: an all-miss
    compressed row comes back as exact zeros, and the caller's ``clamp_min``
    normalisation leaves it as zeros rather than turning it into a NaN or a
    uniform distribution."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(4)
    b, s, heads, d, w, c = 1, 4, 16, 64, 64, 64
    q = torch.randn(b, s, heads, d, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device=DEVICE, dtype=torch.bfloat16)
    sink = torch.zeros(heads, device=DEVICE, dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device=DEVICE, dtype=torch.int32)
    topk[0, 0, w:] = -1  # query 0 keeps a valid window and loses every compressed slot
    scale = d**-0.5

    _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
    raw = sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)
    assert (raw[0, 0] == 0).all()

    normalised = raw / raw.sum(-1, keepdim=True).clamp_min(1e-20)
    assert (normalised[0, 0] == 0).all()
    assert not torch.isnan(normalised).any()
    # Still local: the queries with valid compressed slots are unaffected.
    assert (raw[0, 1:] > 0).any()
    assert torch.allclose(normalised[0, 1:].sum(-1), torch.ones(s - 1, device=DEVICE), atol=1e-5)
    # And the reference agrees on the same input, so the contract is one contract.
    assert (reference_compressed_target(q, kv, sink, topk, w, scale)[0, 0] == 0).all()


def test_target_kernel_rejects_more_than_64_heads():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    q = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(1, 128, 64, device=DEVICE, dtype=torch.bfloat16)
    topk = torch.zeros(1, 2, 64, device=DEVICE, dtype=torch.int32)
    lse = torch.zeros(1, 2, 128, device=DEVICE, dtype=torch.float32)
    # Matching on "64" alone would also be satisfied by the head-multiple assert
    # or by any message that happens to mention a shape of 64.
    with pytest.raises(RuntimeError, match="one block owns the head sum"):
        sparse_mqa_target_fwd_interface(q, kv, topk, lse)


def test_target_kernel_rejects_head_counts_the_interface_cannot_emit():
    """``heads % 16 == 0`` on its own admits 48, which no interface call can
    produce -- padding is ``max(next_power_of_2(heads), 16)``, so one of
    {16, 32, 64} -- and which ``T.GemmWarpPolicy.FullRow`` cannot partition: with
    the guard removed, lowering dies on ``Check failed: (m_warp * n_warp ==
    num_warps) is false: m_warp: 3, n_warp: 2, num_warps: 8``. Only a caller that
    bypasses the interface can reach this, which is exactly what this test does.
    """
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target

    with pytest.raises(RuntimeError, match=r"padded to one of \(16, 32, 64\)"):
        sparse_mqa_target(48, 64, 64)


def test_target_kernel_rejects_non_bfloat16_inputs():
    """The kernel hardcodes ``dtype = T.bfloat16``; the interface names that
    constraint instead of letting an fp16 caller fall into tilelang's lowering."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    q = torch.randn(1, 2, 16, 64, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(1, 128, 64, device=DEVICE, dtype=torch.bfloat16)
    topk = torch.zeros(1, 2, 64, device=DEVICE, dtype=torch.int32)
    lse = torch.zeros(1, 2, 16, device=DEVICE, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="bfloat16-only, got q"):
        sparse_mqa_target_fwd_interface(q.to(torch.float16), kv, topk, lse)
    with pytest.raises(RuntimeError, match="bfloat16-only, got kv"):
        sparse_mqa_target_fwd_interface(q, kv.to(torch.float16), topk, lse)


def test_target_kernel_rejects_empty_kv():
    """The gather clamps candidate rows into ``[0, S_kv - 1]``, so an empty kv
    would clamp to row 0 of a tensor with no rows -- an out-of-bounds device read
    that the candidate mask cannot prevent, since it only zeroes the score after
    the gather. ``sparse_mqa_fwd_interface`` guards this; mirror it here."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    q = torch.randn(1, 2, 16, 64, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(1, 0, 64, device=DEVICE, dtype=torch.bfloat16)
    topk = torch.zeros(1, 2, 64, device=DEVICE, dtype=torch.int32)
    lse = torch.zeros(1, 2, 16, device=DEVICE, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="at least one row"):
        sparse_mqa_target_fwd_interface(q, kv, topk, lse)


def test_target_kernel_emits_no_unexpected_warnings():
    """Pin the warning set, so a newly introduced warning is a failure rather than
    an unexplained bump in pytest's summary count.

    ``heads = 32`` is used by no other test in this module, so both kernels below
    are constructed here rather than being served from tilelang's in-process
    cache; that is what makes the construction-time warnings observable at all.
    """
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(5)
    b, s, heads, d, w, c = 1, 2, 32, 64, 64, 64
    q = torch.randn(b, s, heads, d, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device=DEVICE, dtype=torch.bfloat16)
    sink = torch.zeros(heads, device=DEVICE, dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device=DEVICE, dtype=torch.int32)
    scale = d**-0.5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
        sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)

    unexpected = [
        f"{w.category.__name__} at {w.filename}:{w.lineno}: {w.message}"
        for w in caught
        if not any(known in str(w.message) for known in _TOLERATED_WARNING_SUBSTRINGS)
    ]
    assert not unexpected, "unexpected warning(s):\n" + "\n".join(unexpected)


def test_sparse_attn_returns_non_differentiable_lse():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang

    torch.manual_seed(3)
    b, s, heads, d = 1, 8, 16, 64
    q = torch.randn(b, s, heads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(b, 128, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sink = torch.randn(heads, device=DEVICE, dtype=torch.float32, requires_grad=True)
    topk = torch.randint(0, 128, (b, s, 64), device=DEVICE, dtype=torch.int32)

    out_only = sparse_attn_tilelang(q, kv, sink, topk, d**-0.5)
    out, lse = sparse_attn_tilelang(q, kv, sink, topk, d**-0.5, return_lse=True)

    torch.testing.assert_close(out_only, out)
    assert lse.shape == (b, s, heads)
    assert lse.dtype == torch.float32
    assert not lse.requires_grad, "the LSE feeds a detached teacher and must not open a path back into attention"

    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


# Run in a fresh interpreter by
# ``test_indexer_loss_slots_are_off_before_anything_binds_them``. It has to be a
# subprocess: the slots are module globals, so the only place their *unbound* value
# can be observed is a process where nothing has bound them yet, and in-process that
# depends on what else the session ran.
_UNBOUND_SLOT_PROBE = """
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

loss = modeling.veomni_dsa_indexer_loss
coef = modeling.veomni_dsa_indexer_loss_coef
assert loss.value is False, f"unbound {loss.field_name} reads {loss.value!r}, expected False"
assert coef.value == 1.0, f"unbound {coef.field_name} reads {coef.value!r}, expected 1.0"
assert isinstance(coef.value, float), f"unbound {coef.field_name} is a {type(coef.value).__name__}, expected float"
assert modeling._indexer_loss_enabled(object()) is False, "the gate is enabled with nothing bound"
print("UNBOUND_SLOTS_OK")
"""


def test_indexer_loss_slots_are_off_before_anything_binds_them():
    """``OpsConfigSlot``'s own constructor default is the string ``"eager"``, so the
    two slots added for the indexer loss pass an explicit ``default=``.

    That deviation is load-bearing and silent: delete both ``default=`` arguments and
    an unbound ``dsa_indexer_loss`` reads ``"eager"``, which is truthy, so the gate
    turns *on* for a model whose config never asked for the loss — while the
    coefficient becomes a string that would raise only once something multiplied by
    it. Nothing else observes this, because every test that touches the gate binds
    the slots first, which is exactly what erases the state under test.

    Hence the subprocess: a process where nothing has bound anything is the only
    honest way to read an unbound slot, and it does not depend on which tests ran
    before this one. The expected values are written out here rather than read from
    ``OpsImplementationConfig``'s field defaults — the point is that the two
    declarations agree, so deriving one from the other would pin nothing.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _UNBOUND_SLOT_PROBE],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "UNBOUND_SLOTS_OK" in result.stdout, (
        "the indexer-loss slots are not off before binding; "
        f"probe exited {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-500:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
    )


@_HDFS_STDENV_DEPRECATION_FILTER
@pytest.mark.parametrize("coef", [-1.0, -1e-8, float("nan"), float("inf"), float("-inf")])
def test_indexer_loss_coef_rejects_negative_and_non_finite(coef):
    """The coefficient is multiplied into the indexer KL before it joins the total
    loss, so a negative value trains the indexer *away* from the teacher and a
    non-finite one wipes out every other term in the sum. Both would show up as a
    loss curve rather than as an error, which is why the bound is checked where the
    rest of ``OpsImplementationConfig`` validates itself — at config-parse time,
    before any of it reaches a model.

    ``load_balancing_loss_implementation`` is pinned to eager only so this test does
    not depend on ``triton`` being installed; the field is unrelated to the bound
    under test.
    """
    from veomni.arguments.arguments_types import OpsImplementationConfig

    with pytest.raises(ValueError, match="dsa_indexer_loss_coef"):
        OpsImplementationConfig(load_balancing_loss_implementation="eager", dsa_indexer_loss_coef=coef)


@_HDFS_STDENV_DEPRECATION_FILTER
@pytest.mark.parametrize("coef", [0.0, 0.5, 1.0, 100.0])
def test_indexer_loss_coef_accepts_finite_non_negative_weights(coef):
    """The other half of the bound: zero is a legitimate way to switch the term off
    without changing the flag, and the check must not reject the ordinary weights."""
    from veomni.arguments.arguments_types import OpsImplementationConfig

    config = OpsImplementationConfig(load_balancing_loss_implementation="eager", dsa_indexer_loss_coef=coef)
    assert config.dsa_indexer_loss_coef == coef


@contextlib.contextmanager
def _ops_config_slots_bound(pre_state):
    """Put the generated module's ``OpsConfigSlot``s in a known state, and undo every
    binding afterwards.

    The slots are globals on a module shared with the rest of the session, so without
    this a test leaks its configuration into whatever runs next — and a refusal test
    that reads a value some earlier test happened to leave behind proves nothing about
    the refusal it names. Binding ``pre_state`` closes the same hole from the other
    side: a test that binds only some of the slots reads the rest from here, not from
    the session. Both directions go through ``bind``, the same public path the tests
    use.

    The ``isinstance`` filter that collects the slots to restore would happily match
    nothing at all — after a slot rename or a move out of the generated module —
    leaving a teardown loop over an empty list and no failure anywhere, so the setup
    asserts that every slot named in ``pre_state`` was found.
    """
    from types import SimpleNamespace

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
    from veomni.ops.dispatch import OpsConfigSlot

    saved = [(slot, slot.value) for slot in vars(modeling).values() if isinstance(slot, OpsConfigSlot)]
    collected = {slot.field_name: slot for slot, _ in saved}
    missing = sorted(pre_state.keys() - collected.keys())
    assert not missing, (
        f"no OpsConfigSlot found on {modeling.__name__} for {missing}; this is the only "
        "thing keeping the tests that bind slots independent of each other and of the "
        "session, and it just restored nothing"
    )
    for field_name, value in pre_state.items():
        collected[field_name].bind(SimpleNamespace(**{field_name: value}))
    try:
        yield
    finally:
        for slot, value in saved:
            slot.bind(SimpleNamespace(**{slot.field_name: value}))


class TestIndexerLossGate:
    """``_indexer_loss_enabled``: the three refusals, the flag-off path, and the one
    configuration that enables the loss.

    Each refusal guards a configuration that would otherwise train the indexer on a
    wrong signal or on none while the loss curve looked entirely reasonable, so a
    test here has to fail when *its own* refusal is deleted. Two things are needed
    for that, and both are easy to get wrong because the gate reads three module
    globals in sequence:

    * every test binds *all* the slots the gate reads, so a deleted refusal falls
      through to a supported configuration rather than into the next refusal, and
    * every ``match=`` names the field its own refusal is about. ``tilelang`` alone
      does not: it appears in the indexer refusal and the attention refusal both.

    The first of those is enforced by the fixture below rather than left to each
    test: it binds all three slots to a known pre-state, so a test that binds two of
    them reads the third from that pre-state instead of from whatever the session
    left behind.

    The class exists to scope the autouse fixture below to this group.
    """

    # Every test here imports the generated modeling module; see the note on
    # ``_HDFS_STDENV_DEPRECATION_FILTER`` for what that drags in and why it is
    # filtered rather than left to accumulate in pytest's summary.
    pytestmark = _HDFS_STDENV_DEPRECATION_FILTER

    # The slots ``_indexer_loss_enabled`` reads, and the pre-state the fixture puts
    # them in before every test: a configuration that is supported but off. Spelled
    # out here rather than read back from the module, so a slot that is renamed or
    # moved out of the generated module fails the fixture's setup assertion instead
    # of quietly dropping out of the collected set.
    _GATE_SLOT_PRE_STATE = {
        "dsa_indexer_loss": False,
        "dsa_indexer_implementation": "eager",
        "dsa_attention_implementation": "eager",
    }

    @pytest.fixture(autouse=True)
    def _restore_ops_config_slots(self):
        """Bind the pre-state above before each test and restore every slot after it.

        See ``_ops_config_slots_bound`` for why both directions are load-bearing. It
        lives at module scope because the tests below this class bind slots too and
        need exactly the same protection.
        """
        with _ops_config_slots_bound(self._GATE_SLOT_PRE_STATE):
            yield

    def test_indexer_loss_refuses_ulysses(self, monkeypatch):
        """Ulysses shards heads across ranks, so the head sum inside the teacher would
        only cover this rank's shard. That is a *wrong* teacher rather than a missing
        one, and it would still produce a decreasing loss curve, so the gate has to
        refuse rather than warn.

        The match pins the message's two actionable halves: the size observed, and the
        one size that is supported.

        The state is a real ``ParallelState`` rather than a namespace carrying an
        ``ulysses_size`` attribute: a duck-typed stand-in keeps this test green
        through a rename of the field, while the gate's ``state.ulysses_size`` would
        raise ``AttributeError`` in production. Two accommodations are needed to build
        one in a single process, both with precedent in
        ``tests/parallel/context_parallel/test_dsv4_cp_parallel_state.py``:
        ``ParallelState.world_size`` reads ``torch.distributed`` directly, and a
        sequence-parallel state requires a non-``None`` device mesh that nothing here
        touches.
        """
        from types import SimpleNamespace
        from unittest import mock
        from unittest.mock import MagicMock

        from veomni.distributed.parallel_state import ParallelState
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
        monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 2)
        state = ParallelState(dp_size=1, ulysses_size=2, device_type="cpu", device_mesh=MagicMock())

        modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=True))
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))
        with mock.patch(
            "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu.get_parallel_state",
            return_value=state,
        ):
            with pytest.raises(ValueError, match=r"requires ulysses_size=1, got ulysses_size=2"):
                modeling._indexer_loss_enabled(object())

    def test_indexer_loss_refuses_eager_indexer(self):
        """The eager indexer discards its scores, so there would be no student to
        train against and the KL would have nothing to say.

        Attention is bound to the *supported* value so that deleting the indexer
        refusal leaves nothing else to raise, and the match names
        ``dsa_indexer_implementation``, which appears in no other refusal.
        """
        from types import SimpleNamespace

        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=True))
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="eager"))
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))
        with pytest.raises(ValueError, match="dsa_indexer_implementation"):
            modeling._indexer_loss_enabled(object())

    def test_indexer_loss_refuses_eager_attention(self):
        """The teacher is read off the TileLang attention LSE, which the eager
        attention path never computes. Unlike the eager indexer, this one has a
        student and no teacher, but the outcome is the same: a loss that cannot mean
        what it appears to mean.

        The indexer is bound to the supported value so the earlier refusal cannot
        stand in for this one, and the match names ``dsa_attention_implementation``.
        """
        from types import SimpleNamespace

        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=True))
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="eager"))
        with pytest.raises(ValueError, match="dsa_attention_implementation"):
            modeling._indexer_loss_enabled(object())

    def test_indexer_loss_off_by_default(self):
        """With the flag off, none of the refusals above fire: eager everything is a
        perfectly ordinary configuration until someone asks for the loss."""
        from types import SimpleNamespace

        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=False))
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="eager"))
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="eager"))
        assert modeling._indexer_loss_enabled(object()) is False

    def test_indexer_loss_enabled_on_the_supported_configuration(self):
        """The one configuration this whole flag exists for, and the only test that
        reaches the gate's ``return True``.

        The three refusals raise before that line and the flag-off test returns from
        the early guard, so without this test ``return True`` could read ``return
        False`` and nothing would notice — the indexer loss would silently never be
        built on precisely the setup it was written for. That is the same "plausible
        curve, nothing trained" failure the refusals were written to prevent, so the
        success path is pinned as tightly as they are: ``is True`` rather than a
        truthiness check, since a gate returning a truthy non-``bool`` is not the
        contract the call sites read.

        ``ulysses_size=1`` is the default of a real ``ParallelState``, and at that
        size the state needs no mesh and no world of more than one rank, so nothing
        has to be accommodated to build it.
        """
        from types import SimpleNamespace
        from unittest import mock

        from veomni.distributed.parallel_state import ParallelState
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=True))
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))
        state = ParallelState(dp_size=1, ulysses_size=1, device_type="cpu")
        assert state.ulysses_size == 1  # the premise of this test, not an assumption about the default
        with mock.patch(
            "veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu.get_parallel_state",
            return_value=state,
        ):
            assert modeling._indexer_loss_enabled(object()) is True


_TOY_CONFIG_DIR = Path(__file__).resolve().parents[1] / "toy_config" / "deepseek_v4_toy"


def _dsv4_indexer_test_config():
    """The toy DeepSeek-V4 config, re-geometried to the 4-layer checkpoint's indexer.

    ``index_n_heads=64`` / ``index_head_dim=128`` / ``index_topk=512`` over the CSA
    compression rate of 4 are the values ``DeepSeek-V4-Flash-Base-4L/config.json``
    ships, and the indexer forward's TileLang gate reads every one of them: the head
    count and the head dim decide whether the kernel is eligible at all, and
    ``index_topk`` against the compressed length decides how wide the score the loss
    trains on is. The toy config's own ``index_n_heads=8`` would take the head count
    off production's boundary of 64.

    ``hidden_size`` (256) and ``q_lora_rank`` (64) stay at the toy config's values
    rather than the checkpoint's 4096 / 1024: nothing under test reads them beyond the
    widths of the indexer's two input projections.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(_TOY_CONFIG_DIR))
    config.index_n_heads = 64
    config.index_head_dim = 128
    config.index_topk = 512
    return config


def _build_test_indexer(device="cuda", dtype=torch.bfloat16):
    """A ``DeepseekV4Indexer`` on the geometry above, ready to call.

    ``position_bias`` is a ``torch.empty`` parameter in the upstream constructor, so a
    directly built indexer scores against uninitialised memory until it is zeroed.

    Returns the config alongside the module because the caller needs both
    ``hidden_size`` and ``q_lora_rank`` to build inputs, and they differ — the indexer
    projects ``hidden_states`` for its keys, gates and per-head weights but
    ``q_residual`` for its queries, so one ``randn_like`` cannot serve for both.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = _dsv4_indexer_test_config()
    indexer = modeling.DeepseekV4Indexer(config).to(device=device, dtype=dtype)
    with torch.no_grad():
        torch.nn.init.zeros_(indexer.position_bias)
    return indexer, config


def _build_test_csa_compressor(device="cuda", dtype=torch.bfloat16):
    """A ``DeepseekV4CSACompressor`` around an indexer of the same geometry.

    Both modules declare ``position_bias`` as ``torch.empty``, and the compressor's own
    is the easy one to forget because it sits one level above the module under test.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = _dsv4_indexer_test_config()
    compressor = modeling.DeepseekV4CSACompressor(config).to(device=device, dtype=dtype)
    with torch.no_grad():
        torch.nn.init.zeros_(compressor.position_bias)
        torch.nn.init.zeros_(compressor.indexer.position_bias)
    return compressor, config


def _run_indexer(indexer, hidden_states, q_residual, position_ids=None, **kwargs):
    """Call the indexer exactly as ``DeepseekV4CSACompressor.forward`` calls it.

    Positional, in the compressor's argument order, with the compressor's ``None``
    cache and its layer index, so that a change to the real call site's shape breaks
    these tests rather than leaving them agreeing with a signature nothing uses. The
    compressor's own two call sites are exercised directly by
    ``test_csa_compressor_carries_the_indexer_scores``.
    """
    if position_ids is None:
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    return indexer(hidden_states, q_residual, position_ids, None, 0, **kwargs)


def _bind_indexer_loss(*, enabled):
    """Bind every slot ``_indexer_loss_enabled`` reads, with the two implementation
    slots on the one configuration the loss supports.

    All three, always: a test that left one unbound would read it from whatever ran
    before, and a deleted guard would then fall into a neighbouring refusal instead of
    through to the behaviour under test.
    """
    from types import SimpleNamespace

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    modeling.veomni_dsa_indexer_loss.bind(SimpleNamespace(dsa_indexer_loss=enabled))
    modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
    modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))


@contextlib.contextmanager
def _single_rank_parallel_state():
    """Pin the ambient parallel state to a real single-rank ``ParallelState``.

    Both ``_indexer_loss_enabled`` and the indexer forward resolve
    ``get_parallel_state()``, and an uninitialised process only returns a default
    single-rank state *because* nothing has initialised one — which is a fact about the
    session, not about the code under test. A real ``ParallelState`` rather than a
    namespace, for the reason ``test_indexer_loss_refuses_ulysses`` gives: a duck-typed
    stand-in survives a rename of a field the production path would raise on.
    """
    from unittest import mock

    from veomni.distributed.parallel_state import ParallelState

    state = ParallelState(dp_size=1, ulysses_size=1)
    with mock.patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=state):
        yield state


@contextlib.contextmanager
def _counting_tilelang_indexer():
    """Count the calls that reach the TileLang Lightning Indexer kernel.

    Without this, a test that means to exercise the kernel branch would still pass if
    the forward quietly took the eager fallback — which returns a bare index tensor
    too, so the arity assertions below would be satisfied by the path they are not
    about.
    """
    from unittest import mock

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    calls = []
    real_kernel = modeling.v4_lighting_indexer

    def _counting(*args, **kwargs):
        calls.append(None)
        return real_kernel(*args, **kwargs)

    with mock.patch(f"{_PATCHED_MODULE}.v4_lighting_indexer", _counting):
        yield calls


class TestIndexerScoresAndDecoupling:
    """The indexer returning its scores, and detaching its own inputs to pay for it.

    Returning ``index_score`` is what gives the KL a student to train; the detach is
    what stops that KL from reaching the language-modelling objective. They belong in
    one group because they are one change: before it, the indexer's forward returned
    integer indices and the graph was severed by accident.

    Every test here binds all three of the gate's slots through ``_bind_indexer_loss``
    and pins the parallel state, and the class-scoped fixture restores every slot
    afterwards, so nothing leaks into the rest of the session.
    """

    # Every test here imports the generated modeling module; see the note on
    # ``_HDFS_STDENV_DEPRECATION_FILTER`` for what that drags in.
    pytestmark = _HDFS_STDENV_DEPRECATION_FILTER

    # Supported but off, matching ``TestIndexerLossGate``: a test that failed to bind a
    # slot reads it from here rather than from whatever the session left behind.
    _SLOT_PRE_STATE = {
        "dsa_indexer_loss": False,
        "dsa_indexer_implementation": "eager",
        "dsa_attention_implementation": "eager",
    }

    @pytest.fixture(autouse=True)
    def _restore_ops_config_slots(self):
        with _ops_config_slots_bound(self._SLOT_PRE_STATE):
            yield

    def test_indexer_detaches_its_input(self):
        """The indexer must not backpropagate into the main model.

        DeepSeek-V3.2 §2.1: "we detach the indexer input from the computational graph
        for separate optimization." Until this change the graph was severed only by
        accident, because the forward returned integer indices, which carry no
        gradient. Returning ``index_score`` ends that accident: without an explicit
        detach the KL would flow back through the indexer's projections into
        ``hidden_states`` and perturb the LM objective.

        The parameter half of the assertions is not a formality either. A detach
        placed one line too late — after the projections rather than before them —
        would still leave ``hidden_states.grad`` unset while cutting the indexer off
        from its own gradient, so the loss would train nothing. All three of
        ``q_b_proj`` / ``kv_proj`` / ``weights_proj`` are checked because they are
        exactly the three tensors ``v4_lighting_indexer`` differentiates: the query,
        the compressed key and the per-head weight. ``grad.abs().sum() > 0`` rather
        than a ``grad_fn`` check, because a graph that exists and carries zeros trains
        nothing.
        """
        _require_tilelang_cuda()
        _bind_indexer_loss(enabled=True)

        seq_len = 2048
        indexer, config = _build_test_indexer(device="cuda")
        hidden = torch.randn(1, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_residual = torch.randn(
            1, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        with _single_rank_parallel_state(), _counting_tilelang_indexer() as kernel_calls:
            top_k_indices, index_score = _run_indexer(indexer, hidden, q_residual)
        assert len(kernel_calls) == 1, "the TileLang scorer is the only path that has scores to return"

        # The contract Task 5 reads: one score per selected slot, in fp32.
        assert index_score.shape == top_k_indices.shape
        assert index_score.dtype == torch.float32
        # ``-inf`` marks the invalid slots and ``nan_to_num``'s own backward zeroes the
        # gradient there, so this reduction stays finite and touches only real slots.
        index_score.float().nan_to_num(neginf=0.0).sum().backward()

        assert hidden.grad is None, "indexer input must be detached"
        assert q_residual.grad is None, "indexer q_residual must be detached"
        for name in ("q_b_proj", "kv_proj", "weights_proj"):
            grad = getattr(indexer, name).weight.grad
            assert grad is not None, f"{name} received no gradient from the indexer score"
            assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"

    def test_indexer_returns_indices_only_with_the_loss_off(self):
        """Default off means the return *arity* is what it was, not just the values.

        This is the only test that pins the off side of that branch. Invert it and the
        two-tuple leaks into every flag-off forward: ``DeepseekV4CSACompressor`` would
        keep working, because it unpacks by the same gate, and the break would surface
        somewhere else entirely — the CP and Ulysses suites call the indexer directly.
        """
        _require_tilelang_cuda()
        _bind_indexer_loss(enabled=False)

        seq_len = 2048
        indexer, config = _build_test_indexer(device="cuda")
        hidden = torch.randn(1, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_residual = torch.randn(1, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)

        with _single_rank_parallel_state(), _counting_tilelang_indexer() as kernel_calls:
            result = _run_indexer(indexer, hidden, q_residual)

        assert len(kernel_calls) == 1, "the eager fallback returns a bare tensor too, so it pins nothing here"
        assert isinstance(result, torch.Tensor), f"expected a bare index tensor, got {type(result).__name__}"
        assert result.dtype == torch.long
        assert result.shape == (1, seq_len, min(config.index_topk, seq_len // indexer.compress_rate))

    def test_indexer_refuses_the_eager_fallback_under_the_loss(self):
        """A configuration that passes every construction-time check can still miss the
        kernel at runtime, and the eager path discards its scores.

        ``use_tilelang`` is decided per call out of dtypes, devices and shapes, so this
        refusal cannot live where Task 3's configuration refusals live. The miss staged
        here is the device — CPU tensors — but an fp32 activation dtype or a head count
        outside the kernel's range lands at the same line. Silence is the worst outcome
        available: the loss would train on nothing at all while its curve looked
        entirely reasonable.

        The match is on ``fell back to the eager path``, which no other message in the
        generated module contains. ``eager`` alone would also be satisfied by the
        ``dsa_indexer_implementation`` refusal, and a match on TileLang by the
        attention forward's own runtime refusal a few hundred lines down.
        """
        _bind_indexer_loss(enabled=True)

        indexer, config = _build_test_indexer(device="cpu", dtype=torch.float32)
        hidden = torch.randn(1, 32, config.hidden_size)
        q_residual = torch.randn(1, 32, config.q_lora_rank)

        with _single_rank_parallel_state():
            with pytest.raises(RuntimeError, match="fell back to the eager path"):
                _run_indexer(indexer, hidden, q_residual)

    def test_indexer_eager_fallback_still_runs_with_the_loss_off(self):
        """The refusal above is conditional, and this is the only test that says so.

        The eager scorer is the production path for cache/decode and for any layout the
        kernel declines, so a refusal that fired regardless of the flag would break
        inference for every DeepSeek-V4 model in the tree — while the suite stayed
        green on the enabled path, because that path never reaches the eager scorer.
        """
        _bind_indexer_loss(enabled=False)

        indexer, config = _build_test_indexer(device="cpu", dtype=torch.float32)
        hidden = torch.randn(1, 32, config.hidden_size)
        q_residual = torch.randn(1, 32, config.q_lora_rank)

        with _single_rank_parallel_state():
            result = _run_indexer(indexer, hidden, q_residual)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32 // indexer.compress_rate)

    @pytest.mark.parametrize("packed", [False, True])
    def test_csa_compressor_carries_the_indexer_scores(self, packed):
        """``CompressedCandidates.indexer_scores`` is how the scores reach the loss.

        Both of ``DeepseekV4CSACompressor.forward``'s indexer call sites are exercised
        — the packed one and the contiguous one — because each builds its own
        ``CompressedCandidates`` and a dropped field at one is invisible from the
        other.

        The flag-off half is not a bonus. It pins that the compressor unpacks by the
        same gate the indexer returns by, and the index comparison pins that turning
        the loss on leaves the selection the model attends over untouched: a detach
        cannot change a forward value, so exact equality is the right bar and any
        tolerance would hide a real perturbation of the LM path.
        """
        _require_tilelang_cuda()

        seq_len = 2048
        compressor, config = _build_test_csa_compressor(device="cuda")
        hidden = torch.randn(1, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_residual = torch.randn(1, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        if packed:
            from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

            sequence_slices = ((0, 1024), (1024, seq_len))
            position_ids = torch.cat(
                [torch.arange(end - start, device="cuda") for start, end in sequence_slices]
            ).unsqueeze(0)
            packed_kwargs = {
                "packed_sequence_slices": sequence_slices,
                "packed_compression_metadata": build_packed_compression_metadata(
                    hidden, position_ids, sequence_slices, (compressor.compress_rate,)
                ),
            }
        else:
            position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
            packed_kwargs = {}

        def _candidates():
            with _single_rank_parallel_state():
                _, _, candidates = compressor(
                    hidden, q_residual, position_ids, None, 0, return_topk_indices=True, **packed_kwargs
                )
            return candidates

        _bind_indexer_loss(enabled=False)
        without_loss = _candidates()
        assert without_loss.indexer_scores is None, "the scores must not appear with the loss off"

        _bind_indexer_loss(enabled=True)
        with_loss = _candidates()
        scores = with_loss.indexer_scores
        assert scores is not None, "the compressor dropped the indexer scores on the floor"
        assert scores.shape == with_loss.topk_indices.shape
        assert scores.dtype == torch.float32
        # The kernel marks a miss with ``-1`` and scores it ``-inf``. The loss reads the
        # two together, so anything but an exact correspondence means they are
        # describing different slots.
        assert (with_loss.topk_indices < 0).any(), "no invalid slot here, so the -inf correspondence is untested"
        assert torch.equal(torch.isneginf(scores), with_loss.topk_indices < 0)
        assert torch.equal(with_loss.topk_indices, without_loss.topk_indices)
