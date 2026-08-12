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


def _build_test_attention(device="cuda", seq_len=4096, layer_type="compressed_sparse_attention", seed=0):
    """One ``DeepseekV4Attention`` of the requested layer type, plus the kwargs its
    real call site hands it.

    The kwargs mirror ``DeepseekV4Model.forward``'s call into the decoder layer,
    which forwards them verbatim to ``self_attn``: the ``position_embeddings`` dict
    keyed by rope layer type, global ``position_ids``, ``attention_mask`` and
    ``past_key_values``. Mirrored rather than invented so that a change to the real
    call site's shape breaks these tests instead of leaving them agreeing with a
    signature nothing uses.

    ``attention_mask=None`` is the mask-free sparse path, which is the one bf16
    TileLang training takes -- ``DeepseekV4Model.forward`` withholds the dense mask
    exactly when the sparse index builder can validate candidates on its own. It also
    keeps a ``[1, 1, 4096, 5120]`` mask off a device this test shares.

    ``sinks`` and both ``position_bias`` parameters are ``torch.empty`` in the
    upstream constructors, so a directly built layer attends against uninitialised
    memory until they are written. The sinks are drawn per-head rather than set to a
    constant so that they cannot all cancel out of the teacher's denominators; note
    though that they are *not* what separates the paper's teacher from a
    compressed-only one. Measured on this configuration, forcing every sink to the
    same value moves that separation from 1.9633 to 1.9620 — it is the per-head query
    content that makes the summed compressed mass differ across heads, and
    ``test_kl_uses_the_paper_correct_teacher`` asserts the separation itself rather
    than trusting either.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = _dsv4_indexer_test_config()
    if layer_type not in config.layer_types:
        # The 4-layer checkpoint this config mirrors has no sliding layer, and the
        # gate has to keep one on its two-value return just as it does an HCA layer.
        config.layer_types = [*config.layer_types[:-1], layer_type]
    layer_idx = config.layer_types.index(layer_type)
    torch.manual_seed(seed)
    attn = modeling.DeepseekV4Attention(config, layer_idx=layer_idx).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        attn.sinks.normal_(0.0, 1.0)
        for name, param in attn.named_parameters():
            if name.endswith("position_bias"):
                param.zero_()

    hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary = modeling.DeepseekV4RotaryEmbedding(config).to(device=device)
    inputs = {
        "hidden_states": hidden,
        "position_embeddings": {
            rope_type: rotary(hidden, position_ids=position_ids, layer_type=rope_type)
            for rope_type in ("main", "compress")
        },
        "position_ids": position_ids,
        "attention_mask": None,
        "past_key_values": None,
    }
    return attn, inputs


class _DsaKernelProbe:
    """What the TileLang attention, indexer-teacher and Lightning Indexer kernels were
    handed, and what they returned.

    Reading the tensors at the kernel boundary is deliberate. Rebuilding ``q`` and
    ``kv`` inside the test would mean copying the attention forward's projection,
    per-head normalisation, RoPE and compressor concatenation out of the module under
    test, and a copy like that agrees with the implementation by construction. What
    the reference has to supply independently is the *teacher* -- one softmax over
    ``[selected slots ‖ sink]`` -- and ``reference_compressed_target`` does.

    ``index_score`` is captured from the indexer kernel rather than from the attention
    forward's own use of it, so a forward that handed the KL a permuted or otherwise
    mismatched student is compared against the selection the indexer really made.
    """

    def __init__(self):
        self.attention = []
        self.target = []
        self.indexer = []

    def one(self, name):
        calls = getattr(self, name)
        assert len(calls) == 1, f"expected exactly one {name} kernel call, got {len(calls)}"
        return calls[0]


@contextlib.contextmanager
def _probe_dsa_kernels():
    """Record every DSA kernel call the generated module makes, without changing one.

    The wrappers bind their arguments by name, so a call site that renames or
    reorders one fails here rather than being recorded under the wrong key.
    """
    from unittest import mock

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    probe = _DsaKernelProbe()
    real_attention = modeling.sparse_attn_tilelang
    real_target = modeling.sparse_mqa_target_fwd
    real_indexer = modeling.v4_lighting_indexer

    def _attention(q, kv, attn_sink, topk_idxs, sm_scale=None, return_lse=False):
        result = real_attention(q, kv, attn_sink, topk_idxs, sm_scale, return_lse)
        lse = result[1] if return_lse else None
        probe.attention.append(
            {"q": q, "kv": kv, "sink": attn_sink, "topk": topk_idxs, "sm_scale": sm_scale, "lse": lse}
        )
        return result

    def _target(q, kv, topk_idxs, lse, sm_scale=None):
        probe.target.append({"q": q, "kv": kv, "topk": topk_idxs, "lse": lse, "sm_scale": sm_scale})
        return real_target(q, kv, topk_idxs, lse, sm_scale)

    def _indexer(*args, **kwargs):
        index_score, topk_indices = real_indexer(*args, **kwargs)
        probe.indexer.append({"index_score": index_score, "topk_indices": topk_indices})
        return index_score, topk_indices

    with (
        mock.patch(f"{_PATCHED_MODULE}.sparse_attn_tilelang", _attention),
        mock.patch(f"{_PATCHED_MODULE}.sparse_mqa_target_fwd", _target),
        mock.patch(f"{_PATCHED_MODULE}.v4_lighting_indexer", _indexer),
    ):
        yield probe


def _reference_compressed_only_target(q, kv, topk_idxs, compressed_start, sm_scale):
    """The teacher this loss must **not** build: Megatron's compressed-only variant.

    One softmax per head over the compressed slice alone, so the sliding window and
    the attention sink are absent from the denominator (NVIDIA/Megatron-LM#5776).
    Written here so that ``test_kl_uses_the_paper_correct_teacher`` can assert its own
    premise -- that the two teachers are far enough apart for its tolerance to tell
    them apart -- instead of taking that on trust.

    All-miss compressed rows are neutralised on the way in, as
    ``indexer_kl_terms`` does: without the sink in the denominator such a row is an
    all-``-inf`` softmax, i.e. NaN, and a NaN teacher would make the separation below
    unreadable rather than small.

    Args / returns as ``reference_compressed_target``, minus the sink.
    """
    idx = topk_idxs[..., compressed_start:]
    b = q.shape[0]
    valid = idx >= 0
    batch_index = torch.arange(b, device=kv.device).view(b, 1, 1)
    gathered = kv[batch_index, idx.clamp_min(0).long()]
    logits = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    all_missing = ~torch.isfinite(logits).any(-1, keepdim=True)
    probs = torch.softmax(torch.where(all_missing, torch.zeros_like(logits), logits), dim=-1)
    compressed = torch.where(all_missing, torch.zeros_like(probs), probs).sum(dim=2)  # head sum -> [B, S, C]
    return compressed / compressed.sum(-1, keepdim=True).clamp_min(1e-20)


def _reference_compressed_only_target_by_query_chunk(q, kv, topk_idxs, compressed_start, sm_scale, chunk=512):
    """``_reference_compressed_only_target`` over query chunks, for the reason
    ``_reference_target_by_query_chunk`` gives."""
    return torch.cat(
        [
            _reference_compressed_only_target(
                q[:, start : start + chunk],
                kv,
                topk_idxs[:, start : start + chunk],
                compressed_start,
                sm_scale,
            )
            for start in range(0, q.shape[1], chunk)
        ],
        dim=1,
    )


def _reference_target_by_query_chunk(q, kv, sink, topk_idxs, compressed_start, sm_scale, chunk=512):
    """``reference_compressed_target`` over query chunks, concatenated.

    The reference gathers ``[B, S, W + C, D]`` in one allocation, which at S=4096,
    W + C = 544 and D = 64 is 285 MB in bf16 and another 570 MB once it upcasts --
    on a device shared with whatever else is running. Query rows are independent, so
    chunking moves the peak and nothing else.
    """
    return torch.cat(
        [
            reference_compressed_target(
                q[:, start : start + chunk],
                kv,
                sink,
                topk_idxs[:, start : start + chunk],
                compressed_start,
                sm_scale,
            )
            for start in range(0, q.shape[1], chunk)
        ],
        dim=1,
    )


@_HDFS_STDENV_DEPRECATION_FILTER
def test_indexer_kl_terms_matches_hand_computation():
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    target = torch.tensor([[[0.5, 0.5, 0.0]]])
    index_score = torch.tensor([[[0.0, 0.0, float("-inf")]]])
    # softmax over the two finite slots is [0.5, 0.5]; KL(t || q) == 0.
    assert torch.allclose(modeling.indexer_kl_terms(index_score, target), torch.zeros(1, 1), atol=1e-6)

    index_score = torch.tensor([[[1.0, 0.0, float("-inf")]]])
    q0 = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    expected = 0.5 * (torch.log(torch.tensor(0.5)) - torch.log(q0[0])) + 0.5 * (
        torch.log(torch.tensor(0.5)) - torch.log(q0[1])
    )
    assert torch.allclose(modeling.indexer_kl_terms(index_score, target), expected.view(1, 1), atol=1e-6)


@_HDFS_STDENV_DEPRECATION_FILTER
def test_indexer_kl_terms_gradient_is_finite_when_a_query_sees_no_compressed_slot():
    """A query row whose *every* compressed slot is a miss must not poison the
    gradient, and the forward value cannot be trusted to reveal it.

    Such rows are not a corner case: the first ``compress_rate - 1`` positions of
    every packed sample have no complete compression window behind them, so the
    indexer's causal range is empty, its top-k comes back all ``-1``, and its score
    row is entirely ``-inf``. ``log_softmax`` of that row is NaN, and the NaN is
    invisible in the forward — ``torch.where`` selects the zero branch, so the KL
    for the row is exactly 0, which is the right answer. The backward is where it
    bites: ``log_softmax``'s own backward computes ``g - softmax * g.sum(-1)`` with
    ``softmax = exp(NaN)``, so a zero incoming gradient still comes out NaN. That
    NaN survives the indexer's backward too, because its kernel forms
    ``grad * relu(logits)`` and ``NaN * 0`` is NaN, so ``weights_proj`` ends up with
    a NaN gradient while the loss curve stays finite.

    Row 1 is here to keep the fix honest: the row-wise neutralisation must be
    confined to the all-miss rows, so row 1's gradient has to match what it gets
    when it is the only row present.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    neg_inf = float("-inf")
    index_score = torch.tensor([[[neg_inf, neg_inf], [0.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [0.4, 0.6]]])

    kl = modeling.indexer_kl_terms(index_score, target)
    assert torch.allclose(kl[0, 0], torch.zeros(())), "an all-miss row has no teacher mass and no KL"
    kl.sum().backward()

    assert torch.isfinite(index_score.grad).all(), f"non-finite gradient: {index_score.grad}"
    assert (index_score.grad[0, 0] == 0).all(), "an all-miss row must not push the indexer anywhere"

    alone = index_score.detach()[:, 1:].clone().requires_grad_(True)
    modeling.indexer_kl_terms(alone, target[:, 1:]).sum().backward()
    torch.testing.assert_close(index_score.grad[:, 1:], alone.grad)
    assert alone.grad.abs().sum() > 0, "row 1 carries no gradient, so it pins nothing here"


@_HDFS_STDENV_DEPRECATION_FILTER
def test_indexer_kl_terms_ignores_zero_target_slots():
    """A -inf score paired with zero target mass must contribute 0, not NaN."""
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    target = torch.tensor([[[1.0, 0.0]]])
    index_score = torch.tensor([[[0.0, float("-inf")]]])
    kl = modeling.indexer_kl_terms(index_score, target)
    assert torch.isfinite(kl).all()
    assert torch.allclose(kl, torch.zeros(1, 1), atol=1e-6)


class TestAttentionForwardIndexerKL:
    """The attention forward assembling the KL: the teacher it builds, the slice it
    takes, the student it pairs with, and the arity it returns.

    Every test binds all three of the gate's slots through ``_bind_indexer_loss`` and
    pins the parallel state, and the fixture restores every slot afterwards.
    """

    # Every test here imports the generated modeling module; see the note on
    # ``_HDFS_STDENV_DEPRECATION_FILTER`` for what that drags in.
    pytestmark = _HDFS_STDENV_DEPRECATION_FILTER

    # Supported but off, matching the two classes above.
    _SLOT_PRE_STATE = {
        "dsa_indexer_loss": False,
        "dsa_indexer_implementation": "eager",
        "dsa_attention_implementation": "eager",
    }

    # 4096 tokens at the CSA compression rate of 4 give 1024 compressed slots while
    # the indexer selects ``index_topk = 512``, so the top-k genuinely ranks. At the
    # 2048 the other tests in this file use, the two are equal and every causally
    # visible slot is selected -- a teacher that never exercises selection, against
    # which a student misaligned with its own selection would still look right.
    _RANKING_SEQ_LEN = 4096
    # Where ranking is beside the point, 2048 is used instead: it is half the work and
    # it keeps ``sliding_window + top_k`` at 544, so it reuses the same compiled
    # TileLang kernels rather than paying for a second set.
    _CHEAP_SEQ_LEN = 2048

    @pytest.fixture(autouse=True)
    def _restore_ops_config_slots(self):
        with _ops_config_slots_bound(self._SLOT_PRE_STATE):
            yield

    def test_kl_uses_the_paper_correct_teacher(self):
        """The KL the forward returns must match one built from the independent
        reference teacher -- a single softmax over ``[selected slots ‖ sink]``, which
        carries the sliding window and the attention sink in its denominator.

        This is the assertion that separates the paper's teacher from Megatron's
        (NVIDIA/Megatron-LM#5776): a compressed-only denominator is a plausible,
        decreasing loss that trains the indexer toward the wrong distribution. The
        reference is deliberately LSE-free, so it cannot reproduce that mistake by
        sharing it.
        """
        _require_tilelang_cuda()
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        _bind_indexer_loss(enabled=True)
        seq_len = self._RANKING_SEQ_LEN
        attn, inputs = _build_test_attention(device="cuda", seq_len=seq_len)
        with _single_rank_parallel_state(), _probe_dsa_kernels() as probe:
            _, _, kl_sum = attn(**inputs)

        assert kl_sum.shape == (), f"the KL must be a scalar, got {tuple(kl_sum.shape)}"
        assert kl_sum.dtype == torch.float32

        attention, target, indexer = probe.one("attention"), probe.one("target"), probe.one("indexer")
        # The premise of the sequence length: the selection is a strict subset of what
        # is causally visible for a large share of the query rows, so the KL is being
        # asked about a ranking rather than about "everything visible".
        visible = (inputs["position_ids"] + 1) // attn.compressor.compress_rate
        selected = (indexer["topk_indices"] >= 0).sum(-1)
        ranked_rows = int((visible > selected).sum())
        assert ranked_rows > seq_len // 4, (
            f"only {ranked_rows} of {seq_len} query rows had more visible compressed slots than the "
            "indexer selected, so this test barely exercises the top-k"
        )

        compressed_start = attention["topk"].shape[-1] - target["topk"].shape[-1]
        expected_target = _reference_target_by_query_chunk(
            attention["q"],
            attention["kv"],
            attention["sink"],
            attention["topk"],
            compressed_start,
            attn.scaling,
        )
        expected = modeling.indexer_kl_terms(indexer["index_score"], expected_target).sum()

        # Measured on GB200: the kernel teacher and this reference put the summed KL
        # 6.1e-5 apart on a KL of 413.35, i.e. 1.5e-7 relative, both consuming the
        # same bf16 inputs and accumulating in fp32. ``assert_close`` admits
        # ``atol + rtol * |expected|``, so these bounds accept 5.1e-2 -- ~840x the
        # observed gap, which is headroom for a different summation order.
        atol, rtol = 1e-2, 1e-4
        admitted = atol + rtol * expected.abs()

        # The other half of that arithmetic, and the premise this whole test rests on:
        # the wrong teacher has to sit *outside* what the bound accepts, or the
        # comparison above would pass against the mistake it exists to catch. The
        # separation is not free -- for a single head the LSE cancels under the L1
        # renormalisation at the end of the forward, and the two teachers coincide
        # exactly -- so it is asserted rather than assumed, the way the ranking premise
        # above is. Measured here: 1.96, i.e. 38x the bound.
        wrong_target = _reference_compressed_only_target_by_query_chunk(
            attention["q"], attention["kv"], attention["topk"], compressed_start, attn.scaling
        )
        wrong = modeling.indexer_kl_terms(indexer["index_score"], wrong_target).sum()
        separation = (expected - wrong).abs()
        assert separation > 10 * admitted, (
            f"the compressed-only teacher is only {separation:.3e} from the correct one while the "
            f"tolerance below admits {admitted:.3e}, so this test can no longer tell them apart"
        )

        torch.testing.assert_close(kl_sum, expected, atol=atol, rtol=rtol)

    def test_target_reads_the_full_window_lse_and_the_trailing_compressed_slice(self):
        """The three structural contracts the numeric test above can only observe
        through their consequences.

        1. The LSE must come from a forward fed the *full* window+compressed index
           tensor: that is the only reason the teacher's per-head denominator is the
           true CSA denominator. A second forward over a compressed-only index list
           would produce the Megatron variant.
        2. The compressed entries are the *trailing* range of the index tensor, which
           is what ``[:, :, -width:]`` relies on. Both index builders end at
           ``torch.cat((sliding_indices, compressed_indices), dim=-1)``.
        3. ``index_score[..., j]`` and the teacher's slot ``j`` must be the same slot,
           or the KL trains the student's score for one slot toward another slot's
           probability. The chain is checked link by link, against the selection the
           indexer kernel really returned.
        """
        _require_tilelang_cuda()

        _bind_indexer_loss(enabled=True)
        seq_len = self._RANKING_SEQ_LEN
        attn, inputs = _build_test_attention(device="cuda", seq_len=seq_len)
        with _single_rank_parallel_state(), _probe_dsa_kernels() as probe:
            attn(**inputs)

        attention, target, indexer = probe.one("attention"), probe.one("target"), probe.one("indexer")
        width = target["topk"].shape[-1]

        # (1) One attention call, over sink + sliding window + compressed entries, and
        # the LSE the teacher consumed is that call's own -- not a second forward's.
        assert attention["topk"].shape[-1] == attn.sliding_window + width, (
            "the attention forward's index tensor no longer spans window + compressed slots, "
            "so its LSE is not the full CSA denominator the teacher needs"
        )
        assert target["lse"] is attention["lse"], "the teacher must consume the attention forward's own LSE"
        assert target["q"] is attention["q"] and target["kv"] is attention["kv"]
        assert target["sm_scale"] == attention["sm_scale"] == attn.scaling

        # (2) The teacher's slots are the trailing range of the attention forward's.
        assert torch.equal(target["topk"], attention["topk"][:, :, -width:])
        assert not torch.equal(target["topk"], attention["topk"][:, :, :width]), (
            "the leading and trailing slices coincide here, so this test cannot tell them apart"
        )

        # (3) Those slots are exactly the indexer's selection, lifted past the
        # full-resolution KV rows, in the indexer's own order -- so slot ``j`` of the
        # teacher is the slot ``index_score[..., j]`` scored.
        selection = indexer["topk_indices"].to(torch.int32)
        assert width == selection.shape[-1] == indexer["index_score"].shape[-1]
        expected_slice = torch.where(selection >= 0, selection + seq_len, torch.full_like(selection, -1))
        assert torch.equal(target["topk"], expected_slice)
        assert (selection < 0).any(), "no miss in this selection, so the -1 half of the mapping is untested"

    def test_the_kl_survives_the_dense_mask_path(self):
        """The other index-building path: a dense ``attention_mask``.

        Every other test here passes ``attention_mask=None``, which is the mask-free
        path bf16 TileLang training takes. When ``DeepseekV4Model.forward`` cannot
        validate candidates from packed metadata alone it hands the layer the dense
        sliding-window mask instead, and the layer then builds its candidates with
        ``build_sparse_attention_indices`` and filters them through
        ``mask_sparse_attention_indices``. This test is coverage of that path: the KL
        has to come out finite, positive and trainable there too.

        What the filtering step does *not* do is worth recording here, because the
        obvious guess is wrong. Measured on this configuration it changes nothing at
        all: the layer extends the dense mask with the compressor's own ``block_bias``
        before handing it over, and the indexer's selection is already a subset of the
        blocks that bias allows, so the call invalidates 0 of the 1048576 compressed
        slots and returns a tensor bit-identical to its input. The ``-1`` slots the
        premise below looks for are therefore the indexer's own misses carried through
        the builder -- the same ones the mask-free path produces, scored ``-inf`` by
        the indexer as well as zero-massed in the teacher -- and not a teacher/student
        asymmetry this path creates.

        The mask is built the way the model builds it, not by hand, so that a change to
        the real mask's shape or dtype breaks this test rather than leaving it agreeing
        with a mask nothing produces.
        """
        _require_tilelang_cuda()
        from unittest import mock

        from transformers.masking_utils import create_sliding_window_causal_mask

        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        _bind_indexer_loss(enabled=True)
        seq_len = self._CHEAP_SEQ_LEN
        attn, inputs = _build_test_attention(device="cuda", seq_len=seq_len)
        # The mask builder dispatches on the config's attention implementation, and a
        # directly built layer leaves it unset; "eager" is what the model configures
        # here and it is the interface this file's patched forward replaces.
        attn.config._attn_implementation = "eager"
        inputs["attention_mask"] = create_sliding_window_causal_mask(
            config=attn.config,
            inputs_embeds=inputs["hidden_states"],
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device="cuda"),
            past_key_values=None,
            position_ids=inputs["position_ids"],
        )
        assert inputs["attention_mask"] is not None, "no dense mask here, so this test is the mask-free path again"

        with (
            _single_rank_parallel_state(),
            _probe_dsa_kernels() as probe,
            mock.patch(
                f"{_PATCHED_MODULE}.mask_sparse_attention_indices",
                wraps=modeling.mask_sparse_attention_indices,
            ) as masking,
        ):
            _, _, kl_sum = attn(**inputs)

        # The premise: the dense path really ran -- the mask-free path never reaches
        # this call, so without it this test is that path again under another name.
        assert masking.call_count == 1, f"the dense masking step ran {masking.call_count} times, expected once"
        target_width = probe.one("target")["topk"].shape[-1]
        compressed_slots = probe.one("attention")["topk"][:, :, -target_width:]
        assert (compressed_slots < 0).any(), (
            "no missing compressed slot, so the KL's zero-mass masking is not exercised here"
        )

        assert torch.isfinite(kl_sum), f"the dense-mask path made the KL {kl_sum}"
        assert kl_sum > 0, "the KL collapsed to zero on the dense-mask path"
        kl_sum.backward()
        for name in ("q_b_proj", "kv_proj", "weights_proj"):
            grad = getattr(attn.compressor.indexer, name).weight.grad
            assert grad is not None, f"indexer.{name} received no gradient from the KL"
            assert torch.isfinite(grad).all(), f"indexer.{name} received a non-finite gradient from the KL"
            assert grad.abs().sum() > 0, f"indexer.{name} received an all-zero gradient from the KL"

    def test_the_flag_off_forward_returns_two_values(self):
        """Default off means the return *arity* is what it was.

        This is the only test that pins the off side of the branch. Invert it and
        every flag-off DeepSeek-V4 forward returns a 3-tuple into a decoder layer
        that unpacks two, breaking inference for every model in the tree -- while the
        enabled path stays green, because it never takes this branch.
        """
        _require_tilelang_cuda()

        _bind_indexer_loss(enabled=False)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN)
        with _single_rank_parallel_state(), _probe_dsa_kernels() as probe:
            result = attn(**inputs)

        assert len(result) == 2, f"expected (output, attn_weights), got {len(result)} values"
        assert not probe.target, "the teacher kernel ran with the loss off"
        assert probe.one("attention")["lse"] is None, "the LSE was computed with nothing to consume it"

    def test_the_language_model_path_is_bit_identical_with_the_loss_on(self):
        """Turning the loss on must not perturb the model it is attached to.

        The two runs share their parameters and their input, and the enabled path
        reaches the same attention kernel with the same arguments, so the outputs are
        equal *bitwise*: a detach cannot change a forward value, and any tolerance
        here would hide a real perturbation of the LM objective. This is the
        forward-value half of the decoupling; the gradient half is
        ``test_the_kl_gradient_reaches_the_indexer_and_stops_there``.
        """
        _require_tilelang_cuda()

        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN)
        with _single_rank_parallel_state():
            _bind_indexer_loss(enabled=False)
            output_off, weights_off = attn(**inputs)
            _bind_indexer_loss(enabled=True)
            output_on, weights_on, kl_sum = attn(**inputs)

        assert torch.equal(output_off, output_on), "the indexer loss moved the attention output"
        assert weights_off is None and weights_on is None
        assert torch.isfinite(kl_sum) and kl_sum > 0, f"the KL is not a usable objective: {kl_sum}"

    @pytest.mark.parametrize("layer_type", ["heavily_compressed_attention", "sliding_attention"])
    def test_a_non_csa_layer_keeps_the_two_tuple(self, layer_type):
        """Only a compressed-sparse-attention layer carries a Lightning Indexer, so
        only it can build the KL -- and the other layer types must pass the flag-on
        forward through unchanged rather than raise on it.

        Both directions are easy to get wrong. An HCA layer's compressor returns a
        perfectly ordinary ``CompressedCandidates`` -- carrying the causal
        ``range_starts`` / ``range_ends`` its index builder needs -- with no
        ``indexer_scores``, because it has no indexer to produce them; gating on the
        candidates being present would make every HCA layer raise "the CSA compressor
        produced no indexer scores" the moment the flag went on, which is three of the
        four layers here. A sliding layer has no compressor at all, so it would reach
        the same refusal through ``compressed_candidates is None``.

        The gate keys on ``self.layer_type`` rather than on the compressor having an
        attribute called ``indexer``, because a rename of that attribute has to break
        loudly instead of turning the auxiliary objective into a silent no-op.
        """
        _require_tilelang_cuda()

        _bind_indexer_loss(enabled=True)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN, layer_type=layer_type)
        assert attn.layer_type != "compressed_sparse_attention", "this test is about the layers the gate excludes"
        assert getattr(attn.compressor, "indexer", None) is None, "this layer type is supposed to have no indexer"
        with _single_rank_parallel_state(), _probe_dsa_kernels() as probe:
            result = attn(**inputs)

        assert len(result) == 2, f"a {layer_type} layer has no student to train; expected 2 values, got {len(result)}"
        assert not probe.target, "the teacher kernel ran for a layer with no indexer"

    def test_a_compressor_that_drops_the_scores_is_refused(self):
        """The scores travelling from the indexer to the loss pass through a
        ``CompressedCandidates`` field that no index builder reads, so losing them
        would show up as a loss of exactly zero -- a plausible curve, nothing trained.

        Staged by narrowing the compressor's own return rather than by editing the
        module, because the field is what the contract is about.
        """
        _require_tilelang_cuda()
        from unittest import mock

        _bind_indexer_loss(enabled=True)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN)
        real_forward = attn.compressor.forward

        def _without_scores(*args, **kwargs):
            compressed_kv, block_bias, candidates = real_forward(*args, **kwargs)
            return compressed_kv, block_bias, candidates._replace(indexer_scores=None)

        with _single_rank_parallel_state(), mock.patch.object(attn.compressor, "forward", _without_scores):
            with pytest.raises(RuntimeError, match="produced no indexer scores"):
                attn(**inputs)

    def test_scores_and_indices_of_different_widths_are_refused(self):
        """``[:, :, -width:]`` takes the compressed slice on the strength of the
        indexer's score width, so a width that disagrees with the selection would
        silently slide the slice into the sliding-window entries and train the indexer
        against window probabilities.

        Staged as a one-slot-narrower score tensor. This pins the width invariant
        only, which is all the production assertion claims; a *reordering* of the
        index tensor leaves both widths equal and is caught instead by
        ``test_target_reads_the_full_window_lse_and_the_trailing_compressed_slice``.
        """
        _require_tilelang_cuda()
        from unittest import mock

        _bind_indexer_loss(enabled=True)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN)
        real_forward = attn.compressor.forward

        def _narrower_scores(*args, **kwargs):
            compressed_kv, block_bias, candidates = real_forward(*args, **kwargs)
            return (
                compressed_kv,
                block_bias,
                candidates._replace(indexer_scores=candidates.indexer_scores[..., :-1]),
            )

        with _single_rank_parallel_state(), mock.patch.object(attn.compressor, "forward", _narrower_scores):
            with pytest.raises(AssertionError, match="must be the same width"):
                attn(**inputs)

    def test_a_declined_tilelang_dispatch_under_the_loss_is_refused(self):
        """The teacher's LSE comes out of the TileLang kernel, so a dispatch declined
        at runtime leaves the loss with no teacher at all.

        ``_indexer_loss_enabled`` can only check the *configured* implementation; the
        dispatch additionally requires bf16 CUDA tensors and declines silently on
        anything else. Without this refusal the caller would unpack three values from
        a two-value eager return and report a bare "not enough values to unpack",
        pointing at the loss's plumbing rather than at the dtype that caused it.
        """
        _require_tilelang_cuda()
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        _bind_indexer_loss(enabled=True)
        # fp32 is the realistic way to lose the dispatch: the kernel is bf16-only, and
        # the attention forward hands it whatever dtype the module was built in.
        query = torch.randn(1, 4, 8, 64, device="cuda")
        kv = torch.randn(1, 1, 24, 64, device="cuda")
        with pytest.raises(RuntimeError, match="needs the TileLang sparse attention dispatch"):
            modeling.eager_attention_forward(
                None, query, kv, kv, None, 0.1, sparse_topk_indices=None, indexer_target_width=8
            )

    def test_the_kl_gradient_reaches_the_indexer_and_stops_there(self):
        """The KL has to be a trainable objective for the indexer and *only* for the
        indexer, and it has to be finite.

        Three separate ways this can fail silently, all pinned here:

        * a KL detached from the indexer's own graph would log a plausible decreasing
          number while training nothing, so the assertion is on parameter gradients
          rather than on ``grad_fn``, and on their magnitude rather than their
          presence -- a graph carrying zeros trains nothing either;
        * a teacher or an LSE that stayed attached to the attention query would make
          the auxiliary objective a second gradient path into the language model,
          which ``hidden_states.grad is None`` is what rules out; and
        * the ``-inf`` score rows of the first positions of the sequence make the KL's
          gradient NaN unless they are neutralised, and NaN parameter gradients here
          would be blamed on anything but the auxiliary loss.
        """
        _require_tilelang_cuda()

        _bind_indexer_loss(enabled=True)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._RANKING_SEQ_LEN)
        with _single_rank_parallel_state():
            _, _, kl_sum = attn(**inputs)
        kl_sum.backward()

        assert inputs["hidden_states"].grad is None, "the indexer KL reached the language-model input"
        for name in ("q_b_proj", "kv_proj", "weights_proj"):
            grad = getattr(attn.compressor.indexer, name).weight.grad
            assert grad is not None, f"indexer.{name} received no gradient from the KL"
            assert torch.isfinite(grad).all(), f"indexer.{name} received a non-finite gradient from the KL"
            assert grad.abs().sum() > 0, f"indexer.{name} received an all-zero gradient from the KL"
        # Nothing outside the indexer may move on this objective. The compressor and
        # the attention projections sit on the language-model path.
        for module_name in ("compressor.kv_proj", "compressor.gate_proj", "q_b_proj", "kv_proj"):
            module = attn
            for part in module_name.split("."):
                module = getattr(module, part)
            assert module.weight.grad is None, f"{module_name} moved on the indexer objective"

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize(
        "layer_type", ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
    )
    def test_the_shared_gate_predicts_the_attention_return_arity(self, layer_type, enabled):
        """``_builds_indexer_kl`` is the *only* thing that may decide the arity, and
        every caller above reads it rather than re-deriving it.

        Three call sites act on this predicate -- the attention forward that returns
        the third value, the decoder layer that unpacks it, and the model loop that
        accumulates it -- and they are in three different functions. A copy of the
        expression in any of them can go stale independently: the decoder layer gating
        on ``_indexer_loss_enabled`` alone (the whole expression before Task 5's fix
        round narrowed it) would three-unpack the two-tuple every sliding and HCA layer
        returns, which is three of the four layers of the reference checkpoint.

        Reading the predicate twice would prove nothing, so the assertion is against
        ``len(attn(...))``: what the attention forward actually returned.
        """
        _require_tilelang_cuda()
        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        _bind_indexer_loss(enabled=enabled)
        attn, inputs = _build_test_attention(device="cuda", seq_len=self._CHEAP_SEQ_LEN, layer_type=layer_type)
        with _single_rank_parallel_state():
            verdict = modeling._builds_indexer_kl(attn)
            result = attn(**inputs)

        assert verdict is (len(result) == 3), (
            f"the gate says {verdict} for a {layer_type} layer with the flag {enabled}, "
            f"while the forward returned {len(result)} values"
        )
        # And that the verdict itself is the intended one, so that a predicate stuck at
        # ``False`` against a forward stuck at two values cannot agree vacuously.
        assert verdict is (enabled and layer_type == "compressed_sparse_attention")


# The reference 4-layer DeepSeek-V4 checkpoint. Its ``compress_ratios`` of
# ``[0, 0, 4, 128]`` is the layer schedule this whole task is about: two sliding
# layers, one CSA layer at rate 4 and one HCA layer at rate 128, i.e. exactly one
# layer carrying a Lightning Indexer and three that must pass a flag-on forward
# through untouched.
_CHECKPOINT_4L = Path("/mnt/hdfs/__MERLIN_USER_DIR__/veomni/models/DeepSeek/DeepSeek-V4-Flash-Base-4L")


def _4layer_test_config():
    """The 4-layer checkpoint's own config, with the widths nothing here reads reduced.

    Everything the indexer loss touches comes from the checkpoint verbatim:
    ``layer_types`` (from its legacy ``compress_ratios``), ``compress_rates``,
    ``sliding_window``, and the three fields the indexer's TileLang gate reads --
    ``index_n_heads=64``, ``index_head_dim=128``, ``index_topk=512``. What is reduced
    is width alone: hidden size, head dim, the two LoRA ranks, the expert count and
    width, and the vocabulary.

    Measured, not assumed. At the checkpoint's own widths this model is 27.4B
    parameters, 54.8 GB in bf16 before gradients; ``test_the_indexer_objective...``
    and its neighbours build several models per session, and one build alone leaves
    no room for a backward on the 189 GB this device has. At the widths below the
    model is 7.8M parameters and builds in 0.4 s, while every gate, kernel dispatch
    and layer type under test is still the checkpoint's own.

    Skips rather than fails when the checkpoint is absent: it lives on an HDFS FUSE
    mount that a machine running these tests need not have.
    """
    from transformers import AutoConfig

    if not (_CHECKPOINT_4L / "config.json").is_file():
        pytest.skip(f"the reference 4-layer DeepSeek-V4 checkpoint is not present at {_CHECKPOINT_4L}")

    config = AutoConfig.from_pretrained(str(_CHECKPOINT_4L))
    config.hidden_size = 256
    config.head_dim = 64
    config.q_lora_rank = 64
    config.o_lora_rank = 64
    config.n_routed_experts = 8
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 128
    config.vocab_size = 1024
    # An MTP head is a second stack of layers with its own attention, and nothing in
    # this file reads it; the toy config drops it the same way.
    config.num_nextn_predict_layers = 0
    # The checkpoint ships fp8 block-quantised experts. ``from_pretrained`` would act
    # on this; building from the config alone must not, and the weights are random
    # here in any case.
    config.quantization_config = None
    return config


@contextlib.contextmanager
def _veomni_loss_mapping_installed():
    """Install VeOmni's loss wrappers into HuggingFace's ``LOSS_MAPPING``, and undo it.

    ``DeepseekV4ForCausalLM.forward`` unpacks three values from
    ``self.loss_function``, which is VeOmni's contract and not HuggingFace's -- with
    HF's stock ``ForCausalLMLoss`` in the mapping the forward fails on the unpack
    long before it reaches anything this file is about. ``build_foundation_model``
    installs these through ``apply_ops_config``; a test that builds the model
    directly has to do it itself.

    Restored afterwards for the reason ``_ops_config_slots_bound`` gives: the mapping
    is a module-level dict shared with the rest of the session.
    """
    from transformers.loss.loss_utils import LOSS_MAPPING

    from veomni.ops.kernels.cross_entropy import install_loss_mapping

    saved = dict(LOSS_MAPPING)
    install_loss_mapping("eager")
    try:
        yield
    finally:
        LOSS_MAPPING.clear()
        LOSS_MAPPING.update(saved)


def _build_4layer_test_model(
    device="cuda", seq_len=4096, indexer_loss=True, enable_reentrant=False, seed=0, labels=True
):
    """Returns ``(model, batch)`` for the 4-layer CSA checkpoint's configuration.

    Builds the config above (skipping when the checkpoint is absent), binds
    ``dsa_indexer_loss`` / ``dsa_indexer_implementation`` / ``dsa_attention_implementation``,
    enables gradient checkpointing with the given ``enable_reentrant``, and seeds both
    the model init and the batch from ``seed`` so two calls with the same seed are
    comparable.

    ``seq_len=4096`` is not incidental: at the CSA compression rate of 4 it gives
    ``compressed_len == 1024 > index_topk == 512``, so the top-k actually ranks. At
    2048 the indexer would select every causally visible slot and a student permuted
    within the valid region would still look right.

    The batch is the packed one production trains on -- one sample filling the row,
    global ``position_ids``, and ``cu_seq_lens_q`` spanning it -- which is what puts
    ``DeepseekV4Model.forward`` on the mask-free TileLang path. ``labels`` are the
    inputs themselves: what is under test is which objectives the gradient reaches,
    not what the model predicts.

    ``enable_reentrant=True`` uses PyTorch's own reentrant ``CheckpointFunction``
    rather than the copy at ``veomni/distributed/checkpoint.py`` that
    ``build_parallelize_model`` installs. That copy reads ``run_function.__self__``,
    while ``GradientCheckpointingLayer.__call__`` hands the checkpoint a
    ``functools.partial``, so it raises ``AttributeError: 'functools.partial' object
    has no attribute '__self__'`` here -- with the indexer loss *off* as well as on,
    i.e. an existing incompatibility with this transformers version rather than
    anything this file introduces. Both implementations run the first forward inside
    ``torch.no_grad()``, which is the property these tests are about.
    """
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = _4layer_test_config()
    _bind_indexer_loss(enabled=indexer_loss)

    torch.manual_seed(seed)
    with torch.device(device):
        model = modeling.DeepseekV4ForCausalLM(config)
    model = model.to(dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": enable_reentrant})
    model.train()

    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    batch = {
        "input_ids": input_ids,
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "use_cache": False,
    }
    if labels:
        batch["labels"] = input_ids.clone()
    return model, batch


def _csa_layers(model):
    """The layers carrying a Lightning Indexer, i.e. the ones that can build a KL."""
    return [
        layer
        for layer in model.model.layers
        if getattr(getattr(layer.self_attn, "compressor", None), "indexer", None) is not None
    ]


def _model_only_batch(batch):
    """``batch`` without the fields only ``DeepseekV4ForCausalLM.forward`` takes."""
    return {key: value for key, value in batch.items() if key != "labels"}


class TestFourLayerModelIndexerKL:
    """The KL leaving the attention layer and arriving in the total loss: the decoder
    layer's arity, the model loop's accumulation, the reduction and the coefficient.

    Every test here runs a whole ``DeepseekV4ForCausalLM`` forward on the reference
    checkpoint's layer schedule, so three of its four layers exercise the pass-through
    the gate has to leave alone and one exercises the KL itself.
    """

    # Every test here imports the generated modeling module; see the note on
    # ``_HDFS_STDENV_DEPRECATION_FILTER`` for what that drags in.
    pytestmark = _HDFS_STDENV_DEPRECATION_FILTER

    # Supported but off, matching the classes above. The coefficient is here too:
    # every test below reads it through the loss, and one of them binds it to 0, so a
    # test that took it from the session rather than from here would be reading
    # whatever ran before it.
    _SLOT_PRE_STATE = {
        "dsa_indexer_loss": False,
        "dsa_indexer_implementation": "eager",
        "dsa_attention_implementation": "eager",
        "dsa_indexer_loss_coef": 1.0,
    }

    @pytest.fixture(autouse=True)
    def _restore_ops_config_slots_and_loss_mapping(self):
        with _ops_config_slots_bound(self._SLOT_PRE_STATE), _veomni_loss_mapping_installed():
            yield

    @pytest.mark.parametrize("enable_reentrant", [False, True])
    def test_indexer_receives_gradient_under_checkpointing(self, enable_reentrant):
        """The silent-failure guard, and this task's acceptance criterion: a flag-on
        forward *and backward* completing on the reference schedule.

        Asserts parameter gradients, not ``kl.grad_fn is not None``. Under reentrant
        checkpointing the first forward runs inside ``torch.no_grad()``, so any tensor
        smuggled out through a side channel rather than through the layer's return
        value is graph-less -- and a ``grad_fn`` proxy would not notice. The KL would
        still log a plausible decreasing number while the indexer learned nothing.
        """
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096, enable_reentrant=enable_reentrant)
        with _single_rank_parallel_state():
            out = model(**batch)
            out.loss.backward()

        csa = _csa_layers(model)
        assert csa, "test model must contain at least one CSA layer"
        for layer in csa:
            for name in ("q_b_proj", "kv_proj", "gate_proj", "weights_proj"):
                param = getattr(layer.self_attn.compressor.indexer, name).weight
                assert param.grad is not None, f"{name} received no gradient"
                assert param.grad.abs().sum() > 0, f"{name} received an all-zero gradient"

    def test_only_csa_layers_carry_an_indexer(self):
        """The reference checkpoint has ``compress_ratios [0, 0, 4, 128]``: two sliding
        layers, one CSA layer at rate 4, one HCA layer at rate 128. Only the CSA layer
        has an indexer, so only it may contribute to the KL, and the total must be a
        finite positive number rather than a silent zero."""
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096)

        with_indexer = [
            i
            for i, layer in enumerate(model.model.layers)
            if getattr(getattr(layer.self_attn, "compressor", None), "indexer", None) is not None
        ]
        assert with_indexer == [2], f"expected exactly the rate-4 layer to have an indexer, got {with_indexer}"

        with _single_rank_parallel_state():
            out = model(**batch)
        kl = out.aux_metrics["indexer_kl"]
        assert torch.isfinite(kl).all()
        assert kl > 0, "a zero KL here means the teacher and the indexer already agree exactly, which is not credible"

    def test_the_model_output_carries_the_kl_where_a_round_trip_can_find_it(self):
        """The KL leaves ``DeepseekV4Model.forward`` in a declared field, not as an
        attribute assigned onto the output object.

        ``ModelOutput.__setattr__`` writes into the underlying dict only for keys that
        already exist; a new name becomes a plain instance attribute, invisible to
        ``keys()``, to ``to_tuple()`` and to pytree flattening, and dropped by any
        round-trip. The immediate read in ``ForCausalLM.forward`` would still work, so
        nothing about the loss value would look wrong -- and this repository has been
        bitten by exactly that before, with FSDP2's pre-backward unshard hook, which
        finds the tensors it must unshard for by walking that same flattened output
        (see the note above ``MoeCausalLMOutputWithLogProbs``'s import in the patch
        config).

        A flat-out ``kl > 0`` assertion elsewhere does not cover this: the value is
        right either way. What has to hold is that the tensor is *reachable* the way
        every generic consumer reaches it.
        """
        _require_tilelang_cuda()
        from torch.utils import _pytree as pytree

        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096)
        with _single_rank_parallel_state():
            outputs = model.model(**_model_only_batch(batch))

        assert outputs.indexer_kl_total is not None, "the model dropped the KL"
        assert "indexer_kl_total" in outputs, "the KL is an ad-hoc attribute, not a field of the output"
        assert outputs.indexer_query_tokens == batch["input_ids"].numel()

        leaves, spec = pytree.tree_flatten(outputs)
        assert any(leaf is outputs.indexer_kl_total for leaf in leaves), (
            "the KL is invisible to pytree flattening, which is how FSDP2 finds the tensors "
            "a backward will need unsharded"
        )
        rebuilt = pytree.tree_unflatten(leaves, spec)
        assert rebuilt.indexer_kl_total is outputs.indexer_kl_total
        assert rebuilt.indexer_query_tokens == outputs.indexer_query_tokens

    def test_the_reported_kl_is_the_per_token_mean_of_the_layer_sum(self):
        """What reaches the loss is a *mean* over local query rows, not the sum the
        attention layer returns.

        ``reduce_sequence_parallel_loss`` takes a local mean and the local token count
        and re-weights by that count across the group; handing it a sum instead trains
        perfectly well on one rank and converges to the wrong cross-rank weighting,
        which no single-process test of the value alone would ever notice. The two
        differ here by a factor of 4096, so the check is exact rather than statistical.
        """
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096)
        with _single_rank_parallel_state():
            outputs = model.model(**_model_only_batch(batch))
            out = model(**batch)

        tokens = batch["input_ids"].numel()
        assert outputs.indexer_query_tokens == tokens
        torch.testing.assert_close(out.aux_metrics["indexer_kl"], outputs.indexer_kl_total / tokens, atol=0, rtol=1e-6)
        assert not out.aux_metrics["indexer_kl"].requires_grad, "the reported metric must be detached"

    def test_the_coefficient_scales_what_the_loss_receives(self):
        """The KL enters the loss weighted by ``dsa_indexer_loss_coef``, and by nothing
        else.

        Run at coefficient 0 the total loss must be *bitwise* what the flag-off run
        produced -- ``x + 0.0 * kl`` is exactly ``x`` for any finite KL -- which pins
        two things at once: that the coefficient is really applied, and that turning
        the loss on perturbs no part of the language-model objective. Run at 1 the
        difference from that baseline must be the reported KL itself.
        """
        _require_tilelang_cuda()
        from types import SimpleNamespace

        from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

        def loss_at(indexer_loss, coef):
            model, batch = _build_4layer_test_model(device="cuda", seq_len=4096, indexer_loss=indexer_loss, seed=0)
            modeling.veomni_dsa_indexer_loss_coef.bind(SimpleNamespace(dsa_indexer_loss_coef=coef))
            with _single_rank_parallel_state():
                out = model(**batch)
            return out

        off = loss_at(indexer_loss=False, coef=1.0)
        assert off.aux_metrics is None, "the flag-off forward reported an indexer metric"

        zero_weighted = loss_at(indexer_loss=True, coef=0.0)
        assert torch.equal(zero_weighted.loss, off.loss), (
            "at coefficient 0 the total loss moved, so either the coefficient is ignored "
            "or the language-model objective was perturbed"
        )

        weighted = loss_at(indexer_loss=True, coef=1.0)
        difference = weighted.loss - off.loss
        reported = weighted.aux_metrics["indexer_kl"].to(off.loss.dtype)
        # Subtracting two losses of the same magnitude cancels, and what survives is
        # the KL plus a few ULPs of the larger number. Bound the comparison by those
        # ULPs rather than by a relative tolerance on the KL itself, and assert the KL
        # clears the bound -- otherwise this comparison would pass on a KL of zero.
        admitted = 8 * torch.finfo(off.loss.dtype).eps * off.loss.detach().abs().item()
        assert reported > 100 * admitted, (
            f"the reported KL {float(reported):.3e} is within the {float(admitted):.3e} of "
            "floating-point noise this subtraction carries, so it cannot be told from zero here"
        )
        torch.testing.assert_close(difference, reported, atol=admitted, rtol=0)

    def test_the_indexer_objective_moves_only_the_indexer(self):
        """Decoupling at model scope: the auxiliary objective may move indexer
        parameters and nothing else.

        The KL leaving ``DeepseekV4Model.forward`` in a declared field is what makes
        this exact. Comparing two whole runs' gradients would not be: measured on this
        model, two identical flag-off runs already disagree on 77 of their 102
        gradients at the bf16 ULP level, because the backward's reductions are not
        deterministic -- so an equality comparison there cannot separate a leak from
        the noise floor. Backpropagating the KL alone has no noise floor at all: a
        parameter outside the indexer either receives a gradient from it or does not.
        """
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096)
        with _single_rank_parallel_state():
            outputs = model.model(**_model_only_batch(batch))
        outputs.indexer_kl_total.backward()

        moved = [name for name, param in model.named_parameters() if param.grad is not None]
        assert moved, "the model-level KL trained nothing at all"
        assert all(".indexer." in name for name in moved), (
            f"the indexer objective moved parameters outside the indexer: "
            f"{[name for name in moved if '.indexer.' not in name]}"
        )
        for name in ("q_b_proj", "kv_proj", "gate_proj", "weights_proj"):
            grad = getattr(_csa_layers(model)[0].self_attn.compressor.indexer, name).weight.grad
            assert grad is not None and grad.abs().sum() > 0, f"indexer.{name} received no usable gradient"

    def test_the_flag_off_model_forward_is_untouched(self):
        """The off side of every branch this task adds, at model scope.

        With the flag off the decoder layer must still return a bare tensor, the model
        output must carry no indexer fields, and the causal-LM output must report no
        aux metrics. Inverting any of those gates would break every flag-off DeepSeek-V4
        forward in the tree while leaving the enabled path green.
        """
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096, indexer_loss=False)
        with _single_rank_parallel_state():
            outputs = model.model(**_model_only_batch(batch))
            out = model(**batch)

        assert outputs.indexer_kl_total is None
        assert outputs.indexer_query_tokens is None
        assert list(outputs.keys()) == ["last_hidden_state"], (
            f"the flag-off model output grew fields: {list(outputs.keys())}"
        )
        assert out.aux_metrics is None
        assert torch.isfinite(out.loss)
        for layer in _csa_layers(model):
            for name in ("q_b_proj", "kv_proj", "gate_proj", "weights_proj"):
                param = getattr(layer.self_attn.compressor.indexer, name).weight
                assert param.grad is None

        out.loss.backward()
        for layer in _csa_layers(model):
            for name in ("q_b_proj", "kv_proj", "gate_proj", "weights_proj"):
                param = getattr(layer.self_attn.compressor.indexer, name).weight
                assert param.grad is None, f"indexer.{name} received a gradient with the loss disabled"

    def test_the_kl_is_reported_but_not_added_when_there_are_no_labels(self):
        """Inference has no loss to add the KL to.

        ``loss`` is ``None`` without labels, so a fold-in that did not check would
        raise ``TypeError: unsupported operand type(s) for +: 'NoneType'``. The metric
        is still reported, which is what makes this distinguishable from skipping the
        KL entirely.
        """
        _require_tilelang_cuda()
        model, batch = _build_4layer_test_model(device="cuda", seq_len=4096, labels=False)
        with _single_rank_parallel_state():
            out = model(**batch)

        assert out.loss is None
        assert out.aux_metrics is not None and out.aux_metrics["indexer_kl"] > 0
