"""CPU equivalence tests for Qwen3.5 shared-prefix training.

Each mixer is run twice on the same compact rows: once through the plan, once
by expanding to the packed layout and calling the reference kernel per packed
sequence. Outputs and the gradients that flow back into the compact rows must
match. The reference kernels are naive float64 loops, so any difference is a
layout bug, not rounding.
"""

import pytest
import torch
import torch.nn.functional as F

from veomni.models.transformers.qwen3_5.shared_prefix import build_shared_prefix_plan


torch.manual_seed(0)
DT = torch.float64


def _cu(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def ref_conv(x, cu_seqlens, weight, activation=True):
    """Depthwise causal conv per packed sequence; x [1, T, D], weight [D, W]."""
    width = weight.shape[1]
    out = []
    cu = cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        seq = F.pad(x[0, a:b].t(), (width - 1, 0))  # [D, L + W - 1]
        y = F.conv1d(seq[None], weight[:, None, :], groups=weight.shape[0])[0].t()
        out.append(F.silu(y) if activation else y)
    return torch.cat(out)[None]


def ref_gdn(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None):
    """Recurrent gated delta rule per packed sequence. q,k [1,T,H,K]; v [1,T,H,V]; g,beta [1,T,H]."""
    cu = cu_seqlens.tolist()
    heads, dk, dv = q.shape[2], q.shape[3], v.shape[3]
    outs, finals = [], []
    for n, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        state = q.new_zeros(heads, dk, dv) if initial_state is None else initial_state[n]
        for t in range(a, b):
            state = state * g[0, t].exp()[:, None, None]
            kt, vt = k[0, t], v[0, t]
            pred = torch.einsum("hk,hkv->hv", kt, state)
            state = state + torch.einsum("hk,hv->hkv", kt, beta[0, t][:, None] * (vt - pred))
            outs.append(torch.einsum("hk,hkv->hv", q[0, t], state))
        finals.append(state)
    return torch.stack(outs)[None], (torch.stack(finals) if output_final_state else None)


def ref_attention(q, k, v, cu_q, cu_k):
    """Bottom-right causal attention per packed sequence. q [1,Tq,H,d], k,v [1,Tk,H,d]."""
    out = []
    for (qa, qb), (ka, kb) in zip(
        zip(cu_q[:-1].tolist(), cu_q[1:].tolist()), zip(cu_k[:-1].tolist(), cu_k[1:].tolist())
    ):
        lq, lk = qb - qa, kb - ka
        scores = torch.einsum("qhd,khd->hqk", q[0, qa:qb], k[0, ka:kb]) / q.shape[-1] ** 0.5
        mask = torch.arange(lk)[None, :] > (torch.arange(lq)[:, None] + lk - lq)
        scores = scores.masked_fill(mask, float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[0, ka:kb]))
    return torch.cat(out)[None]


def _batch(prefixes, suffix_lengths, singles=(), vocab=50):
    """Packed ids: for each group a shared random prefix and distinct suffixes; plus singleton sequences."""
    seqs = []
    for prefix_len, suffixes in zip(prefixes, suffix_lengths):
        prefix = torch.randint(1, vocab, (prefix_len,))
        for s in suffixes:
            suffix = torch.randint(1, vocab, (s,))
            suffix[0] = vocab + len(seqs)  # make suffixes diverge at their first token
            seqs.append(torch.cat([prefix, suffix]))
    for length in singles:
        seqs.append(torch.randint(1, vocab, (length,)))
    perm = torch.randperm(len(seqs))  # groups need not be contiguous
    seqs = [seqs[i] for i in perm]
    ids = torch.cat(seqs)[None]
    return ids, _cu([len(s) for s in seqs])


CASES = [
    ([130], [[5, 9, 1]], ()),  # unaligned prefix, tail replay
    ([128], [[7, 3]], ()),  # aligned prefix, empty tail
    ([70, 200], [[4, 4, 4, 4], [11, 2]], (33, 90)),  # two groups + singletons
    ([65], [[1, 1]], (64, 3)),  # tiny suffixes
]


@pytest.mark.parametrize("prefixes,suffixes,singles", CASES)
def test_plan_layout(prefixes, suffixes, singles):
    ids, cu = _batch(prefixes, suffixes, singles)
    plan = build_shared_prefix_plan(ids, cu, conv_kernel_size=4)
    assert plan is not None
    saved = sum(p * (len(s) - 1) for p, s in zip(prefixes, suffixes))
    assert plan.compact_length == ids.shape[1] - saved
    assert torch.equal(plan.expand(plan.compact(ids)), ids)


def test_no_sharing_returns_none():
    ids, cu = _batch([], [], (100, 80, 70))
    assert build_shared_prefix_plan(ids, cu) is None


def test_short_prefix_not_grouped():
    ids, cu = _batch([30], [[5, 5]], (100,))
    assert build_shared_prefix_plan(ids, cu, min_prefix_length=64) is None


def test_mismatched_positions_not_grouped():
    ids, cu = _batch([100], [[5, 5]])
    pos = torch.cat([torch.arange(b - a) for a, b in zip(cu[:-1].tolist(), cu[1:].tolist())])[None]
    pos[0, cu[1] + 10] += 1  # second member's prefix positions differ
    assert build_shared_prefix_plan(ids, cu, position_ids=pos) is None


def _check(plan, run_plan, run_ref, compact_inputs):
    leaves = [t.clone().requires_grad_() for t in compact_inputs]
    out = run_plan(*leaves)
    grad_out = torch.randn_like(plan.expand(out))
    (plan.expand(out) * grad_out).sum().backward()
    grads = [t.grad for t in leaves]

    leaves_ref = [t.clone().requires_grad_() for t in compact_inputs]
    out_ref = run_ref(*(plan.expand(t) for t in leaves_ref))
    (out_ref * grad_out).sum().backward()
    torch.testing.assert_close(plan.expand(out), out_ref, rtol=1e-10, atol=1e-10)
    for g, t in zip(grads, leaves_ref):
        torch.testing.assert_close(g, t.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("prefixes,suffixes,singles", CASES)
def test_conv_matches_packed(prefixes, suffixes, singles):
    ids, cu = _batch(prefixes, suffixes, singles)
    plan = build_shared_prefix_plan(ids, cu, conv_kernel_size=4)
    weight = torch.randn(6, 4, dtype=DT)
    x = torch.randn(1, plan.compact_length, 6, dtype=DT)

    def conv(x, cu_seqlens):
        return ref_conv(x, cu_seqlens, weight)

    _check(plan, lambda x: plan.causal_conv1d(conv, x), lambda x: ref_conv(x, cu, weight), [x])


@pytest.mark.parametrize("prefixes,suffixes,singles", CASES)
def test_gated_delta_rule_matches_packed(prefixes, suffixes, singles):
    ids, cu = _batch(prefixes, suffixes, singles)
    plan = build_shared_prefix_plan(ids, cu)
    c, h, dk, dv = plan.compact_length, 2, 3, 4
    q = torch.randn(1, c, h, dk, dtype=DT)
    k = F.normalize(torch.randn(1, c, h, dk, dtype=DT), dim=-1)
    v = torch.randn(1, c, h, dv, dtype=DT)
    g = -torch.rand(1, c, h, dtype=DT) * 0.1
    beta = torch.rand(1, c, h, dtype=DT)

    def run_plan(q, k, v, g, beta):
        return plan.gated_delta_rule(ref_gdn, q, k, v, g, beta)

    def run_ref(q, k, v, g, beta):
        return ref_gdn(q, k, v, g, beta, cu_seqlens=cu)[0]

    _check(plan, run_plan, run_ref, [q, k, v, g, beta])


@pytest.mark.parametrize("prefixes,suffixes,singles", CASES)
def test_attention_matches_packed(prefixes, suffixes, singles):
    ids, cu = _batch(prefixes, suffixes, singles)
    plan = build_shared_prefix_plan(ids, cu)
    c, h, d = plan.compact_length, 2, 8
    q, k, v = (torch.randn(1, c, h, d, dtype=DT) for _ in range(3))

    def run_plan(q, k, v):
        kk, vv = plan.attention_kv(k, v, seq_dim=1)
        kw = plan.attention_kwargs()
        return ref_attention(q, kk, vv, kw["cu_seq_lens_q"], kw["cu_seq_lens_k"])

    def run_ref(q, k, v):
        return ref_attention(q, k, v, cu, cu)

    _check(plan, run_plan, run_ref, [q, k, v])


def test_kernel_calls_independent_of_group_size():
    for n in (2, 8):
        ids, cu = _batch([130, 130], [[3] * n, [5] * n])
        plan = build_shared_prefix_plan(ids, cu)
        calls = []

        def gdn(*args, _calls=calls, **kwargs):
            _calls.append("gdn")
            return ref_gdn(*args, **kwargs)

        c = plan.compact_length
        t = [torch.randn(1, c, 1, 2, dtype=DT) for _ in range(3)]
        plan.gated_delta_rule(gdn, *t, -torch.rand(1, c, 1, dtype=DT), torch.rand(1, c, 1, dtype=DT))
        assert calls == ["gdn", "gdn"]


@pytest.mark.parametrize("prefixes,suffixes,singles", CASES)
def test_lm_rows_match_packed(prefixes, suffixes, singles):
    ids, cu = _batch(prefixes, suffixes, singles)
    plan = build_shared_prefix_plan(ids, cu)
    shift = torch.roll(ids, -1, 1)
    shift[0, cu[1:] - 1] = -100  # last token of each sequence predicts nothing
    vocab, dim = 120, 5
    head = torch.randn(vocab, dim, dtype=DT)
    hidden = torch.randn(1, plan.compact_length, dim, dtype=DT)

    def logp(h, labels):
        lp = (h[0] @ head.t()).log_softmax(-1)
        return torch.where(labels >= 0, lp.gather(1, labels.clamp_min(0)[:, None])[:, 0], 0.0)

    rows, labels, inverse = plan.lm_rows(shift)
    shared = logp(hidden.index_select(1, rows), labels).index_select(0, inverse)
    ref = logp(plan.expand(hidden), shift[0])
    torch.testing.assert_close(shared, ref, rtol=1e-12, atol=1e-12)
    groups = [g for g in plan.groups if not g.is_singleton]
    saved = sum((g.prefix_length - 1) * (len(g.starts) - 1) for g in groups)
    assert rows.numel() == ids.shape[1] - saved


def test_groups_sharing_a_system_prompt_stay_separate():
    """Two prompts with a common head must form two groups with their full prompts as prefixes."""
    torch.manual_seed(3)
    system = torch.randint(1, 50, (100,))
    seqs = []
    for _ in range(2):
        prompt = torch.cat([system, torch.randint(50, 99, (300,))])
        for _ in range(4):
            seqs.append(torch.cat([prompt, torch.tensor([100 + len(seqs)]), torch.randint(1, 99, (9,))]))
    ids = torch.cat(seqs)[None]
    cu = _cu([len(s) for s in seqs])
    plan = build_shared_prefix_plan(ids, cu)
    shared = sorted((len(g.starts), g.prefix_length) for g in plan.groups if not g.is_singleton)
    assert shared == [(4, 400), (4, 400)]
    assert plan.compact_length == ids.shape[1] - 2 * 3 * 400
