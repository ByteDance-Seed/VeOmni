# Fused MoE Kernel Notes

This note records implementation invariants for VeOmni's GPU fused MoE kernels.
The DeepSeek-V4 hash-MoE CI failure exposed one of these invariants, but the
invariant itself is generic and applies to the non-EP `fused_triton` MoE path.

## Scatter/gather layout

The non-EP fused MoE path expands tokens by their top-k routes before running
per-expert grouped GEMMs:

```text
original hidden states: [tokens, hidden]
scattered activations: [tokens * topk, hidden]
```

`scatter_index` maps each `(token, topk_slot)` pair into the expert-grouped
scattered tensor. Later, `moe_gather()` sums the routed expert outputs back to
`[tokens, hidden]`.

## Backward grouped-GEMM launch bound

For `group_gemm_same_nk`, `max_M` is a per-expert launch bound. It must cover
the largest expert group in the scattered tensor. The total scattered row count
is a conservative safe bound:

```python
grad_fc2_output = moe_scatter(grad_output, scatter_index)
num_scattered_tokens = grad_fc2_output.shape[0]
```

Use `num_scattered_tokens` for the downstream `group_gemm_same_nk(...,
max_M=...)` calls in non-EP backward. The matching `group_gemm_same_mn(...,
max_K=...)` value is kept consistent with the scattered input row count, but the
stale-row correctness issue is about `group_gemm_same_nk` under-covering its
output rows.

Do not use the original token count as the backward grouped-GEMM bound. It is
only safe when every expert group has at most `tokens` scattered rows.

## Why ordinary top-k routing may hide the bug

Consider `tokens = 128` and `topk = 2`.

If each token routes to two different experts in a balanced pattern:

```text
token0 -> expert0, expert1
token1 -> expert0, expert1
...
token127 -> expert0, expert1
```

After grouping by expert:

```text
expert0: 128 rows
expert1: 128 rows
```

Each expert group has at most `tokens` rows, so a stale bound like
`max_M = tokens` may still cover every group. The implementation would still be
relying on the wrong invariant, but this routing pattern may not expose it.

## Duplicate top-k routes

Some routers can assign multiple top-k slots of the same token to the same
expert. DeepSeek-V4 hash-MoE can do this when the token-to-expert lookup maps
both slots to the same expert.

For `tokens = 128` and `topk = 2`:

```text
token0 -> expert0, expert0
token1 -> expert0, expert0
...
token127 -> expert0, expert0
```

After grouping by expert:

```text
expert0: 256 rows
```

With `BLOCK_M = 128`, a stale `max_M = 128` launches enough blocks for only the
first 128 rows of that expert group. Rows 128-255 can remain uncomputed or
stale, and later gather/gradient operations may consume arbitrary values,
including `inf` or `nan`. A visible symptom is a non-finite gradient norm such
as `gnorm = nan`.

## Regression test pattern

A focused regression test should explicitly construct duplicate top-k routes:

```text
selected_experts = zeros([tokens, topk])
routing_weights = full([tokens, topk], weight)
```

Then compare split-fc1 fused, merged-fc1 fused, and eager MoE backward results,
and assert all produced gradients are finite.
