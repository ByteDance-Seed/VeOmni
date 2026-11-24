"""What the deterministic token permutation costs, and when the atomics it replaced drift.

Both directions of the expert-parallel permutation sum over a token's `top_k` entries.
Spelled with atomics -- `scatter_add_` on the token index for the unpermute, autograd's
`index_add` behind a duplicated `index_select` for the permute's backward -- the order
varies from call to call. `sum_entries_per_token` groups an entry with its token's others
and sums over that axis instead.

Two questions this answers, both of which have been guessed wrong before:

1. What the fixed order costs. Measured on one A800 at 8192 tokens x 2048 hidden,
   top-k 8, bf16:

       unpermute           4.51 ms -> 2.32 ms     peak 864 MiB -> 641 MiB
       permute backward    7.12 ms -> 2.03 ms     peak 352 MiB -> 673 MiB

   Nothing is traded away for the determinism. Atomics contending over `top_k`
   duplicate destinations cost more than the gather the grouping needs; only the
   backward pays anything, in transient memory, for the permuted copy it materialises.

2. When the old spelling can be caught drifting, which is not everywhere. The
   unpermute's atomics accumulate in FP32 and disagree by ~1e-6, so a bfloat16 result
   -- the production width -- discards it and repeats bit for bit in isolation. The
   permute's backward accumulated in bfloat16 itself and drifts in every width. A
   regression test that only checks bfloat16 forwards therefore proves nothing.

       python scripts/bench_moe_token_permutation.py

Requires a GPU.
"""

from functools import partial

import torch

from veomni.distributed.moe.moe_utils import permute, unpermute


COST_SHAPE = dict(num_tokens=8192, hidden=2048, num_experts=64, top_k=8)
DRIFT_TRIALS = 16


def build_routing(num_tokens, hidden, num_experts, top_k, dtype):
    """The tensors the expert-parallel path builds, in its entry order."""
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    weights, experts = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)

    routing_map = torch.zeros(num_tokens, num_experts, device="cuda")
    routing_map.scatter_(1, experts, 1.0)
    routing_map = routing_map.T.contiguous()

    weights_idx = torch.zeros(num_tokens, num_experts, device="cuda", dtype=dtype)
    weights_idx.scatter_(1, experts, weights.to(dtype))

    token_indices = torch.arange(num_tokens, device="cuda").unsqueeze(0).expand(num_experts, -1)
    mapping = token_indices.masked_select(routing_map.bool())
    hidden_states = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    return hidden_states, weights_idx, mapping, routing_map


def unpermute_with_atomics(entries, weights_idx, shape, mapping, routing_map, top_k):
    """The reduction this replaced: FP32 atomics on the token index."""
    weights = weights_idx.T.contiguous().masked_select(routing_map.bool())
    terms = entries * weights.unsqueeze(-1)
    out = torch.zeros(shape, device=entries.device, dtype=torch.float32)
    out.scatter_add_(0, mapping.unsqueeze(1).expand(-1, shape[-1]), terms.float())
    return out.to(terms.dtype)


def permute_backward_with_atomics(hidden_states, routing_map, upstream, top_k):
    """The gradient this replaced: a bare index_select, inverted by autograd."""
    num_experts, num_tokens = routing_map.shape
    token_indices = torch.arange(num_tokens, device="cuda").unsqueeze(0).expand(num_experts, -1)
    mapping = token_indices.masked_select(routing_map.bool())
    leaf = hidden_states.detach().clone().requires_grad_(True)
    leaf.index_select(0, mapping).backward(upstream)
    return leaf.grad


def permute_backward(hidden_states, routing_map, upstream, top_k):
    leaf = hidden_states.detach().clone().requires_grad_(True)
    permute(leaf, routing_map, top_k)[0].backward(upstream)
    return leaf.grad


def measure(call):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(50):
        out = call()
    end.record()
    torch.cuda.synchronize()

    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    del out
    return start.elapsed_time(end) / 50, peak


def report_cost():
    num_tokens, hidden, num_experts, top_k = COST_SHAPE.values()
    hidden_states, weights_idx, mapping, routing_map = build_routing(
        num_tokens, hidden, num_experts, top_k, torch.bfloat16
    )
    entries = torch.randn(mapping.numel(), hidden, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(mapping.numel(), hidden, device="cuda", dtype=torch.bfloat16)
    shape = torch.Size((num_tokens, hidden))

    print(f"cost at {num_tokens} tokens x {hidden} hidden, top-k {top_k}, bfloat16\n")
    print(f"{'':<20}{'atomics':>22}{'grouped sum':>22}")
    for name, atomic, grouped in (
        (
            "unpermute",
            partial(unpermute_with_atomics, entries, weights_idx, shape, mapping, routing_map, top_k),
            partial(unpermute, entries, weights_idx, shape, mapping, routing_map, top_k),
        ),
        (
            "permute backward",
            partial(permute_backward_with_atomics, hidden_states, routing_map, upstream, top_k),
            partial(permute_backward, hidden_states, routing_map, upstream, top_k),
        ),
    ):
        was_ms, was_mib = measure(atomic)
        now_ms, now_mib = measure(grouped)
        print(f"{name:<20}{f'{was_ms:.2f} ms / {was_mib:.0f} MiB':>22}{f'{now_ms:.2f} ms / {now_mib:.0f} MiB':>22}")


def report_drift():
    print(f"\nhow often the atomics disagree with themselves over {DRIFT_TRIALS} repeats\n")
    print(f"{'dtype':<10}{'experts/top_k':<16}{'unpermute':>14}{'permute bwd':>14}{'worst':>12}")

    for dtype in (torch.float32, torch.bfloat16):
        for num_experts, top_k in ((32, 4), (128, 8)):
            hidden_states, weights_idx, mapping, routing_map = build_routing(512, 128, num_experts, top_k, dtype)
            entries = torch.randn(mapping.numel(), 128, device="cuda", dtype=dtype)
            upstream = torch.randn(mapping.numel(), 128, device="cuda", dtype=dtype)
            shape = torch.Size((512, 128))

            worst = 0.0
            counts = []
            for call in (
                partial(unpermute_with_atomics, entries, weights_idx, shape, mapping, routing_map, top_k),
                partial(permute_backward_with_atomics, hidden_states, routing_map, upstream, top_k),
            ):
                first, differing = call(), 0
                for _ in range(DRIFT_TRIALS):
                    again = call()
                    if not torch.equal(again, first):
                        differing += 1
                        worst = max(worst, (again.double() - first.double()).abs().max().item())
                counts.append(differing)

            print(
                f"{str(dtype).replace('torch.', ''):<10}{f'{num_experts}/{top_k}':<16}"
                f"{f'{counts[0]}/{DRIFT_TRIALS}':>14}{f'{counts[1]}/{DRIFT_TRIALS}':>14}{worst:>12.2e}"
            )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures an accumulation order, which needs a GPU")
    torch.manual_seed(0)
    report_cost()
    report_drift()


if __name__ == "__main__":
    main()
