# Encoder balance scheduling costs

The shared greedy sorter in `veomni/utils/data_balance/balance_sorting_algo.py`
supports module-specific scheduling costs without changing item ownership,
payload schemas, or token accounting.

Existing three-argument callers retain descending-length sorting and quadratic
load accumulation. New callers can choose a nonnegative `cost_exponent` (for
example, `1` for linear token cost) or a `cost_fn` that takes the selected length
column and returns one finite, nonnegative cost per item. A callable overrides
the exponent; its values are sorted and accumulated directly, not squared again.
The callable must be deterministic across participating ranks. Lengths and
scheduling costs are separate from consumed-token counters.

```python
linear_assignment = post_mbs_balancing_greedy_without_pad(
    item_table, num_replicas=8, dim=2, cost_exponent=1
)
custom_assignment = post_mbs_balancing_greedy_without_pad(
    item_table, num_replicas=8, dim=2, cost_fn=module_cost
)
```

Empty batches and fewer items than replicas produce empty per-rank tables with
the same column count, dtype, and device. This does not by itself make a legacy
exchange caller empty-rank-safe; that caller must support empty payloads too.

The existing distributed reverse test defaults to eight ranks, as in CI. For a
smaller test machine, set `VEOMNI_BALANCE_WORLD_SIZE=4` when invoking pytest;
the worker derives its DP topology from the torchrun world size.

This is the independent shared-cost step of
[#1074](https://github.com/ByteDance-Seed/VeOmni/issues/1074), following the
[maintainer's split](https://github.com/ByteDance-Seed/VeOmni/issues/1074#issuecomment-5693718137).
It does not introduce the V2 mixin, strategy resolver, YAML override, or model
consumers. Those belong to subsequent `omniv2` PRs. Dataloader-level compute
packing remains a separate follow-up.
