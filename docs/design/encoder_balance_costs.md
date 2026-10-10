# Encoder balance scheduling costs

The shared greedy sorter in `veomni/utils/data_balance/balance_sorting_algo.py`
supports module-specific scheduling costs without changing item ownership,
payload schemas, or token accounting.

Existing three-argument callers retain descending-length sorting and quadratic
load accumulation. New callers can choose a nonnegative `cost_exponent` (for
example, `1` for linear token cost) or a `cost_fn(table, dim)` that receives a cloned full item table and returns
finite, nonnegative costs shaped `[N]` or `[N, 1]`. The callable and an explicitly
supplied exponent are mutually exclusive. Callable costs are sorted and
accumulated directly, not squared again. The `[N, K]` return shape reserves room
for vector scheduling; `K > 1` currently raises `NotImplementedError`, rather
than silently collapsing separate module costs into one scalar.
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

The default path copies the sorted rows to the host once and derives validation
and costs from those rows. A callable requires one additional transfer for its
costs. An integer exponent (including the default `2`) preserves exact Python
integer costs; `2.0` uses floating-point exponentiation instead.
