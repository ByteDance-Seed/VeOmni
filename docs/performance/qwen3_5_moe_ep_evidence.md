# Qwen3.5-MoE EP evidence index

The [current validation report](qwen3_5_moe_ep_load_balance.md) is the public
summary for the replica-storage and native NPU precision implementation.
It includes complete GPU paired timing rows, per-rank scalar curves and native
NPU load distributions, with explicit source and software identities.

| Question | Current evidence |
| --- | --- |
| Does balancing actually execute? | GPU ON steps and 480 native NPU plans show positive replicas/moves and improved physical loads. |
| Is precision preserved? | GPU 100-step and NPU 20-step loss/gradient-norm curves are exact; native NPU partial-migration cases preserve owner operands and gradients. |
| Does GPU throughput regress? | Observed paired mean throughput does not regress; variability and the slow-OFF sensitivity are retained. A repeatable 5% gain is not established. |
| Does NPU throughput regress? | Yes. Current native NPU Memory is slower than matched OFF, so NPU performance acceptance remains open. |
| What does Memory save? | It avoids concatenating the full original and replica weights, keeping allocated memory near OFF; this does not guarantee a speedup. |
| What is untested? | Full-depth and additional topology/data combinations; no broad GPU/NPU bitwise-equivalence claim. |

See the [design](../design/qwen3_5_moe_ep_load_balance.md) for lifecycle and
arithmetic invariants, and the [training guide](../examples/qwen3_5.md#qwen35-moe-ep-load-balance-experiments)
for configuration and comparison templates. Older intermediate experiment
ledgers are excluded from this concise publication rather than relabeled as
validation of the current source.
