# Declarative encoder data balance

SeedOmni's accelerated modules opt in by multiple inheritance from
`DataBalanceMixin`, separately from `BaseMixin`. The training executor discovers
that capability with `isinstance` and opens a per-node call scope. Native HF
graph execution and inference do not open the scope or exchange DP items.

Each call-site declares a `DataBalanceSpec` containing:

- Item-partitioned input tensors and their per-item split metadata keys.
- A scheduling metric and its cost exponent or cost callable.
- Output split metadata and the tensor/list output names requiring an inverse.
- Structural capabilities: global attention, non-overlapping patchification and
  support for whole-item partitioning.

The cost metric and output splits are independent. A VAE can schedule latent
counts without pretending its batch-dimensional output has that many rows.
Callable costs are accumulated directly by the shared sorter from #1191; the
exchange does not implement a second bin-packer or square those costs again.
Qwen3-VL declares frame-wise quadratic attention costs
`t * (h * w / merge_area) ** 2`, accumulated with exponent one. Treating a
video clip as a single quadratic sequence would overestimate its cost by `t`.
Its meter likewise stashes full frame lengths, while output restoration uses
the clip's total merged-token length.

## Strategy and scope

`resolve_balance_strategy` chooses `sp_slice`, `dp_balance` or `replicate` from
the declaration and the module's current topology. Global, non-overlapping
attention retains token-wise SP-slice when SP is enabled. Whole-item modules
can balance across DP ranks; modules without that capability replicate.

The primary SP strategy and cross-DP optimization are orthogonal. Token-wise
SP already balances a ViT *within* its SP group, but different DP groups can
have very different image workloads. Qwen3-VL therefore balances whole items
across DP before its existing SP slice and reverses outputs after SP gather.
With `dp_size=1`, no DP exchange is issued. Conv/window/temporal structures
must not claim the global, non-overlapping capability; this PR does not wire
any such consumer or implement halo exchange.

`model_config.data_balance_override: off` is an explicit diagnostic escape
hatch disabling only cross-DP balancing; it does not disable the module's SP
execution. `auto` (or absence of the key) derives participation from structure.
There is no flag in `auto.py` and no requirement for a YAML opt-in.

The module reads its scoped `dp_group` inside the pre-forward hook. The
immutable routing plan owns that exact group's forward/backward exchanges,
independently of whichever module's state is ambient during backward.
All owners call the inverse even when their original batch was empty. Packed
and carrier consumers must preserve dummy/empty gradient anchors downstream.
Schemas and metadata errors are checked collectively before payload exchange.

## Qwen3-VL ordering

Both `forward` and `pack_encode` use the same declaration:

1. Read worker-built grids; retain real and dummy items alike. Stash full
   merged-token lengths in the meter before routing or slicing.
2. Balance packed patches and grid rows across the module-local DP group.
3. Rebuild ViT metadata and recompute the post-balance own length; pad/slice SP.
4. Execute one ViT forward; gather and de-pad SP outputs.
5. Inverse-route embeddings and deepstack tensors, then scatter to the original
   carrier/packed positions and fold the original dummy anchors.

Never short-circuit the routed ViT using the original owner's dummy flag: that
owner can now be computing real items belonging to another DP rank.

## Validation and remaining gates

`tests/parallel/encoder_data_balance/test_module_balance.py` covers declarative
strategies, independent metric/output lengths, invocation cleanup, transport,
empty owners/destinations/global batches, deepstack and gradients. It also
contains a four-accelerator real FSDP2 toy Qwen3-VL gate at DP4/SP1, DP2/SP2 and
DP1/SP4 for ordinary and packed paths: bit-identical on/off loss, token counts
and restored outputs. Its synthetic downstream CE makes it a module-training
integration test, not a complete text+vision launcher e2e.

The development validation also ran the actual `OmniTrainer`, standard raw
ShareGPT/image preprocessing and live ordinary/packed Qwen graphs with native
two-layer split checkpoints and the official tokenizer. All six path/topology
combinations (ordinary/packed at DP4/SP1, DP2/SP2 and DP1/SP4) completed two
optimizer steps with balancing off and auto. Full-precision train metric
dictionaries and per-module token counts matched exactly on all four ranks at
both steps. This required separate fixes for accumulation-loss scaling and
the packed text encoder's pre-SP token meter. Packed embedding counts include
visual placeholder positions; they are not the ordinary text-only encode count.
These tiny-checkpoint runs prove same-topology on/off correctness, not
production-scale performance or bit-identical gradients across topologies.

Part 2 of [#1074](https://github.com/ByteDance-Seed/VeOmni/issues/1074), following
[the maintainer's split](https://github.com/ByteDance-Seed/VeOmni/issues/1074#issuecomment-5693718137).
PR1 is [#1191](https://github.com/ByteDance-Seed/VeOmni/pull/1191). The full launcher
loss/token parity gate above is verified; final regression and review remain
required before submitting this integration.
Bagel SP inverse permutation and skewed per-rank wall-time/MFU remain PR3,
after mixin-contract review.

The legacy `Qwen3VLEncoderDataBalance` compatibility entry points now delegate
to the same `ModuleDataBalancer` and immutable `BalancePlan`: there is no second
all-to-all transport. They retain independent image/video slots, raw-patch
quadratic scheduling costs, spatially merged output splits and deepstack VJPs.
Only new module consumers use the declarative API; the compatibility shim still
captures its construction-time DP group and replaces a slot on repeated calls.
It is not an invitation to wire legacy opt-in flags into SeedOmni V2.
Bagel, production performance and eventual removal of compatibility flags/shim
remain separate work; this change does not claim the whole issue is complete.

Tensor fields must agree on trailing shape, dtype and `requires_grad` across
the DP group, including empty owners. Rank-asymmetric autograd would omit
backward collectives and is rejected before payload routing. Malformed
field names, split conversions and scalar tensors also fail collectively.
