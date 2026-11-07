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

## Integrating a module

Declaring a spec does not route anything. The executor only opens the scope, so
a module must call `data_balance_pre` and `data_balance_post` from its own hooks.
The module owns the spec, its per-item metadata and these two calls. The mixin
owns the routing plan, the DP exchanges and their inverse.

| Function | Called by | Takes | Returns |
|----------|-----------|-------|---------|
| [`data_balance_scope()`](../../veomni/models/seed_omni/mixins/data_balance_mixin.py#L98) | Framework: [`execute_train_node`](../../veomni/models/seed_omni/accelerated/utils/executor.py#L61), once per training node. Do not call it from a module. | Nothing. | A context manager. It raises on reentry, or on exit if a routed call-site was not restored. |
| [`data_balance_pre(method, tensors, metadata)`](../../veomni/models/seed_omni/mixins/data_balance_mixin.py#L111) | Module: the `@pre_forward` hook, before SP padding or slicing. | `tensors` must hold every `fields` name of the spec. `metadata` must hold each field's `lengths_key`, plus `output_lengths_key` and `cost_lengths_key`, with one entry per local item. | `(tensors, moved)`. Declared fields are routed and other keys pass through. If `moved` is false the inputs are unchanged: no spec for `method`, no open scope, not training, or the strategy does not balance DP. |
| [`data_balance_post(method, outputs)`](../../veomni/models/seed_omni/mixins/data_balance_mixin.py#L140) | Module: the `@post_forward` hook, after SP gather and de-padding, before scattering results to their items. | `outputs` must hold every `output_names` entry of the spec, as a packed tensor or a list of packed tensors. Their leading dimension must equal the routed items' total output length. | A copy of `outputs` with those entries back in the original owner and item order. If `pre` did not route `method`, `outputs` comes back as is. |

Every DP rank must make both calls, including a rank with no items, because the
exchanges and their backward are collectives. Do not skip the model on a rank
whose own batch was empty: use `moved` to clear its dummy flag, since it can
receive another rank's items.

Qwen3-VL is the reference integration:

- Spec: [`_vision_balance_spec`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/accelerated.py#L174)
- Metadata and `data_balance_pre`: [`_balance_vision_inputs`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/accelerated.py#L184)
- Pre hooks: [`forward_pre`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/accelerated.py#L85) and [`pack_encode_pre`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/packed.py#L56)
- Post hooks: [`forward_post`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/accelerated.py#L147) and [`pack_encode_post`](../../veomni/models/seed_omni/modules/qwen3vl/vision/accelerated/packed.py#L114)

A minimal module with one output token per input token:

```python
from typing import Any

import torch

from veomni.models.seed_omni.mixins import (
    BaseMixin,
    DataBalanceMixin,
    DataBalanceSpec,
    ItemTensorField,
    ModuleStructure,
    TrainingModuleMixin,
    post_forward,
    pre_forward,
)


class BalanceMixin(DataBalanceMixin):
    data_balance_specs = {
        "forward": DataBalanceSpec(
            fields=(ItemTensorField("pixel_values", "lengths"),),
            output_names=("embeds",),
            output_lengths_key="lengths",
            cost_lengths_key="lengths",
            structure=ModuleStructure(global_attention=True, non_overlapping_patchify=True),
        )
    }


class TrainingMixin(TrainingModuleMixin):
    @pre_forward("forward")
    def forward_pre(self, pixel_values: torch.Tensor, lengths: list[int]) -> dict[str, Any]:
        inputs, _ = self.data_balance_pre("forward", {"pixel_values": pixel_values}, {"lengths": lengths})
        # SP padding and slicing would go here, on the routed tensors.
        return {"pixel_values": inputs["pixel_values"]}

    @post_forward("forward")
    def forward_post(self, embeds: torch.Tensor) -> dict[str, Any]:
        # SP gather and de-padding would go here, before the inverse.
        return self.data_balance_post("forward", {"embeds": embeds})


class VeOmniMixin(BaseMixin, TrainingMixin, BalanceMixin):
    pass
```

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
execution. A bare YAML or CLI `off` parses as boolean `False` and is accepted
as `off`. `auto` (or absence of the key) derives participation from structure.
There is no flag in `auto.py` and no requirement for a YAML opt-in.

The module reads its scoped `dp_group` inside the pre-forward hook. The
immutable routing plan owns that exact group's forward/backward exchanges,
independently of whichever module's state is ambient during backward.
All owners call the inverse even when their original batch was empty. Packed
and carrier consumers must preserve dummy/empty gradient anchors downstream.

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
after mixin-contract review. V1 migration/deletion remains PR4; this change does
not claim that legacy retirement or the whole issue is complete.

All DP ranks run the same module, so tensor fields agree on trailing shape,
dtype and `requires_grad` across the DP group, including empty owners. The
balancer relies on that contract and gathers only item lengths and costs.
