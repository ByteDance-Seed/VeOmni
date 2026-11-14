# Async Activation Offload Targets

Models may set `_activation_offload_target_classes` to keep async activation
offload targets separate from FSDP wrapping targets. This is an optional model
class attribute, not a configuration field. The model owns a list of exact
module class-name strings; the framework reads it during automatic discovery
in [`_get_auto_offload_modules`](../../veomni/distributed/async_offload.py#L525).
If the attribute is absent, discovery uses `model._no_split_modules`. An empty
list or a list that matches no modules raises an error. Explicit
`activation_offload_modules` name patterns take precedence.

The [H3 model](../../veomni/models/diffusers/minimax_h3/minimax_h3_transformer/modeling_minimax_h3_transformer.py#L64)
illustrates the distinction:

```python
_no_split_modules = ["MiniMaxH3DiTBlock", "MiniMaxH3DiT"]
_activation_offload_target_classes = ["MiniMaxH3DiTBlock"]
```

FSDP2 still wraps both classes. Automatic offload wraps only the blocks,
avoiding nested saved-tensor hooks around the parent DiT and its checkpointed
children. [`MiniMaxH3DiTBlock`](../../veomni/models/diffusers/minimax_h3/minimax_h3_core/minimax_h3_dit.py#L429)
owns the gradient-checkpoint boundary, so the offload hook wraps the checkpoint
call and backward still recomputes the block. A model without the override
retains the existing `_no_split_modules` behavior.

One build step follows this path:

1. [`VeOmniModelRuntime._build_parallelized_model`](../../veomni/models/model_runtime.py#L401)
   calls [`_apply_async_activation_offload`](../../veomni/models/model_runtime.py#L341)
   before FSDP2 wrapping when async activation offload is enabled.
   That method calls [`apply_async_activation_offload`](../../veomni/distributed/async_offload.py#L673).
   1. Explicit name patterns select modules when supplied.
   2. Otherwise, automatic discovery selects exact class-name matches from
      `_activation_offload_target_classes` or the `_no_split_modules` fallback.
2. FSDP2 independently uses `_no_split_modules` for wrapping.

Keep the override limited to non-overlapping modules that own the checkpoint
boundary to be offloaded. Names must match `type(module).__name__` exactly.
The [H3 regression test](../../tests/models/test_minimax_h3_packing.py#L1206)
checks that blocks are selected and the parent DiT is not; the
[two-rank FSDP2 test](../../tests/models/test_minimax_h3_packing.py#L1444)
checks that checkpoint recomputation survives async offload. The
[generic offload tests](../../tests/distributed/test_async_offload_unit.py#L159)
cover fallback discovery and missing targets.
