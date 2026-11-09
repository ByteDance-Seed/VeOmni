# Module Sampling State (`rng_state_dict`)

A SeedOmni module that draws its own per-step randomness from a private
generator can have that generator saved and restored with the module, so a
resumed run continues the same stream instead of replaying it from the seed.
Like [`OfflineEncodingMixin`](offline_encoding.md), this is a protocol, not a
base class: a module opts in by defining two methods.

- Save: [`ModuleRuntime.extra_state`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L651)
- Load: [`ModuleRuntime.load_extra_state`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L664)
- Where it is written: [`ModelCheckpointManager._extra_state`](../../../veomni/models/checkpoint/manager.py#L177)
- Example: [`Cosmos3FlowConnector.rng_state_dict`](../../../veomni/models/seed_omni/modules/cosmos3/flow_connector/modeling.py#L195)

## Why a module needs it

The trainer checkpoints the global torch and device RNGs, which covers every
draw made from them. A diffusion objective usually wants its own stream
instead: the Cosmos3 flow connector draws sigma, the noise and the clean-prefix
mask from a CPU `torch.Generator` seeded `seed + dp_rank`, so the draws do not
depend on how many other global-RNG calls happen elsewhere in the step. Without
saving it, a resumed run would restart that generator from its seed and train
on the draws of the first steps again.

## The contract

```python
class MyConnector(PretrainedOmniModule):
    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator": self._generator.get_state()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._generator.set_state(state["generator"])
```

- `ModuleRuntime.extra_state()` returns `{"module_rng_state": module.rng_state_dict()}`
  when the module defines `rng_state_dict`, and nothing otherwise. The module's
  checkpoint manager merges it into that module's per-rank `extra_state` file,
  next to its lr-scheduler state, so each rank restores its own stream.
- `ModuleRuntime.load_extra_state()` calls `module.load_rng_state_dict(...)`
  inside the module's own `ParallelState`, so a generator that is created on
  first use and seeded from `dp_rank` reads this module's rank, not the
  orchestrator's.
- The pair mirrors the DiT condition model's, so a model ported from
  `DiTTrainer` keeps its method names.

## A fully frozen module saves nothing

`ModuleRuntime.build_checkpoint` gives a module with no trainable parameter no
checkpoint manager, because it has no weights, optimizer or schedule to save.
Its `extra_state` is therefore never written, and its stream restarts on every
resume. `build_checkpoint` logs a warning when a fully frozen module defines
`rng_state_dict`.

Put a sampling stream on a module that trains. Cosmos3 keeps its sigma / noise
draws in the flow connector, which always trains, rather than in the frozen VAE
or in the MoT, which is frozen as a whole when its generation tower is frozen
too.
