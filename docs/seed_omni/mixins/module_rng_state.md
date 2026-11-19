# Module Extra State (`rng_state_dict`)

A SeedOmni module may contribute rank-local state to its own checkpoint. The
runtime stores that state in the module's per-rank `extra_state` file, so a
resumed run can restore private state such as a sampling generator instead of
replaying it from the seed. This page uses RNG state as the example of the
broader module extra-state protocol. Like [`OfflineEncodingMixin`](offline_encoding.md),
this is a protocol, not a base class: a module opts in by defining the relevant
methods.

- Runtime save/load: [`ModuleRuntime`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py)
- Checkpoint writer: [`ModelCheckpointManager`](../../../veomni/models/checkpoint/manager.py)
- Example module: [`Cosmos3FlowConnector`](../../../veomni/models/seed_omni/modules/cosmos3/flow_connector/modeling.py)

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
    # Generic hooks for arbitrary module-owned state:
    def extra_state_dict(self) -> dict[str, object]:
        return {"generator": self._generator.get_state()}

    def load_extra_state_dict(self, state: dict[str, object]) -> None:
        self._generator.set_state(state["generator"])

    # Existing RNG-only modules may keep this compatibility pair instead.
    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator": self._generator.get_state()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._generator.set_state(state["generator"])
```

- `ModuleRuntime.extra_state()` returns an arbitrary module-owned dict under
  `module_extra_state` when the module implements the generic
  `extra_state_dict` hook. Existing RNG-only modules use the compatibility
  `module_rng_state` key; another module-owned state must use its own key.
- The runtime reserves `lr_scheduler` in the same outer dict. Module-owned state
  stays in its own namespace; separate modules cannot collide because each
  module has its own per-rank `extra_state` file under `model/<module>/`.
- `ModelCheckpointManager` writes that file once per rank. On resume,
  `ModuleRuntime.load_extra_state()` restores module-owned entries inside the
  module's own `ParallelState`, so a generator seeded from `dp_rank` reads this
  module's rank, not the orchestrator's.
- The RNG pair mirrors the DiT condition model's, so a model ported from
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
