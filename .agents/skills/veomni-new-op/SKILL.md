---
name: veomni-new-op
description: "Use this skill when adding a new optimized kernel or operator to veomni/ops/. Covers the full lifecycle: understanding VeOmni's ops architecture (KERNEL_REGISTRY + OpSlot dispatch, with a thin function-pointer shim for a few legacy global ops), implementing the kernel, registering it, adding tests, and documenting it. Trigger: 'add op', 'new kernel', 'add attention variant', 'new fused op', 'add triton kernel', 'optimize operator'."
---

## Before You Start

1. Read `.agents/knowledge/constraints.md` — especially the "Hardware" section
   (NPU guards, device-agnostic helpers) and "Module-level OpSlots are shared by
   every model instance" under "Trainer Extensions".
2. Read `docs/design/kernel_selection.md` and `docs/design/unified_kernel_registry.md` — understand the kernel lifecycle, the `KERNEL_REGISTRY`, and `OpSlot` dispatch.
3. Familiarize yourself with the ops architecture below.

## VeOmni Ops Architecture

Most VeOmni ops in v5 are **registry-driven**: a kernel registers itself in
`veomni.ops.kernel_registry.KERNEL_REGISTRY` and is dispatched at model-build
time through `OpSlot` instances declared in the patchgen-generated modeling
files (see `veomni/ops/dispatch.py` and `_bind_veomni_ops()` in
`veomni/models/auto.py`).

```
veomni/ops/
├── __init__.py          # apply_ops_patch / apply_ops_config entry points
├── kernel_registry.py   # KERNEL_REGISTRY (the single source of truth)
├── dispatch.py          # OpSlot + binding helpers
├── config/              # legacy OpSpec/BackendSpec registry: apply_global_ops()
│                        # + apply_per_model_patches() for device_patch.py models
├── kernels/             # all registry-driven kernels
│   ├── attention/       # FA2/3/4 + sequence-parallel wrappers
│   ├── cross_entropy/   # eager + liger fused CE
│   ├── deepseek_sparse_attention/
│   ├── deepseek_v4/     # TileLang sparse attention / indexer
│   ├── load_balancing_loss/
│   ├── mhc/             # TileKernels DeepSeek V4 adapters
│   ├── moe/             # fused MoE (group_gemm / quack / npu_group_gemm)
│   ├── rms_norm/        # eager / liger / batch-invariant
│   ├── rotary/          # default / triton-deterministic
│   ├── swiglu/          # eager / liger
│   └── gated_delta_rule/
├── batch_invariant_ops/ # ATen-level interception for bitwise determinism
├── liger/               # Liger kernel adapters
└── platform/            # NPU-specific helpers
```

**Three mechanisms coexist.** Pick the first one unless you have a concrete
reason not to:

1. **`KERNEL_REGISTRY` + `OpSlot`** (preferred for new ops). Each kernel
   registers itself under a `(slot_name, variant)` pair (e.g.
   `("cross_entropy_loss", "causal")`, `("moe_experts", "standard")`).
   Patchgen-generated modeling code declares matching `OpSlot` instances; at
   model-build time `_bind_veomni_ops()` walks the generated module, finds
   each `OpSlot`, and binds it to the concrete registry entry chosen by
   `OpsImplementationConfig` (`config/registry.py`).
2. **Legacy global function pointer shim** (kept for a few global ops that
   are dispatched outside generated modeling). Public-API functions like
   `fused_moe_forward` and `load_balancing_loss` still expose a thin pointer
   that is rebound by `apply_ops_config()` so call sites in non-patchgen code
   (DeepSeek MLA inference paths, NPU custom forwards) can keep importing the
   public name without going through an `OpSlot`.
3. **Per-model `device_patch.py`** via `OpSpec`/`BackendSpec` in
   `ops/config/registry.py`. `apply_per_model_patches(hf_module, model_name,
   targets={op: attr})` setattr-replaces attributes on an HF module. Used by the
   models that have no patchgen-generated file (`wan`) or that need a runtime
   device-specific swap after generation (`deepseek_v3`, `deepseek_v4`). Those
   three `device_patch.py` files are its only callers. Do not extend this for
   new kernels.

Mechanism 1 covers any kernel living inside a patchgen-generated modeling file.
Use 2 only when the kernel must be callable from unpatched (or
non-Transformers) Python code, and 3 only when touching a model that already
ships a `device_patch.py`.

## Phase 1: Design

1. **Determine op category**:
   - **Registry-driven kernel** (the common case, used inside patchgen-generated modeling): register under a `(slot_name, variant)` in `KERNEL_REGISTRY` and add a matching `OpSlot` in the relevant `<model>_patch_gen_config.py`. No global mutation; selection is driven by `OpsImplementationConfig`.
   - **Global op with public API** (e.g. `fused_moe_forward`, `load_balancing_loss`): expose a public function in `veomni/ops/__init__.py` and rebind it from `apply_ops_config()` based on the active `OpsImplementationConfig`. Only use this when a non-patchgen call site (NPU MLA forward, manual inference scripts, etc.) needs to import the kernel directly.
   - **Library op** (no dispatch — called directly by model code): just create the module, no registry entry needed.
   - **NPU variant**: add alongside the GPU implementation behind an `is_torch_npu_available()` guard.

2. **Decide selection mechanism**: read `docs/design/kernel_selection.md` and `docs/design/unified_kernel_registry.md` to determine if you need:
   - Config field in `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`)
   - Environment variable
   - Both

3. **Determine binding timing**:
   - **Model build time** (default): registry entries are resolved by `_bind_veomni_ops()` in `veomni/models/auto.py` when a model is constructed. New kernels just need to register themselves at import time.
   - **`apply_ops_config()` time**: legacy global ops (rebound function pointers) are wired in `veomni/ops/__init__.py::apply_ops_config(ops_config)`.

## Phase 2: Implement

1. **Create the op directory** under `veomni/ops/kernels/<op_name>/`.

2. **Implement each kernel variant** in its own file (e.g. `triton_kernel.py`, `eager.py`, `npu_kernel.py`). Each variant declares a concrete function with the kernel's canonical signature.

3. **Register the kernel** in `veomni/ops/kernels/<op_name>/__init__.py`:
   ```python
   from veomni.ops.kernel_registry import KERNEL_REGISTRY

   from .eager import my_op_eager
   from .triton_kernel import my_op_triton

   KERNEL_REGISTRY.register(slot="my_op", variant="eager")(my_op_eager)
   KERNEL_REGISTRY.register(slot="my_op", variant="triton")(my_op_triton)
   ```

   Then declare a matching `OpSlot` in the patchgen config of every model that uses it:
   ```python
   from veomni.ops.dispatch import OpSlot
   veomni_my_op = OpSlot("my_op", "eager")  # default variant
   ```
   `_bind_veomni_ops()` will swap this for the registry entry selected by `OpsImplementationConfig`.

4. **Wire the config field** (if the user needs to choose a variant):
   - Add a field to `OpsImplementationConfig` in `veomni/arguments/arguments_types.py`.
   - In `veomni/ops/config/registry.py`, map the new config field to the `(slot, variant)` tuple consumed by `_bind_veomni_ops()`.

5. **For legacy global ops** (only when needed): add the public function to `veomni/ops/__init__.py` and rebind it from `apply_ops_config(ops_config)`.

6. **Async Ulysses split wrappers** (only for `rms_norm` and `rotary_pos_emb`): compound Functions cannot call `OpSlot`. They use no-autograd `(output, saved)` / `backward` pairs in `veomni/distributed/sequence_parallel/op_wrappers.py`. A new backend or variant must either add a matching wrapper there, or be left off `_SUPPORTED_IMPLEMENTATIONS` / `_SUPPORTED_VARIANTS` so `get_op_wrapper` rejects it. `KERNEL_REGISTRY` coverage is not enough.

7. **NPU support**:
   - Always guard NPU imports with `is_torch_npu_available()`.
   - Put NPU implementations in a separate file (e.g., `npu_kernel.py`).
   - Register the NPU variant under the same slot with a distinct variant name.

## Phase 3: Test

1. **Add unit tests** to `tests/ops/`:
   - Test correctness: compare output against a reference implementation (eager PyTorch)
   - Test numerical precision: verify tolerance for bf16/fp16
   - Test edge cases: empty inputs, single-element tensors, extreme shapes

2. **Add benchmark** (optional but recommended for performance-critical ops):
   - Use `veomni/ops/kernels/moe/_kernels/utils/benchmark_utils.py` as reference
   - Compare against baseline implementation

3. Run: `pytest tests/ops/ -v`

## Phase 4: Document

1. **Update `docs/design/kernel_selection.md`**:
   - Add the new op to the Quick Reference table
   - Describe the selection mechanism

2. **Update `.agents/knowledge/architecture.md`** if the op adds a new subdirectory to `veomni/ops/`.

## Phase 5: Finalize

1. Run `/veomni-review` skill.
2. Run `make quality`.
3. Verify the new variant shows up in `KERNEL_REGISTRY.dump()` and that the relevant `OpSlot` is rebound after `build_foundation_model`.

## Common Pitfalls

- **Forgetting to register in `KERNEL_REGISTRY`**: the variant is invisible to `_bind_veomni_ops()` and `OpSlot` will fall through to its default — you'll silently exercise the wrong kernel.
- **Forgetting to add the matching `OpSlot` to the patchgen config**: registering a kernel alone has no effect — generated modeling code must declare an `OpSlot` for it to be picked up.
- **Unconditional NPU imports**: importing NPU modules without an `is_torch_npu_available()` guard crashes on GPU-only environments.
- **Binding at wrong time**: registry entries are resolved when `build_foundation_model` runs `_bind_veomni_ops()`. Kernels that depend on per-model config must be picked at that point — not at module-import time.
- **New `rms_norm` / `rotary_pos_emb` backend without an async wrapper**: `OpSlot` will bind, but async Ulysses goes through `op_wrappers.py`, not the registry callable. Add a split wrapper or confirm `get_op_wrapper` rejects the new name; do not derive the supported set from `KERNEL_REGISTRY`.
- **Sequence parallel interaction**: ops that touch attention or loss must handle sequence parallel correctly — use `get_parallel_state().sp_enabled` to check and dispatch.
- **Mixed precision**: fused kernels often require specific dtypes (bf16/fp16). Add assertions at the public API level to catch dtype mismatches early.
- **Not exporting public APIs**: if the op provides a public function (legacy global ops), export it from `veomni/ops/__init__.py`'s `__all__`.
