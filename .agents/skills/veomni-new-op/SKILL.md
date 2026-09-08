---
name: veomni-new-op
description: "Add or optimize a tensor-level kernel in veomni/kernels, register its variants, integrate it with models_kernel, and add numerical and registry tests. Trigger: 'add op', 'new kernel', 'add attention variant', 'new fused op', 'add triton kernel', 'optimize operator'."
---

## Before You Start

1. Read `.agents/knowledge/constraints.md`, especially device guards and patchgen rules.
2. Read `veomni/kernels/README.md` and `docs/design/kernel_selection.md`.
3. Inspect the closest family under `veomni/kernels/_kernels/` and its tests under `tests/kernels/`.

## Kernel Architecture

`veomni.kernels.KERNEL_REGISTRY` is the tensor-kernel source of truth. Each
row has the identity `(kernel, variant, implementation, device)`. Callers
select the public `(kernel, variant, implementation)` triple; the registry
derives the device from the row's `KernelRequirement` and resolves the current
device before a device-agnostic row.

A row provides exactly one of these forms:

- raw `forward` and `backward` functions, from which the registry generates a
  `torch.autograd.Function` wrapper; or
- an opaque `wrapper` for eager PyTorch or a library API that already owns its
  autograd behavior.

For a raw pair, `forward` returns `(output, SavedState)` and `backward` returns
one gradient entry per positional tensor input. Pass tensors positionally and
non-tensor attributes by keyword.

Model classes construct an instance-local `VeomniKernel` handle and call it
directly. Input normalization, HuggingFace-compatible signatures, and loss
policy belong in `veomni/models_kernel/`; do not add consumer-specific adapters
to the registry. The public CLI/YAML field remains
`model.ops_implementation`, while model builders receive it through the
`kernels_implementation` keyword.

Use these separate mechanisms only when their semantics require them:

- `veomni/kernels/batch_invariant/` for the opt-in ATen patch controlled by
  `set_batch_invariant_mode(...)`;
- `veomni/kernels/install.py` for idempotent process-wide integrations such as
  registration with a third-party framework;
- `veomni/distributed/hccl_premul_sum.py` for the NPU collective compatibility
  patch used by distributed ExtraParallel code.

## Phase 1: Design

1. Define the stable tensor contract and decide whether consumer-specific
   preprocessing belongs in `models_kernel`.
2. Choose the kernel name, semantic variant, implementation name, and device
   requirement. A variant changes the tensor contract; an implementation keeps
   that contract and changes how it is computed.
3. Decide whether the implementation is a raw pair or an opaque wrapper. Never
   provide both forms for one row.
4. If users must select it, add or extend the appropriate field in
   `OpsImplementationConfig`. Preserve existing CLI/YAML field names.

## Phase 2: Implement

1. Create or extend a family under
   `veomni/kernels/_kernels/<kernel_name>/`.
2. Keep implementations in variant/device-oriented modules consistent with the
   neighboring families.
3. Register every row through `register_kernel`:

   ```python
   from veomni.kernels import register_kernel
   from veomni.kernels.requirement import CudaKernelRequirement

   register_kernel("example", "standard", "eager", wrapper=eager_example)
   register_kernel(
       "example",
       "standard",
       "triton",
       forward=triton_forward,
       backward=triton_backward,
       requirement=CudaKernelRequirement(min_cc=80),
   )
   ```

4. Import the family from `veomni/kernels/_kernels/__init__.py` so registration
   happens when `veomni.kernels` is imported.
5. In each consuming model, construct a `VeomniKernel` from
   `resolve_kernel_impl(...)` and store it on the model/module instance.
6. For a compound raw kernel, call the nested row's raw `forward`/`backward`
   and use the saved-state helpers in `veomni/kernels/compound.py`; do not call
   the nested autograd wrapper.
7. Guard optional device packages and attach an explicit requirement. A
   registered implementation must fail clearly when its requirement is not
   satisfied; it must not silently fall back to eager.
8. Add English module, class, and function docstrings. VeOmni-owned kernel code
   is checked by `tests/kernels/base/test_kernel_documentation.py`.

## Phase 3: Test

1. Add tests under `tests/kernels/<kernel_name>/` for:
   - forward parity against an independent eager reference;
   - backward parity for every differentiable input;
   - dtype/shape/edge contracts;
   - registration and device requirements;
   - explicit failure for unknown or unavailable implementations.
2. Put consumer-specific normalization and model wiring tests under
   `tests/models_kernel/` instead of duplicating them in the raw-kernel suite.
3. Run the family tests plus the registry and documentation guards:

   ```bash
   pytest -q tests/kernels/<kernel_name>/ tests/kernels/base/
   ```

4. Add a benchmark only when performance is part of the acceptance criteria;
   compare it with the canonical eager contract.

## Phase 4: Document

1. Update `veomni/kernels/README.md` with the family, variants, and supported
   implementations.
2. Update `docs/design/kernel_selection.md` when selection behavior changes.
3. Update `.agents/knowledge/architecture.md` when the layout or call chain
   changes.

## Phase 5: Finalize

1. Run `/veomni-review`.
2. Run `make quality` and the relevant kernel/model integration tests.
3. Verify `KERNEL_REGISTRY.list_registered(...)` and
   `list_available(...)` report the expected rows.

## Common Pitfalls

- Confusing a semantic variant with an implementation backend.
- Adding model-specific reshaping or reduction policy to the tensor registry.
- Calling a nested autograd wrapper from a compound custom-autograd function.
- Importing NPU/CUDA-only libraries without guards or requirements.
- Registering a row without importing its family from `_kernels/__init__.py`.
- Letting an unavailable optimized implementation fall back silently.
- Editing patchgen-generated files instead of their patch config.
- Renaming `ops_implementation` while changing the internal kernel plumbing.
