---
name: veomni-new-op
description: "Add or optimize a tensor-level kernel in veomni/ops, register its variants, integrate it with models, and add numerical and registry tests. Trigger: 'add op', 'new kernel', 'add attention variant', 'new fused op', 'add triton kernel', 'optimize operator'."
---

## Before You Start

1. Read `.agents/knowledge/constraints.md` (and `.agents/knowledge/topics/qat_and_ops.md` if touching QAT / low-precision / activations).
2. Read `veomni/ops/README.md` and `docs/design/op_selection.md` (and `docs/design/module_config.md` for config ownership).
3. Inspect the closest family under `veomni/ops/kernels/` and its tests under `tests/ops/`.

## Ops Architecture

- `veomni.ops.OP_REGISTRY` keys rows by `(op, variant, implementation, device)`. Callers select `(op, variant, implementation)`; the registry derives device from `KernelRequirement` and resolves the current device before device-agnostic rows.
- Each row provides **exactly one** form:
  - Raw `forward` + `backward` (`forward` returns `(output, SavedState)`; `backward` returns one grad per positional tensor input; tensors positional, non-tensor attrs by keyword) → registry builds `torch.autograd.Function`.
  - Opaque `wrapper` (eager PyTorch or library API owning its own autograd).
- **Variant** = tensor contract / model geometry; **implementation** = backend (`eager` is an implementation under every variant, never a variant).
- Modeling constructs instance-local `VeomniOp` handles from `self.config.ops_config.<field>` (public CLI/YAML field: `model.ops_implementation`). Keep input normalization, HF signatures, and loss policy in `veomni/models/`, never in the tensor registry.
- Separate mechanisms only when required: `veomni/ops/batch_invariant/` (opt-in deterministic ATen patch), `veomni/ops/install.py` (idempotent process-wide third-party registration), `veomni/distributed/hccl_premul_sum.py` (NPU collective compat).

## Workflow

### 1. Design & Implement
1. Define the tensor contract, `(op, variant, implementation)`, device requirement, and form (raw pair vs wrapper). Extend `OpsImplementationConfig` only if users must select a new field.
2. Add modules under `veomni/ops/kernels/<kernel_name>/` matching neighboring layout and register rows via `register_op`:
   ```python
   from veomni.ops import register_op
   from veomni.ops.platform import GpuKernelRequirement, NvidiaGpuPlatform

   register_op("example", "standard", "eager", wrapper=eager_example)
   register_op(
       "example", "standard", "triton",
       forward=triton_forward, backward=triton_backward,
       requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=80),)),
   )
   ```
3. Import the family from `veomni/ops/kernels/__init__.py` so registration runs when `veomni.ops` is imported.
4. For compound raw kernels, call nested raw `forward`/`backward` + `veomni/ops/compound.py` saved-state helpers (never the nested autograd wrapper).
5. Lazy-import optional device packages and attach explicit `KernelRequirement`; unavailable explicit selections must **raise**, never silently fall back to `eager`.
6. Bind instance-local `VeomniOp` handles in consuming models (`self.config.ops_config.<field>`). Never import a specific kernel module directly from `modeling_<m>.py`.
7. Add English docstrings on module, module-level classes, public functions, and `forward`/`backward`/`wrapper`/`__call__` (gated by `tests/ops/base/test_op_documentation.py`).

### 2. Test
1. Add tests under `tests/ops/<kernel_name>/` (auto-collected by GPU and NPU unit manifests) covering:
   - Forward & backward parity against an independent eager reference for every differentiable input
   - Dtype/shape/edge contracts, distinct variants, and EP/non-EP or split/merged layouts when applicable
   - Registration, `KernelRequirement`, explicit failure on unknown/unavailable impls, and `torch.compile` inference mode (especially for 1-tensor ops)
2. Put model wiring / normalization tests under `tests/models/` (and E2E representative in `tests/e2e/test_e2e_parallel.py` when introducing a new integrated distributed path; see `docs/transformers_v5/testing_new_model.md`).
3. Run:
   ```bash
   pytest -q tests/ops/<kernel_name>/ tests/ops/base/
   ```

### 3. Document & Finalize
1. Update `veomni/ops/README.md` (and `docs/design/op_selection.md` if selection semantics changed).
2. Verify `OP_REGISTRY.list_registered(...)` and `list_available(...)`, run `make quality`, and run `/veomni-review` before opening the PR.

## Common Pitfalls

- Confusing semantic variant with implementation backend, or leaking model policy into the op registry.
- Calling a nested autograd wrapper inside a compound custom autograd function.
- Forgetting to import the new family in `veomni/ops/kernels/__init__.py` or letting unavailable selections silently fall back to `eager`.
- On NPU, unweighted RMSNorm supports only `eager` (explicit `torch_npu`/`npu` must raise); attention `eager` preserves model-local eager math or forwards to `veomni_sdpa`.
