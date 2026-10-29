---
name: veomni-new-op
description: "Add or optimize a tensor-level kernel in veomni/ops, register its variants, integrate it with models, and add numerical and registry tests. Trigger: 'add op', 'new kernel', 'add attention variant', 'new fused op', 'add triton kernel', 'optimize operator'."
---

## Before You Start

1. Read `.agents/knowledge/constraints.md`, especially device guards and patchgen rules.
2. Read `veomni/ops/README.md` and `docs/design/op_selection.md`.
3. Inspect the closest family under `veomni/ops/kernels/` and its tests under `tests/ops/`.

## Ops Architecture

`veomni.ops.OP_REGISTRY` is the tensor-kernel source of truth. Each
row has the identity `(op, variant, implementation, device)`. Callers
select the public `(op, variant, implementation)` triple; the registry
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

Model classes construct an instance-local `VeomniOp` handle and call it
directly. Input normalization, HuggingFace-compatible signatures, and loss
policy belong in `veomni/models/`; do not add consumer-specific adapters
to the registry. The public CLI/YAML field remains
`model.ops_implementation`, while model builders receive it through the
`ops_implementation` keyword.

Use these separate mechanisms only when their semantics require them:

- `veomni/ops/batch_invariant/` for the opt-in ATen patch controlled by
  `set_batch_invariant_mode(...)`;
- `veomni/ops/install.py` for idempotent process-wide integrations such as
  registration with a third-party framework;
- `veomni/distributed/hccl_premul_sum.py` for the NPU collective compatibility
  patch used by distributed ExtraParallel code.

## Phase 1: Design

1. Define the stable tensor contract and decide whether consumer-specific
   preprocessing belongs in `models`.
2. Choose the op name, semantic variant, implementation name, and device
   requirement. A variant changes the tensor contract; an implementation keeps
   that contract and changes how it is computed.
3. Decide whether the implementation is a raw pair or an opaque wrapper. Never
   provide both forms for one row.
4. If users must select it, add or extend the appropriate field in
   `OpsImplementationConfig`. Preserve existing CLI/YAML field names.

## Phase 2: Implement

1. Create or extend a family under
   `veomni/ops/kernels/<kernel_name>/`.
2. Keep implementations in variant/device-oriented modules consistent with the
   neighboring families.
3. Register every row through `register_op`:

   ```python
   from veomni.ops import register_op
   from veomni.ops.platform import GpuKernelRequirement, NvidiaGpuPlatform

   register_op("example", "standard", "eager", wrapper=eager_example)
   register_op(
       "example",
       "standard",
       "triton",
       forward=triton_forward,
       backward=triton_backward,
       requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=80),)),
   )
   ```

4. Import the family from `veomni/ops/kernels/__init__.py` so registration
   happens when `veomni.ops` is imported.
5. In each consuming model, construct a `VeomniOp` from
   `config.ops_config.<field>` and store it on the model/module instance.
6. For a compound raw kernel, call the nested row's raw `forward`/`backward`
   and use the saved-state helpers in `veomni/ops/compound.py`; do not call
   the nested autograd wrapper.
7. Guard optional device packages and attach an explicit requirement. A
   registered implementation must fail clearly when its requirement is not
   satisfied; it must not silently fall back to eager.
8. Add English module, module-level class, and public function docstrings.
   Nested closures and ordinary `_private` helpers are not gated; `forward` /
   `backward` / `wrapper` / `__call__` still are. Checked by
   `tests/ops/base/test_op_documentation.py`.

## Phase 3: Test

1. Add tests under `tests/ops/<kernel_name>/` for:
   - forward parity against an independent eager reference;
   - backward parity for every differentiable input;
   - dtype/shape/edge contracts;
   - registration and device requirements;
   - explicit failure for unknown or unavailable implementations.
2. Put consumer-specific normalization and model wiring tests under
   `tests/models/` instead of duplicating them in the raw-kernel suite.
   Cover distinct variants and custom backward implementations separately,
   including EP/non-EP and split/merged layouts when implemented. Standard
   Quack and GPT-OSS Quack share GEMM but not the expert computation. An
   EP-local compute test does not qualify token all-to-all; add a model/backend
   E2E representative when that integrated path is new. Follow the
   [new-model coverage policy](../../../docs/transformers_v5/testing_new_model.md)
   for representative selection and capability exceptions.
3. Run the family tests plus the registry and documentation guards:

   GPU and NPU unit manifests select `tests/ops/`, with hardware prerequisites
   and selection filters controlling individual cases. Check both manifests
   before adding an entry; an already-collected case needs no workflow edit.

   ```bash
   pytest -q tests/ops/<kernel_name>/ tests/ops/base/
   ```

4. Add a benchmark only when performance is part of the acceptance criteria;
   compare it with the canonical eager contract.

## Phase 4: Document

1. Update `veomni/ops/README.md` with the family, variants, and supported
   implementations.
2. Update `docs/design/op_selection.md` when selection behavior changes.
3. Update `.agents/knowledge/architecture.md` when the layout or call chain
   changes.

## Phase 5: Finalize

1. Run `/veomni-review`.
2. Run `make quality` and the relevant kernel/model integration tests.
3. Verify `OP_REGISTRY.list_registered(...)` and
   `list_available(...)` report the expected rows.

## Common Pitfalls

- Confusing a semantic variant with an implementation backend.
- Adding model-specific reshaping or reduction policy to the tensor registry.
- Calling a nested autograd wrapper from a compound custom-autograd function.
- Importing NPU/CUDA-only libraries without guards or requirements.
- Registering a row without importing its family from `kernels/__init__.py`.
- Letting an unavailable optimized implementation fall back silently.
- Editing patchgen-generated files instead of their patch config.
- Renaming `ops_implementation` while changing the internal kernel plumbing.

## Configuration ownership

Model ops selections belong to `VeOmniPretrainedConfig.ops_implementation`.
Read `config.ops_config.<field>` when binding a handle. HF config methods are
never patched globally; standalone module tests must construct an adapted config
with `get_veomni_config_class` or opt in with `as_veomni_config`.
Attention's HF accessors alias the same dictionary; do not add synchronization
fields or repair attention state in builders. See `docs/design/module_config.md`.

Pass selections directly to the owning variant. On NPU, unweighted RMSNorm
supports only `eager`; explicit `torch_npu`/`npu` must raise. The attention
`eager` row preserves model-local eager math when present and otherwise
forwards to `veomni_sdpa`; consumers do not remap eager themselves.
