---
name: veomni-new-op
description: "Use this skill when adding a new optimized kernel or operator to veomni/ops/. Covers the full lifecycle: understanding VeOmni's ops architecture (monkey-patch + global function pointer pattern), implementing the kernel, registering it, adding tests, and documenting it. Trigger: 'add op', 'new kernel', 'add attention variant', 'new fused op', 'add triton kernel', 'optimize operator'."
---

## Before You Start

1. Read `.agents/knowledge/constraints.md` — especially rules about NPU guards (#19, #20).
2. Read `docs/design/kernel_selection.md` — understand the kernel lifecycle and selection mechanisms.
3. Familiarize yourself with the ops architecture below.

## VeOmni Ops Architecture

VeOmni ops use a **global function pointer + monkey-patch** pattern:

```
veomni/ops/<op_name>/
├── __init__.py          # Public API function + apply_veomni_*_patch()
├── <impl_a>.py          # Implementation A (e.g., triton kernel)
├── <impl_b>.py          # Implementation B (e.g., eager PyTorch fallback)
└── <npu_impl>.py        # NPU variant (optional)
```

**Key pattern**: Each op module defines:
1. A **global function pointer** (e.g., `_fused_moe_forward = None`) — starts as None.
2. A **public API function** (e.g., `fused_moe_forward()`) — dispatches through the pointer.
3. A **patch function** (e.g., `apply_veomni_fused_moe_patch()`) — binds the pointer to a concrete implementation at runtime.

Patch functions are called from:
- `veomni/ops/__init__.py` -> `apply_ops_patch()` (import time, for attention/loss/load-balancing)
- `veomni/models/auto.py` -> `build_foundation_model()` (model build time, for MoE)

### Existing Ops

| Op | Directory | Patch time | Implementations |
|----|-----------|------------|-----------------|
| Flash Attention (FA2/3/4 + SP) | `flash_attn/` | import | transformers FA, with sequence parallel wrappers |
| Cross-Entropy Loss | `fused_cross_entropy/` | import | eager (PyTorch), liger_kernel (fused) |
| Load Balancing Loss | `fused_load_balancing_loss/` | import | torch_native, triton_kernel |
| Fused MoE | `fused_moe/` | model build | group_gemm (triton), quack_gemm (CUTLASS), npu_group_gemm |
| Group GEMM | `group_gemm/` | N/A (library) | triton kernels + benchmark utils |
| Batch Invariant Ops | `batch_invariant_ops/` | N/A (utility) | numerical stability helpers |
| DiT RoPE | `dit/rope_wan/` | N/A (direct import) | Wan rotary embedding |
| NPU Patches | `npu_patch/` | conditional | hccl_premul_sum, npu_fused_operator |

## Phase 1: Design

1. **Determine op category**:
   - **Monkey-patch op** (replaces a transformers/torch function globally): follow the function pointer pattern. Registered in `apply_ops_patch()` or `build_foundation_model()`.
   - **Library op** (called directly by model code): just create the module, no patch needed.
   - **NPU variant**: add alongside GPU implementation with `is_torch_npu_available()` guard.

2. **Decide selection mechanism**: read `docs/design/kernel_selection.md` to determine if you need:
   - Config field in `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`)
   - Environment variable
   - Both

3. **Determine patch timing**:
   - **Import time**: for ops that replace transformers internals globally (attention, loss). Add to `apply_ops_patch()` in `veomni/ops/__init__.py`.
   - **Model build time**: for ops that depend on model config (MoE implementation). Add to `build_foundation_model()`.

## Phase 2: Implement

1. **Create the op directory**: `veomni/ops/<op_name>/`

2. **Write `__init__.py`** following the pattern:
   ```python
   _my_op = None  # global function pointer

   def my_op(...):
       """Public API — dispatches through the pointer."""
       if _my_op is None:
           raise NotImplementedError("...")
       return _my_op(...)

   def apply_veomni_my_op_patch():
       """Bind the function pointer to a concrete implementation."""
       global _my_op
       if is_torch_npu_available():
           from .npu_impl import npu_my_op
           _my_op = npu_my_op
       else:
           from .default_impl import default_my_op
           _my_op = default_my_op
   ```

3. **Write implementations** in separate files (e.g., `triton_impl.py`, `eager.py`, `npu_impl.py`).

4. **Register in the ops system**:
   - Add import to `veomni/ops/__init__.py`
   - If monkey-patch op: add the `(alias, function_pointer)` tuple to `build_ALL_OPS()`
   - If import-time patch: call `apply_veomni_*_patch()` from `apply_ops_patch()`
   - If build-time patch: call from `build_foundation_model()` in `veomni/models/auto.py`

5. **NPU support**:
   - Always guard NPU imports with `is_torch_npu_available()`
   - Put NPU implementations in a separate file (e.g., `npu_impl.py`)
   - NPU patches live in `veomni/ops/npu_patch/` if they are general-purpose

## Phase 3: Test

1. **Add unit tests** to `tests/ops/`:
   - Test correctness: compare output against a reference implementation (eager PyTorch)
   - Test numerical precision: verify tolerance for bf16/fp16
   - Test edge cases: empty inputs, single-element tensors, extreme shapes

2. **Add benchmark** (optional but recommended for performance-critical ops):
   - Use `veomni/ops/group_gemm/utils/benchmark_utils.py` as reference
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
3. Verify `build_ALL_OPS()` and `format_kernel_functions()` include the new op.

## Common Pitfalls

- **Forgetting to register in `build_ALL_OPS()`**: the op will work but won't appear in the ops format output, making debugging harder.
- **Unconditional NPU imports**: importing NPU modules without `is_torch_npu_available()` guard crashes on GPU-only environments.
- **Patching at wrong time**: import-time patches happen before model config is available. If your op depends on model config, it must be patched at model build time.
- **Sequence parallel interaction**: ops that touch attention or loss must handle sequence parallel correctly — use `get_parallel_state().sp_enabled` to check and dispatch.
- **Mixed precision**: fused kernels often require specific dtypes (bf16/fp16). Add assertions at the public API level to catch dtype mismatches early.
- **Not updating `__all__`**: if the op provides a public API function, export it from `veomni/ops/__init__.py`.
