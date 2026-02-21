# VeOmni Flash Attention Custom Name Adapter (Transformers 5.x)

## Problem Background

VeOmni uses custom attention implementation names:

- `veomni_flash_attention_2_with_sp`
- `veomni_flash_attention_3_with_sp`
- `veomni_flash_attention_4_with_sp`

These names are registered into `ALL_ATTENTION_FUNCTIONS` and routed to VeOmni's SP-aware attention wrapper.

With Transformers 5.x, model init and flash-attention preload logic may still call
`transformers.modeling_flash_attention_utils._lazy_imports(...)` for the configured implementation string.
For non-native names, `_lazy_imports` falls back to hub-kernel loading and can fail with:

`ValueError: Could not find the currently requested flash attention implementation at veomni_flash_attention_2_with_sp`

even though VeOmni already registered the custom attention function.

## Why This Happens

The failure path is:

1. Model config keeps VeOmni custom name in `_attn_implementation`.
2. Transformers flash preload code tries to resolve low-level flash kernels from the implementation string.
3. Custom VeOmni names are not hub kernel identifiers.
4. Hub fallback returns no valid kernel entry for this name.
5. `_lazy_imports` raises before normal `ALL_ATTENTION_FUNCTIONS` dispatch takes effect.

## Adapter Strategy Implemented

Instead of patching `_lazy_imports` directly, VeOmni patches:

`transformers.integrations.hub_kernels.load_and_register_attn_kernel`

and intercepts VeOmni custom names only.
This compatibility adapter is applied only when `transformers>=5.0.0`.

For VeOmni names, the adapter returns a local kernel-like object exposing:

- `flash_attn_func`
- `flash_attn_varlen_func`

mapped to local FA2/FA3/FA4 backends:

- `veomni_flash_attention_2_with_sp` -> `flash_attn.flash_attn_func` / `flash_attn.flash_attn_varlen_func`
- `veomni_flash_attention_3_with_sp` -> `flash_attn_interface.flash_attn_func` / `flash_attn_interface.flash_attn_varlen_func`
- `veomni_flash_attention_4_with_sp` -> `flash_attn.cute.flash_attn_func` / `flash_attn.cute.flash_attn_varlen_func`

For simplicity, paged VeOmni aliases (for example `paged|veomni_flash_attention_2_with_sp`) are not handled by this adapter.

All non-VeOmni implementations are delegated to the original Transformers loader unchanged.

## Design Goals

- Keep VeOmni custom implementation names unchanged.
- Keep existing VeOmni `ALL_ATTENTION_FUNCTIONS.register(...)` behavior unchanged.
- Avoid hub-kernel lookup for VeOmni private names.
- Minimize patch surface by touching a single integration point.
- Fail fast with clear ImportError when required FA backend is missing.

## Expected Runtime Behavior

After `import veomni`:

- VeOmni custom names remain registered in `ALL_ATTENTION_FUNCTIONS`.
- `_lazy_imports("veomni_flash_attention_2_with_sp")` and `_lazy_imports("veomni_flash_attention_4_with_sp")` can resolve through the adapter.
- No spurious "kernel hub name not found" error for VeOmni custom names.
- Paged VeOmni aliases are outside the adapter scope.

## Notes

- This adapter is a compatibility bridge for Transformers 5.x behavior around flash preload.
- It does not change VeOmni SP attention semantics.
- It does not require the `kernels` Python package for VeOmni custom names.
- FA2 and FA3 have dedicated branches in `_lazy_imports` (both v4 and v5) and are
  resolved directly without reaching the hub-kernel path. The adapter is therefore a
  no-op for those two in practice, but is kept for safety.
- FA4 (`veomni_flash_attention_4_with_sp`) has no such branch in `_lazy_imports` and
  always falls through to the hub-kernel path in Transformers v5. The adapter is the
  **critical** component that makes FA4 usable. FA4 is not supported on Transformers v4.
- FA4 requires the `flash-attn-cute` package (`flash_attn.cute`). To install Transformers v5
  and FA4 together, run:
  ```
  uv sync --extra gpu --extra fa4 --extra transformers5-exp --no-group transformers-stable
  ```
