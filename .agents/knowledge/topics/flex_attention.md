# FlexAttention & Pinned FA4 Traps

Quick-check invariants for `veomni/ops/kernels/attention/standard/flex/` and `veomni/ops/kernels/attention/mask/flex.py`. Full mechanism, layouts, and hardware scope live in `docs/transformers_v5/veomni_attention_interface.md`.

- **Compilation**: compile with `dynamic=True, fullgraph=True` (no static fallback), deriving bounds from post-Ulysses Q/K shapes and keeping cache exhaustion explicit. Keep `torch.compile(create_block_mask)` enabled by default (eager construction allocates $O(T^2)$ intermediates).
- **Inputs & visibility**: reject `output_attentions=True` before Ulysses/dispatch (second return is optional LSE, never attention weights). Normalize 3D `[T,H,D]` via views before validation/dispatch/Ulysses and restore layout on exit.
- **Pinned FA4 (`4.0.0b16`) shim**: install before compilation; apply Q-only clamping only on physically padded Q tiles (accounting for PackGQA) with a proven complete K tile (never trust incoming `False` as proof). Materialize zero-stride Q/K/V views after KV expansion (native b16 omits broadcast strides from its cache key) and bump separate JIT-cache namespaces on shader changes. Validate padded reads with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
- **CuteDSL callbacks**: index device buffers by token positions (constant `buffer[0]` and 0-D scalar tensors fail CuteDSL lowering; skip zero offsets), precompute flattened coordinates outside callbacks (no symbolic `index_expr`), preserve index dtype (`Int32`), and use invocation-local predicates (never temporarily monkey-patch HF module helpers).
