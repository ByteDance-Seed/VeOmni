# Transformers v5 Compatibility Updates

This section documents VeOmni's compatibility work for HuggingFace `transformers>=5.0.0`.

## Included Updates

- [Flash Attention custom-name handling](veomni_flash_attention_kernel_adapter.md): explains why `_lazy_imports` failed for VeOmni custom attention names and how the local hub-kernel loader adapter resolves it.
- [Qwen3 patchgen workflow](patchgen.md): explains the modeling code generation workflow used for Qwen3 GPU patches and regeneration.

```{toctree}
:maxdepth: 1

veomni_flash_attention_kernel_adapter.md
patchgen.md
```
