# Transformers v5 Notes

This section documents VeOmni's integration with HuggingFace
`transformers==5.9.0` (the only supported transformers version).

## Included Notes

- [Flash Attention custom-name handling](veomni_flash_attention_kernel_adapter.md): explains why `_lazy_imports` fails for VeOmni custom attention names and how the local hub-kernel loader adapter resolves it.
- [MoE weight loading](transformers_v5_moe_weight_loading.md): explains how VeOmni expects MoE expert weights to be laid out and documents qwen3_moe handling.
- [Testing a new model](testing_new_model.md): SOP for adding test cases in `test_models_patch.py` and `test_e2e_parallel.py` when onboarding a new model.

The modeling code generation workflow (patchgen) has graduated out of this section — it is no longer tied to the transformers v5 cutover and now lives at [`design/patchgen.md`](../design/patchgen.md).

```{toctree}
:maxdepth: 1

veomni_flash_attention_kernel_adapter.md
transformers_v5_moe_weight_loading.md
testing_new_model.md
```
