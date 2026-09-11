# Transformers v5 Notes

This section documents VeOmni's integration with HuggingFace
`transformers==5.16.1` (the only supported transformers version).

## Included Notes

- [Fused attention interface](veomni_fused_attention.md): documents the unified Flash/Flex facade, config routing, native BlockMask contract, and Ulysses restrictions.
- [Flash Attention custom-name handling](veomni_flash_attention_op_adapter.md): explains why `_lazy_imports` fails for VeOmni custom attention names and how the local hub-kernel loader adapter resolves it.
- [MoE weight loading](transformers_v5_moe_weight_loading.md): explains how VeOmni expects MoE expert weights to be laid out and documents qwen3_moe handling.
- [Testing a new model](testing_new_model.md): SOP for adding test cases in `test_models_patch.py` and `test_e2e_parallel.py` when onboarding a new model.
- [Upgrading the pin 5.9.0 → 5.16.1](upgrade_5_9_to_5_16.md): what upstream changed across the bump, how each affected patch config was migrated, and how to detect the same class of drift next time.

The modeling code generation workflow (patchgen) has graduated out of this section — it is no longer tied to the transformers v5 cutover and now lives at [`design/patchgen.md`](../design/patchgen.md).

```{toctree}
:maxdepth: 1

veomni_flash_attention_op_adapter.md
veomni_fused_attention.md
transformers_v5_moe_weight_loading.md
testing_new_model.md
upgrade_5_9_to_5_16.md
```
