# Transformers Integration

Current model integration targets the Transformers version pinned in
`pyproject.toml`. The current guides are organized by task:

- [Adding a model](../usage/support_new_models/guide_and_checklist.md)
- [Testing a new model](testing_new_model.md)
- [Attention interfaces](veomni_fused_attention.md)
- [Flash Attention custom-name handling](veomni_flash_attention_kernel_adapter.md)
- [MoE checkpoint conversion](transformers_v5_moe_weight_loading.md)
- [Patchgen](../design/patchgen.md)

These pages retain their existing URLs. Historical upgrade notes are listed
separately under [Migrations](../migrations/index.md); they describe a particular
transition and should not be used as the current onboarding procedure.
