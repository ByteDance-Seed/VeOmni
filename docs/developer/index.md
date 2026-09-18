# Developer Guide

Start with the [contribution workflow](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md). This section covers extending and testing the framework; routine training instructions live in the [User Guide](../usage/index.md).

```{toctree}
:maxdepth: 1
:caption: Architecture and extension points

architecture
dependencies
../usage/basic_modules
../usage/trainer
../key_features/model_loader
../key_features/preprocessor_registry
```

```{toctree}
:maxdepth: 1
:caption: Model integration

../usage/support_new_models/guide_and_checklist
multimodal_metadata
../transformers_v5/testing_new_model
../usage/support_new_models/qwen3_vl_example
../usage/support_new_models/qwen3_omni_moe_example
../usage/support_new_models/dit_model_guide
```

```{toctree}
:maxdepth: 1
:caption: Development workflow

lora
../testing
../usage/agent_workflow
```

For documentation changes, use the
[authoring guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/README.md).
Architecture and dependency explanations are maintained here; `.agents/`
contains the additional constraints and procedures used by coding agents.
