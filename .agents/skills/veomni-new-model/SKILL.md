---
name: veomni-new-model
description: "Use this skill when adding support for a new model to VeOmni. Covers the full lifecycle: analyzing the HuggingFace model, creating model patches, defining parallel plans, writing configs, integrating with the trainer, and testing. Trigger: 'add model', 'support new model', 'integrate <model_name>', 'new model support'."
---

## Before You Start: Create Todos

Use TodoWrite to track all phases:

```
Phase 1: Analyze HF model             -> in_progress
Phase 2: Create model patch            -> pending
Phase 3: Define parallel plan          -> pending
Phase 4: Write training config         -> pending
Phase 5: Integrate with trainer        -> pending
Phase 6: Test                          -> pending
```

## Phase 1: Analyze HuggingFace Model

1. **Identify the model** on HuggingFace. Read its `config.json`, `modeling_*.py`, and any processor configs.

2. **Determine model category**:
   - Text-only LLM -> `veomni/models/transformers/<model_name>/`
   - Vision-Language -> `veomni/models/transformers/<model_name>/` + `veomni/data/multimodal/`
   - MoE model -> additional `veomni/distributed/moe/` integration
   - Diffusion model -> `veomni/models/diffusers/<model_name>/`
   - Omni model -> `veomni/models/seed_omni/`

3. **Check existing similar models**: Find the closest existing model in `veomni/models/transformers/` and use it as a reference. E.g., if adding a new Qwen variant, reference `qwen3/` or `qwen3_vl/`.

4. **Identify required patches**: VeOmni uses a patchgen system (`veomni/patchgen/`) to auto-generate model patches from HuggingFace models. Check if a patch spec already exists or if one needs to be created.

## Phase 2: Create Model Patch

1. **Create the model directory**: `veomni/models/transformers/<model_name>/`

2. **Required files**:
   - `__init__.py` — model registration (`MODELING_REGISTRY` / `MODEL_CONFIG_REGISTRY` / `MODEL_PROCESSOR_REGISTRY`)
   - `<model_name>_gpu_patch_gen_config.py` — declarative patchgen config (replace_class / override_method / replace_function / modify_init / add_post_import_block / drop_import_names) defining all VeOmni patches against the upstream HF modeling
   - `<model_name>_npu_patch_gen_config.py` — NPU patchgen config (often just imports the GPU config and applies NPU-specific overrides via `name_map`)
   - `parallel_plan.py` — FSDP / TP / EP sharding plan
   - `generated/patched_modeling_<model_name>_{gpu,npu}.py` — patchgen output (do NOT edit manually)

3. **Patch patterns** — follow existing models:
   - Sequence parallel: declare an `OpSlot` for attention/loss and override `forward` via patchgen
   - MoE: stack per-expert weights (`gate_up_proj [E, 2*I, H]` / `down_proj [E, H, I]`) and add a `veomni_moe_experts_forward` `OpSlot`
   - Cross-entropy: add a `veomni_causal_lm_loss` `OpSlot` and return `CausalLMOutputWithLogProbs`
   - Register the model class in the model package `__init__.py` (no entry in `veomni/models/auto.py` is needed for transformers models — registration happens via the per-model `MODELING_REGISTRY` decorators)

4. **Run patchgen**: `make patchgen` regenerates every `generated/patched_modeling_*.py` from the matching `*_patch_gen_config.py`.

## Phase 3: Define Parallel Plan

1. Create `parallel_plan.py` in the model directory.

2. Define FSDP/FSDP2 sharding strategy:
   - Which layers to wrap (typically transformer blocks)
   - Activation checkpointing granularity
   - Parameter dtype policies

3. If the model is MoE, define expert parallelism plan in addition to FSDP.

4. Reference existing parallel plans for guidance (e.g., `veomni/models/transformers/qwen3/parallel_plan.py`).

## Phase 4: Write Training Config

1. **Model config**: Create `configs/model_configs/<model_family>/<ModelName>.json` matching HuggingFace format.

2. **Training config**: Create YAML in the appropriate directory:
   - Text: `configs/text/<model_name>.yaml`
   - Multimodal: `configs/multimodal/<model_name>/<model_name>.yaml`
   - DiT: `configs/dit/<model_name>.yaml`

3. Config must include: model path, data config, optimizer settings, parallelism config, checkpoint settings.

4. **Verify against existing configs** — match the structure of similar model configs.

## Phase 5: Integrate with Trainer

1. Verify the model works with the appropriate trainer:
   - Text -> `TextTrainer` (`veomni/trainer/text_trainer.py`)
   - VLM -> `VLMTrainer` (`veomni/trainer/vlm_trainer.py`)
   - DiT -> `DitTrainer` (`veomni/trainer/dit_trainer.py`)

2. If the model needs custom data preprocessing:
   - Add transform in `veomni/data/data_transform.py` or `veomni/data/multimodal/`
   - Register the transform for the model

3. If the model needs custom collator logic:
   - Extend `veomni/data/data_collator.py`

4. **VLM only — multimodal metadata precompute**: to keep the ViT forward free
   of host-device CUDA syncs, derive ViT `cu_seqlens` / `max_seqlen` in the
   collator rather than the forward. Follow the checklist in
   `.agents/knowledge/multimodal_metadata.md` ("Adding the hook to a new model"):
   a `collate_multimodal_metadata` patchgen helper + a `get_metadata_collate_func`
   override, the per-modality `vit_metadata` sub-dict threaded through
   Model.forward → ViT.forward (with a runtime fallback), and the model added to
   `_MM_METADATA_WIRED_CASES` in the sync gate test.

## Phase 6: Test

1. **Create toy config**: Add `tests/toy_config/<model_name>_toy/config.json` with minimal parameters for fast testing.

2. **Unit tests**: Add tests in `tests/models/` to verify:
   - Model loads correctly via `veomni.models.auto`
   - Forward pass produces correct output shape
   - Model patch applies without errors

3. **E2e tests** (if feasible): Test a short training run using the toy config.

4. Run `make quality` and `pytest tests/models/`.

5. **Update documentation**:
   - Add usage example to `docs/` (training command, config reference).
   - Update `.agents/knowledge/architecture.md` if the model adds a new module or trainer path.
   - Update supported models table in project `README.md` if applicable.

## Common Pitfalls

- **Model registry**: Registration must happen at import time in `__init__.py`. If the model's `AutoConfig` type is not registered, `build_foundation_model()` will fail.
- **Generated files**: Never edit files in `generated/` directories — they are overwritten by patchgen. Edit the matching `<model>_{gpu,npu}_patch_gen_config.py` and re-run `make patchgen` instead.
- **Tokenizer compatibility**: Some models require specific tokenizer versions or custom chat templates — verify in `veomni/data/chat_template.py`.
- **Transformers version**: All modeling targets `transformers==5.9.0` (pinned by the `transformers-stable` default dependency group). Models register through the patchgen-generated path under `generated/`; do not introduce legacy `modeling_<m>.py` files or `apply_veomni_<m>_patch()` helpers.
