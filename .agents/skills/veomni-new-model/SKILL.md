---
name: veomni-new-model
description: "Use this skill when adding support for a new model to VeOmni. Owns the lifecycle around the modeling itself: analyzing the HuggingFace model, choosing the category, the training config, trainer and data-pipeline integration, tests and docs. The modeling patch itself is delegated to /veomni-patchgen-model. Trigger: 'add model', 'support new model', 'integrate a model', 'new model support'."
---

> The hard part of a new transformers-family model — the patchgen config,
> parallel plan, MoE weight conversion, `__init__.py` registration, codegen —
> lives in `/veomni-patchgen-model`. This skill is the wrapper around it: it
> decides *what* you are adding, then hands off, then does the config, trainer
> and data work that patchgen does not cover.

## Before You Start: Create a Plan

Track the phases with whatever todo/plan tool the running agent provides:

```
Phase 1: Analyze HF model             -> in_progress
Phase 2: Modeling (/veomni-patchgen-model)  -> pending
Phase 3: Write training config         -> pending
Phase 4: Integrate with trainer        -> pending
Phase 5: Test and document             -> pending
```

## Phase 1: Analyze HuggingFace Model

1. **Identify the model** on HuggingFace. Read its `config.json`, `modeling_*.py`, and any processor configs.

2. **Determine model category**:
   - Text-only LLM -> `veomni/models/transformers/<model_name>/`
   - Vision-Language -> `veomni/models/transformers/<model_name>/` + `veomni/data/multimodal/`
   - MoE model -> additional `veomni/distributed/moe/` integration
   - Diffusion model -> `veomni/models/diffusers/<model_name>/`

3. **Check existing similar models**: Find the closest existing model in `veomni/models/transformers/` and use it as a reference. E.g., if adding a new Qwen variant, reference `qwen3/` or `qwen3_vl/`.

4. **Identify required patches**: VeOmni uses a patchgen system (`veomni/patchgen/`) to generate model patches from the HuggingFace modeling. Check whether a sibling model already has a config you can extend via `name_map` — that is usually the difference between a 60-line config and a 1000-line one.

## Phase 2: Modeling — hand off to `/veomni-patchgen-model`

1. **Create the model directory**: `veomni/models/transformers/<model_name>/`.

2. **Switch to `/veomni-patchgen-model`.** It owns the whole modeling surface —
   the `<model_name>_{gpu,npu}_patch_gen_config.py` files, the MoE
   `parallel_plan.py` and `checkpoint_tensor_converter.py`, `__init__.py`
   registration, `make patchgen`, and the model-level test cases — with the
   working examples and the pitfalls that cost the most time. Do not re-derive
   it from this file.

   Note that `parallel_plan.py` is **not** an FSDP wrapping policy: FSDP2 wraps
   generically in `build_parallelize_model()`, and `ParallelPlan`
   (`veomni/distributed/parallel_plan.py`) only describes ExtraParallel
   EP/embedding sharding. Every `parallel_plan.py` in the repo belongs to a MoE
   model; a dense model does not need one.

3. **Exception — non-transformers architectures.** Diffusion models under
   `veomni/models/diffusers/<model_name>/`, and the `flux` / `movqgan` / `wan`
   directories, have no `generated/` output and no patchgen config: they patch
   through `device_patch.py` or direct modeling. Copy the closest existing one
   and skip to Phase 3.

Come back here once the model loads and its registry / patch tests pass.

## Phase 3: Write Training Config

1. **Model config**: Create `configs/model_configs/<model_family>/<ModelName>.json` matching HuggingFace format.

2. **Training config**: Create YAML in the appropriate directory:
   - Text: `configs/text/<model_name>.yaml`
   - Multimodal: `configs/multimodal/<model_name>/<model_name>.yaml`
   - DiT: `configs/dit/<model_name>.yaml`

3. Config must include: model path, data config, optimizer settings, parallelism config, checkpoint settings.

4. **Verify against existing configs** — match the structure of similar model configs.

## Phase 4: Integrate with Trainer

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

## Phase 5: Test and Document

1. **Create toy config**: Add `tests/toy_config/<model_name>_toy/config.json` with minimal parameters for fast testing.

2. **Unit tests**: add cases to the existing enumerated tables rather than new
   files — `tests/models/test_model_registry.py` and
   `tests/models/test_models_patch.py` (`TEST_CASES`) already cover loading via
   `veomni.models.auto`, forward output shape, and patch application. See
   `.agents/knowledge/testing.md` for the full landing-spot table and for why a
   new file outside `tests/ops/` / `tests/data/` will not run in CI unless it is
   wired into the unit-test workflows.

3. **E2e tests** (if feasible): add a `pytest.param` to
   `tests/e2e/test_e2e_parallel.py` using the toy config, rather than a new
   e2e file.

4. Run `make quality` and `pytest tests/models/`.

5. **Update documentation**:
   - Add usage example to `docs/` (training command, config reference).
   - Update `.agents/knowledge/architecture.md` if the model adds a new module or trainer path.
   - Update supported models table in project `README.md` if applicable.

## Common Pitfalls

- **Model registry**: Registration must happen at import time in `__init__.py`. If the model's `AutoConfig` type is not registered, `build_foundation_model()` will fail.
- **Tokenizer compatibility**: Some models require specific tokenizer versions or custom chat templates — verify in `veomni/data/chat_template.py`.
- **Skipping the handoff**: the modeling pitfalls — never editing `generated/`, MoE expert layout, `name_map` reuse, Omni subtree exclusion — are in `/veomni-patchgen-model`, not here. This file deliberately does not restate them, so a summary read of Phase 2 is not enough to write a config.
