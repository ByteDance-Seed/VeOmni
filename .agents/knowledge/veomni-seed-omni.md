# SeedOmni Architecture and Constraints

Read before touching `veomni/models/seed_omni/`, `veomni/trainer/omni/`, `veomni/data/seed_omni/`, `veomni/arguments/omni_arguments_types.py`, `tasks/omni/`, or `configs/seed_omni/`. Pair with `/veomni-seed-omni`.

## 1. Module Map & Entry Points

```text
veomni/
├── arguments/omni_{arguments_types,parser}.py  OmniArguments + build_omni_{model,module}_runtime_args
├── data/seed_omni/                             seedomni_transform, SeedOmniCollator, media fetchers/savers
├── models/seed_omni/
│   ├── configuration_omni.py                   OmniConfig (modules + graphs)
│   ├── modeling_omni.py                        OmniModel (eager graph forward/generate; no veomni runtime imports)
│   ├── processing_omni.py                      OmniProcessor (ordered per-module CPU preprocessors)
│   ├── graphs/                                 TrainingGraph (DAG) and GenerationGraph (FSM)
│   ├── mixins/                                 BaseMixin, TrainingModuleMixin, InferenceModuleMixin, MetricMeterMixin, OfflineEncodingMixin
│   ├── modules/                                module_{configuration,modeling,processing}_base.py + <family>/<submodule>/
│   ├── utils/                                  checkpoint, conversation, convert_registry, hf_layout, offline_cache, graph_profiler, visualize
│   └── accelerated/
│       ├── omni_model/                         OmniModelRuntimeArguments, OmniModelRuntime (graph loops, MultiOptimizer)
│       ├── omni_module/                        OmniModuleRuntimeArguments, ModuleRuntime(VeOmniModelRuntime)
│       └── utils/                              executor (TrainNodeRunner), dispatch (unwrap wrappers), modules
└── trainer/omni/                               OmniTrainer, OmniInferencer
```

Entry points: `tasks/omni/train_omni.py`, `tasks/omni/infer_omni.py`, `tasks/omni/infer_omni_native.py`, `scripts/seed_omni/convert_model.py`. Tests: `pytest tests/seed_omni/`.

## 2. Runtime Hierarchy & Config Resolution

Two build paths expose one `self.model` handle:
- **Bare HF** (`OmniModel.from_pretrained` / `from_config`): plain `PreTrainedModel` graph (all-eager inference). `modeling_omni.py` must stay free of VeOmni runtime imports (`test_modeling_omni_imports_no_veomni_runtime_package`).
- **VeOmni** (`build_omni_model_runtime`): `OmniModelRuntime` composing `OmniModel` + one `ModuleRuntime(VeOmniModelRuntime)` per sub-module + `MultiOptimizer`/`MultiLRScheduler`, passing `TrainNodeRunner` into `OmniModel.forward`.

### Key `ModuleRuntime` Overrides (vs `VeOmniModelRuntime`)

- `module_name` aliases `model_name` and keys `ParallelState`, checkpoint subdir (`<save_path>/global_step_N/<module>/`), and graph node name.
- `build_model` / `build_model_assets`: loads module subfolder and binds preprocessor onto the module (`bind_module_assets`).
- `build_parallelized_model`: honors `customized_build_parallelize_model`; deferred to `OmniModelRuntime` when `fsdp_scope='model'`.
- `build_optimizer` / `build_lr_scheduler`: skipped if frozen; passes optimizer to `module.configure_optimizer(optimizer)` when defined (hooks changing param-group count must stay present on resume).
- `extra_state` / `load_extra_state`: persists per-rank `module_extra_state` (`extra_state_dict`/`load_extra_state_dict`) and legacy `module_rng_state` (`rng_state_dict`/`load_rng_state_dict`) beside `lr_scheduler`; frozen modules have no checkpoint manager.
- `skip_hf_weight_load`: vetoes skip (`False`) on resume for frozen modules with non-empty `state_dict()`.
- `on_lora_matched_nothing`: leaves unmatched sub-module frozen; `_reject_lora_that_matched_nothing` raises only if the whole composed model has zero trainable params.
- `clip_grad_norm` / `__call__`: computes per-module norm (`veomni_omni_module_clip_grad_norm`) and scopes direct calls via `_scoped`.

### Four-Layer Per-Module Config Precedence (`build_omni_module_runtime_args`)

Merged widest-first via `_deep_update` (`tests/seed_omni/arguments/test_module_args.py`):
1. Launcher global `model:` block (`_module_base`)
2. Checkpoint persisted entry (`OmniConfig._module_entries[name]`)
3. `fsdp_mode: eager` default when `for_inference=True`
4. Launcher per-module `modules:` YAML (also defines active module set)

## 3. Hard Constraints (SeedOmni)

1. **Shared `dp_size` under SP/CP (`docs/seed_omni/distributed/sequence_parallel.md`)**
   - Modules may independently pick Ulysses (`ulysses_size=S, cp_size=1`) or CP (`cp_size=S, ulysses_size=1`) as long as `pp_size = tp_size = 1` and every distributed module + `model.accelerator` resolves to the same `dp_size = world_size // (cp_size * ulysses_size)` (`_validate_omni_dp_size`; combining `cp_size > 1` and `ulysses_size > 1` on one module is rejected). Remember to zero out the opposite axis when overriding via `_deep_update`.
   - Dataloader yields `dp_size` distinct shards replicated across each `sp_group`. A module with `sp_size > 1` pads/slices inside `pre_forward` (rebuilding varlen `cu_seqlens` over full padded length first) and all-gathers + narrows in `post_forward`.
   - **Sharded handoff (`accelerator.skip_sp_gather_between_modules=True`)**: opted in per module class via `PretrainedOmniModule.supports_sp_sharded_handoff` (refused if any participating module is `False`). `text_encoder.encode_post`, `vision.forward_post`, `llm.forward_post` scatter `1/S` real-row shards via `split_sp_sharded_tensor` (`veomni/models/seed_omni/modules/base/sp_shard.py`); consumers restore padding via `pad_sp_shard`, read lengths via `item_token_len` (`meta[_OMNI_TOKEN_LEN]`), bypass `AllGather` on single-modality batches (decided from `item.type`/`meta` so all ranks agree), or gather per producer via `gather_sp_sharded_groups`. `ReduceLoss.backward` over `fsdp_group` supplies the `×|dp_sp|` grad scale directly.

2. **Decode losses reduce over `fsdp_group` (`dp_sp`) for global token-weighting**
   - Decode modules (`text_encoder/modeling.py::decode`, `janus/vqvae/modeling.py::_vq_loss`) call `reduce_sequence_parallel_loss(ce_sum / n_valid_local, n_valid_local, group=ps.fsdp_group)`.
   - Must execute symmetrically on all ranks including all-dummy spans (`-100` labels, clamped `n.clamp(min=1)` → graph-connected `0.0` loss). With fused CE (`liger_kernel`/`chunk_loss`), route zero-count spans to eager with `num_items_in_batch=n.clamp(min=1)`.

3. **Metering stashes pre-slice seqlens in `pre_forward`**
   - Call `metric_meter_set_seqlens(method, seqlens)` in `pre_forward` **before** `if sp_size > 1:` slicing (`OmniEnvironMeter.step` reduces over `dp_group`, excluding SP).

4. **Per-module `ParallelState` scoping**
   - Each `ModuleRuntime` registers under `self.module_name` and scopes build, forward (`module_context`), grad-ckpt recompute (`_recompute_context`), optimizer/scheduler, and clipping with `use_parallel_state(self.module_name)`.

5. **Per-module DDP, ExtraParallel, & `fsdp_scope='model'`**
   - DDP under SP follows `.agents/knowledge/topics/distributed.md` (`veomni_omni_module_clip_grad_norm` reduces over `fsdp_group`; `broadcast_buffers=False`; ExtraParallel refused only when `_has_extra_parallel_plan()` is true).
   - With `fsdp_scope='model'`, `OmniModelRuntime` wraps once under `use_parallel_state("base")` using `{child_name: snapshot_path}` and `{child}.{ClassName}` wrap targets (`_scoped_no_split_modules`). Fully frozen modules are materialized, loaded, and returned unwrapped.

6. **Frozen-module DCP resume**
   - Frozen modules with non-empty `state_dict()` keep `skip_hf_weight_load=False` on resume (no checkpoint manager exists for them); parameterless modules may skip.

7. **Dataset-stage `OmniProcessor`, `_OMNI_TOKEN_LEN`, & per-sample dummies**
   - `seedomni_transform` runs `OmniProcessor` per sample in the Dataset stage before dynamic batching; `SeedOmniCollator` only groups lists.
   - Every CPU preprocessor (`ModulePreprocessorBase` / `TextEncoderChatTemplate`) must write `item.meta[_OMNI_TOKEN_LEN]` on both real and `is_dummy=True` items and inject missing-modality dummies per sample when `not inference`.
   - `dyn_bsz_count_mode="total"` sums `_omni_token_len` including dummies; `"effective"` excludes dummies and sums supervised text + assistant-media tokens (`0` for user media).
