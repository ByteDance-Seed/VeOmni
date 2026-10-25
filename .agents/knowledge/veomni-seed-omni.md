# SeedOmni Architecture and Constraints

Read this before modifying `veomni/models/seed_omni/`, `veomni/trainer/omni/`,
`veomni/data/seed_omni/`, `veomni/arguments/omni_arguments_types.py`,
`tasks/omni/`, or `configs/seed_omni/`. Pair with the `/veomni-seed-omni` skill
(`.agents/skills/veomni-seed-omni/SKILL.md`).

---

## 1. Module Map & Entry Points

```text
veomni/
├── arguments/
│   ├── omni_arguments_types.py     OmniArguments + build_omni_{model,module}_runtime_args
│   └── omni_parser.py              parse_omni_args
├── data/seed_omni/                 SeedOmni preprocessing, transform, media fetchers/savers
├── models/seed_omni/
│   ├── configuration_omni.py       OmniConfig (composite PretrainedConfig: modules + graphs)
│   ├── modeling_omni.py            OmniModel (eager graph forward/generate)
│   ├── processing_omni.py          OmniProcessor (ordered per-module CPU preprocessors)
│   ├── graphs/                     TrainingGraph (DAG) and GenerationGraph (FSM)
│   ├── mixins/                     BaseMixin, TrainingModuleMixin, InferenceModuleMixin,
│   │                               MetricMeterMixin, OfflineEncodingMixin
│   ├── modules/
│   │   ├── module_configuration_base.py  OmniModuleConfig (per-module HF descriptor)
│   │   ├── module_modeling_base.py       PretrainedOmniModule
│   │   ├── module_processing_base.py     ModulePreprocessorBase
│   │   └── <family>/<submodule>/         configuration.py, modeling.py, accelerated/, processing.py
│   ├── utils/
│   │   ├── checkpoint.py           OmniModuleCheckpointManager (per-module DCP/HF/LoRA)
│   │   ├── conversation.py         ConversationItem and carrier helpers
│   │   ├── convert_registry.py     OMNI_CONVERT_REGISTRY + convert_checkpoint
│   │   ├── hf_layout.py            OmniHFLayout direct HF load and export
│   │   ├── offline_cache.py        Offline-cache writer and reader
│   │   ├── graph_profiler.py       GraphProfiler (per-node timing/memory)
│   │   └── visualize.py            Mermaid export for training/generation graphs
│   └── accelerated/
│       ├── omni_model/
│       │   ├── omni_model_config.py    OmniModelRuntimeArguments (composite ModelArguments)
│       │   └── omni_model_runtime.py   OmniModelRuntime (composed graph loops, MultiOptimizer)
│       ├── omni_module/
│       │   ├── omni_module_config.py   OmniModuleRuntimeArguments (per-module ModelArguments)
│       │   └── omni_module_runtime.py  ModuleRuntime(VeOmniModelRuntime)
│       └── utils/
│           ├── executor.py             TrainNodeRunner, execute_*_node
│           ├── dispatch.py             unwrap FSDP/DDP/LoRA wrappers, call_graph_endpoint
│           └── modules.py              save_module_subdirectory
└── trainer/omni/
    ├── omni_trainer.py             OmniTrainer
    └── omni_inferencer.py          OmniInferencer
```

| Area | Entry Point / Test Command |
|------|----------------------------|
| Training launch | `tasks/omni/train_omni.py` (`OmniTrainer`) |
| VeOmni inference launch | `tasks/omni/infer_omni.py` (`OmniInferencer`) |
| Native HF inference launch | `tasks/omni/infer_omni_native.py` (`OmniModel.from_pretrained`) |
| Checkpoint conversion | `scripts/seed_omni/convert_model.py` |
| SeedOmni unit tests | `pytest tests/seed_omni/` |

---

## 2. Runtime Hierarchy & Data Flow

There are **two** ways to build a SeedOmni model, and the trainer / inferencer holds **one** model handle (`self.model`):

| Build path | `self.model` | Sub-modules | Used by |
|------------|--------------|-------------|---------|
| **Bare HF** — `OmniModel.from_pretrained(root)` / `from_config(cfg)` | `OmniModel` (a `PreTrainedModel`) | plain `PreTrainedModel` | non-VeOmni users; all-`eager` inference |
| **VeOmni** — `build_omni_model_runtime(runtime_args)` | `OmniModelRuntime` | one `ModuleRuntime` each | training; distributed inference |

`OmniModelRuntime` composes an `OmniModel` and forwards attributes it does not define (`config`, `modules_dict`, `save_pretrained`, `nn.Module` reads) to it, so shared callbacks read `trainer.model` either way. Training uses `OmniModelRuntime` to add `ParallelState` scoping, graph tracing, and metric metering via a `node_runner` (`TrainNodeRunner`) passed to `OmniModel.forward`. The graph walk lives in `modeling_omni.py`, which stays free of VeOmni runtime imports (`tests/seed_omni/model/test_graph.py::test_modeling_omni_imports_no_veomni_runtime_package`).

```text
tasks/omni/train_omni.py             -> OmniTrainer (orchestrator)
└── self.model = OmniModelRuntime    graph loops (forward), trace, metering
    ├── ModuleRuntime × N            per sub-module (FSDP, opt, ckpt, ParallelState)
    └── OmniModel                    eager graph definition (modeling_omni.py)

tasks/omni/infer_omni.py             -> OmniInferencer
└── self.model = OmniModelRuntime (any FSDP2/DDP module) | OmniModel (all eager)
```

### `ModuleRuntime(VeOmniModelRuntime)` overrides

`ModuleRuntime` inherits the single-model build sequence (meta-init, freeze/LoRA, FSDP2/DDP wrap + weight load, optimizer, lr-scheduler, `ParallelState` registration) and overrides where a module differs from a standalone model:

| Override | Module behavior |
|----------|-----------------|
| `build_model` | reads `model_path` (the module's subfolder) |
| `build_model_assets` | binds the preprocessor onto the model (`bind_module_assets`) so graph nodes and HF export can access it |
| `freeze_model` | prefixes the parameter table and VRAM log with the module's name |
| `build_parallelized_model` | supports custom `customized_build_parallelize_model` hooks; deferred to `OmniModelRuntime` when `fsdp_scope='model'` |
| `build_optimizer` / `build_lr_scheduler` | scoped to the module's mesh and skipped when the module is frozen |
| `build_checkpoint` | creates a per-module `OmniModuleCheckpointManager` under `<save_path>/global_step_N/<module>/` when trainable |
| `clip_grad_norm` | computes this module's norm (`veomni_omni_module_clip_grad_norm`) for `OmniModelRuntime` to combine |
| `skip_hf_weight_load` | keeps HF weight loading enabled on resume when the module is frozen and has persistent state |
| `on_lora_matched_nothing` | leaves a module with no matching LoRA targets frozen; `_reject_lora_that_matched_nothing` raises only if the entire composed model has no trainable parameters |
| `save_model_assets` | handled at the composed level by `OmniModelRuntime.save_model_assets` |
| `__call__` | scopes a direct forward to the module's `ParallelState` (`_scoped`) |

`module_name` aliases `model_name` and serves as the `ParallelState` registry key, the checkpoint subdirectory, and the graph node name.

### Config split & four-layer resolution

| Layer | Config source | Owns |
|-------|---------------|------|
| **OmniModel** | `OmniConfig`, projected from `OmniModelRuntimeArguments.to_hf_config()` | graph topology, module wiring |
| **OmniModuleConfig** | `OmniConfig._module_configs[name]`, loaded from the module subfolder (`modules/module_configuration_base.py`); `_module_entries[name]` holds the root overwrite fields | `model_path` / `ops_implementation` / `model_config` / `processor_config` |
| **ModuleRuntime** | `OmniModuleRuntimeArguments` (`accelerated/omni_module/omni_module_config.py`) + launcher `train` | FSDP, optimizer, checkpoint per module |
| **OmniModelRuntime** | `OmniModelRuntimeArguments` (`accelerated/omni_model/omni_model_config.py`) via `build_omni_model_runtime()` | graph loops, module runtimes, optimizer / lr scheduler, graph trace, metering |
| **OmniTrainer** | launcher YAML + `OmniArguments` | dist init, dataloader, train loop, callbacks |

Per-module settings resolve in four layers, widest first, via `_deep_update` in `build_omni_module_runtime_args` (`tests/seed_omni/arguments/test_module_args.py`):
1. the launcher's global `model:` block (`_module_base`),
2. the checkpoint's persisted entry for that module (`OmniConfig._module_entries[name]`),
3. the `fsdp_mode: eager` default when `for_inference=True`,
4. the launcher's per-module `modules:` YAML (which also defines the active module set).

```text
YAML Config -> OmniArguments
                    │
        ┌───────────┴───────────────────────────┐
        v                                       v
build_omni_model_runtime_args()         OmniTrainer._setup()
 └── build_omni_module_runtime_args()           │
     (-> OmniModelRuntimeArguments.modules)     │
        │                               ParallelState registry
        v                                       │
    build_omni_model_runtime()
     ├── ModuleRuntime × N  (FSDP2 wrap, weight load, optimizer, ckpt)
     ├── MultiOptimizer / MultiLRScheduler over the trainable modules
     └── OmniModel          (composed graph definition;
                             config = runtime_config.to_hf_config())
                    │                                 │
                    v                                 v
    trainer.model (single handle)            build_dataloader()
                    │                                 │
                    └───────────────┬─────────────────┘
                                    v
                          OmniTrainer.train()
                    (graph forward/backward, model.optimizer, model.clip_grad_norm(), callbacks)
```

---

## 3. Hard Constraints (SeedOmni)

1. **SeedOmni SP is one shared size, replicated data, single-pass per module**
   - Set `model.accelerator.ulysses_size` (CLI `--model.accelerator.ulysses_size`) on the global accelerator so every module inherits the same SP size through `build_omni_module_runtime_args`.
   - The dataloader is the standard build-time sharded loader: it yields `dp_size = world / sp_size` **distinct** shards and **replicates** each shard across its SP group, so every rank of an SP group holds the same sample with identical tensor shapes and dtypes.
   - A module with `sp_size = S > 1` pads and slices the replicated sample to its `1/S` shard inside `pre_forward` (`sp_pad` + `slice_input_tensor` or `sp_pad_and_slice`, rebuilding varlen/`cu_seqlens` over the full padded sequence first), runs the endpoint once (attention all-to-alls over the SP group internally), and all-gathers the output back to the full sample in `post_forward` (`gather_outputs`, then `narrow` off the SP pad).
   - **Gradient correctness:** `_Gather.backward` all-reduces (SUM) the grad over the SP group and takes this rank's shard; FSDP2's `÷|dp_shard_sp|` grad averaging cancels the `×sp` factor, yielding gradients identical to the non-SP baseline.
   - Design doc: `docs/seed_omni/distributed/sequence_parallel.md`.

2. **SeedOmni decode losses reduce over `dp_sp` (`fsdp_group`) for global token-weighting**
   - To keep loss and gradients identical across DP vs SP splits, decode modules (`text_encoder/modeling.py::decode`, `janus/vqvae/modeling.py::_vq_loss`) compute `local_mean = ce_sum / n_valid_local` and call `reduce_sequence_parallel_loss(local_mean, n_valid_local, group=ps.fsdp_group)`. Because `fsdp_group` spans `dp_sp` (`dp_replicate × dp_shard × sp`), `ReduceLoss.backward`'s `×|dp_sp|` factor cancels FSDP2/HSDP gradient averaging and produces the global token-weighted gradient while replicated SP copies cancel out.
   - Because the reduction collective spans all `fsdp_group` ranks, decode must run symmetrically on every rank (including all-dummy spans, which use `-100` labels and a clamped denominator `n.clamp(min=1)` to produce a graph-connected `0.0` loss). When using fused CE backends (`liger_kernel` / `chunk_loss`), count supervised tokens first and route zero-count spans to the eager branch with `num_items_in_batch=n.clamp(min=1)`.

3. **Per-module metric meter stashes pre-slice (full-sample) seqlens in `pre_forward`**
   - Each metered module calls `metric_meter_set_seqlens(method, seqlens)` inside `pre_forward` **before** the `if sp_size > 1:` slice, recording the full pre-slice sample lengths; the executor's `metric_meter_add(method)` drains that stash right after `pre_forward`.
   - Because `OmniEnvironMeter.step` reduces tokens and `total_flops` over `dp_group` (which excludes the SP dimension), recording full-sample lengths once per DP shard reconstructs the exact global total across any SP size.

4. **SP / DP / CP groups resolve from the current `ParallelState`'s mesh (`ModuleRuntime` scoping)**
   - In `veomni/distributed/sequence_parallel/comm.py`, `get_ulysses_sequence_parallel_group`, `get_unified_sequence_parallel_group`, `get_context_parallel_group`, and `get_data_parallel_group` read `_current_state()` (`_PARALLEL_STATE.{ulysses,sp,cp,dp}_group`, returning `None` when uninitialized).
   - Each `ModuleRuntime` registers its `ParallelState` under `self.module_name` via `init_parallel_state_from_config(accelerator, self.module_name)` and scopes build, forward (`OmniModelRuntime.module_context`), grad-checkpoint recompute (`ModuleRuntime._recompute_context`), optimizer/lr-scheduler build, and grad clipping (`mt.clip_grad_norm(max_norm)`) with `use_parallel_state(self.module_name)`, so `veomni_flash_attention_2_with_sp` and SP slice/gather helpers automatically use the active module's own process groups.

5. **Per-module DDP, ExtraParallel, and `fsdp_scope='model'` composed wrapping**
   - For DDP modules under SP, `veomni_omni_module_clip_grad_norm` all-reduces `p.grad` (with `SUM` then divide, selecting by `requires_grad` and zero-filling missing grads) over `fsdp_group` (`dp_sp`) when `sp_size > 1` before clipping, and `build_parallelize_model()` passes `broadcast_buffers=False` unconditionally (see constraint 7a in `.agents/knowledge/constraints.md`).
   - ExtraParallel is refused on the DDP path only when the module has a parallel plan (`_has_extra_parallel_plan()`) and `ParallelState.any_extra_parallel_enabled` is true, so a plan-less DDP vision tower can inherit the global mesh's `ep` dimension without error.
   - With `fsdp_scope='model'`, `ModuleRuntime` defers the composed wrap and weight load to `OmniModelRuntime`, which runs them under `use_parallel_state("base")` with `weights_path` passed as a `{child_name: snapshot_path}` mapping and wrap targets scoped as `{child}.{ClassName}` (`_scoped_no_split_modules`). Leftover parameters unshard in `OmniModel.forward()` via `nn.Module.__call__`.
   - A fully frozen module is materialized and loaded, then returned unwrapped.

6. **Frozen-module DCP resume (`ModuleRuntime.skip_hf_weight_load`)**
   - SeedOmni decides `skip_hf_weight_load` per `ModuleRuntime` after `_freeze_model_module` and `_setup_lora`. A fully frozen module with a non-empty `state_dict()` has no checkpoint manager from `build_checkpoint`, so `ModuleRuntime.skip_hf_weight_load` vetoes the skip (`False`) and loads its released HF weights during resume. A parameterless module with an empty `state_dict()` may skip HF loading.

7. **Dataset-stage `OmniProcessor`, `_OMNI_TOKEN_LEN`, and per-sample dummy injection for `dyn_bsz`**
   - `OmniTrainer._build_data_transform` builds `OmniProcessor.from_config(...)`, calls `bind_dummy_inputs`, and passes `processor` into `seedomni_transform` (`process_seedomni_example` and `process_seedomni_cached_example`). Every sample is preprocessed at the Dataset stage before dynamic batching; `SeedOmniCollator` is a pure list-grouping collator (`{key: [f[key] for f in features]}`).
   - Every module's CPU preprocessor (`ModulePreprocessorBase` / `TextEncoderChatTemplate`) must set `item.meta[_OMNI_TOKEN_LEN]` (from `veomni.models.seed_omni.utils.conversation`) on every item it processes—both real items and `is_dummy=True` placeholders—and inject missing-modality `is_dummy=True` placeholders **per sample** when `not inference`.
   - `get_length_fn` (`count_mode="total"` and `physical_token_cap`) sums `_omni_token_len` across all items in `sample["conversation_list"]` **including** `is_dummy=True` items; `get_length_by_labels_fn` (`count_mode="effective"`) excludes `is_dummy=True` items and sums supervised tokens (`(labels != IGNORE_INDEX).sum()` for text, `_omni_token_len` for `role == "assistant"` media, `0` for `role == "user"` media).
