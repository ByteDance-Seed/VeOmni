# VeOmni Architecture Overview

This document describes VeOmni's architecture for AI coding agents. Read this to understand where code lives and how components interact.

## Module Map

```
veomni/
├── arguments/          CLI argument parsing (VeOmniArguments dataclass)
├── checkpoint/         DCP-based distributed checkpoint save/load
├── data/               Data pipeline: datasets, collators, transforms, dynamic batching
│   ├── multimodal/     Vision, audio, video preprocessing and chat templates
│   └── diffusion/      Diffusion model data loading
├── distributed/        All parallelism strategies
│   ├── parallel_state.py   init_parallel_state_from_accelerator(), ParallelState, mesh setup
│   ├── torch_parallelize.py  build_parallelize_model(), parallelize_model_fsdp2()
│   ├── parallel_plan.py    ParallelPlan for ExtraParallel (EP, embedding shard)
│   ├── fsdp2/          FSDP2 (composable fully_shard), gradient clipping
│   ├── moe/            MoE expert parallelism: token routing, all-to-all, EPGroupGemm
│   └── sequence_parallel/  Ulysses SP: all-to-all head/seq exchange, async variants
├── models/             Model loading and patching
│   ├── auto.py         High-level API: build_foundation_model, build_tokenizer, build_processor
│   ├── loader.py       Registry-based model loading (MODELING_REGISTRY, MODEL_CONFIG_REGISTRY)
│   ├── model_runtime.py  VeOmniModelRuntime: the model-bound half of a job
│   │                   (build, freeze/LoRA, parallelize, optimizer,
│   │                   lr-scheduler, grad clip, its own ParallelState)
│   ├── transformers/   Per-model patches (one subpackage per model family)
│   └── diffusers/      Diffusion model families (Wan, LTX, Qwen-Image)
├── optim/              Optimizer and LR scheduler construction
│   ├── optimizer.py    build_optimizer() factory + MultiOptimizer wrapper.
│   │                   For optimizer.type=="muon" splits params Muon vs AdamW
│   │                   and (under FSDP+EP) further by ExtraParallel mesh, so
│   │                   the resulting MultiOptimizer holds up to four
│   │                   sub-optimizers: muon_<para>, muon_non_extra_parallel,
│   │                   <para>, non_extra_parallel.
│   ├── muon.py         DistributedMuon: DTensor-aware Muon for 2D dense and
│   │                   3D MoE expert weights, plus the batched_newton_schulz
│   │                   primitive (Keller-Jordan quintic NS over the trailing
│   │                   two dims; 2D path is byte-equivalent to
│   │                   torch.optim._muon._zeropower_via_newtonschulz, 3D path
│   │                   uses baddbmm so each slice keeps the same fused
│   │                   arithmetic). Per-param classifier picks one of
│   │                   {local, fsdp_gather_2d, moe_local_3d, moe_gather_3d};
│   │                   Shard(0) experts run locally with zero comm (opt-in
│   │                   via OptimizerConfig.muon_expert_zero_comm), Shard(d>0)
│   │                   experts go through one all-to-all-gather over the
│   │                   ep_fsdp mesh.
│   └── lr_scheduler.py LR scheduler construction
├── ops/                Optimized kernels and dispatch
│   ├── config/         Unified ops registry + singleton resolved config
│   │   ├── registry.py OpSpec/BackendSpec/OpScope + register_op/apply_*
│   │   └── singleton.py  get_ops_config()/set_ops_config() for patch files
│   ├── kernels/        Kernel implementations (one subdir per op)
│   │   ├── deepseek_v4/  TileLang sparse attention/indexer + precision helpers
│   │   ├── attention/  Flash attention v2/3/4 + SP-aware variants
│   │   ├── cross_entropy/  eager/liger/npu-chunk loss variants
│   │   ├── load_balancing_loss/  eager + triton variants
│   │   ├── mhc/        TileKernels DeepSeek V4 pre/post/head adapters
│   │   ├── rms_norm/   Liger/NPU/batch-invariant Triton RMSNorm
│   │   ├── rotary/     Liger/NPU + DeepSeek V3 deterministic + Wan Triton
│   │   ├── swiglu_mlp/ Liger SwiGLU MLP
│   │   └── moe/        Fused MoE kernels + group_gemm sub-kernels
│   ├── platform/       Platform-specific runtime patches
│   │   └── npu/        HCCL pre-mul sum patch
│   └── batch_invariant_ops/  Mode switch for deterministic ops
├── patchgen/           Auto-generate model patches from HuggingFace models
├── schedulers/         LR scheduler implementations (flow matching)
├── trainer/            Training loop implementations
│   ├── base.py         BaseTrainer (ABC): VeOmniModelRuntime + the job-bound
│   │                   half (distributed init, data pipeline, loop, callbacks)
│   ├── text_trainer.py TextTrainer: LLM SFT training
│   ├── vlm_trainer.py  VLMTrainer: vision-language model training
│   ├── dit_trainer.py  DitTrainer: diffusion transformer training
│   ├── text_dpo_trainer.py  DPO training for text models
│   ├── base_rl_trainer.py   Base RL trainer for RLHF
│   └── callbacks/      Training callbacks (checkpoint, evaluate, trace, etc.)
└── utils/              Shared utilities (logging, device, constants, helpers)
```

## Trainer Hierarchy

A training job splits in two. `VeOmniModelRuntime` (`veomni/models/model_runtime.py`) owns everything bound to *one* model; `BaseTrainer` *holds* one as `trainer.model` and adds everything bound to the *job*, of which there is exactly one. A single-model job is one runtime inside one trainer; a composed omni model is many runtimes under one orchestrator, which is why the trainer composes the runtime rather than inheriting it — `OmniTrainer` holds several and could not inherit them all.

`trainer.model` is therefore the runtime, not the raw module. It forwards unshadowed `nn.Module` APIs (`.parameters()`, `.config`) via `__getattr__` and forwards `trainer.model(**batch)` via `__call__`, so most call sites read unchanged; the real `nn.Module` is at `trainer.model.model`. Code that needs a genuine module — anything doing `isinstance`, or handing it to `fully_shard` / DCP — must go through the runtime's own methods instead of unwrapping at the call site.

What a trainer holds:

```
BaseTrainer (ABC)                     job: data, train_step, loop, callbacks
└── .model: VeOmniModelRuntime        one model: build, freeze/LoRA, parallelize,
    ├── .model: nn.Module             optimizer, lr-scheduler, clip, ParallelState
    └── .checkpoint:                  DCP / HF / LoRA save-load + directory layout
        ModelCheckpointManager
```

Who subclasses it:

```
BaseTrainer (ABC)
├── TextTrainer             -> tasks/train_text.py
├── VLMTrainer              -> tasks/train_vlm.py
├── DitTrainer              -> tasks/train_dit.py
├── TextDPOTrainer          -> tasks/train_text_dpo.py
└── BaseRLTrainer (ABC)
    ├── (text RL)           -> tasks/train_text_rl.py
    └── (VLM RL)            -> tasks/train_vlm_rl.py
```

`VeOmniModelRuntime` contributes the model-bound half. `setup()` runs on construction — registering the mesh, then building over it:
- `build_model()` -> meta-init through the registry-aware loader
- `freeze_model()` / `setup_lora()` -> trainable surface
- `build_parallelized_model()` -> FSDP2/DDP wrap + weight load
- `build_optimizer()` -> optimization
- `build_model_assets()` -> the preprocessor this model reads inputs through, the `model_assets` sidecars an export writes beside its weights, and `chat_template` when the job named one

and past construction:
- `build_lr_scheduler(total_steps)` -> left to the trainer, since `total_steps` is only known once the dataset is built
- `clip_grad_norm()` -> gradient clipping under this model's mesh
- `load()` / `save_dcp()` / `save_hf_or_lora()` -> what this model persists, delegated to the `ModelCheckpointManager` at `checkpoint` (`veomni/models/checkpoint_manager.py`), which owns how: DCP load/save, HF and LoRA export, and the directory layout for all three

Which preprocessor a model gets follows from what the checkpoint holds, not from a declaration: `build_model_assets` always calls `build_processor`, since `AutoProcessor` falls back to `AutoTokenizer` when a repository has no processor to offer. If a real `ProcessorMixin` comes back, `processor` is set and `tokenizer` is taken from inside it (never loaded twice, so the object the data pipeline reads through is the object exported); otherwise only `tokenizer` is set. `processor is not None` is therefore the job's signal that a model sees more than text. `DiTModelRuntime` overrides this to load neither. A path with no preprocessor to load warns rather than raises — a toy config exercising the loop on synthetic batches has none and never asks for one.

The same method assembles `model_assets`, the sidecars an export writes beside the weights: the config always, plus whatever preprocessor loaded. Caching the list is safe because nothing replaces or rewrites those objects afterwards. This mirrors SeedOmni V2's `ModuleRuntime._load_module_assets`. It also builds `chat_template` when the job named one, because the template is the third thing a model needs before it can read text: the tokenizer says how a string becomes ids, the processor how pixels do, and the template how a *conversation* becomes a training sample — including the assistant-only label mask no jinja can express. A trainer therefore never assembles one; it reads `model.chat_template` the way it reads `model.tokenizer`, and always forwards it to `build_data_transform` (transforms take `**kwargs`, so one that has no use for it ignores it). The job picks *which* one via `data.chat_template` and hands the name to the runtime at construction, since only the runtime holds the preprocessor to build it from. The field defaults to `None`, meaning a config asks for a template by naming one and otherwise gets none — how a plaintext job says its data has no conversation to lay out, and how a Qwen-Omni config says it formats prompts through its processor's own template. Both used to be branches inside a trainer (`data_type == "plaintext"` in `TextTrainer`, a `model_type` hardcode in `VLMTrainer`). A model that loaded no preprocessor (a DiT over latents) warns and leaves it unset rather than failing a build with no use for one.

The template is deliberately absent from `model_assets`, and is never written onto the tokenizer either: it is a choice about the *data*, not a property of the checkpoint, so an export keeps whatever jinja the checkpoint shipped with rather than substituting this job's formatting for what the model's authors published.

What the repository ships can be overridden per run through `model.processor_config`, forwarded as kwargs to `build_processor` the way `model_config` overrides the architecture. It is a job-level argument rather than a runtime hook because the value belongs to the run, not the model class — a hardcoded `max_pixels` on `VLMModelRuntime` used to serve this purpose and silently capped every VLM and omni job at one resolution, defeating the per-config `data.mm_configs.image_max_pixels`. Pixel budgets belong in `data.mm_configs`, which resizes before the processor sees the image; two caps in two places means the smaller wins and the config's stated budget is silently ignored.

So `self.model = self.build_model_runtime()` *is* the model build — a trainer never resequences those steps from outside, and needs no `use_parallel_state` scope around anything that follows. A model whose build differs subclasses the runtime and overrides the step that differs: `VLMModelRuntime` (encoder-aware build, tower freezing, separate ViT lr), `DiTModelRuntime` (frozen condition model; nothing to parallelize on an embedding-only run), and `DPOReferenceModelRuntime` (frozen eval replica under `"reference"`; init only builds the module, never optimizer / checkpoint). `TextDPOTrainer` owns both handles as `policy_model` (ParallelState `"policy"`) and `reference_model`. Callbacks bind to the DPO trainer; `.model` is the policy. DPO hands the reference its own `reference_model` args, or reuses `model` when that field is omitted. Each is returned from `build_policy_model_runtime()` / `build_reference_model_runtime()`.

It is usable on its own, with no trainer at all (see `tests/models/test_model_runtime.py`). Construction takes this model's *own* arguments (`ModelRuntimeArguments`), the `ParallelState` name to register under, and the job-wide `TrainingArguments` it still needs for checkpoint paths and the resume decision. Nothing has to find itself inside a larger config: a job composing several models hands each one its own slice, so a single-model trainer and a multi-module omni model share one build sequence.

Checkpointing is split three ways, mirroring SeedOmni V2's `OmniModuleDcpCallback` -> `OmniTrainer.save_dcp` -> `OmniModelRuntime`:

- **When** — `ModelDcpCallback` / `ModelHfCallback` (`veomni/trainer/callbacks/checkpoint_callback.py`). They own the every-N-steps/epochs cadence and call nothing but the trainer.
- **What** — `BaseTrainer.load()` / `save_dcp()` / `save_hf_or_lora()`, one line each, fanning out to `self.model.<same name>()`. A trainer holding a second model (a DPO reference, a distillation teacher) extends the fan-out here without the callbacks learning about it.
- **How** — `VeOmniModelRuntime` forwards to its `ModelCheckpointManager`, which owns the *ordering* (drain async saves, `empty_cache` around the DCP write, barrier, then export) — the part previously duplicated between the V1 callbacks and V2's per-module manager. A multi-module model subclasses it and sets `checkpoint_subfolder` to nest every artifact under the module name.

Only model-bound state (the lr scheduler) travels in a model checkpoint. Job-level state — the dataloader cursor, the rng, the meters — belongs to `GlobalStateCallback` / `RootAssetsCallback` (`veomni/trainer/callbacks/global_state_callback.py`) and gets its own file, because with several models in one job there is one such record but N model checkpoints.

That file is written **per rank**, `trainer_state_rank_{N}.pt`, where V2 writes a single rank-0 `trainer_state.pt`. The cursor in it is rank-local by construction: iterable datasets are `split_dataset_by_node`-sharded on `dp_rank` (`veomni/data/dataset.py:1509`), the multisource sampler filters on `_global_sample_idx % dp_size == dp_rank` (`:596`), and Energon takes `dp_rank` in its `WorkerConfig` (`:1645`). Restoring one rank's cursor everywhere makes every rank resume on rank 0's shard — replaying that slice and skipping the rest. Only the map-style path is rank-agnostic, which is why the single-file version looks correct until an iterable dataset resumes.

`BaseTrainer` adds the job-bound half:
- `_build_dataloader()` -> data pipeline setup
- `train_step()` -> single training step (forward + backward + update)
- `training_loop()` -> main loop with callbacks

Subclasses override specific methods (e.g., `compute_loss()`, custom data transforms) rather than the entire training loop. Note that `TextTrainer`, `VLMTrainer`, `DiTTrainer` and `TextDPOTrainer` *compose* a `BaseTrainer` in `self.base` rather than subclassing it, and drive the build steps one at a time (constraint 24).

**Parallel-state scoping**: `BaseTrainer.setup_distributed(args)` registers `"base"` before seed/determinism — it is a staticmethod because everything it does is job-level and runs before any model exists (a model then derives its own mesh in `VeOmniModelRuntime.setup()`); its `__init__` then scopes the model build to that mesh, so the trainer's remaining build steps need no scope of their own. Run time uses **per-op** wraps with `"base"` (forward / postforward / backward / clip). The inherited `parallel_state` property is a by-name registry lookup, never a stored state object, so the registry stays the single source of truth. See `.agents/knowledge/constraints.md` §7 and `docs/design/local_parallel_state.md`.

## Data Flow

```
YAML Config -> VeOmniArguments -> Trainer
                                    │
                    ┌───────────────┼───────────────┐
                    v               v               v
         build_model_runtime()  build_dataloader()  build_lr_scheduler()
                    │               │               │
                    v               v               v
              VeOmniModelRuntime  Dataset +     Runtime.lr_scheduler
              (build/freeze/      Collator      (needs train_steps)
               parallelize/opt)
                    │               │               │
                    v               v               v
              FSDP2 wrap      Dynamic Batch     Runtime.clip_grad_norm()
                              + Data Transform
                    │               │               │
                    └───────────────┼───────────────┘
                                    v
                            training_loop()
                            (with callbacks)
```

## Model Loading Flow

1. Read `config.json` -> `AutoConfig.from_pretrained()` -> check `MODEL_CONFIG_REGISTRY`
2. If registered: use VeOmni custom config class; else: use HF config
3. Determine model class via `MODELING_REGISTRY` (keyed by `model_type`)
4. Instantiate model on meta device (`init_empty_weights()`)
5. Apply VeOmni patches (flash attention, sequence parallel hooks)
6. Load weights (`load_model_weights()` or `rank0_load_and_broadcast_weights()`)
7. Apply parallelization (`build_parallelize_model()`)

## Parallelization Flow

VeOmni uses FSDP2 exclusively.

1. `init_parallel_state_from_accelerator()` -> global `DeviceMesh` with named dims (`dp_shard`, `ulysses`, `cp`, etc.) + per-ExtraParallel submeshes (`[ep × ep_fsdp]`)
2. Model-specific `parallel_plan.py` -> define EP/embedding weight sharding via `ParallelPlan`
3. `build_parallelize_model()` -> `parallelize_model_fsdp2()`:
   - `ParallelPlan.apply()` wraps EP/embedding params as DTensors on para mesh
   - `fully_shard()` on EP modules with `ep_fsdp` submesh (Shard(1) for hidden dim)
   - `fully_shard()` on each transformer block with `fsdp_mesh`
   - `fully_shard()` on root model
4. SP is orthogonal to FSDP2 — models call Ulysses all-to-all (`gather_seq_scatter_heads` / `gather_heads_scatter_seq`) around attention; the FSDP shard mesh fuses with SP mesh (`dp_shard_sp`)
5. EP token routing is in model MoE code + `moe_layer.py` using `ep_group` from `ParallelState`

## Config Structure

```
configs/
├── text/                   Text model training configs
│   └── <model>.yaml        (model_path, data, optimizer, parallelism, checkpoint)
├── multimodal/             Multimodal training configs
│   └── <model>/
│       └── <model>.yaml
├── dit/                    Diffusion model configs
│   └── <model>.yaml
└── model_configs/          Base model architecture configs
    └── <family>/
        └── <Model>.json    (HuggingFace-compatible config.json)
```

## Testing

```
tests/
├── models/         Model loading, patching, registry tests
├── data/           Data pipeline, collator, transform tests
├── ops/            Kernel operation tests
├── parallel/       Distributed parallelism tests (ulysses, data balance)
├── checkpoints/    Checkpoint save/load tests
├── utils/          Utility function tests
├── e2e/            End-to-end training tests (require GPU)
├── toy_config/     Minimal model configs for fast testing
└── tools/          Test utilities (launch_utils, common_utils)
```

### Test Commands by Change Area

| Change in | Test command |
|-----------|-------------|
| `veomni/models/` | `pytest tests/models/` |
| `veomni/data/` | `pytest tests/data/` |
| `veomni/ops/` | `pytest tests/ops/` |
| `veomni/distributed/` | `pytest tests/parallel/` |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` |
| `veomni/utils/` | `pytest tests/utils/` |
| `veomni/trainer/` | `pytest tests/e2e/` |
| Full regression | `pytest tests/` |

Distributed tests (`tests/parallel/`, `tests/e2e/`) may require multiple GPUs and use `torchrun` or `tests/tools/launch_utils.py`.

## Key Entry Points

| Task | Script | Trainer |
|------|--------|---------|
| Text SFT | `tasks/train_text.py` | `TextTrainer` |
| Text DPO | `tasks/train_text_dpo.py` | `TextDPOTrainer` |
| Text RL | `tasks/train_text_rl.py` | `BaseRLTrainer` |
| VLM SFT | `tasks/train_vlm.py` | `VLMTrainer` |
| VLM RL | `tasks/train_vlm_rl.py` | `BaseRLTrainer` |
| DiT | `tasks/train_dit.py` | `DitTrainer` |
| Inference (text) | `tasks/infer/infer_text.py` | N/A |
| Inference (VLM) | `tasks/infer/infer_qwen2_vl.py` | N/A |
