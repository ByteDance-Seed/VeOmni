# VeOmni Architecture Overview

High-level map of where code lives and how components interact. For deep-dive subsystem designs, follow the linked docs under `docs/design/` and `.agents/knowledge/`.

## Module Map

```text
veomni/
├── arguments/          CLI/YAML argument parsing (VeOmniArguments)
├── checkpoint/         DCP save/load, planners (planner.py, row_crop.py), consolidation
├── data/               Datasets, MainCollator, transforms, dynamic batching
│   ├── multimodal/     Vision/audio/video preprocessing and chat templates
│   ├── diffusion/      Diffusion model data loading
│   └── seed_omni/      SeedOmni preprocessing, transform, and collator
├── distributed/        Parallelism strategies (FSDP2 only)
│   ├── parallel_state.py    init_parallel_state_from_config(), ParallelState, mesh setup
│   ├── torch_parallelize.py build_parallelize_model(), parallelize_model_{fsdp2,ddp}()
│   ├── parallel_plan.py     ParallelPlan for ExtraParallel (EP, embedding, PLE)
│   ├── async_offload.py     Async activation offload (OffloadManager, async_save_on_cpu)
│   ├── emb_parallel/        ShardedEmbedding over emb group
│   ├── fsdp2/               FSDP2 wrapping, clip_grad_norm, fp8_all_gather
│   ├── moe/                 EP token routing, all-to-all, EPGroupGemm
│   ├── context_parallel/    Sparse-attention CP window sharding and collectives
│   ├── sequence_parallel/   Ulysses SP all-to-all exchange and async variants
│   └── hccl_premul_sum.py   Idempotent HCCL PREMUL_SUM compatibility patch
├── models/             Model loading, standalone HF modeling, and runtime wrapper
│   ├── auto.py              build_foundation_model, build_tokenizer, build_processor
│   ├── registry.py          Import-time model/config/processor registries
│   ├── model_runtime.py     VeOmniModelRuntime (model-bound half of a job)
│   ├── checkpoint/          Weight I/O, tensor converters, ModelCheckpointManager
│   ├── transformers/        Standalone HF-derived modeling_<m>.py with VeomniOp handles
│   ├── diffusers/           Diffusion model families
│   ├── seed_omni/           SeedOmni composed model, graphs, mixins, module runtimes
│   ├── utils/               Modeling-side attention helpers and MoE fallbacks
│   └── loss_utils/          Model-facing CE, load-balancing, and chunked-loss policy
├── optim/              build_optimizer() / MultiOptimizer, DistributedMuon, SwapAdamW, LR schedulers
├── ops/                Tensor-native op registry (OP_REGISTRY, VeomniOp)
│   ├── registry.py          OpEntry, register_op, resolve_op, VeomniOp
│   ├── platform/            GPU/NPU/MLU KernelRequirement declarations
│   ├── qat/                 Fake-quantization primitives
│   ├── batch_invariant/     Scoped deterministic ATen patch
│   ├── install.py           Idempotent process-wide attention integration
│   └── kernels/             Per-op/variant eager and optimized rows (incl. attention/standard/flex/)
├── lora/               Linear + MoE-expert LoRA injection and DCP/HF export
├── schedulers/         Flow-matching schedulers
├── trainer/            BaseTrainer, Text/VLM/DiT/DPO/RL trainers, omni/, callbacks/
└── utils/              Logging, device helpers, constants, flops_meter.py, EnvironMeter
```

## Trainer & ModelRuntime Hierarchy

A job splits cleanly into **model-bound** (`VeOmniModelRuntime`) and **job-bound** (`BaseTrainer`):

```text
BaseTrainer (ABC)                         job: dist init, dataloader, loop, callbacks
└── .model: VeOmniModelRuntime            one model: build, freeze/LoRA, parallelize, opt, lr_sched, clip, ParallelState
    ├── .model: nn.Module                 underlying PyTorch module (unshadowed attrs forwarded via __getattr__)
    └── .checkpoint: ModelCheckpointManager  DCP / HF / LoRA save-load + directory layout

Single-model trainers (compose BaseTrainer in self.base):
  TextTrainer / VLMTrainer / DiTTrainer / TextDPOTrainer (policy + frozen reference)
RL trainers (subclass BaseTrainer):
  BaseRLTrainer (ABC) -> tasks/train_{text,vlm}_rl.py
Composed multi-module trainer (independent orchestrator):
  OmniTrainer -> OmniModelRuntime -> ModuleRuntime × N  (see .agents/knowledge/veomni-seed-omni.md)
```

### Runtime Lifecycle & Ownership

- **`VeOmniModelRuntime`** (`veomni/models/model_runtime.py`):
  1. `setup()` registers this model's `ParallelState` by name.
  2. `__init__` runs inside `use_parallel_state(self.model_name)`: `_build_model()` (meta-init) → `_freeze_model_module()` / `_setup_lora()` → `_build_parallelized_model()` (FSDP2/DDP + weight load) → `_build_optimizer()`.
  3. Outside that scope: `_build_model_assets()` (loads `processor` via `build_processor` falling back to `tokenizer`, caches `model_assets`, builds optional `chat_template` without writing it onto the tokenizer) and `build_checkpoint()` (`ModelCheckpointManager`).
  4. Post-init: `_build_lr_scheduler(total_steps)` (after dataloader), `clip_grad_norm()`, `load()` / `save_dcp()` / `save_hf_or_lora()`.
  - Model-specific build differences subclass `VeOmniModelRuntime` (`VLMModelRuntime`, `DiTModelRuntime`, `DPOReferenceModelRuntime`) rather than resequencing steps in the trainer.
- **Checkpointing (3-way split)**:
  - **When**: `CheckpointCallback` (cadence for DCP, HF/LoRA, sidecars) + `GlobalStateCallback` (rank-local `loader/rank_{N}.pt`, `extra_state/rank_{N}.pt`, `checkpoint_manifest.json`).
  - **What**: `BaseTrainer.{load,save_dcp,save_hf_or_lora,save_model_assets}` fans out to `self.model.*`.
  - **How**: `ModelCheckpointManager` (`veomni/models/checkpoint/manager.py`) owns async drain, `empty_cache`, barrier, I/O, and layout (`docs/usage/checkpoint.md`).
- **DiT fixed microbatches**: `DiTTrainer` uses `train.micro_batch_size` with `dyn_bsz=false` and dict-of-lists microbatches from `DiTDataCollator` (`docs/usage/dit_microbatching.md`).
- **FLOPs & profiling**: `veomni/utils/flops_meter.py` registers per-model FLOPs callbacks for `EnvironMeter` (`docs/design/flops_metadata_callback.md`); profiling uses `ProfileConfig` (`veomni/arguments/arguments_types.py`), `create_profiler()` (`veomni/utils/helper.py`), `ProfileTraceCallback` (`veomni/trainer/callbacks/trace_callback.py`), and `scripts/profile/merge_chrome_trace.py`.

## Core Flows

### Data & Training Flow

```text
YAML Config -> VeOmniArguments -> Trainer
  ├── _build_model_runtime()  -> VeOmniModelRuntime (meta -> freeze/LoRA -> FSDP2/DDP -> opt)
  ├── _build_dataloader()     -> Dataset + DataTransform + DynamicBatching + MainCollator
  └── _build_lr_scheduler()   -> runtime.lr_scheduler
        └── training_loop()   -> per-op use_parallel_state(model.parallel_state) + callbacks
```

### Model Loading & Config Ownership

1. `build_foundation_model()` binds `ops_implementation` onto the config and resolves a `VeOmniPretrainedConfig` tree (`docs/design/module_config.md`) without monkey-patching native HF configs.
2. `MODELING_REGISTRY` resolves the class by `model_type` (unless `MODELING_BACKEND=hf`).
3. Instantiates on `meta` (`init_empty_weights()`), validates final `ops_config`, and binds instance-local `VeomniOp` handles (`docs/design/op_selection.md`).
4. Loads weights (`load_model_weights()` / `rank0_load_and_broadcast_weights()`) and parallelizes (`build_parallelize_model()`).
- Public CLI/YAML key is `model.ops_implementation` (sole stored attention selection; HF attention accessors delegate to it).

### Parallelization Flow (FSDP2)

1. `init_parallel_state_from_config()` builds the global `DeviceMesh` and ExtraParallel submeshes (`[ep × ep_fsdp]`, etc.).
2. `ParallelPlan.apply()` wraps ExtraParallel params (EP experts, sharded embeddings, Qwen4-Exp 2D PLE) as DTensors.
3. `parallelize_model_fsdp2()` calls `fully_shard()` on EP modules (`ep_fsdp`), transformer blocks (`fsdp_mesh`), and root.
4. Ulysses SP exchanges seq/heads via all-to-all around attention (`dp_shard_sp` fuses FSDP and SP); EP routes tokens via `veomni/distributed/moe/moe_layer.py`.
- DeepGEMM FP8 MoE & FP8 expert all-gather: `docs/design/deepgemm_moe.md`. Qwen4-Exp 2D PLE: `docs/design/qwen4_exp_ple_2d_parallelism.md`.
- Optional Ascend HyperParallel backend: `fsdp_mode: hyper_fsdp2` and `optimizer.type: hyper_muon` (Text & VLM trainers; `docs/key_features/hyper_parallel.md`).

## Configs, Tests & Entry Points

- **Configs**: `configs/{text,multimodal,dit,seed_omni,model_configs}/`.
- **Entry points**: `tasks/train_{text,text_dpo,text_rl,vlm,vlm_rl,dit}.py`, `tasks/omni/{train,infer}_omni.py`, `tasks/infer/infer_{text,qwen2_vl}.py`.
- **Tests by area**:

| Change in | Command |
|---|---|
| `veomni/models/` | `pytest tests/models/` |
| `veomni/ops/` | `pytest tests/ops/` |
| `veomni/data/` | `pytest tests/data/` |
| `veomni/distributed/` | `pytest tests/parallel/ tests/distributed/` |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` |
| `veomni/trainer/` | `pytest tests/trainer/` (then `tests/e2e/` for loop changes) |
| `veomni/lora/` | `pytest tests/lora/` |
| `veomni/optim/` | `pytest tests/optim/` |
| `veomni/utils/` | `pytest tests/utils/` |
| `veomni/models/seed_omni/` | `pytest tests/seed_omni/` |

**CI selection**: A passing local test does not mean CI runs it. `scripts/ci/accelerator/suites/{gpu,npu}.json` enumerate selections (wholesale dirs: `tests/{ci,arguments,data,checkpoints,ops,models,seed_omni}/`, plus `tests/parallel/context_parallel/` on GPU; everything else requires an explicit entry). Read `.agents/knowledge/testing.md` before adding a test.
