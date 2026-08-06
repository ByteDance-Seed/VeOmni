# VeOmni Architecture Overview

This document describes VeOmni's architecture for AI coding agents. Read this to understand where code lives and how components interact.

## Module Map

```
veomni/
├── arguments/          CLI argument parsing
│   ├── arguments_types.py      V1 VeOmniArguments
│   └── omni_arguments/          V2 OmniArguments (launcher + per-module runtime)
│       ├── arguments_types.py
│       └── parser.py
├── checkpoint/         DCP-based distributed checkpoint save/load
├── data/               Data pipeline: datasets, collators, transforms, dynamic batching
│   ├── multimodal/     Vision, audio, video preprocessing and chat templates
│   └── diffusion/      Diffusion model data loading
├── distributed/        All parallelism strategies
│   ├── parallel_state.py   init_parallel_state(), ParallelState, device mesh setup
│   ├── torch_parallelize.py  build_parallelize_model(), parallelize_model_fsdp2()
│   ├── parallel_plan.py    ParallelPlan for ExtraParallel (EP, embedding shard)
│   ├── fsdp2/          FSDP2 (composable fully_shard), gradient clipping
│   ├── moe/            MoE expert parallelism: token routing, all-to-all, EPGroupGemm
│   └── sequence_parallel/  Ulysses SP: all-to-all head/seq exchange, async variants
├── models/             Model loading and patching
│   ├── auto.py         High-level API: build_foundation_model, build_tokenizer, build_processor
│   ├── loader.py       Registry-based model loading (MODELING_REGISTRY, MODEL_CONFIG_REGISTRY)
│   ├── transformers/   Per-model patches (one subpackage per model family)
│   ├── diffusers/      Diffusion model definitions (Wan T2V)
│   └── seed_omni/      Omni-model architecture (encoder-foundation-decoder)
│       ├── configuration_omni.py   OmniConfig (pure HF PretrainedConfig, checkpoint I/O,
│       │                           all generation scenarios keyed by infer_type)
│       ├── modeling_omni.py        OmniModel (eager graph forward/generate)
│       ├── utils/                  Shared SeedOmni helpers
│       │   ├── checkpoint.py       OmniModuleCheckpointManager (per-module DCP/HF/LoRA)
│       │   ├── graph_profiler.py   GraphProfiler (request-local node timing)
│       │   └── visualize.py        Mermaid export for training/generation graphs
│       └── accelerator/          VeOmni runtime (graph loops + per-module FSDP)
│           ├── omni_model_runtime.py  OmniModelRuntime (composed graph loops)
│           ├── module_runtime.py      ModuleRuntime (per sub-module FSDP/opt/ckpt)
│           ├── executor.py            execute_train_node / execute_generation_node
│           └── dispatch.py            unwrap FSDP/DDP/LoRA wrappers, call_graph_endpoint
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
│   │   ├── attention/  Flash attention v2/3/4 + SP-aware variants
│   │   ├── cross_entropy/  eager/liger/npu-chunk loss variants
│   │   ├── load_balancing_loss/  eager + triton variants
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
│   ├── base.py         BaseTrainer (ABC): the composable training skeleton
│   ├── text_trainer.py TextTrainer: LLM SFT training
│   ├── vlm_trainer.py  VLMTrainer: vision-language model training
│   ├── dit_trainer.py  DitTrainer: diffusion transformer training
│   ├── text_dpo_trainer.py  DPO training for text models
│   ├── base_rl_trainer.py   Base RL trainer for RLHF
│   ├── omni/           SeedOmni V2 orchestrators (OmniTrainer, OmniInferencer)
│   └── callbacks/      Training callbacks (checkpoint, evaluate, trace, etc.)
└── utils/              Shared utilities (logging, device, constants, helpers)
```

## Trainer Hierarchy

### V1 (`veomni.trainer`) — legacy single-model trainers

```
├── TextTrainer          -> tasks/train_text.py
├── VLMTrainer           -> tasks/train_vlm.py
├── DitTrainer           -> tasks/train_dit.py
├── TextDPOTrainer       -> tasks/train_text_dpo.py
└── BaseRLTrainer (ABC)
    ├── (text RL)        -> tasks/train_text_rl.py
    └── (VLM RL)         -> tasks/train_vlm_rl.py
```

`BaseTrainer` provides the composable training skeleton:
- `build_model()` -> model construction and parallelization
- `build_dataloader()` -> data pipeline setup
- `build_optimizer()` / `build_lr_scheduler()` -> optimization
- `train_step()` -> single training step (forward + backward + update)
- `training_loop()` -> main loop with callbacks

Subclasses override specific methods (e.g., `compute_loss()`, custom data transforms) rather than the entire training loop.

**Parallel-state scoping**: standalone trainers register the `"base"` `ParallelState` in `_setup()` and use `with use_parallel_state("base"):` for the complete build and each ambient-dependent runtime operation (forward, post-forward loss, backward, and clipping). See `.agents/knowledge/constraints.md` §7d.

### V2 (`veomni.trainer.omni`) — SeedOmni graph trainers (no BaseTrainer)

There are exactly **two** ways to build a SeedOmni model, and the trainer / inferencer holds exactly **one** model handle for either — `self.model`:

| Build path | `self.model` | Sub-modules | Used by |
|------------|--------------|-------------|---------|
| **Bare HF** — `OmniModel.from_pretrained(root)` / `from_config(cfg)` | `OmniModel` (a `PreTrainedModel`) | plain `PreTrainedModel` | non-VeOmni users; all-`eager` inference |
| **VeOmni** — `OmniModelRuntime.from_runtime_config(cfg)` | `OmniModelRuntime` | one `ModuleRuntime` each | training; distributed inference |

`OmniModelRuntime` composes an `OmniModel` and forwards everything it does not define (`config`, `modules_dict`, `save_pretrained`, `nn.Module` reads) to it, so shared callbacks read `trainer.model` either way. Training **requires** the runtime — only it runs the graph with ParallelState scoping, graph tracing and metric metering.

```
OmniTrainer (orchestrator)           -> tasks/omni/train_omni.py
└── self.model = OmniModelRuntime    graph loops (forward), trace, metering
    ├── ModuleRuntime × N            per sub-module (FSDP, opt, ckpt, ParallelState)
    └── OmniModel                    eager graph definition (modeling_omni.py)

OmniInferencer                       -> tasks/omni/infer_omni.py
└── self.model = OmniModelRuntime (any FSDP2/DDP module) | OmniModel (all eager)
```

V2 reuses lower-level libraries (`distributed/`, `optim/`, `models/`, `data/`, `checkpoint/`) but does **not** inherit `BaseTrainer`. Config split:

| Layer | Config source | Owns |
|-------|---------------|------|
| **OmniModel** | `OmniConfig`, projected from `OmniModelRuntimeArguments.to_hf_config()` | graph topology, module wiring |
| **ModuleRuntime** | slim `OmniModuleRuntimeArguments` (`model` + `accelerator` + `optimizer`) + launcher `train` | FSDP, optimizer, checkpoint per module |
| **OmniModelRuntime** | `OmniModelRuntimeArguments` via `from_model_runtime()` | graph loops, module runtimes, graph trace, metering |
| **OmniTrainer** | launcher YAML + `OmniArguments` | dist init, dataloader, train loop, multi-opt |

Canonical imports:

```python
from veomni.trainer.omni import OmniTrainer, OmniInferencer
from veomni.models.seed_omni.accelerator import OmniModelRuntime
from veomni.models.seed_omni.accelerator.module_runtime import ModuleRuntime
from veomni.omni_arguments.arguments_types import (
    OmniModelRuntimeArguments,
    build_module_runtime_args,
    build_omni_model_runtime,
    resolve_omni_model,
)
from veomni.arguments import OmniArguments
```

**Runtime config vs `OmniConfig`.** `resolve_omni_model()` returns an
`OmniModelRuntimeArguments` — the launcher's whole picture: split-checkpoint root, merged
per-module blocks with absolute paths and `accelerator`/`train` settings, and every graph.
Nothing is discarded. Projecting onto the slim, checkpoint-shaped `OmniConfig` is a
separate explicit step, `.to_hf_config()`, taken only where an HF artefact is needed:
`OmniModelRuntime.from_model_runtime()` and `OmniInferencer` (because `OmniModel` is a
`PreTrainedModel` that needs one for `save_pretrained`), and the checkpoint export script.
Graph-only consumers such as `scripts/visualize_omni_graph.py` use the runtime config
directly and never convert.

`OmniConfig.from_pretrained()` hydrates each module entry into a `PretrainedConfig` by
reading the module subfolder's `config.json`.

`OmniConfig` itself (`configuration_omni.py`) is a plain `PretrainedConfig` and imports
nothing from `veomni.arguments`: it only reads/writes a checkpoint root. Every path from
launcher YAML into an `OmniConfig` goes through `veomni.omni_arguments.arguments_types`
(`resolve_omni_model` / `build_omni_model_runtime`), which also owns the launcher-YAML
helpers shared with `build_module_runtime_args()`. `OmniModelRuntimeArguments` and these
resolution helpers live in the same file as `OmniArguments` (no separate `model_runtime.py`)
so `OmniArguments.model` can be typed directly as `OmniModelRuntimeArguments` without an
import cycle.

**Generation scenarios**: `OmniConfig` holds *every* FSM from `infer.infer_graph` in
`generation_graphs` (`{infer_type: fsm}`), with `infer_type` naming the active one and
the `generation_graph` property returning it. So a checkpoint exported for `infer_gen`
can still run `infer_und` — set `config.infer_type` and rebuild the model. `OmniModel`
binds one FSM at `__init__`, so switching at runtime on an already-built model is not
supported.

**Parallel-state scoping (V2)**: each `ModuleRuntime._setup()` registers its `ParallelState` under its **module name** (registry key = checkpoint subdir). `OmniModelRuntime.module_context` re-enters via `use_parallel_state(module_name)` for every module that registered one (eager inference modules never do, so their nodes run unscoped). The orchestrator never wraps a module's private mesh — it calls `module_runtime._clip_grad_norm()` and reads `module_runtime.parallel_state` for validation.

## Data Flow

### V1 (BaseTrainer)

```
YAML Config -> VeOmniArguments -> Trainer
                                    │
                    ┌───────────────┼───────────────┐
                    v               v               v
              build_model()   build_dataloader()  build_optimizer()
                    │               │               │
                    v               v               v
              HF Model +      Dataset +         Optimizer +
              VeOmni Patch     Collator          LR Scheduler
                    │               │               │
                    v               v               v
              Parallelize     Dynamic Batch     Grad Clip
              (FSDP2)         + Data Transform  (fsdp2/clip_grad_norm)
                    │               │               │
                    └───────────────┼───────────────┘
                                    v
                            training_loop()
                            (with callbacks)
```

### V2 (SeedOmni)

```
YAML Config -> OmniArguments
                    │
        ┌───────────┼───────────────────────────┐
        v           v                           v
build_omni_model_runtime() / resolve_omni_model()  build_module_runtime_args()  OmniTrainer.setup_distributed()
        │                    (per module)                      │
        └───────────┬────────────────┘                  ParallelState registry
                    v                                          │
    OmniModelRuntime.from_model_runtime()  <── build_omni_model()
     ├── ModuleRuntime × N  (FSDP2 wrap, weight load, optimizer, ckpt)
     └── OmniModel          (composed graph definition;
                             config = runtime_config.to_hf_config())
                    │                                 │
                    v                                 v
    trainer.model (single handle)            build_dataloader()
                    │                                 │
                    └───────────────┬─────────────────┘
                                    v
                          OmniTrainer.train()
                    (graph forward/backward, multi-opt, grad clip, callbacks)
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

1. `init_parallel_state()` -> global `DeviceMesh` with named dims (`dp_shard`, `ulysses`, `cp`, etc.) + per-ExtraParallel submeshes (`[ep × ep_fsdp]`)
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
├── seed_omni/      SeedOmni graph, config, runtime, inferencer tests
├── toy_config/     Minimal model configs for fast testing
└── tools/          Test utilities (launch_utils, common_utils)
```

### Test Commands by Change Area

| Change in | Test command |
|-----------|-------------|
| `veomni/models/` | `pytest tests/models/` |
| `veomni/models/seed_omni/` | `pytest tests/seed_omni/` |
| `veomni/trainer/omni/` | `pytest tests/seed_omni/ tests/trainer/test_omni_trainer_context.py` |
| `veomni/arguments/` (Omni) | `pytest tests/seed_omni/test_omni_module_args.py tests/seed_omni/test_omni_offline_cache_args.py` |
| `veomni/data/` | `pytest tests/data/` |
| `veomni/ops/` | `pytest tests/ops/` |
| `veomni/distributed/` | `pytest tests/parallel/` |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` |
| `veomni/utils/` | `pytest tests/utils/` |
| `veomni/trainer/` (V1) | `pytest tests/e2e/` |
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
| Omni (V2) | `tasks/omni/train_omni.py` | `OmniTrainer` |
| Omni infer (V2) | `tasks/omni/infer_omni.py` | `OmniInferencer` |
| Inference (text) | `tasks/infer/infer_text.py` | N/A |
| Inference (VLM) | `tasks/infer/infer_qwen2_vl.py` | N/A |
