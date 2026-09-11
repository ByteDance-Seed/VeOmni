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
│   ├── parallel_state.py   init_parallel_state_from_config(), ParallelState, mesh setup
│   ├── torch_parallelize.py  build_parallelize_model(), parallelize_model_fsdp2()
│   ├── parallel_plan.py    ParallelPlan for ExtraParallel (EP, embedding shard)
│   ├── async_offload.py    Async activation offload (SwapTensor, OffloadManager, async_save_on_cpu)
│   ├── hccl_premul_sum.py  Idempotent HCCL PREMUL_SUM collective compatibility patch
│   ├── fsdp2/          FSDP2 (composable fully_shard), gradient clipping
│   ├── moe/            MoE expert parallelism: token routing, all-to-all, EPGroupGemm
│   └── sequence_parallel/  Ulysses SP: all-to-all head/seq exchange, async variants
├── models_kernel/      Model loading, patchgen configs, and kernel-aware modeling
│   ├── auto.py         High-level API: build_foundation_model, build_tokenizer, build_processor
│   ├── registry.py     Import-time model/config/processor registries
│   ├── checkpoint/     Weight I/O, tensor conversion, and ModelCheckpointManager
│   ├── transformers/   Model classes/configs with instance-local VeomniOp handles
│   ├── diffusers/      Diffusion model families
│   └── loss_utils/     Model-facing CE, load-balancing, and chunked-loss policy
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
├── ops/                Tensor-native op registry and implementations
│   ├── registry.py     OP_REGISTRY, OpEntry, register_op, resolve_op, VeomniOp
│   ├── platform/       GPU/NPU/MLU availability requirements
│   ├── install.py      Idempotent process-wide attention integration
│   ├── batch_invariant/  Scoped deterministic ATen patch
│   ├── qat/            Functional quantization-aware-training helpers
│   └── kernels/        Registered per-op/variant eager and optimized rows
├── lora/               LoRA / PEFT injection: linear + MoE-expert adapters,
│                       DCP + HF-adapter save/load, target mapping
├── patchgen/           Patch specs + codegen driver consumed by the `patchgen`
│                       CLI, which is a separate package under patchgen-pkg/
│                       (installed via the `patchgen` dependency group)
├── schedulers/         LR scheduler implementations (flow matching)
├── trainer/            Training loop implementations
│   ├── base.py         BaseTrainer (ABC): the composable training skeleton
│   ├── text_trainer.py TextTrainer: LLM SFT training
│   ├── vlm_trainer.py  VLMTrainer: vision-language model training
│   ├── dit_trainer.py  DitTrainer: diffusion transformer training
│   ├── text_dpo_trainer.py  DPO training for text models
│   ├── base_rl_trainer.py   Base RL trainer for RLHF
│   └── callbacks/      Training callbacks (checkpoint, evaluate, trace, etc.)
└── utils/              Shared utilities (logging, device, constants, helpers)
```

## Trainer Hierarchy

```
BaseTrainer (ABC)
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

**Checkpointing**: `CheckpointCallback` owns cadence for DCP, HF/LoRA, and the one-shot tokenizer/config sidecars; `GlobalStateCallback` owns the job cursor; `BaseTrainer.load` / `save_dcp` / `save_hf_or_lora` / `save_model_assets` fan out; `ModelCheckpointManager` (`veomni/models_kernel/checkpoint/manager.py`) owns DCP / HF / LoRA I/O, drain-async, `empty_cache`, barrier, and directory layout. Job cursor (dataloader, rng, meters) is not in DCP extra_state.

Subclasses override specific methods (e.g., `compute_loss()`, custom data transforms) rather than the entire training loop.

**Parallel-state scoping**: `_setup()` calls `init_parallel_state_from_config(args.model.accelerator, name="base")` before seed/determinism; then each trainer builds under `use_parallel_state("base")`. Run time uses **per-op** wraps with `"base"` (forward / postforward / backward / clip). No `self.parallel_state` on trainers. See `.agents/knowledge/constraints.md` §7 and `docs/design/local_parallel_state.md`.

## Data Flow

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

## Model Loading Flow

1. `models_kernel.build_foundation_model()` installs the supplied ops selection.
2. Read `config.json` -> `AutoConfig.from_pretrained()` -> check `MODEL_CONFIG_REGISTRY`.
3. Determine the model class via `MODELING_REGISTRY` (keyed by `model_type`); an unregistered model fails explicitly unless `MODELING_BACKEND=hf` selects the upstream class.
4. Instantiate model on meta device (`init_empty_weights()`)
5. Construct instance-local `VeomniOp` handles from the installed selection.
6. Load weights (`load_model_weights()` or `rank0_load_and_broadcast_weights()`)
7. Apply parallelization (`build_parallelize_model()`)

The public configuration field remains `model.ops_implementation`. Trainer and
inference entry points pass it to the model builder as
`ops_implementation`; changing the config field would break existing CLI
and YAML inputs.

## Parallelization Flow

VeOmni uses FSDP2 exclusively.

1. `init_parallel_state_from_config()` -> global `DeviceMesh` with named dims (`dp_shard`, `ulysses`, `cp`, etc.) + per-ExtraParallel submeshes (`[ep × ep_fsdp]`)
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
├── models_kernel/  Kernel-aware model loading, integration, and helper tests
├── kernels/        Registry contracts and per-family kernel tests
├── data/           Data pipeline, collator, transform tests
├── parallel/       Distributed parallelism tests (ulysses, data balance)
├── distributed/    dummy forward, torch.compile, FSDP equivalence, grad ckpt
├── trainer/        Callback / trainer-unit tests (channel loss, step sync, DPO)
├── lora/           LoRA + MoE-LoRA unit, kernel-parity and trainer tests
├── optim/          Muon / optimizer param-group tests
├── checkpoints/    Checkpoint save/load tests
├── utils/          Utility function tests
├── e2e/            End-to-end training tests (require GPU)
├── special_sanity/ Standalone sanity scripts (e.g. device API usage check)
├── testdata/       Fixture assets used by tests
├── toy_config/     Minimal model configs for fast testing
├── train_scripts/  Python training entry scripts launched by tests/e2e/utils.py
└── tools/          Test utilities (launch_utils, common_utils)
```

### Test Commands by Change Area

| Change in | Test command |
|-----------|-------------|
| `veomni/models_kernel/` | `pytest tests/models_kernel/` |
| `veomni/ops/` | `pytest tests/ops/` |
| `veomni/data/` | `pytest tests/data/` |
| `veomni/distributed/` | `pytest tests/parallel/ tests/distributed/` |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` |
| `veomni/utils/` | `pytest tests/utils/` |
| `veomni/trainer/` | `pytest tests/trainer/`, then `pytest tests/e2e/` for loop changes |
| `veomni/lora/` | `pytest tests/lora/` |
| `veomni/optim/` | `pytest tests/optim/` |
| Full regression | `pytest tests/` |

Distributed tests (`tests/parallel/`, `tests/distributed/`, `tests/e2e/`) may require multiple GPUs and use `torchrun` or `tests/tools/launch_utils.py`.

**A passing local run does not mean CI runs it.**
`.github/workflows/gpu_unit_tests.yml` and `npu_unit_tests.yml` enumerate most
test files one by one. Only `tests/data` runs as a whole directory in both
workflows; `tests/ops` and `tests/parallel/context_parallel` run wholesale on
GPU only (the NPU job enumerates three named ops files). The e2e paths belong
to `gpu_e2e_test.yml` / `npu_e2e_test.yml` instead. A new file anywhere else is
invisible to CI until it is added to the workflow that owns its path, usually
both unit workflows. See `.agents/knowledge/testing.md` before adding a test.

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
