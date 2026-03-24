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
│   ├── fsdp/           FSDP (v1) wrapping, gradient clipping, extensions
│   ├── fsdp2/          FSDP2 (composable) wrapping, gradient clipping
│   ├── moe/            MoE expert parallelism: routing, communication, layer wrapping
│   └── sequence_parallel/  Ulysses sequence parallelism: split/gather, async variants
├── models/             Model loading and patching
│   ├── auto.py         High-level API: build_foundation_model, build_tokenizer, build_processor
│   ├── loader.py       Registry-based model loading (MODELING_REGISTRY, MODEL_CONFIG_REGISTRY)
│   ├── transformers/   Per-model patches (one subpackage per model family)
│   ├── diffusers/      Diffusion model definitions (Wan T2V)
│   └── seed_omni/      Omni-model architecture (encoder-foundation-decoder)
├── ops/                Optimized kernels
│   ├── flash_attn/     Flash attention integration
│   ├── fused_cross_entropy/   Fused loss computation
│   ├── fused_moe/      Fused MoE kernels
│   ├── group_gemm/     Group GEMM kernels (triton)
│   └── npu_patch/      NPU-specific operator patches
├── optim/              Optimizer and LR scheduler construction
├── patchgen/           Auto-generate model patches from HuggingFace models
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

Subclasses override specific methods (e.g., `compute_loss()`, custom data transforms) rather than the entire training loop.

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
              (FSDP/FSDP2)    + Data Transform  (veomni_clip_grad_norm)
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

1. `init_parallel_state()` -> set up device meshes for FSDP, SP, EP
2. Model-specific `parallel_plan.py` -> define wrapping policy and sharding spec
3. `build_parallelize_model()`:
   - FSDP (v1): `FullyShardedDataParallel` wrapping with auto-wrap policy
   - FSDP2: `fully_shard()` composable API on each transformer block
   - Sequence Parallel: `parallelize_module()` with Ulysses attention
   - Expert Parallel: shard MoE experts across EP group

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

## Testing Structure

```
tests/
├── models/         Model loading, patching, registry tests
├── data/           Data pipeline, collator, transform tests
├── ops/            Kernel operation tests
├── parallel/       Distributed parallelism tests (ulysses, data balance)
├── checkpoints/    Checkpoint save/load tests
├── utils/          Utility function tests
├── e2e/            End-to-end training tests
├── toy_config/     Minimal model configs for fast testing
└── tools/          Test utilities (launch_utils, common_utils)
```

## Key Entry Points

| Task | Script | Trainer |
|------|--------|---------|
| Text SFT | `tasks/train_text.py` | `TextTrainer` |
| Text DPO | `tasks/train_text_dpo.py` | `TextDPOTrainer` |
| Text RL | `tasks/train_text_rl.py` | `BaseRLTrainer` |
| VLM SFT | `tasks/train_vlm.py` | `VLMTrainer` |
| VLM RL | `tasks/train_vlm_rl.py` | `BaseRLTrainer` |
| DiT | `tasks/train_dit.py` | `DitTrainer` |
| Omni | `tasks/omni/train_omni_model.py` | Custom |
| Inference (text) | `tasks/infer/infer_text.py` | N/A |
| Inference (VLM) | `tasks/infer/infer_qwen2_vl.py` | N/A |
