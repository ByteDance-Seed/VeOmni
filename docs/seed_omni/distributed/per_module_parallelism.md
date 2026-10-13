# SeedOmni Per-Module Parallelism

A SeedOmni job composes modules that differ by orders of magnitude in size: a
few-million-parameter VAE or vision tower, a vocabulary-sized text encoder, a
multi-billion-parameter backbone. Each module therefore chooses its own data
parallel wrap and its own extra parallel groups, while the job as a whole shares
one world, one data stream and one sequence parallel (SP) size. This page covers
what a module may choose, what it must share, and how the runtime keeps the
modules apart. SP has its own page: [Sequence Parallelism](sequence_parallel.md).

## 1. What a module can choose

A module's choices live in its block of the `modules` YAML, under
`accelerator:`. Anything a module does not set falls back to the global
`model.accelerator` (see
[How a module's arguments are resolved](../usage/training_and_inference.md#how-a-modules-arguments-are-resolved)).

| Choice | YAML (inside the module's `accelerator:`) | Typical module |
|---|---|---|
| FSDP2 (default) | `fsdp_config.fsdp_mode: fsdp2` | Backbones and every module large enough that sharding pays for its communication |
| DDP | `fsdp_config.fsdp_mode: ddp`, usually with `init_device: cuda` and `broadcast_model_weights_from_rank0: false` (module level, outside `accelerator:`) | Small encoders, where a full replica per rank is cheaper than FSDP2 all-gathers |
| Eager (inference only) | `fsdp_config.fsdp_mode: eager` | A single-process replica with no mesh; rejected for training |
| Embedding parallel | `extra_parallel_names: ["emb"]`, `extra_parallel_sizes: [N]`, `extra_parallel_placement_innermost: [false]` | A vocabulary-sharded text encoder; see [Sharded Embedding](../../key_features/sharded_embedding.md) |
| Expert parallel | `ep_size: N`, `ep_outside: false` (shorthand for an `ep` extra parallel group) | A MoE backbone |
| Gradient checkpointing | `gradient_checkpointing.enable: false` | Off for cheap modules (text encoders, VAE, flow connector) |

Two per-module properties are not accelerator settings but change how the
module is wrapped and saved:

- **Frozen modules.** A module decides which of its parameters train (for
  example `model_config.freeze: true` on the Qwen3-VL vision tower in the
  visual-instruction-tuning recipe, or a LoRA config that matches nothing in
  that module). A module with no trainable parameters gets no optimizer, no
  lr scheduler and no checkpoint manager, and it loads its released HF weights
  on resume because no DCP checkpoint will restore it.
- **Offline-cache readers.** A module that reads precomputed outputs from an
  offline cache is built with `requires_grad_(False)`; see
  [Offline Encoding](../mixins/offline_encoding.md).

## 2. What the job shares

- **The world.** The trainer initializes the process group once; every module
  builds its mesh over the full world.
- **The data stream.** `OmniTrainer` builds one dataloader, under the job's
  `base` parallel state derived from `model.accelerator`. Every module consumes
  the same micro-batch, so the data-parallel layout of the loader is the job's,
  not any module's.
- **The SP size.** `model.accelerator.ulysses_size` is inherited by every
  module and must not be overridden per module. The reason is the shared data
  stream above; see
  [Why one SP size](sequence_parallel.md#2-why-one-sp-size-for-the-whole-graph).
- **The step.** One `loss.backward()` runs over the whole graph, followed by
  one optimizer step per module; gradient accumulation and the global batch are
  job-level settings.

## 3. How the runtime keeps modules apart

**One `ParallelState` per module.** `ModuleRuntime.setup()` calls
`init_parallel_state_from_config(mesh_accelerator, module_name)`, which builds
the module's device mesh and process groups and registers them under the module
name. States are cached by topology, so modules whose resolved accelerator
matches share one object, and the job's own state is registered as `base`.

**Scoped execution.** Each module's forward runs inside
`OmniModelRuntime.module_context(name)`, and the module's optimizer build,
gradient clipping and gradient-checkpoint recompute enter
`use_parallel_state(name)` themselves. SP, DP and extra parallel process groups
are resolved from the current `ParallelState` rather than from globals, so
attention all-to-alls, expert dispatch and vocabulary-parallel collectives all
reach the module's own groups without per-module bookkeeping.

**Gradient clipping.** `veomni_omni_module_clip_grad_norm` reduces each module's
norm over the groups its wrap requires:

- FSDP2: the local shard's p-th power sum, all-reduced over `fsdp_group`;
- FSDP2 with an extra parallel group: non-extra-parallel parameters over
  `fsdp_group`, extra-parallel parameters over that group's FSDP mesh and then
  the group itself;
- DDP without SP: the local sum, since DDP already all-reduced the gradients;
- DDP with SP: gradients are first averaged over `fsdp_group`, which includes
  the SP ranks (constraint 7a in `.agents/knowledge/constraints.md`).

`train.optimizer.grad_clip_scope` selects `per_module` (each module clipped to
`max_grad_norm` on its own, the default) or `global` (one coefficient for all
modules). See [Arguments](../../usage/arguments.md).

**Checkpoints.** Each trainable module saves its own DCP shards and its own HF
weights under `hf_ckpt/<module>/`; see
[Checkpointing](../../usage/checkpoint.md). A run loaded from an upstream HF
checkpoint instead exports one flat `hf_ckpt/` in that checkpoint's layout
([Upstream Checkpoint Layout](../usage/hf_layout.md)).

**One FSDP2 tree instead.** `--model.accelerator.fsdp_config.fsdp_scope model`
wraps the composed `OmniModel` once instead of each module. The mesh, init device
and wrap settings then come from the top-level accelerator, the wrap targets are
each child's `_no_split_modules` scoped as `{child}.{ClassName}`, and leftover
parameters unshard in `OmniModel.forward()`. It cannot be combined with eager
modules.

## 4. Recipes in the repository

| Recipe | Module | Training wrap | Notes |
|---|---|---|---|
| Janus 1.3B | `janus_siglip` | DDP | `init_device: cuda` |
| | `janus_vqvae` | FSDP2 | eager attention, no gradient checkpointing; the codec is frozen by the module |
| | `janus_text_encoder` | FSDP2 + `emb` 4 | no gradient checkpointing |
| | `janus_llama` | FSDP2 | |
| Qwen3 0.6B | `qwen3_text_encoder`, `qwen3_llm` | FSDP2 | text encoder without gradient checkpointing |
| Qwen3-30B-A3B | `qwen3_text_encoder` | FSDP2 | |
| | `qwen3_moe_llm` | FSDP2 + `ep` 4 | |
| Qwen3-VL 2B | `qwen3vl_vision`, `qwen3vl_text_encoder`, `qwen3vl_llm` | FSDP2 | text encoder without gradient checkpointing |
| BAGEL 7B MoT | all five modules | FSDP2 | VAE, text encoder and flow connector without gradient checkpointing |

The `modules_train.yaml` beside each recipe's `base.yaml` under
`configs/seed_omni/` is the source of truth.

## 5. Inference

Inference resolves each module's wrap the same way, with eager as the default
for modules that set nothing. A launch where every module is eager runs as one
process; a launch with any FSDP2, DDP or extra-parallel module needs `torchrun`
(`bash train.sh`). Under `torchrun`, eager replicas pin to the rank's own device
rather than fanning out with `device_map="auto"`. Recipes that support both keep
two files, `infer/modules_infer_eager.yaml` and `infer/modules_infer_fsdp.yaml`.
The launch commands are in
[Training and Inference](../usage/training_and_inference.md#5-inference).
