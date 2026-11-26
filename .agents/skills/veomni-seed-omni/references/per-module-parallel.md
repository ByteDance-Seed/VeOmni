# Per-Module Parallelism

Read before editing per-module `accelerator` blocks, DDP/FSDP2 config, extra parallel dims, or distributed/eager inference configs. User-facing docs: `docs/seed_omni/distributed/per_module_parallelism.md` and `docs/seed_omni/distributed/sequence_parallel.md`. Hard constraints: `.agents/knowledge/veomni-seed-omni.md` (§3).

## Config Ownership & Patterns

- Top-level `accelerator` defines `"base"`; per-module `accelerator:` blocks in `modules_*.yaml` override via `_deep_update`.
- Each `ModuleRuntime` registers its `ParallelState` under `self.module_name` (identical topologies share the cached mesh in `_PARALLEL_STATE_CACHE`) and scopes execution via `use_parallel_state(self.module_name)`.
- `accelerator.fsdp_config.fsdp_scope: model` wraps `OmniModel` once under `"base"` using `{child}.{ClassName}` targets; leftover params unshard on `OmniModel.forward()`.

```yaml
text_encoder:
  accelerator:
    extra_parallel_sizes: [4]
    extra_parallel_names: ["emb"]
    extra_parallel_placement_innermost: [false]
vision_encoder:
  accelerator:
    fsdp_config:
      fsdp_mode: ddp
backbone:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
```

## Runtime Rules

- Module code calling `get_parallel_state()` must execute inside the module's `ParallelState` scope (`module_context` / `_recompute_context`).
- Sharded modules expose `get_parallel_plan()`. Mixed DDP/FSDP2/ExtraParallel models use `veomni_omni_module_clip_grad_norm`.
- DDP wrappers do not proxy all module attributes — unwrap for `pre_forward` / `post_forward` while calling the wrapper for forward.
- Nested HF configs (e.g. `Qwen3VLVisionEncoderConfig.vision_config`, `Qwen3VLLlmConfig.text_config`) must declare `sub_configs` so `attn_implementation` propagates to the inner model.

## SP / CP Patterns (`ulysses_size` / `cp_size`)

All distributed modules and `model.accelerator` must resolve to the same `dp_size` (`pp_size = tp_size = 1`; never enable both `cp_size > 1` and `ulysses_size > 1` on one module — explicitly set the other axis to `1` when overriding).

- **Standard SP/CP module flow** (inside `pre_forward` / `post_forward` when `get_parallel_state().sp_size > 1`):
  1. Stash full pre-slice seqlens via `metric_meter_set_seqlens` first.
  2. Pad/slice replicated sample to `1/S` (`sp_pad` + `slice_input_tensor` / `sp_pad_and_slice`), rebuilding varlen `cu_seqlens` over full padded length first.
  3. Run forward once (attention reads scoped `ulysses_group` / `cp_group` from `ParallelState`).
  4. In `post_forward`: `gather_outputs` + `narrow` off SP pad (or scatter `1/S` shards when `skip_sp_gather_between_modules=True`, per `.agents/knowledge/veomni-seed-omni.md` §3).
- **Shard axis by attention type**:
  - *Sequence dim*: text backbones (`qwen3/llm`, `janus/llama`, `qwen3vl/llm` incl. DeepStack embeds & 3-row M-RoPE) and Qwen3-VL ViT (`qwen3vl/vision` flat patches).
  - *Batch dim*: non-Ulysses vision towers (`janus/siglip`, `janus/vqvae`) slice the replicated image batch.

## Inference

- Default is `eager` for modules without a distributed accelerator; distributed launch is needed when any module uses FSDP/DDP.
- Under `torchrun`, pin eager full-replica modules to each rank's local device. Keep separate `modules_infer_{eager,fsdp}.yaml` files when both modes are supported.
