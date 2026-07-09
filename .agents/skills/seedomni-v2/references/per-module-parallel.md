# Per-Module Parallelism

Read this before editing per-module `accelerator` blocks, DDP/FSDP2 behavior,
extra parallel dimensions, or distributed/eager inference configs.

## Config Ownership

- Top-level `accelerator` defines the default/global topology.
- A module may override with its own `accelerator:` block in a modules YAML.
- Same topology should reuse the global `ParallelState`.
- Different topology gets a module-local `ParallelState`.

## Common Patterns

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

- Module code that calls `get_parallel_state()` must run under the module's
  parallel-state scope.
- A module with sharded weights should expose `get_parallel_plan()`.
- DDP wraps modules and does not proxy all attributes. Hook dispatch may need to
  unwrap for `pre_forward` / `post_forward` while calling the wrapper for actual
  forward.
- Mixed DDP/FSDP2/ExtraParallel modules require the Omni-specific grad clipping
  path, not a naive `clip_grad_norm_` over all parameters.
- Per-module extra-parallel lists may need deduplication before mesh creation;
  check the current trainer helper before changing this area.

## Sequence Parallel (Ulysses)

SP is **per-module**. The outer (orchestrator) trainer ALWAYS runs with SP
disabled (`ulysses_size == 1`, `cp_size == 1`) — it does no per-token compute, so
`OmniTrainer.__init__` raises if the outer `sp_size != 1`. Do NOT pass a top-level
`accelerator.ulysses_size`; declare SP on the modules you want sharded via their
own `accelerator.ulysses_size`. Every rank loads its own DP shard directly (no
rank-0-only dataloader / broadcast).

A module declaring `ulysses_size = S > 1` has an SP group of `S` ranks each
holding a **distinct** per-rank sequence; it must aggregate those `S` sequences,
shard, run, then narrow back per-rank. The primitives
(`distributed/sequence_parallel`):

- `pre_forward`: `sp_gather_seqs(x, dim)` → `(full, seg_lengths, sp_rank)`
  (autograd-aware; no-op when the module has no SP), then `sp_pad` +
  `slice_input_tensor` / `sp_pad_and_slice` to this rank's chunk.
- `post_forward`: `gather_outputs(...)` (drop the SP pad tail), then
  `sp_take_own_seq(full, dim, seg_lengths, sp_rank)` to narrow back to
  this rank's own segment before scattering onto its items.

Which dim shards depends on the module's attention:
- **Sequence-dim (Ulysses)** — text backbones (`qwen3/llm`, `janus/llama`) shard
  the packed token sequence; the Qwen3-VL ViT (`qwen3vl/vision`) shards the flat
  patch sequence. Gather only the DATA (pixels **and** `grid_thw`, both through
  `sp_gather_seqs` so they share group order), then DERIVE the ViT
  cu_seqlens metadata from the post-gather `grid_thw` — mirroring the text
  backbone deriving its FA `cu_seqlens` from the gathered `position_ids`. Do not
  gather/rebuild the metadata separately.
- **Batch-dim** — SigLIP (`janus/siglip`), whose ViT does not honor Ulysses,
  shards the image batch instead.

**Per-module group isolation.** Modules may run at *different* SP sizes in the
same graph (e.g. ViT `ulysses_size=2` while the LLM runs `ulysses_size=4`). This
works because the SP/DP/CP process groups are **ParallelState-local**: the
`comm.py` getters (`get_ulysses_sequence_parallel_group`,
`get_unified_sequence_parallel_group`, …) resolve from
`get_parallel_state().{ulysses,sp,cp,dp}_group` (the current state's device-mesh
subgroup) — there are no group globals. Since `use_parallel_state` (via the
graph's `_module_scope`) already scopes `_PARALLEL_STATE` per module for its
forward and its grad-checkpoint recompute, the global attention integration
(`veomni_flash_attention_2_with_sp`, which only reaches the group through
`get_ulysses_sequence_parallel_group()`) automatically all-to-alls over the
scoped module's own group. No key bookkeeping. (No group-injection seam: SP unit
tests build a real state via `init_parallel_state(dp_size=1, ulysses_size=world_size)`
and toggle the no-SP path with `set_parallel_state(None)`.)

**Nested-config gotcha:** an SP module whose real weights live in a nested HF
config (e.g. `Qwen3VLVisionEncoderConfig.vision_config`,
`Qwen3VLLlmConfig.text_config`) MUST declare `sub_configs = {"<name>": <Cfg>}` on
the top-level config so transformers propagates `attn_implementation`
(`veomni_flash_attention_2_with_sp`) to the inner model. Without it the inner
model silently falls back to SDPA — the Ulysses all-to-all lives in the
flash-attention path, so SDPA leaves the sliced chunk ungathered while the
cu_seqlens describe the full sequence (a `split_with_sizes` size mismatch).

Metering must stash **pre-gather / pre-slice own-data** seqlens (see
`module-contract.md` + constraint 7c). All backbones (`qwen3/llm`, `janus/llama`,
`qwen3vl/llm`) and vision towers (`qwen3vl/vision`, `janus/siglip`, `janus/vqvae`)
support `module_sp > 1`; the Qwen3-VL in-model backbone (`qwen3vl/llm`) aggregates
its distinct per-rank sequences including DeepStack visual embeds + 3-row M-RoPE
`position_ids` (reconciling the mixed real/dummy case with a MAX all-reduce).

## Inference

- Eager inference is the default for modules without a distributed accelerator.
- A module needs distributed launch when its inference accelerator uses non-eager
  FSDP/DDP.
- Under `torchrun`, eager full-replica modules should pin to the rank's device;
  avoid `device_map="auto"` fanning every rank across all GPUs.
- Keep separate module config files for eager and distributed inference when
  both modes are supported.
