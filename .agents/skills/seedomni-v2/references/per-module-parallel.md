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

> **Architecture (2026-07-20, implemented — Arch B):** SeedOmni V2 uses classic
> single-pass Ulysses at ONE **uniform** SP size shared by the outer trainer and
> every module. The earlier looped per-module SP (Arch A: outer SP=1, distinct
> per-rank samples, per-sample loop + gather-to-owner + activation offload/ckpt +
> `fsdp2_ac_patch`) is **deleted**. Decision record + deferred future work
> (data-balance, compute-packing, audio/video halo): `docs/seed_omni/module_level_sp.md`.

SP is **uniform**. Set the SP size on the outer trainer
(`accelerator.ulysses_size`); `OmniTrainer._validate_uniform_sp()` raises at
build time unless every module's `ulysses_size` equals it. Do NOT add per-module
`ulysses_size` overrides — modules inherit the outer size. The dataloader is the
standard `BaseTrainer` build-time sharded loader: it yields `dp_size = world / sp`
**distinct** shards and **replicates** each shard across its SP group, so every
rank of an SP group holds the SAME sample.

A module with `sp_size = S > 1` runs classic single-pass Ulysses, with the SP
slice/gather living INSIDE its own `pre_forward` / `post_forward` (no separate
`sp_pre_forward` / `sp_post_forward` hooks — same shape as veomni v1 single-model
SP):

    kwargs = module.pre_forward(**data)    # if sp_size>1: pad + slice replicated sample to 1/S
    out    = module(**kwargs)              # ONE forward; attention all-to-alls over the SP group
    out    = module.post_forward(**out)    # if sp_size>1: gather_outputs → all-gather + strip pad; then SP-agnostic write-back

Forward AND backward both peak at ≈ `1/S` — no loop, no per-sample checkpoint, no
CPU offload. `gather_outputs` uses autograd-aware `_Gather`; its backward
all-reduce over the SP group is cancelled by FSDP2's grad averaging over the
`dp_shard_sp` mesh, so gradients match the non-SP baseline (constraints 7a/7b).

Primitives (`distributed/sequence_parallel`), each called from an
`if get_parallel_state().sp_size > 1:` branch:

- In `pre_forward` (after stashing the full pre-slice seqlens for the meter):
  `sp_pad` + `slice_input_tensor` / `sp_pad_and_slice` to this rank's `1/S` chunk
  (rebuild any varlen `cu_seqlens` over the FULL padded sequence first); stash the
  sample's real pre-pad length for the gather.
- In `post_forward`: `gather_outputs(chunk, gather_dim, group)` to all-gather the
  full sequence on every rank, then `narrow` off the SP pad, then fall through to
  the SP-agnostic write-back.

Which dim shards depends on the module's attention:
- **Sequence-dim (Ulysses)** — text backbones (`qwen3/llm`, `janus/llama`,
  `qwen3vl/llm`) shard the packed token sequence; the Qwen3-VL ViT
  (`qwen3vl/vision`) shards the flat patch sequence and rebuilds its ViT
  `cu_seqlens` from the (replicated) `grid_thw`.
- **Batch-dim** — SigLIP (`janus/siglip`) / VQVAE (`janus/vqvae`), whose attention
  does not honor Ulysses, shard the (replicated) image batch instead — slicing the
  replicated batch is exactly per-rank image balance. DDP modules AVG param grads
  over `fsdp_group` in `veomni_clip_grad_norm` /
  `veomni_omni_module_clip_grad_norm` when `sp_size > 1`, and set
  `broadcast_buffers=False` under SP (constraints 7e).

**Per-module group isolation.** Each module builds its OWN `ParallelState`/device
mesh, so even at the same uniform SP size the modules hold distinct SP subgroup
objects. This is safe because the SP/DP/CP process groups are
**ParallelState-local**: the `comm.py` getters
(`get_ulysses_sequence_parallel_group`, `get_unified_sequence_parallel_group`, …)
resolve from `get_parallel_state().{ulysses,sp,cp,dp}_group` (the current state's
device-mesh subgroup) — there are no group globals. Since `use_parallel_state`
(via the graph's `_module_scope`) already scopes `_PARALLEL_STATE` per module for
its forward and its grad-checkpoint recompute, the global attention integration
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

Metering must stash **pre-slice full-sample** seqlens in `pre_forward` (see
`module-contract.md` + constraint 7c). All backbones (`qwen3/llm`, `janus/llama`,
`qwen3vl/llm`), the text encoder (`base/text_encoder`) and vision towers
(`qwen3vl/vision`, `janus/siglip`, `janus/vqvae`) support `sp_size > 1`. The
Qwen3-VL in-model backbone (`qwen3vl/llm`) slices the packed sequence INCLUDING
DeepStack visual embeds + 3-row M-RoPE `position_ids`; since the sample is
replicated (each rank self-describing), `visual_pos_masks` / per-layer DeepStack
embeds are normalized to real tensors before the slice (no cross-rank
reconciliation).

## Inference

- Eager inference is the default for modules without a distributed accelerator.
- A module needs distributed launch when its inference accelerator uses non-eager
  FSDP/DDP.
- Under `torchrun`, eager full-replica modules should pin to the rank's device;
  avoid `device_map="auto"` fanning every rank across all GPUs.
- Keep separate module config files for eager and distributed inference when
  both modes are supported.
