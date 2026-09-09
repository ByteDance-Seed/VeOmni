# VLM and Omni specifics

Read this when the model has a vision, audio or speech tower. It covers
pruning inactive subtrees (Omni), the multimodal forward and metadata
contract, and the VLM/Omni-only pitfalls. This is *in addition to* the
SKILL.md spine, not a replacement for it.

## Phase 0: upstream contracts

On top of the contracts the spine lists, check these four before writing any
patch. Each one has silently broken a migration before.

- Helper-method signatures (e.g. `get_placeholder_mask` takes `inputs_embeds`
  + `image_features` / `video_features`).
- Return-shape conventions: e.g. `get_{image,video}_features.pooler_output`
  is a `tuple[per-image tensor]` after `torch.split`, not a flat tensor.
- Packed position-ids contract (`[4, bs, seq-len]` with prepended
  `text_position_ids`).
- **RoPE shape collapse** — VLMs use `apply_interleaved_mrope` (and similar
  helpers) that collapse the leading 3-axis of mrope before layers see
  cos/sin, so the shape is `(bs, seq_len, head_dim)`. Any SP path that gathers
  cos/sin across the sequence dim (async Ulysses, ring attention) must use
  the correct `gather_dim`. Grep upstream for `interleaved_mrope`,
  `mrope_section`, or any pre-attention RoPE reshape before writing the patch.

## Phase 2: pruning inactive subtrees (Omni)

**Pruning inactive subtrees** (e.g. talker / code2wav in an omni model where
training only uses the thinker): use `config.exclude_from_output(<Class>, ...)`
to drop classes entirely from the generated file. This has three downstream
ripples you must clean up in the same patch config — otherwise `make quality`
or `import` will fail on the regenerated output:

- **`_init_weights` `isinstance(...)` branches** — upstream's
  `<M>PreTrainedModel._init_weights` typically has one `elif isinstance(module,
  <ExcludedClass>)` branch per leaf init. Override it
  (`@config.override_method("<M>PreTrainedModel._init_weights")`) and drop
  every branch that references an excluded class.
- **Public methods whose bodies reference excluded classes** — e.g.
  `enable_talker` constructs the talker. Override it to
  `raise NotImplementedError("<what>. Use upstream transformers for <purpose>.")`
  so callers get a clear message instead of an F821/NameError at import.
- **`__all__` is auto-filtered** by `veomni/patchgen/codegen.py` — any excluded
  class name is removed from the generated `__all__` list automatically, so
  you don't need a manual `drop_import_names` dance for it.
- **Transitively-dead helper classes** — activations / small utility modules
  used *only* by classes you just excluded will still land in the generated
  file as dead code. Grep the generated output for each excluded class's
  private helpers and add them to `exclude_from_output` too. Example:
  `SnakeBeta` is only referenced by `Qwen3OmniMoeCode2WavDecoderResidualUnit`;
  excluding Code2Wav without also excluding `SnakeBeta` leaves ~40 lines of
  dead code in `generated/`. For qwen2_5_omni's BigVGAN vocoder,
  `UpSample1d`/`DownSample1d` are referenced **both** by Token2Wav residual
  blocks (caught by exclusion) **and** by the base `_init_weights` method
  via `isinstance` checks (NOT caught — `ast.walk` doesn't trace
  `isinstance` strings). After excluding the speech subtree, always
  `rg "isinstance\(.*<excluded_class>" generated/` and override the methods
  that still reference excluded names.
- **`_init_weights` referencing excluded classes** — base `PreTrainedModel._init_weights`
  often has `isinstance(module, <SpeechHeadClass>)` / `<UpSample1d>` /
  `<SnakeBeta>` branches that init excluded modules. These do not generate a
  patchgen warning but explode at first model build with `NameError: name 'X'
  is not defined` (ruff also flags as `F821`). Always override `_init_weights`
  to drop branches that touch excluded classes — see qwen2_5_omni's override
  that strips `UpSample1d`/`DownSample1d` branches.
- **Upstream `generate()` with mutable default arg** — Omni models like
  qwen2_5_omni define `generate(..., talker_eos_token_id: list[int] = [8292, 8294], ...)`
  which `ruff B006` rejects when copied verbatim into the generated file.
  Since the speech path is excluded anyway, override `<M>ForConditionalGeneration.generate`
  to raise `NotImplementedError("...generate is disabled in the VeOmni
  training modeling (talker / token2wav are excluded). Use upstream
  transformers for TTS generation.")`. This double-serves to kill the lint
  and make the contract explicit.

See `qwen3_omni_moe_gpu_patch_gen_config.py` (MoE thinker) and
`qwen2_5_omni_gpu_patch_gen_config.py` (dense thinker) for the canonical
templates. Both exclude the whole speech subtree plus the dead-after-exclusion
activations (`SnakeBeta` for qwen3_omni_moe; `UpSample1d`/`DownSample1d` for
qwen2_5_omni's BigVGAN), override `_init_weights` to drop the excluded-module
branches, override `enable_talker` to raise, and (for qwen2_5_omni) also
override `ForConditionalGeneration.generate` to raise `NotImplementedError`
— upstream's `generate(...)` signature has a mutable default arg
(`talker_eos_token_id: list[int] = [...]`) that trips `ruff B006` in the
generated file, and the TTS path is excluded anyway.

## Phase 2: multimodal forward and metadata

- **VLM/multimodal forward** — replicate qwen3_5_moe's pattern (VLM+MoE) or
  qwen3_vl's (VLM, non-MoE): pop LM-level flash-attn kwargs before ViT call,
  transpose seq↔head layout for Ulysses SP, shard image/video embeds, shard
  placeholder masks, and transpose back. Add
  `@config.override_method("<M>ForConditionalGeneration.get_position_id_func")`
  via an `add_post_import_block` that defines the helper `get_position_id` in
  generated scope (module-level, so multiprocessing can pickle it).
- **Multimodal metadata precompute** — to keep the ViT forward host-device-sync
  free, derive ViT `cu_seqlens` / `max_seqlen` in the collator, not the forward.
  See `.agents/knowledge/multimodal_metadata.md` for the full contract. Checklist
  for a new VLM:
  1. Add a module-level `collate_multimodal_metadata(batch, sp_pad)` helper
     (`@config.add_helper`) — read `batch["image_grid_thw"]` / `["video_grid_thw"]`,
     `.tolist()`, derive `vit_*_cu_seqlens` / `vit_*_max_seqlen` (+ the `sp_pad`
     tail entry), write `batch["multimodal_metadata"]`.
  2. `@config.override_method("<M>ForConditionalGeneration.get_metadata_collate_func")`
     returning that helper (or a `partial` over it if the formula needs config).
  3. Optional `get_extra_collate_infos` `override_method` for audio / extra
     feature tensors (Omni).
  4. Model.forward: pop `multimodal_metadata`, build the per-modality
     `vit_metadata` sub-dict (`grid_thw_list` / `cu_seqlens` / `max_seqlen`),
     pass to `get_image_features` / `get_video_features`.
  5. ViT.forward: pop the single `vit_metadata` kwarg; consume the precomputed
     values **with a runtime fallback** (in-forward `.tolist()` / cu_seqlens
     build) for callers that bypass `MainCollator`.
  6. `dummy_forward` (FSDP path): build the `vit_metadata` sub-dict host-side.
  7. Add the model to `_MM_METADATA_WIRED_CASES` in
     `tests/models/test_model_forward_no_implicit_sync.py`.
  When SP is enabled and you need to all-gather `input_ids` (or any tensor that
  went through `MainCollator`'s `pack_dim=-1` path) back to full seq on each
  rank, use `torch.cat(list, dim=1)` — the collator's `PackingCollator.__call__`
  does `torch.cat(..., dim=pack_dim).unsqueeze(0)` (see
  `veomni/data/data_collator.py:246-248`), so the shape at model forward is
  `[1, seq_per_rank]`, not flat `[seq_per_rank]`. Using `dim=0` would wrongly
  produce `[sp_size, seq_per_rank]` and silently break downstream mask slicing.

## Pitfalls

- **VLM `vocab_size` lookup** — top-level VLM configs use
  `config.text_config.vocab_size`, not `config.vocab_size`. Same for
  `num_experts`, `num_experts_per_tok`, `router_aux_loss_coef` on VLM-MoE.
- **Non-picklable helpers inside override bodies** — VLM `get_position_id_func`
  returns a `partial` over a helper; that helper must be at module scope in the
  generated file (injected via `add_post_import_block`), not a local closure,
  or DataLoader worker processes will fail to pickle it.
- **SP + `compute_3d_position_ids` on-the-fly is incorrect** — under Ulysses SP
  the `input_ids` / `inputs_embeds` arriving at `<VLM>Model.forward` are per-rank
  slices; computing mrope positions on them produces positions that drift across
  ranks. VeOmni training expects precomputed position_ids via `get_position_id_func`
  in the data transform. If your patched `Model.forward` has a fallback branch
  that calls `compute_3d_position_ids` (or equivalent) when `position_ids is
  None`, raise a clear `RuntimeError` under `get_parallel_state().sp_enabled`
  rather than silently returning wrong positions. This keeps inference /
  generation (single-rank, SP off) working while fail-fast-ing under SP.
- **Hardcoded shapes in `<M>VisionModel.dummy_forward`** — compute pixel row
  size and `grid_thw` from `self.config.patch_size` / `temporal_patch_size` /
  `in_channels` and `self.spatial_merge_size`, not from the model variant you
  first tested. Grids must be multiples of `spatial_merge_size` (merger
  requirement); under SP, scale one spatial dim by `sp_size` so the post-slice
  seq length stays a multiple of `sp_size`.
- **`self.dtype` / cached `_dummy_data` in `dummy_forward` is wrong under
  FSDP2 + MixedPrecisionConfig** — `self.dtype` returns the *first parameter's*
  dtype, which under FSDP2+MixedPrecision is the stored dtype (fp32), not the
  per-call compute dtype (bf16) the framework casts weights to at forward time.
  If `dummy_forward` allocates inputs via `torch.zeros(..., dtype=self.dtype)`
  or caches a `_dummy_data` buffer at `__init__`, the first conv/linear on a
  text-only rank crashes with "Input type (float) and bias type
  (c10::BFloat16) should be the same", while the multimodal rank hangs on the
  collective — masquerading as an NCCL hang. Always look up dtype from a live
  parameter at call time and don't cache dummy tensors across calls. The
  exact attribute is **model-specific** and copy-pasting the wrong one is a
  classic-silently-broken bug:
  - qwen3_omni_moe audio: `dtype = self.conv2d1.weight.dtype` (2D conv front-end)
  - qwen2_5_omni audio: `dtype = self.conv1.weight.dtype` (1D conv front-end —
    qwen3_omni_moe-style `conv2d1` does not exist on this model)
  - qwen2_5_omni / qwen3_omni_moe vision: `dtype = self.patch_embed.proj.weight.dtype`
  See the audio / vision `dummy_forward` patches in
  `qwen2_5_omni_gpu_patch_gen_config.py` and
  `qwen3_omni_moe_gpu_patch_gen_config.py`.
- **FSDP2 "hang" may be a rank-asymmetric crash** — when one rank crashes
  inside a collective-spanning forward (dtype mismatch, shape mismatch,
  unexpected `None`), the surviving ranks block on the never-completing
  collective and the test wall-clocks to SIGTERM. Re-run with
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` to force the per-rank exception to surface;
  once you see the real traceback on the crashing rank, fix *that* rather than
  hunting for deadlocks in the happy-path code.
- **`gather_dim` for cos/sin in async Ulysses attention paths** — the correct
  seq dim depends on whether a pre-attention RoPE reshape has happened. In
  Qwen3-VL v5, `apply_interleaved_mrope` runs before attention and collapses
  the leading 3-axis, so cos/sin arriving at async Ulysses is
  `(bs, seq_len, head_dim)` → `gather_dim=1`. Don't blindly copy `gather_dim`
  from a sibling model; read the upstream RoPE path first.
- **`TypeError: expected string or buffer` when manually exercising
  `MODEL_CONFIG_REGISTRY` before `MODELING_REGISTRY` (Omni models with patched
  configs)** — calling `MODEL_CONFIG_REGISTRY.get("<m>")()` *before*
  `MODELING_REGISTRY.get("<m>")()` causes the config-registration monkey patch
  to fire first; transformers' `@auto_docstring` then tries to read the patched
  config class's source via `CONFIG_MAPPING` and gets a live Python object
  instead of a source string. This blows up inside upstream
  `transformers/utils/auto_docstring.py`. **Not a real bug** — the natural model
  build order (`build_foundation_model_from_config(...)` → `MODELING_REGISTRY`
  first, which imports modeling and triggers the config import transitively)
  hits the right order and the error never fires. Only matters if your smoke
  test calls the registries directly in the wrong order. Confirmed on
  qwen2_5_omni / qwen3_omni_moe.
