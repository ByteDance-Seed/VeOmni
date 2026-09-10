# Upgrading the transformers pin: 5.9.0 → 5.16.1

The `transformers-stable` dependency group moved from `transformers==5.9.0` to
`transformers==5.16.1`. Seven upstream minor releases changed contracts that
VeOmni's patchgen configs depend on, so the bump was not a lockfile-only
change: 15 of the 29 patch configs failed to regenerate, and several of the
remaining ones drifted silently. This note records what upstream changed and
how each VeOmni patch was migrated, so the next bump can follow the same path.

## Final pin: 5.16.1

The final upgrade targets [transformers 5.16.1](https://github.com/huggingface/transformers/releases/tag/v5.16.1).
Compared with 5.16.0, an AST audit of all 18 upstream modeling modules used by
VeOmni's 29 patch configs found no function-signature changes. DeepSeek-V3 and
GLM-MoE-DSA initialize unused query projections to `None`; regeneration carries
those updates into both GPU and NPU outputs. The shared HF loading code also
restores standalone tensor-parallel arguments for backward compatibility.

The upgrade includes the current main-branch DeepSeek-V4 QAT, context-parallel,
and indexer-loss changes. VeOmni retains the scoring projection at
`indexer.weights_proj` so existing DCP model and optimizer keys remain stable.
The HF converter maps the new `indexer.scorer.weights_proj` name back to this
layout, and the projection remains excluded from FP8 conversion. Omni RoPE
helpers also accept the global audio/video flag used by HF generation while
retaining the per-video placeholder convention for training. Thinker forwards
pass RoPE arguments by keyword: adding the global flag before `audio_seqlens`
otherwise shifts audio lengths and video timing into the wrong parameters.
The upgrade contract tests exercise forward without precomputed `position_ids`,
including multiple silent videos and the explicit global `False` mode.

## Finding the drift

Three signals together cover most of the risk:

1. **`patchgen <config>` (not `--dry-run`).** Generation runs the patched file
   through `ruff check`, so any patch body that references a symbol upstream no
   longer defines fails loudly with `F821` / `F811`. `--dry-run` returns before
   that step and therefore reports success on files that cannot import.
2. **An AST diff of the patch targets.** Ruff cannot see a patch whose *target*
   still exists but whose signature changed — that patch keeps applying and
   quietly runs against the wrong contract. Parse the 5.9 and 5.16 modeling
   modules, index every `Class.method`, and compare the parameter lists of the
   targets named in each config's `override_method` / `replace_class` /
   `replace_function` calls. Targets absent from *both* versions are
   VeOmni-added methods and are expected.
3. **Importing every generated module.** Neither of the above catches an
   attribute that moved *within* a class (`DeepseekV4Indexer.weights_proj` →
   `indexer.scorer.weights_proj`), and `@auto_docstring` validates patched
   signatures and return-dataclass docstrings while the class body executes, so
   some breakage only appears at import. `tests/models/test_generated_modeling_imports.py`
   was added for this: the bitwise logits suite only builds the GPU families that
   have toy configs, so an NPU-only generated file can be broken with no test
   noticing — which is exactly what happened to `patched_modeling_qwen3_5_npu.py`.

The bitwise `tests/models/test_models_logits_equal_v5.py` suite is the gate that
confirms a migration landed: it builds toy configs (no checkpoints needed) and
compares VeOmni's generated modeling against pristine HF layer by layer.

One trap worth calling out, because ruff, signature diffs and the logits suite
all miss it: transformers 5.16's `PreTrainedModel._initialize_weights` skips
`_init_weights` entirely for modules with no direct parameters when
`is_custom_code` is true — and it is true for VeOmni's out-of-tree generated
modeling. Any `_init_weights` branch keyed on a *container* module (a
`SparseMoeBlock` that only holds submodules) silently stops firing, leaving its
parameters at their `torch.empty` values. `torch.empty` usually returns finite
garbage, so a finiteness check does not catch it; compare parameter statistics
against pristine HF instead. Qwen3-Omni-MoE hit this: expert weights came out
all-zero (`std=0.0`) against HF's `std=0.02`.

## Upstream changes and the VeOmni response

### `typing.Optional` removed from modeling imports

Upstream converted annotations to PEP 604 and dropped `from typing import
Optional`. Patch bodies that still wrote `Optional[X]` became undefined names in
the generated file. All affected configs (deepseek_v4, qwen2, qwen2_5_omni,
qwen3_moe, qwen3_omni_moe, seed_oss) now use `X | None`.

### Vision attention takes `position_embeddings` and `max_seqlen`

`*VisionAttention.forward` dropped `rotary_pos_emb` and gained
`max_seqlen: int | None`, resolved through
`get_max_seqlen(cu_seqlens, config, kwargs={"max_seqlen": max_seqlen})`. The
vision model computes the value once via `get_vision_attention_seqlens` and
threads it through the blocks — the same host-sync hoisting VeOmni's
`max_seqlen` patches were added for. Upstream keeps a `@deprecate_kwarg` shim on
the *block*, so callers passing `rotary_pos_emb=` still work (with a
`FutureWarning`) until v5.20; the *attention* override had to be ported or it
received `position_embeddings=None`.

Note for VeOmni's SP path: `get_max_seqlen` gates on HF's
`is_flash_attention_requested`, which does not recognise the custom
`veomni_flash_attention_*` names, so it returns `None` for them. The ported
overrides use the caller-supplied value when present and otherwise reduce
`cu_seqlens` locally.

### `get_{image,video}_features` split `pooler_output` per item

Both helpers now return `pooler_output` as a `list` of per-image / per-video
tensors. VeOmni needs the flat tensor for the SP all-to-all and indexes a single
`n_*_tokens` slice, so qwen2_5_omni and qwen3_omni_moe gained the same
"skip the split" override qwen3_vl and qwen3_5 already carried.

### Qwen3.5 linear-attention plumbing

The Qwen3-VL, Qwen3-VL-MoE and Qwen3.5 composite constructors now use
`AutoModel.from_config` upstream. In generated modeling this still resolves
to HuggingFace's original towers, silently bypassing VeOmni's vision and text
patches. Override these constructors on GPU and NPU to instantiate the local
generated classes with `_from_config`, as Qwen3.5-MoE already does. Without
this, sequence-parallel image tokens meet unsharded position embeddings and
fail with a shape mismatch. `test_generated_multimodal_children` checks the
actual child classes through the model registry; the Qwen VLM e2e cases
exercise their sequence-parallel forward and backward paths.

- `causal_conv1d_fn` / `causal_conv1d_update` / `torch_chunk_gated_delta_rule` /
  `torch_recurrent_gated_delta_rule` are module-level torch implementations
  decorated with `@use_kernel_func_from_hub_with_fallback(...)`, no longer
  conditional imports. `FusedRMSNormGated`, `is_fast_path_available`, and
  `torch_causal_conv1d_update` are gone. The `drop_import_names` call and the
  `<name> = None` post-import placeholders were removed — they had nothing left
  to neutralise and collided with the new definitions. VeOmni's OpSlot dispatch
  in `Qwen3_5GatedDeltaNet.__init__` is unchanged in intent.
- `Qwen3_5DecoderLayer.layer_type` was renamed to `block_type`
  (`Qwen3_5GatedDeltaNet.layer_type` kept its name).
- `Qwen3_5TextModel._update_linear_attn_mask` was removed; `forward` builds a
  per-attention-type mask mapping with `create_causal_mask` /
  `create_recurrent_attention_mask`. VeOmni's override was dropped: upstream's
  helper decides the cached-forward case from shapes and *trims* the 2-D mask to
  the local sequence rather than discarding it, which is more correct for
  chunked prefill. It still reads `torch.all(attention_mask == 1)` behind an
  `is_tracing` guard, so that host sync is upstream behaviour now.

### MoE auxiliary-loss memory comparison

HF's Qwen3-MoE load-balancing loss now accumulates routing statistics per
layer without a one-hot expert mask. Its no-grad forward can use less memory
than VeOmni's Triton implementation, which concatenates the layer logits.
The memory regression test therefore measures training forward with
`requires_grad=True`, where HF retains softmax activations for backward.
It subtracts the live input allocation from each measured peak and warms up
both implementations. This is a forward-only comparison: the fused kernel's
full forward/backward peak can exceed HF's for multi-layer inputs. Numerical
forward and backward parity remain checked separately against the new HF
reference.

### DeepSeek-V3 MoE

`DeepseekV3NaiveMoe` was renamed `DeepseekV3Experts` (and carries
`@use_experts_implementation`), and the family-specific top-k math moved from
`DeepseekV3MoE.route_tokens_to_experts` into `DeepseekV3TopkRouter.forward`,
which now returns `(router_logits, topk_weights, topk_indices)`. VeOmni's
fused-MoE `replace_class` retargeted, and both the router and MoE forward
overrides were rebuilt on the new bodies while keeping the fp32-router
`autocast(enabled=False)` guard and the load-balance monitor hook.

### DeepSeek-V4 indexer scoring head

`softmax_scale`, `weights_scaling` and `weights_proj` moved off
`DeepseekV4Indexer` into a `DeepseekV4IndexerScorer` submodule reachable as
`self.scorer`. The final VeOmni patches keep these attributes on the indexer to preserve
existing DCP model and optimizer keys. Both GPU and NPU constructors create the scoring projection directly on
the indexer.
The checkpoint converter accepts both the new HF scorer key and the existing
VeOmni key.

### MLA value padding for flash attention

Upstream removed the `qk_head_dim != v_head_dim` value padding from each MLA
model's `forward` and moved it into HF's `integrations/flash_attention.py`
wrapper. VeOmni registers its own `veomni_flash_attention_*` implementations that
replace that wrapper, so `veomni/ops/kernels/attention/flash.py` now carries the
pad-and-crop itself; without it DeepSeek-V3/V4 training fails with
`RuntimeError: v must have shape (total_k, num_heads_k, head_size)`. The padding
is applied after the Ulysses all-to-all so the padded columns are not
communicated.

### GLM-MoE-DSA

The largest port:

- The per-tensor `apply_rotary_pos_emb` helper was replaced by
  `apply_rotary_pos_emb_interleave(q, k, cos, sin, ...)`, which rotates both
  rope streams in one call.
- The indexer key cache moved off a per-module `_cached_keys` buffer onto the
  shared cache via `past_key_values.update_indexer(k, layer_idx)`.
- `GlmMoeDsaIndexer.forward` and `GlmMoeDsaAttention.forward` both gained
  `position_ids`; the indexer takes `past_key_values` in place of `use_cache`,
  and applies causality from `position_ids` when no mask is supplied.
- Shared indexer layers now set `self.indexer = None` instead of branching on
  `skip_topk` inside `forward`, and `next_skip_topk` no longer exists — the
  attention returns `topk_indices` unconditionally.
- The `qk_head_dim != v_head_dim` flash-attention value padding is gone; the
  class sets `_supports_flash_attn = False`, so that branch was unreachable.
- Eager/SDPA now fold a boolean "not selected" mask into the additive attention
  mask, and the sparse indices are passed only to the flash-mla path.

That last point needed a guard. Upstream can pass `indices=` straight through
because its only non-eager/sdpa path is a flash-mla kernel that consumes them.
VeOmni also registers `veomni_flash_attention_*` names, and those swallow
`indices` into `**kwargs` — which would run *dense* attention with the DSA
selection silently discarded. Two places now reject that: the patched GPU
attention forward raises for any implementation that is not eager/sdpa/flash-mla,
and `veomni/ops/kernels/attention/flash.py` refuses a non-`None` `indices`
outright. The second one is what covers the glm_moe_dsa **NPU** build, whose
config does not patch the attention forward at all, and any future DSA family.

## Final 5.16.1 verification

- Locked GPU environment sync, `uv lock --check`, `make quality`, and both
  documentation path checks pass.
- All 29 patchgen configs regenerate and pass the drift check.
- After restoring the generated Qwen child models, all 13 HF/VeOmni
  forward/backward parity cases pass with a 40 GiB allocator limit (31.78 GiB
  peak). Reference weights stay on CPU between modes, and the DeepSeek-V4
  reference translates HF's scorer key while retaining strict loading.
- The restored towers pass all 52 implicit-sync/logits checks and 30 VLM
  freezing, LoRA and log-probability checks. Upgrade import/compatibility
  tests are now explicitly listed in both unit-test workflows.
- Generated imports, call-site signatures, implicit-sync checks, and bitwise
  logits parity: 111 passed, 1 skipped because `torch_npu` is unavailable.
  All 34 logits cases pass.
- Checkpoint converter, QAT, indexer loss, and upgrade compatibility: 265 passed.
  The new contracts cover GPU/NPU model and optimizer names, FP8 exclusions,
  and Omni text/silent-video generation positions.
- DeepSeek-V4 CP/Ulysses: 43 passed; after preserving the indexer keys, all
  5 focused indexer parallel cases pass again.
- The broad model suite retains three failures reproduced with the original
  5.16.0 commit: the DeepSeek-V4 q-norm reference assertion and two Flux
  CPU/Triton non-causal attention tests. These are not introduced by 5.16.1.
- All 3 DeepSeek-V4 end-to-end smoke cases pass with all 8 H20 GPUs visible,
  covering eager and packed TileLang training with/without gradient checkpointing.
- All 12 eight-GPU trainer checkpoint cases pass, including MoE EP variants,
  DeepSeek-V4, HSDP, and HF safetensors export.
- NPU hardware execution was not available on the H20 validation host.

## Initial 5.16.0 verification

These results describe the original migration before the final patch bump.

- `patchgen --check` — clean, no drift across all 29 configs.
- `pytest tests/models/test_models_logits_equal_v5.py` — 34/34 pass (19/34
  before the migration).
- `pytest tests/models/test_generated_modeling_imports.py` — 29 pass, 1 skipped
  (needs `torch_npu`).
- `pytest tests/models/test_model_forward_no_implicit_sync.py` initially
  passed after dropping three allowlist entries. Those sites were hidden by
  the `AutoModel` constructor regression, rather than removed upstream.
  Restoring the generated towers makes them observable again; the final
  5.16.1 fix restores their original fallback/position-copy classifications.
- `pytest tests/checkpoints/` — clean; the DeepSeek save/load cases are what
  surfaced the missing MLA value padding.
- `make quality` — clean.

Run `make` targets with the project venv activated: the repo pins ruff `0.13.2`
(`.github/workflows/check_pr_lint.yml`), and a newer ruff on `PATH` formats some
files differently. `patchgen` also shells out to whichever `ruff` it finds.

## Known follow-ups

- **Time-sensitive:** `get_window_index` (qwen2_5_vl, qwen2_5_omni) and
  `fast_pos_embed_interpolate` (qwen3_omni_moe) emit `FutureWarning`s naming
  v5.11 as their removal version — the pin is already past that, so they are
  living on borrowed time and can disappear in any release. They still exist in
  5.16 and the three vision paths work, so porting them to
  `transformers.vision_utils.get_vision_window_index` /
  `get_vision_interpolation_indices_and_weights` was deliberately left out of
  this bump to keep it reviewable, but it should land before the next one rather
  than waiting for a build break.
- `create_recurrent_attention_mask`'s all-ones `torch.all` reduction is a host
  sync on every forward that passes a 2-D mask — the reduction runs before the
  answer is known, so it fires whether or not the batch is padded. It is once
  per forward rather than once per layer, so the cost is bounded, and
  `test_model_forward_no_implicit_sync.py` cannot catch it because that gate only
  attributes syncs originating inside `generated/`. Re-optimising it belongs in a
  separate change.
- Qwen3.5's vision attention still consumes VeOmni's own `vision_max_seqlen`
  kwarg (injected by the patched ViT forward and popped before the attention
  interface call) rather than upstream's native `max_seqlen` parameter. Both
  mechanisms are self-consistent, so this is redundancy rather than a bug, but
  the qwen3_5 / qwen3_5_moe configs should converge on the upstream kwarg the
  way qwen2_5_omni and qwen3_omni_moe now do.
