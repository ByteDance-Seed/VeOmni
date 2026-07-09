# Hard Constraints

Violating any of these causes silent bugs, crashes, or incorrect training results. Check before every code change.

## Model Loading & Registry

1. **Model registration must happen at import time**
   - `MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`, and `MODEL_PROCESSOR_REGISTRY` in `veomni/models/loader.py` are populated when model `__init__.py` files are imported.
   - Moving registrations into functions or delaying them breaks `build_foundation_model()`.
   - All model `__init__.py` files must import and register their modeling classes at module level.

2. **Model config `model_type` must match registry key**
   - The `model_type` field in a model's `config.json` is used as the lookup key in registries.
   - Mismatches cause fallback to vanilla HuggingFace loading, which misses VeOmni patches (flash attention, sequence parallel).

3. **Patchgen-generated files must not be edited manually**
   - Files under `veomni/models/transformers/*/generated/` are created by the `patchgen` CLI (entry point installed by the `patchgen` package).
   - Manual edits are silently overwritten on the next patchgen run.
   - To change generated behavior, edit the patch spec (`patch_spec.py`) or the modeling patch file (`modeling_*_patch.py`).

4. **Transformers version: pinned to v5.9.0**
   - VeOmni installs `transformers==5.9.0` via the `transformers-stable`
     default dependency group in `pyproject.toml`.
   - The legacy v4 path was removed; all modeling under
     `veomni/models/transformers/<m>/` is patchgen-generated.
   - `is_transformers_version_greater_or_equal_to()` from
     `veomni/utils/import_utils.py` is retained only for forward-looking
     gates (for HF APIs newer than the current pin) — do **not** add new
     version gates for versions `<= 5.9.0` (the legacy `>= 5.0.0` …
     `>= 5.8.x` interval is dead code).
   - Patchgen regeneration must be done with `transformers==5.9.0` installed.

## Distributed Training

VeOmni uses FSDP2 exclusively. FSDP1 has been removed.

Core entry points:
- `veomni/distributed/parallel_state.py` — `init_parallel_state()`, `ParallelState` dataclass
- `veomni/distributed/torch_parallelize.py` — `build_parallelize_model()`, `parallelize_model_fsdp2()`
- `veomni/distributed/parallel_plan.py` — `ParallelPlan`, `SpecInfo`

### FSDP2

5. **FSDP2 uses PyTorch composable `fully_shard()` API**
   - `parallelize_model_fsdp2()` in `torch_parallelize.py` calls `fully_shard()` on each transformer block, then on the root model.
   - The FSDP mesh comes from `ParallelState.fsdp_mesh`, which is a view of the global device mesh (can be `dp_shard`, `dp_shard_sp`, or include `dp_replicate` for HSDP).
   - When SP is enabled, the FSDP shard mesh fuses with the SP mesh (`dp_shard_sp`) so sequence-parallel ranks co-shard via FSDP.
   - Gradient clipping: `veomni/distributed/fsdp2/clip_grad_norm.py` — handles DTensor grads and ExtraParallel param groups.

6. **Device mesh initialization (`init_parallel_state()`)**
   - Builds a global `DeviceMesh` with named dimensions: `pp`, `dp_replicate`, `dp_shard`, `ulysses`, `cp`, `tp` (each included only if size > 1).
   - Flattens subviews for common usage: `dp` (all data-parallel), `sp` (ulysses+cp), `dp_shard_sp` (FSDP shard × SP), `dp_sp` (for loss/grad sync across SP+DP).
   - For each ExtraParallel name (e.g. `ep`), builds a `[para_size × para_fsdp_size]` submesh via `init_para_mesh_matrix()`.

### Sequence Parallel (Ulysses)

7. **SP uses all-to-all head/sequence exchange, not all-gather**
   - Implementation: `veomni/distributed/sequence_parallel/ulysses.py`
   - `gather_seq_scatter_heads(qkv)` — before attention: each rank sends sequence chunks, receives head chunks → **full sequence, subset of heads** per rank.
   - `gather_heads_scatter_seq(output)` — after attention: inverse exchange → **full heads, subset of sequence** per rank.
   - Underlying primitive: `_SeqAllToAll` (autograd-aware `all_to_all_tensor`).
   - Async variants in `async_ulysses*.py` for DiT and pipelined QKV/output projections.
   - Data slicing: `veomni/distributed/sequence_parallel/data.py` — `sp_pad_and_slice()`, `slice_input_tensor()`, `gather_outputs()`.
   - Loss reduction: `reduce_sequence_parallel_loss()` in `loss.py` aggregates across SP ranks.
   - Process groups: `comm.py`'s getters (`get_ulysses_sequence_parallel_group`, `get_unified_sequence_parallel_group`, `get_context_parallel_group`, `get_data_parallel_group`) resolve from the *current* `ParallelState`'s device mesh (`get_parallel_state().{ulysses,sp,cp,dp}_group`) — there are NO group globals. See 7d.

7-outer. **The outer (orchestrator) trainer ALWAYS runs with SP disabled (`ulysses_size == 1`, `cp_size == 1`); SP is a per-module concern**
   - The orchestrator does no per-token sequence compute — it only loads data, assembles the conversation carrier, dispatches to modules and aggregates loss — so outer SP buys nothing and would only force each SP rank to load-and-broadcast a replicated batch. `OmniTrainer.__init__` raises if `get_parallel_state().sp_size != 1`.
   - Consequence: **every rank loads its own DP shard directly** (no rank-0-only dataloader, no `broadcast_object_list`, no `StopIteration` sentinel). A module declaring `ulysses_size = S > 1` gathers its SP group's `S` *distinct* per-rank sequences into one packed sequence (`sp_gather_seqs`), runs SP over it, and narrows the output back per-rank (`sp_take_own_seq`). All `S` ranks hold distinct samples — there is no outer replication to dedupe.

7a. **Per-module Omni SP (`module_sp > 1`) must unify dtype before the cross-rank `all_gather`**
   - A module may run SP even though the orchestrator does not. `sp_gather_seqs()` (`data.py`) all-gathers the module SP group's `module_sp` *distinct* per-rank sequences via `_GatherConcatSP`.
   - Unlike classic Ulysses SP (where all SP ranks hold the SAME replicated packed sequence, hence identical dtype), here ranks hold **different samples**. A rank's packed `inputs_embeds` promotes to a dtype that depends on which modalities its local samples contain — e.g. bf16 text embeds (FSDP2 mixed precision) vs float32 image embeds (a DDP vision tower like `janus_siglip`). `naflatten` promotes the packed tensor to float32 if any component is float32.
   - **`all_gather` requires an identical dtype on every rank.** A mismatch (float32 vs bfloat16) makes NCCL read mismatched byte counts and **silently corrupts** the gathered buffer (NaN garbage) — no hang, no error unless `TORCH_DISTRIBUTED_DEBUG=DETAIL`. It surfaces only as a delayed loss NaN once garbage lands on a label token.
   - Fix: `sp_gather_seqs()` calls `_sp_unify_dtype()` to all-gather the per-rank dtype and cast every rank to the promoted common dtype before `_GatherConcatSP`. This is numerically a no-op vs the non-SP path (FSDP2 `cast_forward_inputs` re-casts to the compute dtype at the module boundary). Any new autograd collective that combines *distinct* per-rank data in Omni SP must apply the same dtype-unify guard.

7b. **Omni decode losses reduce over `dp_sp` (`fsdp_group`), NOT just `sp_group` — token-weighting must be global for SP transparency**
   - Per-module SP is meant to be *accuracy-transparent*: the loss/gradient must be identical no matter how the global batch is split into DP vs SP. A per-rank `ce_sum / n_valid` (the classic DP path) gives a **mean-of-means** that over-weights ranks holding few valid tokens — e.g. one rank with 8 text tokens @ loss 13.1 and another with 1926 @ loss 1.6 both get 1/4 weight, so `sp1` reports `text_encoder.decode` ≈ 4.47 while `allsp4` (SP4, token-weighted) reports the correct 1.55. Same batch, ~3x apart.
   - Fix (`text_encoder/modeling.py::decode`, `janus/vqvae/modeling.py::_vq_loss`): **always** compute `local_mean = ce_sum / n_valid_local` then `reduce_sequence_parallel_loss(local_mean, n_valid_local, group=ps.fsdp_group)`. `fsdp_group` is the `dp_sp` mesh (dp_replicate × dp_shard × sp) = exactly the mesh FSDP2 reduce-scatter (÷dp_shard_sp) + HSDP all-reduce (÷dp_replicate) average grads over, so `ReduceLoss`'s backward `×|dp_sp|` cancels it and yields the true **global token-weighted** gradient. Result: `sp1`/`llama_sp4`/`allsp4` align to ≤ bf16 ulp (`max|dloss|` 2.95 → 0.06).
   - **Consequence — the collective now spans every DP rank, so decode must run symmetrically on ALL `fsdp_group` ranks.** Any per-rank early-return before the reduce (e.g. vqvae's old `is_dummy` fast-path, which was only made group-uniform over `sp_group`) will desync/hang when dummy-ness varies across DP. `vqvae.decode` therefore always routes through `_vq_loss`; an all-dummy span is handled by its `-100` labels (CE 0, clamped denom → 0.0, `generation_head` still runs as the FSDP anchor). This changes DP loss semantics for Omni training from mean-of-means to global token-weighted (bagel excluded, out of SP scope).

7c. **Per-module metric meter must stash PRE-gather / PRE-slice (own-data) seqlens in `pre_forward`, not measure the forward `data`**
   - `MetricMeterMixin` no longer reads the module's forward `data` for token counts. Each metered module calls `metric_meter_set_seqlens(method, seqlens)` **inside `pre_forward`, before any SP gather/slice**; the default `metric_meter_token_lengths` just drains that stash. Backbones (`janus_llama`, `qwen3`, `qwen3_moe`) build it from the pre-gather `position_ids`, `text_encoder` from the pre-gather `input_ids`, `janus_siglip`/`janus_vqvae` from the pre-gather image count, `bagel/siglip_navit` from `token_lens` (no SP → already full).
   - **Why pre-gather, not post-gather:** `metric_meter_add` runs *after* `pre_forward` (`TrainingGraph.step`), so its `data` is this rank's SP shard — measuring it under-counts by ~`sp`. But stashing the *post*-`sp_gather_seqs` full is also wrong: `OmniEnvironMeter.add` reduces tokens **and** `total_flops` over the **`dp_group`** (`omni_helper.py`), and per-module SP gathers `module_sp` *distinct* DP ranks — so a post-gather full would be summed `module_sp` times, over-counting by exactly `module_sp`. The invariant is "each rank reports only its OWN data (full, un-sliced); DP-sum reconstructs the global total". This holds for FLOPs/MFU too (each device does `1/sp` of the work; `world_size` in the MFU denominator absorbs the sp factor).
   - Verified SP-transparent: `sp1` == `llama_sp4` == `all_sp4` per-module `trace/<module>/consume_tokens` match to the digit. A regression here shows up as tokens/MFU scaling with `sp`, not as a crash.

7d. **SP / DP / CP groups are ParallelState-local (resolved from the current state's mesh), NOT module-level globals**
   - Why this matters: the global attention integration (`veomni_flash_attention_2_with_sp`) has no handle on the `ParallelState` — it reaches its all-to-all group only through the process-level getter `get_ulysses_sequence_parallel_group()`. If that getter reads a mutable global slot set once at build time, then two Omni modules with *distinct* SP sizes both > 1 (e.g. ViT `ulysses_size=2`, LLM `ulysses_size=4`) clobber each other: the later-built module wins the slot and the earlier module's attention all-to-alls over the wrong ranks (silent corruption, or a `split_with_sizes`/shape mismatch). The old design "worked" only while every SP-enabled module shared ONE SP size.
   - Fix (the principled one): `comm.py` keeps NO group globals. `get_ulysses_sequence_parallel_group` / `get_unified_sequence_parallel_group` / `get_context_parallel_group` / `get_data_parallel_group` resolve from the *current* state's `{ulysses,sp,cp,dp}_group` (the state's device-mesh subgroup) — exactly how `fsdp_group` already worked. The helper `_current_state()` reads the `_PARALLEL_STATE` module global **directly** (not via `get_parallel_state()`), so an UNINITIALIZED process resolves to `None` — i.e. "no groups" — and every getter is `None`-safe (`ps.X_group if ps is not None else None`). This avoids `get_parallel_state()` constructing a default `ParallelState` that validates against the world size and raises for pre-init / unit-test code. Since `use_parallel_state` (via the graph `_module_scope`) already scopes `_PARALLEL_STATE` per module for its forward *and* its grad-checkpoint recompute (`_scope_recompute_to_parallel_state`), each module's attention automatically gets its own group with zero key bookkeeping. The `ParallelState.{dp,sp,ulysses,cp}_group` properties are mesh-only (return `None` when the dim is absent) so the getters never recurse.
   - Consequences: (a) meshless SP init is gone — `ParallelState.__post_init__` raises if `sp_enabled and device_mesh is None`; always build via `init_parallel_state`. (b) The `_ULYSSES_GROUP_KEY` registry, `init_sequence_parallel`, `UlyssesGroupKeyManager`, `get_ulysses_group_key_context`, `*_by_key`, cpu-group and `set_{data,context,unified}_*` setters were all deleted. (c) ALL group setters are gone, including the former `set_ulysses_sequence_parallel_group` test seam (no `_ULYSSES_SP_GROUP_OVERRIDE`). SP unit tests (`tests/parallel/ulysses/*`) build a real state via `init_parallel_state(dp_size=1, ulysses_size=world_size)` and switch to the no-SP reference path with `set_parallel_state(None)` — no group injection into production code. (d) The redistribution helpers (`sp_gather_seqs`/`sp_take_own_seq`) take the group explicitly from `ps.sp_group` (the scoped module state); there is no orchestrator-state read (`get_global_parallel_state`/`set_global_parallel_state` were removed with the outer-SP layer — see 7-outer).
   - Non-omni trainers self-scope too (same design as the Omni orchestrator, but caller = the trainer itself): `BaseTrainer` captures its state (`self.parallel_state = init_parallel_state(...)` in `_setup`) and scopes **three regions** — a build region (`with use_parallel_state(self.parallel_state):` around the whole build sequence in `__init__`, right after `_setup()`), the callback delegates (`on_train_begin/end`, `on_epoch_begin/end`, `on_step_begin/end`), and a run region (`forward_backward_step`). `TextTrainer` / `VLMTrainer` / `DiTTrainer` / `TextDPOTrainer` (all compose `BaseTrainer` via `BaseTrainer.__new__`) mirror this with `self.base.parallel_state`. This is a no-op for the single-model case (one global state) but keeps the standalone trainers consistent with the per-module design, so a subclass that later drives multiple modules with distinct states scopes each callback (e.g. DCP checkpointing, which reads the current `ParallelState` at save time) to the owning module. It does NOT affect the Omni path: `OmniModuleTrainer` skips `BaseTrainer.__init__` (uses `__new__`) and does its own per-module scoping (build at construction, plus per-callback and grad-checkpoint-recompute scopes), and `OmniTrainer` has its own `forward_backward_step` — neither calls `BaseTrainer.forward_backward_step`.
   - Cache robustness: `init_parallel_state()` caches states by topology in the module-level `_PARALLEL_STATE_CACHE` and only (re)establishes the default `_PARALLEL_STATE` global when it is `None` — on both the build path and the cache-hit path (a same-topology hit no longer asserts the global is set, since tests reset `_PARALLEL_STATE = None` at teardown without clearing the cache).
   - Verified: outer SP1 + ViT SP2 + LLM SP4 (`qwen3_0.6b/modules_train_visual_instruction_tuning_hetero_sp.yaml`) trains cleanly (loss decreases, no shape mismatch). A regression here surfaces as a wrong-ranks all-to-all (`split_with_sizes` mismatch or delayed loss NaN), only in configs with ≥2 distinct SP sizes > 1.

### Expert Parallel (MoE)

8. **EP shards expert weights and exchanges tokens via all-to-all**
   - Weight sharding: `ParallelPlan` in `parallel_plan.py` defines which expert parameters get `Shard(0)` on the EP mesh. `ParallelPlan.apply()` wraps matching params as DTensors and redistributes to local shards.
   - Token routing: `veomni/distributed/moe/moe_layer.py` — `preprocess()` computes dispatch counts, `token_pre_all2all()` / `tokens_post_all2all()` exchange tokens between EP ranks via `all_to_all` / `all_to_all_async` in `moe/comm.py`.
   - Expert computation: `EPGroupGemm` runs fused expert MLP on grouped tokens per rank.
   - Device mesh: `init_parallel_state()` builds `[ep × ep_fsdp]` submesh; accessed via `ParallelState.extra_parallel_mesh("ep")`, `ep_group`, `ep_rank`.
   - In FSDP2: expert modules get `fully_shard()` on the `ep_fsdp` submesh with `Shard(1)` placement so hidden-dim sharding composes with EP's dim-0 sharding.

## Data Pipeline

Core files:
- `veomni/data/data_collator.py` — `MainCollator` (3-stage pipeline)
- `veomni/data/dynamic_batching.py` — sample packing with token budgets
- `veomni/data/data_transform.py` — dataset transform registry
- `veomni/data/chat_template.py` — chat template with label masking
- `veomni/utils/seqlen_pos_transform_utils.py` — FA kwargs computation

### MainCollator Pipeline

9. **MainCollator is a 3-stage pipeline, not a single function**
   - Stage 1: `PrecomputePositionIDsCollator` — fills `position_ids = torch.arange(seq_len)` if absent.
   - Stage 2: `PackingCollator` — concatenates micro-batch samples along sequence dim using `DataCollateInfo` rules from `DEFAULT_DATA_COLLATE_INFO`. Sets `labels[0]` of each non-first sample to `IGNORE_INDEX` at pack boundaries.
   - Stage 3: `SequenceParallelCollator` (only when SP enabled) — label shift, SP padding/slicing, FA kwargs, then position_ids slicing.

### Conventions

10. **`position_ids == 0` marks segment boundaries for FlashAttention varlen**
    - `add_flash_attention_kwargs_from_position_ids()` finds indices where `position_ids == 0` → builds `cu_seq_lens_q/k` for `flash_attn_varlen`.
    - These must be in the batch dict **before** the model forward pass. Recomputing per-layer causes host-device sync.
    - Multimodal models may have 3D position_ids `(B, dim, L)` — FA uses the first row `[:, 0, :]`.

11. **Dynamic batching token counting must match `dyn_bsz_count_mode`**
    - Default / legacy behavior (`train.dyn_bsz_count_mode="total"`) uses `attention_mask.sum()` as the length function in `DynamicBatchingSizeDataset` and `DynBszBuffer`.
    - Optional effective-token mode (`"effective"`) uses `(labels != IGNORE_INDEX).sum()` when `labels` are present, and falls back to `attention_mask.sum()` otherwise.
    - With FA varlen, `attention_mask` is still expected to be all-ones over packed length; boundaries come from `position_ids` and `cu_seq_lens`.
    - When SP is enabled, `attention_mask` must use `sp_pad_value=1` (asserted in `MainCollator.__post_init__`).
    - In effective-token mode, dynamic batching still applies a hard physical-token cap of `micro_batch_size * max_seq_len` during micro-batch selection to avoid unbounded prompt-heavy batches; a single sample may still exceed the cap by itself and should be controlled by preprocessing.

12. **`IGNORE_INDEX` (-100) for loss masking**
    - Labels set to `IGNORE_INDEX` are excluded from loss computation.
    - Chat templates set `IGNORE_INDEX` on non-target turns (prompts, system messages).
    - `PackingCollator` sets `IGNORE_INDEX` on the first token of each packed sample (after the first) to prevent cross-sample supervision.
    - Custom data transforms must preserve this convention.

13. **SP collation ordering is load-bearing**
    - `SequenceParallelCollator` executes in strict order: pad → slice batch tensors → compute FA kwargs on **full** `position_ids` → slice `position_ids` last.
    - Reordering causes incorrect `cu_seq_lens` or misaligned position/label tensors.

14. **Dynamic batching packs samples by token budget**
    - `DynamicBatchingSizeDataset` (preferred) / `DynBszBuffer` (legacy): per-worker buffer, yields when token sum ≥ `micro_batch_seq_length`.
    - `_get_micro_batch` greedily adds samples that fit. Supports `state_dict` / `load_state_dict` for checkpoint resumption.
    - Position IDs in packed sequences must encode segment boundaries (see constraint 10).

### Multimodal Data

15. **Multimodal preprocessing pipeline (`veomni/data/multimodal/`)**
    - `encode_multimodal_sample()` in `multimodal_transform.py` orchestrates: `conv_preprocess()` → `fetch_images/videos/audios` → `process_mm_data()` → processor tokenization.
    - Images: load → RGB PIL → `smart_resize` (pixel min/max, scale_factor for grid alignment, max aspect ratio).
    - Videos: `torchcodec` decode → `calculate_frame_indices` (FPS, min/max frames, `frame_factor`/`frame_factor_remainder` for VAE-friendly counts); optional paired audio.
    - Audio: `librosa` at configurable `sample_rate` (default 16kHz).
    - Placeholder IDs: `TYPE2INDEX` maps modality tokens (e.g. image input → `-200`, output → `-201`). `mask_input_ids()` replaces these with `0` for text embedding and exposes `{modality}_{input|output}_mask`.

## Checkpoint

16. **DCP checkpoint keys must match model state dict**
    - `veomni/checkpoint/dcp_checkpointer.py` uses PyTorch's DCP (`torch.distributed.checkpoint`).
    - Renaming model parameters or changing the model structure between save and load breaks checkpoint loading.
    - Extra state is saved per-rank via `_EXTRA_STATE_FORMAT` — changing rank count requires checkpoint resharding.

17. **Checkpoint save/load requires all ranks to participate**
    - DCP operations are collective — all ranks must call save/load simultaneously.
    - Calling checkpoint operations from only rank 0 causes deadlocks.

## Code Quality

18. **Ruff must pass before commit**
    - `make quality` runs `ruff check` and `ruff format --check`.
    - Pre-commit hooks enforce this automatically (`pre-commit run --all-files`).

19. **All comments and docstrings must be in English**
    - No Chinese or other non-English text in code comments. This is enforced by project convention.

20. **PR title must follow format: `[{modules}] {type}: {description}`**
    - Allowed modules and types are defined in `.github/workflows/check_pr_title.yml` (single source of truth).
    - CI checks PR titles automatically on every PR.

## Hardware

21. **NPU (Ascend) code paths require guards**
    - NPU-specific code must be guarded with `is_torch_npu_available()` or `IS_NPU_AVAILABLE`.
    - NPU kernels live in `veomni/ops/kernels/{rms_norm,rotary}/npu.py` and `veomni/ops/platform/npu/` — they must not be imported on GPU-only environments.

22. **Device-agnostic code must use `veomni.utils.device` helpers**
    - Use `get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()` instead of direct `torch.cuda.*` calls.
    - Direct CUDA calls break NPU compatibility.
