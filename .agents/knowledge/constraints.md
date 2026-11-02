# Hard Constraints

Violating any of these causes silent bugs, crashes, or incorrect training results. Check before every code change.

## Model Loading & Registry

1. **Supported model registration must happen at import time**
   - `MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`, and `MODEL_PROCESSOR_REGISTRY` in `veomni/models/registry.py` are populated when model `__init__.py` files are imported.
   - Moving registrations into functions or delaying them breaks `build_foundation_model()`.
   - Every model imported by `veomni/models/transformers/__init__.py` must
     register its modeling classes at module level. Placeholder packages for
     pending model migrations must not be imported there or exposed through the
     public registry.

2. **Model config `model_type` must match registry key**
   - The `model_type` field in a model's `config.json` is used as the lookup key in registries.
   - Mismatches fail explicitly under the VeOmni backend. Set `MODELING_BACKEND=hf`
     only when loading the upstream HuggingFace class without VeOmni model kernels.

3. **Patchgen-generated files must not be edited manually**
   - Files under `veomni/models/transformers/*/generated/` are created by the `patchgen` CLI (entry point installed by the `patchgen` package).
   - Manual edits are silently overwritten on the next patchgen run.
   - To change generated behavior, edit the patch spec (`patch_spec.py`) or the modeling patch file (`modeling_*_patch.py`).

4. **Transformers version: pinned to v5.16.1**
   - VeOmni installs `transformers==5.16.1` via the `transformers-stable`
     default dependency group in `pyproject.toml`.
   - The legacy v4 path was removed; all modeling under
     `veomni/models/transformers/<m>/` is patchgen-generated.
   - `is_transformers_version_greater_or_equal_to()` from
     `veomni/utils/import_utils.py` is retained only for forward-looking
     gates (for HF APIs newer than the current pin) — do **not** add new
     version gates for versions `<= 5.16.1` (the legacy `>= 5.0.0` …
     `>= 5.15.x` interval is dead code).
   - Patchgen regeneration must be done with `transformers==5.16.1` installed.

## Distributed Training

VeOmni uses FSDP2 exclusively. FSDP1 has been removed.

Core entry points:
- `veomni/distributed/parallel_state.py` — `init_parallel_state_from_config()`, `ParallelState` dataclass
- `veomni/distributed/torch_parallelize.py` — `build_parallelize_model()`, `parallelize_model_fsdp2()`
- `veomni/distributed/parallel_plan.py` — `ParallelPlan`, `SpecInfo`

### FSDP2

5. **FSDP2 uses PyTorch composable `fully_shard()` API**
   - `parallelize_model_fsdp2()` in `torch_parallelize.py` calls `fully_shard()` on each transformer block, then on the root model.
   - The FSDP mesh comes from `ParallelState.fsdp_mesh`, which is a view of the global device mesh (can be `dp_shard`, `dp_shard_sp`, or include `dp_replicate` for HSDP).
   - When SP is enabled, the FSDP shard mesh fuses with the SP mesh (`dp_shard_sp`) so sequence-parallel ranks co-shard via FSDP.
   - Gradient clipping: `veomni/distributed/fsdp2/clip_grad_norm.py` — handles DTensor grads and ExtraParallel param groups.

6. **Device mesh initialization (`init_parallel_state_from_config()`)**
   - Builds a global `DeviceMesh` with named dimensions: `pp`, `dp_replicate`, `dp_shard`, `cp`, `ulysses`, `tp` (`dp_shard` is always included; other dimensions only when size > 1). Ulysses follows CP so its ranks are adjacent when both axes are used.
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
   - Loss reduction: `reduce_sequence_parallel_loss()` in `loss.py` aggregates across SP ranks (optional `group=` arg; defaults to the current state's unified SP group).
   - Gradient scale: the data-parallel gradient reduction averages over the SP ranks too (FSDP2 over `dp_shard_sp`, DDP over `dp_sp` at clip time), so each SP rank's backward must produce `sp_size ×` its share of the gradient. `ReduceLoss.backward` multiplies by `sp_world_size`. A model that gathers its output and computes a loss replicated on every SP rank must call `_Gather` with `sum_grad=True` (the `gather_outputs()` default) or `grad_scale=True, sum_grad=False`. Plain `sum_grad=False` silently scales every gradient by `1/sp_size` while the loss still matches (MiniMax H3 had this bug); compare SP gradients, not only losses, against a single-rank reference.
   - Process groups: `comm.py` has NO group globals. Its getters (`get_ulysses_sequence_parallel_group`, `get_unified_sequence_parallel_group`, `get_context_parallel_group`, `get_data_parallel_group`) resolve from the *current* `ParallelState`'s device mesh (`get_parallel_state().{ulysses,sp,cp,dp}_group`) — exactly how `fsdp_group` already worked. There are no group setters, not even a test seam: SP unit tests build a real state with `_init_parallel_state`. Meshless SP is unsupported — `ParallelState.__post_init__` raises if `sp_enabled and device_mesh is None`; always build via `init_parallel_state_from_config`.
   - Local parallel state: `BaseTrainer._setup(args)` registers `"base"` **before** seed/determinism env vars (`NCCL_DETERMINISTIC`, etc.). Initialization caches by topology and registers under `name`; duplicate name → debug and return existing (`_setup` then `VeOmniModelRuntime.setup` both register `"base"` on a single-model job). Never store the ParallelState **object**: `VeOmniModelRuntime.parallel_state` is a by-name registry lookup on every access. Build and clip open `use_parallel_state(self.model_name)`. Run-time trainer ops open `use_parallel_state(self.model.parallel_state)` (forward, postforward, backward) so they do not hard-code `"base"`. Only the *name* is held on the runtime. `clear_parallel_state()` after `destroy_process_group()`. Tests that need a topology no job config can express call `_init_parallel_state` directly.
   - **Build scope**: `_setup(args)` registers `"base"`. `VeOmniModelRuntime.setup()` only registers this model's mesh; `__init__` then wraps meta-init, freeze, FSDP2/EP wrap + weight load, and optimizer in `use_parallel_state(self.model_name)`. A trainer must NOT open a build scope of its own or resequence those steps from outside: a model whose build differs subclasses the runtime.
   - **Run scope (per-op, not whole `train_step`)**: each ambient-dependent op wraps `use_parallel_state(<that model's parallel_state>)`. On the one-model trainer that is `self.model.parallel_state` around forward, `postforward`/`mean_global_loss`, and `loss.backward()`. A multi-model trainer overrides `forward_backward_step` and wraps each model's forward/reduce/backward itself (DPO: policy vs reference). Grad clipping is `VeOmniModelRuntime.clip_grad_norm()`. When an API takes explicit `group=`, pass `get_parallel_state_by_name(<that name>).sp_group` and skip ambient. Do not `(loss_a + loss_b).backward()` under one ambient state when A and B have different SP groups — Ulysses backward reads ambient `get_parallel_state()`.
   - **Callbacks**: cache `Callback.parallel_state` at construction (ChannelLossComputer too). Hook dispatch must not depend on ambient ParallelState.
   - See `docs/design/local_parallel_state.md`.

7a. **DDP under SP: average grads over `fsdp_group` before clipping, and never broadcast buffers**
   - `build_parallelize_model()` wraps DDP with `process_group=dp_group`. Enabling Ulysses shrinks `dp` (`dp_size = world / sp_size`), so at `dp_size == 1` DDP's all-reduce is a no-op while each rank still holds gradients from only its `1/sp` slice of the sequence — the optimizer then steps on a partial gradient, silently, with no shape error. `veomni_clip_grad_norm()` therefore all-reduces `p.grad` over `fsdp_group` (`dp_sp`) and divides by the group size before clipping whenever `dp_mode == "ddp"` and `sp_size > 1`. FSDP2 does not have this bug because it already reduces over `fsdp_group`; `dp_sp` spans `dp_replicate × dp_shard × ulysses × cp`, so one reduction covers plain DP, HSDP and SP alike and leaves the two dp modes numerically equivalent. Reducing over `dp_sp` on top of DDP's own `dp` reduction is not a double count — every rank of a `dp` group enters with the same value, so the second average reproduces the plain mean over `dp_sp`. Any new grad-clip entry point that can see a DDP-wrapped module owes the same reduction.
   - Reduce with `SUM` then divide — **not** `ReduceOp.AVG`, which the NPU backend does not support (same reason as `veomni/utils/dist_utils.py::all_reduce`).
   - Reduce over `dp_sp` even though `sp` alone would be arithmetically identical whenever DDP has already averaged over `dp` (every rank of a `dp` group then holds the same value, so the `dp` extent contributes nothing). `dp_sp` costs a wider group — `dp` is usually the cross-node dimension — and buys independence from *whether* DDP reduced: `no_sync()`-style gradient accumulation, which nothing uses today, would leave `dp` peers unreduced at clip time and make the narrow version silently wrong in exactly the way this constraint exists to prevent. Revisit only with a benchmark. Both versions issue one collective per parameter tensor, uncoalesced; bucketing (as DDP itself does at 25 MB) is the obvious optimization if this ever shows up in a profile.
   - Select the gradients to reduce by `requires_grad`, not by `grad is not None`, and zero-fill a missing one. It is one collective per gradient, so a parameter that goes unused on some ranks only would otherwise desynchronize the sequence and hang. The fill is unreachable today — the wrap sets no `find_unused_parameters`, so DDP's own reducer already fails on an unused trainable parameter before the clip runs — and keying on `requires_grad` is what keeps the reduction correct rather than dependent on that.
   - Convert a DTensor grad norm with `full_tensor()` before `.item()`, as the FSDP2 path does: `.item()` on a sharded or partial DTensor reads this rank's piece rather than the global norm. Only the *reported* norm is at stake — `clip_grad_norm_()` computes and applies the clip internally, and does so globally for DTensors. Defensive either way: `build_parallelize_model()` calls `parallelize_module()` with no `parallelize_plan`, which torch warns about and treats as a no-op, so nothing on the DDP path is a DTensor today. Same for the `to_local()` in `_allreduce_ddp_sp_grads()`.
   - Pass `broadcast_buffers=False` **unconditionally**, never conditioned on `sp_size`. FSDP2/HSDP performs *no* buffer sync at all (`torch.distributed.fsdp` contains no `_sync_module_states` / `_broadcast_coalesced`; `fully_shard` only places buffers on the device). A module's buffer semantics must not change with the `fsdp_mode` a config happened to pick.
   - Nothing is lost. Config-derived buffers (SigLIP `position_ids`, static rope tables) are already identical on every rank, and dynamic-rope `inv_freq` is deliberately recomputed from *this* rank's sequence length by `modeling_rope_utils.py` — pushing rank0's copy would actively corrupt the others. Buffers get no gradients, so the gradient all-reduce above is not a fallback for them; but neither is the broadcast.
   - Genuinely replicated mutable state (`nn.BatchNorm*` running stats) is not fixed by `broadcast_buffers=True` either: it overwrites every rank with rank0's copy, i.e. discards the other ranks' statistics rather than aggregating them. `SyncBatchNorm` is the real fix — it all-reduces the statistics inside forward, so it holds under DDP, FSDP2 and HSDP alike; under SP its `process_group` must span dp+sp, since Ulysses splits the sequence and `dp_group` excludes the SP peers. The repo has zero `nn.BatchNorm*` and zero `nn.SyncBatchNorm` today — the single instance lived in the `janus` model deleted with the SeedOmni V1 stack — so this is a rule for new code rather than a live bug: a `SyncBatchNorm` added on the default world group needs that `process_group` argument before it is trained under SP.

7b. **DDP must materialize and load meta-init weights itself, before the wrap**
   - `model.accelerator.init_device` defaults to `"meta"` and only `fsdp_mode == "fsdp2"` is asserted to use it, so a `ddp` config reaches `build_parallelize_model()` with an empty model. DDP registers gradient hooks and broadcasts rank0's parameters, but it materializes nothing and loads nothing, and `BaseTrainer` has no load step of its own — so `parallelize_model_ddp()` owns that pass, exactly as `parallelize_model_fsdp2()` does. Both call `_materialize_and_load_weights()`, which is the single place where the choice between random init, an HF snapshot and a checkpoint resume is made; a new dp mode owes the same call. Omit it and DDP's constructor dies on `Tensor.item() cannot be called on meta tensors`.
   - The gate is `param.is_meta`, not `init_device`. The flag states an intent the model builder is free to ignore — `tests/data/*` construct their model eagerly and leave the flag at its `meta` default — and materializing a model that already holds real weights discards them, or raises outright on a plain `nn.Module` with no `init_weights`. A model built under `cuda` arrives with weights already loaded (`empty_init=False` in `veomni/models/auto.py`) and must be left alone. `parallelize_model_fsdp2()` keeps its call unconditional because `arguments_types.py` asserts `init_device == "meta"` for fsdp2, so a real model cannot reach it.
   - `init_device == "cpu"` is refused for `ddp`, by an assert in `AcceleratorConfig._validate_init_device()` alongside the one that pins fsdp2 to `meta` — not in `parallelize_model_ddp()`. Parse time is the right place: every rank fails together, before a model is built or a snapshot read. It never worked for the wrap: `device_ids=[local_rank]` has been passed since the first commit, and torch rejects that together with a CPU module (`torch/nn/parallel/distributed.py`), so rank0's CPU replica cannot be wrapped while every other rank builds empty and skips the load (`veomni/models/loader.py`). It was FSDP1's `sync_module_states` recipe (rank0 reads, the wrapper broadcasts), dropped in #756; fsdp2 replaced it with `broadcast_model_weights_from_rank0`.
   - Dropping `"cpu"` from the field's `Literal` is *not* what enforces this, and no `Literal` in the arguments layer enforces anything. The parser turns it into argparse `choices`, which covers the CLI only; a YAML value goes straight to the dataclass constructor via `_instantiate_recursive()` (`veomni/arguments/parser.py`), and annotations are never checked at runtime. Any config value that must actually be rejected needs an explicit assert in `__post_init__`.
   - Two other users of `"cpu"` are unaffected and must not be swept up in that: `build_foundation_model(init_device="cpu")` is a live public API (several tests build on CPU that way), and `materialize_device="cpu"` is how fsdp2 CPU offload reaches `load_model_weights()`. The `init_device` parameters in `veomni/models/checkpoint/weights.py` are fed by the latter, not by `model.accelerator.init_device`.
   - `should_skip_hf_weight_load` must be honoured here too: a distributed-checkpoint resume is about to overwrite every parameter, so reading the HF snapshot doubles peak memory, and the snapshot may not exist at all.
   - `broadcast_model_weights_from_rank0` is honoured here under both FSDP2 and DDP. `rank0_load_and_broadcast_weights()` broadcasts over the default (world) group from global rank0 (`dist.broadcast(..., src=0)`, no `group=`), which is what a DDP replica wants. It defaults to `True`.
   - After the load pass, `parallelize_model_ddp()` re-checks `param.is_meta` and raises with the offending parameter names. A loader that leaves one behind would otherwise surface inside DDP's constructor as `Tensor.item() cannot be called on meta tensors`, which names neither the parameter nor the cause.
   - ExtraParallel is refused on the DDP path, keyed on **the model's plan** (`_has_extra_parallel_plan()`), not on `ParallelState.any_extra_parallel_enabled`. Only `parallelize_model_fsdp2()` applies the plan that shards expert weights, so DDP experts are whole and loading a sharded-expert config into them — previously prevented only by the meta crash — would silently produce full tensors. But the mesh alone does not identify such a model: a SeedOmni V2 sub-module's accelerator is `_deep_update(global, override)`, so a DDP vision tower inherits the backbone's ep dim while owning no experts, and refusing on the mesh would block it. The same predicate gates `ep_sharded_stream_load` in `_materialize_and_load_weights()`: a plan-less model skips the fast path with a log line (there was never one to take), while a model that *does* have a plan still lets the loader's `NotImplementedError` propagate, because that one means the checkpoint layout is unsupported — the distinction `tests/utils/test_moe_ep_sharded_load_matrix.py` pins for `nonmerged x ep_sharded`. A per-expert checkpoint is supported only when the model's converter implements `fused_expert_target` + `for_expert_range` (each rank then reads and stacks just its own experts); a converter without them still raises.

### Expert Parallel (MoE)

8. **EP shards expert weights and exchanges tokens via all-to-all**
   - Weight sharding: `ParallelPlan` in `parallel_plan.py` defines which expert parameters get `Shard(0)` on the EP mesh. `ParallelPlan.apply()` wraps matching params as DTensors and redistributes to local shards.
   - Synchronous activation offload must skip `FP8ExpertWeight` containers saved by DeepGEMM: native FSDP owns their payload/scale storage and repopulates it before backward. Copying the opaque container to CPU is unsupported; it must not consume the activation offloader's GPU budget either. Ordinary activations still follow the configured offload policy.
   - Token routing: `veomni/distributed/moe/moe_layer.py` — `preprocess()` computes dispatch counts, `token_pre_all2all()` / `tokens_post_all2all()` exchange tokens between EP ranks via `all_to_all` / `all_to_all_async` in `moe/comm.py`.
   - Expert computation: `EPGroupGemm` runs fused expert MLP on grouped tokens per rank.
   - Device mesh: `init_parallel_state_from_config()` builds `[ep × ep_fsdp]` submesh; accessed via `ParallelState.extra_parallel_mesh("ep")`, `ep_group`, `ep_rank`.
   - In FSDP2: expert modules get `fully_shard()` on the `ep_fsdp` submesh with `Shard(1)` by default. `expert_fsdp_shard_dim=0` explicitly selects whole-expert FSDP shards after validating the EP plan and even local expert divisibility; `None` preserves the existing placement/Muon policy. Record the selected FSDP dimension in `SpecInfo`. If EP/FSDP split the same tensor dimension, DCP must use `_StridedShard(dim, split_factor=ep_size)` on the FSDP mesh axis: physical sharding is EP-first even though the mesh lists FSDP first. Two ordinary `Shard(0)` placements silently permute global expert IDs. VeOmni DCP save/load planners record per-tensor source EP/FSDP dimensions, sizes and expert order in `planner_data`; load compatibility depends on saved coordinates, not the target shard dimension. Unmarked legacy chunks that split only the expert axis are ambiguous and must be migrated using the original training layout; do not bypass the planners for production resumes.

   - LoRA adapter export must pass both `SpecInfo.placement.dim` and `SpecInfo.fsdp_shard_dim` to `restore_extra_parallel_dim`, just like regular model checkpoints. Assuming dim-1 FSDP for whole-expert shards corrupts global adapter shapes and expert order during safetensors consolidation.

8a. **Persistent 2D ExtraParallel parameters require a complete 2D mesh**
   - Persistent ExtraParallel parameters currently require a 2D `(<para>_fsdp, <para>)` mesh covering the complete world. HSDP replicas and FSDP CPU offload remain unsupported.
   - There is currently no `sequence_parallel_persistent_names` plan field and no explicit SP allowlist or SP-specific rejection in `torch_parallelize.py`. Therefore this constraint must not be read as validation that persistent ExtraParallel parameters are safe for every SP model; adding another such model requires reviewing its optimizer, gradient, and checkpoint semantics under SP.
   - Qwen4-Exp's `ple` dimension is currently the only persistent ExtraParallel use. Its lookup, gradient scaling, optimizer, and checkpoint paths operate over the complete flattened PLE mesh while inputs remain sequence-sharded.

8b. **Qwen4-Exp QSA indices are global full-resolution token indices**
   - `qsa_attention_implementation` selects blocks with local queries and globally gathered pooled keys, returning local `[B,L,K]` selections for each rank's query shard while `Qwen4ExpTextAttention` keeps `query_states` local (`[B,H_q,L,D]`) and all-gathers `key_states`/`value_states` (`[B,H_kv,S,D]`) over the active sequence- or context-parallel group.
   - Index values address global full-resolution K/V directly; `-1` is the only invalid-slot sentinel. Do not rebase them by rank or lift them past a compressed-KV segment.
   - Packed block boundaries come exclusively from global `cu_seq_lens_q`. Blocks must never cross a packed-sample boundary or be derived from an SP/CP shard boundary.
   - Keep selections compact through distributed collection and KV all-gather. Unsupported layouts must fail before entering collectives.
   - The eager backend deliberately expands the compact indices to an `[B,1,L,S]` mask and runs dense GQA. It is a numerical reference with quadratic memory/runtime scaling, not evidence that a fused backend implements gather semantics.

8c. **Differentiable SP collectives must remain in every rank's autograd graph**
   - Rank-local packed boundaries can make a gathered halo unused on some ranks. Preserve a zero-valued dependency on the gathered tensor so every rank executes the same backward collective sequence; otherwise a later all-reduce can be matched against another rank's gather backward and fail with inconsistent collective counts.

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
    - Default / legacy behavior (`train.dyn_bsz_count_mode="total"`) uses `get_length_fn` (`attention_mask.sum()` with `input_ids` fallback, or the sum of `item.meta["_omni_token_len"]` across all items—including `is_dummy=True` placeholders—for SeedOmni `conversation_list` samples) in `DynamicBatchingSizeDataset` and `DynBszBuffer`.
    - Optional effective-token mode (`"effective"`) uses `get_length_by_labels_fn` (`(labels != IGNORE_INDEX).sum()` when `labels` are present, or non-dummy supervised text/assistant-media token lengths for SeedOmni `conversation_list` samples, falling back to `get_length_fn` otherwise).
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

15. **Multimodal preprocessing pipeline (`veomni/data/multimodal/` + `veomni/data/data_transform.py`)**
    - The two orchestrators differ in where tokenization and label masking happen — do not assume a single shared order.
      - `process_sample_qwen_vl()`: `conv_preprocess()` → `fetch_images` / `fetch_videos_metadata` → `processor.image_processor` / `processor.video_processor` for pixel features only → `chat_template.encode_messages()`, which does **both** tokenization and label masking.
      - `process_sample_qwen_omni()`: takes no chat template. `conv_preprocess()` → `fetch_images/videos/audios` → `processor(text=..., images=..., videos=..., audios=...)` for tokenization → labels masked inline by a user/assistant token scan.
    - Images: load → RGB PIL → `smart_resize` (pixel min/max, scale_factor for grid alignment, max aspect ratio).
    - Videos: `torchcodec` decode → `calculate_frame_indices` (FPS, min/max frames, `frame_factor`/`frame_factor_remainder` for VAE-friendly counts); optional paired audio.
    - Audio: `librosa` at configurable `sample_rate` (default 16kHz).
    - Placeholder IDs: `veomni/utils/constants.py` defines the negative placeholders (`IMAGE_INPUT_INDEX = -200`, `VIDEO_INPUT_INDEX = -300`, `AUDIO_INPUT_INDEX = -400`; `TYPE2INDEX` groups them by input/output). `MultimodalChatTemplate` writes them into `input_ids`, and `process_sample_qwen_vl()` derives `image_mask` / `video_mask` from them, then zeroes the placeholders before text embedding. `process_sample_qwen_omni()` instead derives `image_mask` / `video_mask` / `audio_mask` from the model's own multimodal token ids. The mask keys are `{modality}_mask` — the V1 `{modality}_{input|output}_mask` convention went away with the SeedOmni V1 stack.

## Checkpoint

16. **DCP checkpoint keys must match model state dict**
    - `veomni/checkpoint/dcp_checkpointer.py` uses PyTorch's DCP (`torch.distributed.checkpoint`).
    - Renaming model parameters or changing the model structure between save and load breaks checkpoint loading.

17. **Checkpoint save/load requires all ranks to participate**
    - DCP operations are collective — all ranks must call save/load simultaneously.
    - Calling checkpoint operations from only rank 0 causes deadlocks.
    - ``lr_scheduler.pt`` is replicated: rank 0 writes the file, but every rank
      still joins the save reduction and the promotion collectives. The sidecar
      is written before ``dcp.save`` / ``dcp.async_save``, so DCP's ``.metadata``
      (the resume completeness marker) lands last. Each step writes a new
      ``global_step_{N}/``; a failed save has no ``.metadata`` and is skipped.
      A resume that expects a scheduler and finds no ``lr_scheduler.pt`` falls
      back to ``extra_state/`` via ``veomni/checkpoint/legacy_v0_1_12.py``
      (VeOmni 0.1.12). Delete that module and its two imports to drop the
      fallback; the load then raises.
    - ``trainer_state_rank_{R}.pt`` stays per-rank: the dataloader cursor and RNG
      are rank-local. Changing world size still requires a matching cursor file
      per rank. On-disk layout: ``docs/usage/checkpoint.md``.

18. **Distributed HF safetensors consolidation must support non-floating tensors**
    - PyTorch 2.9–2.11 computes consolidated tensor byte sizes with `torch.finfo`, which crashes for valid integer and boolean buffers such as DeepSeek V4 `tid2eid`.
    - `apply_dcp_consolidation_patch()` in `veomni/checkpoint/dcp_consolidation.py` replaces the metadata parser with `Tensor.element_size()` and verifies the upstream private-function source hash before patching.
    - Offline DCP-to-HF conversion may cast `save_dtype` only onto floating tensors; integer and boolean buffers must retain their original dtype, and shard-size planning must use their original element sizes.
    - Do not remove this patch during torch upgrades until the new upstream consolidator is verified with sharded integer-tensor save/load coverage.

## Code Quality

19. **Ruff must pass before commit**
    - `make quality` runs `ruff check` and `ruff format --check`.
    - Pre-commit hooks enforce this automatically (`pre-commit run --all-files`).

20. **All comments and docstrings must be in English**
    - No Chinese or other non-English text in code comments. This is enforced by project convention.

21. **PR title must follow format: `[{modules}] {type}: {description}`**
    - Allowed modules and types are defined in `.github/workflows/check_pr_title.yml` (single source of truth).
    - CI checks PR titles automatically on every PR.

## Hardware

22. **Non-CUDA accelerator code paths require guards**
    - There are three backends today: CUDA, Ascend NPU and Cambricon MLU. Guard vendor-specific code with `is_torch_npu_available()` / `IS_NPU_AVAILABLE` or `is_torch_mlu_available()` / `IS_MLU_AVAILABLE` (`veomni/utils/import_utils.py`, `veomni/utils/device.py`).
    - Vendor kernels live in their op family under `veomni/ops/kernels/`. The HCCL compatibility patch lives in `veomni/distributed/hccl_premul_sum.py`. Vendor-only dependencies must not be imported on hosts without that runtime.
    - Device-type sets, not `== "cuda"`, decide dispatch: `MOE_TRITON_DEVICE_TYPES` in `veomni/utils/device.py` is `("cuda", "mlu")` today. Adding a backend means auditing those sets, not just adding a branch.

23. **Device-agnostic code must use `veomni.utils.device` helpers**
   - Use `get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()` instead of direct `torch.cuda.*` calls.
   - Direct CUDA calls break NPU and MLU compatibility.
   - The opt-in DeepGEMM FP8 MoE backend is CUDA-only and requires one CUDA device per process. The pinned DeepGEMM `5081f85` caches device-bound kernel handles without a device key; switching GPUs can fail with `CUDA_ERROR_INVALID_HANDLE` even on identical GPUs. `GroupedLayout.from_counts()` enforces this restriction and sets the operand device context before accessing DeepGEMM. Other DeepGEMM calls in the worker must use the same device. `OpsImplementationConfig` rejects `moe_implementation='deepgemm_fp8'` together with non-`none` QAT to prevent double quantization before expert GEMMs. Test non-current-device handling in a fresh spawned worker; standard one-worker-per-GPU EP/FSDP2 is supported.

## Trainer Extensions

24. **Trainer callback lifecycle changes must cover composed trainers**
   - `TextDPOTrainer` and `DiTTrainer` compose a `BaseTrainer` and override `forward_backward_step()`; they do not inherit the base implementation.
   - Lifecycle work added only inside `BaseTrainer.forward_backward_step()` is skipped by these trainers. Update every supported override or reject the unsupported trainer explicitly.

25. **Kernel handles belong to model instances**
   - Modeling classes construct `VeomniOp` handles from `self.config.ops_config` and store them on the instance. Do not introduce mutable module-level dispatch callables.
   - Model-specific input normalization and loss policy stay in `models`; the registered wrapper remains a tensor-level contract shared by every consumer.
   - Attention-free containers and embedding/projecting modules override HF's `_check_and_adjust_attn_implementation` to return the canonical selection. Do not advertise a backend capability they do not execute, preload its kernel on the container, or bypass checks on the shared model base; attention-owning components validate themselves. A model with its own deferred dispatch (MiniMax packed attention) may return the canonical selection from this hook only when its bind boundary validates the backend before execution.
   - Native configs created inside VeOmni models must explicitly opt in before calling VeOmni helpers; the native HF classes are never globally patched. Remote adapters export a native copy to preserve their source files and `auto_map`.

26. **DCP full resume skips HF weight materialization**
    - When `train.checkpoint.load_path` is set and the run is not LoRA/PEFT, `BaseTrainer` / omni train pass `should_skip_hf_weight_load=True` into `build_parallelize_model`, which forwards it to `parallelize_model_fsdp2` / `parallelize_model_ddp`.
    - The model is materialized without an HF weight read; parameters are restored by DCP in `CheckpointCallback.on_train_begin`.
    - Materialize through `_to_empty_preserving_nonpersistent_buffers()`, never bare `to_empty()`, on the random-init path as much as the resume path. `init_empty_weights()` patches `register_parameter` only, so a meta-built model holds *real* buffer values and `to_empty()` swaps every one for uninitialized memory. What restores them is narrower than it looks: DCP saves `state_dict()`, which omits `persistent=False`, and HF's `_init_weights` recomputes a rope table only for a module exposing `original_inv_freq` — which leaves Gemma3's per-layer-type `{type}_inv_freq`, its `embed_scale` and the Omni audio tower's sinusoidal `positional_embedding` with nothing behind them. A buffer built from a parameter is itself on meta, has no data to copy out of, and is skipped with a warning; no model registers one today. The same rule applies to the HuggingFace-load path in `veomni/models/checkpoint/weights.py`.
    - LoRA/PEFT must not set `should_skip_hf_weight_load` (and `_materialize_and_load_weights()` raises if both are set): LoRA DCP is trainable-only and still needs the HF base from `model.model_path`.
    - After DCP load, `empty_cache()` is called to reduce first-step NCCL OOM risk from allocator fragmentation on near-OOM MoE jobs.

## Environment Reproducibility

27. **MagiAttention is an SM100+ optional extra**
    - MagiAttention requires `--extra gpu --extra magi` and NVIDIA SM100 or newer.
      SM90 uses FlexAttention's FA4 backend instead.

28. **In-place collective reductions in backward must own their gradient buffer**
    - Autograd can pass the same incoming gradient to multiple branches. `.contiguous()` does not copy an already contiguous tensor, so reducing that tensor in place can silently change a sibling branch's gradient and the caller's `grad_outputs`.
    - `_Gather.backward` uses out-of-place NCCL reduce-scatter (`reduce_scatter_tensor` for equal shards, `reduce_scatter` for uneven shards) into a freshly allocated local `grad_input` buffer, and clones `grad_output` (`grad_output.clone(memory_format=torch.contiguous_format)`) before running an in-place `dist.all_reduce` on other backends (`_GatherConcatSP.backward` likewise clones before `dist.all_reduce`). Regression tests in `tests/parallel/ulysses/test_all_gather.py` cover shared gradients across layouts for `_Gather`.

29. **Saved RoPE table metadata must stay stable across checkpoint recomputation**
    - FSDP2 can cast fixed vision cos/sin tables to BF16 on the first block forward while non-reentrant checkpoint recomputation sees the original FP32 tables. Saving them unchanged makes checkpoint backward raise a metadata mismatch.
    - Full RoPE eager and Liger adapters normalize fixed tables saved for backward to `q.dtype`, independently of the forward compute tables. Trainable tables retain the eager gradient path.
    - Cover both table-dtype change directions and text/vision layouts in `tests/ops/rope/test_rope.py`. `tests/models/transformers/qwen/test_qwen2_vl.py` additionally runs the default Liger vision path with real two-rank FSDP2 and BF16 mixed precision.

## FlexAttention

Flex is a package under `veomni/ops/kernels/attention/standard/flex/`: `__init__.py` owns dispatch/Ulysses, `_inputs.py` owns input handling, `_fa4.py` owns compiled execution, and `_fa4_sm90_compat.py` owns the pinned bounds shim. The mask builder remains in `attention/mask/flex.py`.

30. **FLASH compilation uses tensor-derived bounds**
    - BlockMask construction has a separate cached `torch.compile(create_block_mask)` using automatic dynamic generalization. Test the real public `cu_seqlens` path, not only attention with prebuilt masks or compile spies. Keep packed compilation enabled by default; a Python length argument alone does not prove unbounded recompilation. Explicit eager construction can allocate token-square intermediates; see [compiled masks](../../docs/transformers_v5/veomni_attention_interface.md#compiled-mask-construction).
    - Always use `dynamic=True, fullgraph=True`, with no compile-mode parameter or internal static fallback. Derive bounds from normalized, post-Ulysses Q/K shapes inside the wrapper; preserve sparse buffers and `mask_mod`, and require exact equality with the caller's original bounds.
    - Keep cache exhaustion explicit; do not raise global cache limits to handle variable lengths. Outer whole-model dynamic compilation and additional Python scalar captures remain unsupported; dtype/head/callback guards can still specialize.
    - Prefer indexed per-token metadata buffers. The pinned CuteDSL lowering can also fail on zero-dimensional scalar-tensor captures; converting a Python scalar to a scalar tensor alone does not establish callback compatibility.
    - See [dynamic compilation](../../docs/transformers_v5/veomni_attention_interface.md#dynamic-compilation) for the supported compile boundary and mask predicate design.

30a. **Q/K/V layouts and visibility are explicit**
    - Reject `output_attentions=True` at the shared Flex entry before Ulysses or backend selection. The second result is optional LSE, never attention weights; returning LSE from Triton does not satisfy an attention-weight request.
    - Accept matching 3D `[T,H,D]` or 4D `[B,H,S,D]` inputs. Normalize 3D with views before validation, dispatch and Ulysses; never infer axes from sizes. Keep Q/K lengths independent and K/V lengths/heads consistent.
    - BlockMask owns packed-document visibility. Restore output/LSE to the caller's layout after inverse exchange; see [input layouts and mask semantics](../../docs/transformers_v5/veomni_attention_interface.md#input-layouts-and-mask-semantics).

30b. **Pinned FA4 auxiliary callbacks require bounds protection**
    - Install the version/signature-gated `4.0.0b16` compatibility forwarder before compilation. Auxiliary metadata reads must be safe for padded tile positions; validate with allocator-disabled memcheck (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`).
    - Apply Q-only protection only on physically padded Q tiles (account for PackGQA). Check K completeness on every Q tile before either fast path. Q-only callback clamping requires a proven complete K tile; ragged/negative K tiles retain native full bounds. Never use the incoming false flag as proof of K bounds.
    - Use separate JIT-cache namespaces and bump them when shader semantics change. Retain old cache objects and prepare both replacement caches before publishing installation.
    - Native b16 omits broadcast strides from its forward cache key. Materialize zero-stride Q/K/V views after KV expansion, bump the cache namespace, and test mixed layouts through the same production cache. Automatic FLASH routing must retain Triton for unsupported sparse dimensions, missing full-block tables, architecture-mismatched BlockMask tiles (128×128 on SM90; 256×128 on SM100/110), and trainable score bias. Explicit backend selection retains native validation.
    - See [FA4 partial-tile compatibility](../../docs/transformers_v5/veomni_attention_interface.md#fa4-partial-tile-compatibility) for the mechanism, hardware validation scope and upgrade requirements.

30c. **Framework callbacks must preserve FLASH's scalar and index contracts**
    - Sliding-window, softcap and offset callbacks use device buffers indexed by token positions, including uniform broadcast views. A constant `buffer[0]` index fails the pinned CuteDSL SSA conversion. Skip zero offsets entirely.
    - Represent packed rows as metadata tensor dimensions; precompute flattened-coordinate arithmetic outside the callback. The pinned lowering rejects symbolic row-width arithmetic (`index_expr`), and even a dead integer comparison can leave a rejected capture. Preserve the caller's index dtype when forwarding mapped positions into another predicate (Int32 on FLASH).
    - Cover the public builder and real forced FLASH outputs/QKV gradients/graph reuse, including changing scalar values, packed rows, cached offsets, default mask compilation and outer static compilation.
    - Build masks with invocation-local predicates and explicit `create_block_mask` arguments. Never temporarily replace HF module helpers: overlapping exits can restore another invocation's device/length closure and affect native HF consumers. Preserve pinned HF padding/offset composition; test overlapping success/error paths and native HF calls during construction.

## DeepGEMM FP8 all-gather

- Native Torch 2.11 nonzero-dimension FSDP copy-out reshapes each extension
  output using the parameter shard shape. Compact block scales must therefore
  use their own all-gather; only the same-shaped FP8 payload goes through the
  native extension output list. Do not add scales as a second output for Shard(1).
- Weight shard boundaries must align to complete 128x128 blocks. Preserve
  BF16 rounding before quantizing FP32 master shards to match BF16 parameter
  all-gather numerics. Quantizing FP32 directly changes scales and rounding.
- Install parameter extensions after materialization and before FSDP lazy
  initialization. Normal full reshard uses `post_forward_mesh_info is mesh_info`;
  a non-None post-forward mesh alone does not mean integer partial resharding.
- Keep the FP8 container opaque outside DeepGEMM; optimizer and checkpoint
  operations use the ordinary master shards. Do not move/replace wrapped
  parameters after hook registration.
- Transport and copy FP8 payloads as `uint8` views, restoring E4M3 views for
  GEMM. The generic ATen float8-copy path can spend more time converting and
  moving values than FP8 all-gather saves. Use the tiled byte transpose for
  dgrad; square block scales transpose without requantization.
