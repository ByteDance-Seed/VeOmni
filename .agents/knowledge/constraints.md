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

7-outer. **SP is UNIFORM across the outer trainer and every module (Arch B); the dataloader REPLICATES each DP shard across the SP group**
   - **Decision (2026-07-20, implemented):** SeedOmni V2 uses classic single-pass Ulysses at ONE shared SP size (Arch B). The earlier looped per-module SP (Arch A: outer SP=1, per-sample loop + gather-to-owner + activation offload/ckpt) is deleted. Rationale + experiments: `docs/seed_omni/module_level_sp.md`, `docs/seed_omni/sp_loop_memory_experiments.md`.
   - The outer trainer carries the SP size via the **top-level** `accelerator.ulysses_size` (CLI `--accelerator.ulysses_size`). For `OmniArguments`, only the top-level `accelerator` block is canonical — do not nest accelerator under `train`. Its dataloader is the standard build-time sharded loader: it yields `dp_size = world / sp_size` **distinct** shards and **replicates** each shard across its SP group (no per-modality slicing in the collator — slicing happens inside each SP module). So every rank of an SP group holds the SAME sample.
   - **SP is per-module:** each module's ``accelerator.ulysses_size`` / ``enable_async`` come from its own YAML (or model-runtime defaults merged at resolve time). There is no launcher-wide SP override onto all modules — future teacher/student model runtimes keep independent parallel states.
   - Consequence: a module with `sp_size = S > 1` **slices** the replicated sample to its `1/S` shard inside its own `pre_forward` (an `if get_parallel_state().sp_size > 1:` branch), runs the endpoint ONCE (attention all-to-alls over the SP group internally), then **all-gathers** the output back to the full sample on every rank inside its own `post_forward` (same branch → `gather_outputs`). There are NO separate `sp_pre_forward` / `sp_post_forward` hooks — SP is fully contained in the module's `pre_forward` / `post_forward` (matching veomni v1 single-model SP). Forward AND backward both peak at ≈ `1/S` — no loop, no per-sample checkpoint, no CPU offload. Modeling stays SP-unaware.

7a. **Classic single-pass Ulysses: replicated input → slice → one forward → all-gather output**
   - Because all SP ranks hold the SAME replicated sample, their packed tensors have identical shape and dtype — the distinct-per-rank dtype-unify hazard of the old `sp_gather_seqs` path no longer exists (that whole `_GatherConcatSP` / `_sp_unify_dtype` machinery is deleted).
   - The `pre_forward` SP branch pads the sample to a multiple of `sp_size`, rebuilds any varlen/`cu_seqlens` metadata over the FULL padded sequence, and hands this rank only its contiguous `1/sp` chunk (`sp_pad` + `slice_input_tensor`, or `sp_pad_and_slice`). The `post_forward` SP branch all-gathers the shards (`gather_outputs`, autograd-aware `_Gather`) and narrows off the SP pad, then falls through to the SP-agnostic write-back. Downstream nodes then run identically on the replicated full data.
   - **Gradient correctness:** `_Gather.backward` all-reduces (SUM) the grad over the SP group then takes this rank's shard — the `×sp` this introduces is cancelled by FSDP2's `÷|dp_shard_sp|` grad averaging (params are sharded/reduced over a mesh that INCLUDES sp), yielding gradients identical to the non-SP baseline. This is the same invariant that makes VeOmni's single-model Ulysses correct; see 7b for the loss side.

7b. **Omni decode losses reduce over `dp_sp` (`fsdp_group`), NOT just `sp_group` — token-weighting must be global for SP transparency**
   - Per-module SP is *accuracy-transparent*: the loss/gradient must be identical no matter how the global batch is split into DP vs SP. A per-rank `ce_sum / n_valid` (the classic DP path) gives a **mean-of-means** that over-weights ranks holding few valid tokens — e.g. one rank with 8 text tokens @ loss 13.1 and another with 1926 @ loss 1.6 both get 1/4 weight, so `sp1` reports `text_encoder.decode` ≈ 4.47 while `sp4` (token-weighted) reports the correct 1.55. Same batch, ~3x apart.
   - Fix (`text_encoder/modeling.py::decode`, `janus/vqvae/modeling.py::_vq_loss`): **always** compute `local_mean = ce_sum / n_valid_local` then `reduce_sequence_parallel_loss(local_mean, n_valid_local, group=ps.fsdp_group)`. `fsdp_group` is the `dp_sp` mesh (dp_replicate × dp_shard × sp) = exactly the mesh FSDP2 reduce-scatter (÷dp_shard_sp) + HSDP all-reduce (÷dp_replicate) average grads over, so `ReduceLoss`'s backward `×|dp_sp|` cancels it and yields the true **global token-weighted** gradient. Under Arch B the SP ranks hold the SAME replicated sample, so their `(ce_sum, n_valid)` are identical; the `dp_sp` reduce then weights each distinct DP shard once and the sp replicas cancel — same result as the non-SP baseline (`max|dloss|` 2.95 → 0.06).
   - **Consequence — the collective spans every DP rank, so decode must run symmetrically on ALL `fsdp_group` ranks.** Any per-rank early-return before the reduce (e.g. vqvae's old `is_dummy` fast-path) will desync/hang when dummy-ness varies across DP. `vqvae.decode` therefore always routes through `_vq_loss`; an all-dummy span is handled by its `-100` labels (CE 0, clamped denom → 0.0, `generation_head` still runs as the FSDP anchor).
   - **The clamped denominator is the caller's job — no fused CE backend provides it.** Liger divides by its own `n_non_ignore` and `chunk_loss` recounts the labels, both unclamped, so handing a fully-`-100` span to `self.loss_function` yields `0/0` = NaN. `ReduceLoss` masks that rank's NaN out of the forward value but its backward still multiplies the kernel's NaN grad by 0. `text_encoder/accelerated.py::decode` therefore counts supervised tokens itself and routes a zero-count span to the eager branch with an explicit `num_items_in_batch=n.clamp(min=1)` — matching the native `decode` exactly (0.0, still graph-connected so the head's zero grads exist on every rank). Any new fused-loss call site owes the same guard. This changes DP loss semantics for Omni training from mean-of-means to global token-weighted (bagel excluded, out of SP scope).

7c. **Per-module metric meter must stash PRE-slice (full-sample) seqlens in `pre_forward`, not measure the sliced forward `data`**
   - `MetricMeterMixin` no longer reads the module's forward `data` for token counts. Each metered module calls `metric_meter_set_seqlens(method, seqlens)` **inside `pre_forward`, before the SP slice** on `MetricMeterMixin`; the default `metric_meter_token_lengths` just drains that stash.
   - **Why pre-slice / full-sample:** the `pre_forward` SP branch slices to this rank's `1/sp` shard, so a length read *after* that branch under-counts by ~`sp` (always call `metric_meter_set_seqlens` BEFORE the `if sp_size > 1:` slice). Stash the FULL (pre-slice) per-sample lengths instead: `OmniEnvironMeter.add` reduces tokens **and** `total_flops` over the **`dp_group`** (`omni_helper.py`), which EXCLUDES the sp dim — so the sp ranks that all hold the same replicated sample are not double-counted, and the full-sample value counted once per DP shard reconstructs the global total. This holds for FLOPs/MFU too (each device does `1/sp` of the work; `world_size` in the MFU denominator — which includes sp — absorbs the sp factor, matching the non-SP run).
   - Verified SP-transparent: `sp1` == `sp4` per-module `trace/<module>/consume_tokens` match to the digit. A regression here shows up as tokens/MFU scaling with `sp`, not as a crash.

7e. **DDP never broadcasts buffers (FSDP2 / HSDP parity); average grads over `fsdp_group` in clip**
   - Under uniform SP a DDP-wrapped module (e.g. `janus_siglip`) still runs a single forward/backward (classic single-pass Ulysses, no per-sample loop). DDP wraps with `process_group=dp_group`; enabling SP shrinks `dp` (`dp_size = world / sp_size`), so at `dp_size == 1` that allreduce is a no-op while each rank still holds gradients from only its `1/sp` slice — the optimizer then steps on a partial gradient, silently, with no shape error. FSDP2 already syncs over `fsdp_group` (`dp_sp`); for DDP, `veomni_clip_grad_norm` / `veomni_omni_module_clip_grad_norm` all-reduce `p.grad` over `fsdp_group` when `sp_size > 1` before clipping. `dp_sp` spans `dp_replicate × dp_shard × ulysses × cp`, so one reduction covers plain DP, HSDP and SP alike and leaves the two dp modes numerically equivalent. It is not a double count on top of DDP's own `dp` reduction — every rank of a `dp` group enters with the same value, so the second average reproduces the plain mean over `dp_sp`. Any new grad-clip entry point that can see a DDP-wrapped module owes the same reduction.
   - Reduce with `SUM` then divide — **not** `ReduceOp.AVG`, which the NPU backend does not support (same reason as `veomni/utils/dist_utils.py::all_reduce`).
   - Reduce over `dp_sp` even though `sp` alone would be arithmetically identical whenever DDP has already averaged over `dp`. `dp_sp` costs a wider group — `dp` is usually the cross-node dimension — and buys independence from *whether* DDP reduced: `no_sync()`-style gradient accumulation would leave `dp` peers unreduced at clip time and make the narrow version silently wrong in exactly the way this constraint exists to prevent. Revisit only with a benchmark. Both versions issue one collective per parameter tensor, uncoalesced; bucketing (as DDP itself does at 25 MB) is the obvious optimization if this ever shows up in a profile.
   - Select the gradients to reduce by `requires_grad`, not by `grad is not None`, and zero-fill a missing one. It is one collective per gradient, so a parameter that goes unused on some ranks only would otherwise desynchronize the sequence and hang — and the wrap passes `find_unused_parameters=True`, so DDP itself tolerates exactly that case and will not fail first.
   - Convert a DTensor grad norm with `full_tensor()` before `.item()`, as the FSDP2 path does: `.item()` on a sharded or partial DTensor reads this rank's piece rather than the global norm. Only the *reported* norm is at stake — `clip_grad_norm_()` computes and applies the clip internally, and does so globally for DTensors. Defensive either way: `build_parallelize_model()` calls `parallelize_module()` with no `parallelize_plan`, which torch warns about and treats as a no-op, so nothing on the DDP path is a DTensor today. Same for the `to_local()` in `_allreduce_ddp_sp_grads()`.
   - Pass `broadcast_buffers=False` **unconditionally** — NOT conditioned on `sp_size`, which was unrelated to the hazard it guarded. The reason is dp_mode parity: `ParallelState.fsdp_mesh` treats DDP as HSDP with a single `dp_replicate` dim, and FSDP2/HSDP has NO buffer sync at all (torch's whole `distributed/fsdp/` tree contains no `_sync_module_states` / `_broadcast_coalesced`; `fully_shard` only places buffers on the device). A module's buffer behaviour must not change with the `fsdp_mode` a config happened to pick.
   - Nothing is lost by not broadcasting. Config-derived buffers are already per-rank correct: SigLIP `position_ids` and a static rope table come out identical on every rank, and dynamic-rope `inv_freq` is deliberately recomputed from *this* rank's sequence length (`modeling_rope_utils.py` re-registers it during forward), so pushing rank0's copy would actively CORRUPT the others. Buffers get no gradients, so the gradient allreduce is not a fallback for them — but neither is the broadcast.
   - Genuinely replicated mutable state (`nn.BatchNorm*` running stats) is NOT fixed by `broadcast_buffers=True` either: it overwrites every rank with rank0's copy, i.e. DISCARDS the other ranks' statistics instead of aggregating them, and it breaks any module owning ≥2 graph nodes — `call_graph_endpoint` enters the wrapper once per node (`encode`/`decode` are trampolined through `wrapped(**kwargs)`), and the in-place pre-forward `copy_` then hits a buffer the first node's autograd graph saved for backward (PyTorch #22095 / #66504). `SyncBatchNorm` is the one real fix: it all-reduces the statistics inside forward, so it is immune to the FSDP2 gap, the HSDP replicate dim and the ≥2-node DDP case alike. Under SP its `process_group` must span dp+sp, since Ulysses splits the sequence and `dp_group` excludes SP peers. The repo currently has zero plain `nn.BatchNorm*` (generative decoders use GroupNorm, which has no buffers), so this is forward-looking, not a live bug.

7d. **SP / DP / CP groups are ParallelState-local (resolved from the current state's mesh), NOT module-level globals**
   - Why this matters: the global attention integration (`veomni_flash_attention_2_with_sp`) has no handle on the `ParallelState` — it reaches its all-to-all group only through the process-level getter `get_ulysses_sequence_parallel_group()`. Each Omni module builds its OWN `ParallelState`/device mesh (hence its own SP subgroup object), even though uniform SP means they share one SP *size* (7-outer). If the getter read a mutable global slot set once at build time, the last-built module would win the slot and every other module's attention would all-to-all over the wrong ranks (silent corruption, or a `split_with_sizes`/shape mismatch). Resolving from the current state avoids this and keeps modules independent.
   - Fix (the principled one): `comm.py` keeps NO group globals. `get_ulysses_sequence_parallel_group` / `get_unified_sequence_parallel_group` / `get_context_parallel_group` / `get_data_parallel_group` resolve from the *current* state's `{ulysses,sp,cp,dp}_group` (the state's device-mesh subgroup) — exactly how `fsdp_group` already worked. The helper `_current_state()` reads the `_PARALLEL_STATE` module global **directly** (not via `get_parallel_state()`), so an UNINITIALIZED process resolves to `None` — i.e. "no groups" — and every getter is `None`-safe (`ps.X_group if ps is not None else None`). This avoids `get_parallel_state()` constructing a default `ParallelState` that validates against the world size and raises for pre-init / unit-test code. Since `use_parallel_state` (via the graph `module_context`) already scopes `_PARALLEL_STATE` per module for its forward *and* its grad-checkpoint recompute (`_scope_recompute_to_parallel_state`), each module's attention automatically gets its own group with zero key bookkeeping. The `ParallelState.{dp,sp,ulysses,cp}_group` properties are mesh-only (return `None` when the dim is absent) so the getters never recurse.
   - Consequences: (a) meshless SP init is gone — `ParallelState.__post_init__` raises if `sp_enabled and device_mesh is None`; always build via `init_parallel_state`. (b) The `_ULYSSES_GROUP_KEY` registry, `init_sequence_parallel`, `UlyssesGroupKeyManager`, `get_ulysses_group_key_context`, `*_by_key`, cpu-group and `set_{data,context,unified}_*` setters were all deleted. (c) ALL group setters are gone, including the former `set_ulysses_sequence_parallel_group` test seam (no `_ULYSSES_SP_GROUP_OVERRIDE`). SP unit tests (`tests/parallel/ulysses/*`) build a real state via `init_parallel_state(dp_size=1, ulysses_size=world_size)` and switch to the no-SP reference path with `set_parallel_state(None)` — no group injection into production code. (d) The SP slice/gather helpers (`slice_input_tensor` / `gather_outputs`, plus `sp_pad` / `sp_pad_and_slice`) resolve their group from `ps.sp_group` (the scoped module state) or the current-state getters; there is no orchestrator-state read.
   - Standalone trainers register the `"base"` state in `BaseTrainer._setup()` before determinism environment variables, then use `use_parallel_state("base")` for the whole build and each ambient-dependent runtime operation (forward, post-forward loss, backward, and clipping). `TextTrainer`, `VLMTrainer`, `DiTTrainer`, and `TextDPOTrainer` mirror this because they compose `BaseTrainer` via `__new__`. Callback objects capture their own state at construction, so hook dispatch does not depend on an ambient scope. SeedOmni V2 (`veomni.trainer.omni`) uses the SAME named-registry mechanism: each `ModuleRuntime._setup()` registers its ParallelState under its **module name** (`self.module_name` — the single identity that is BOTH the registry key and the `<module>/` checkpoint subdir, unique per `OmniConfig` and distinct from `"base"`) via `init_parallel_state(name=self.module_name)`, and every scope site re-enters it by name — `use_parallel_state(self.module_name)` (module build, grad-ckpt recompute, `OmniModel.module_context`, per-module optimizer/lr-scheduler build). The scoping is the module runtime's OWN concern: its `_build_optimizer` / `_build_lr_scheduler` enter their state internally (via `self._scoped()`) and it exposes `clip_grad_norm()` + a read-only `parallel_state` property (a registry lookup by name, NO stored handle), so `OmniTrainer` calls these plainly and NEVER wraps the module's private mesh — per-module grad-clip calls `mt.clip_grad_norm(max_norm)`, and any SP-size read goes through `mt.parallel_state.sp_size`. `is_parallel_state_registered(name)` gates which modules `OmniModel` scopes (see below). The registry is the single source of truth for the ParallelState object; `OmniModel` only stores the *set of registered names* (`set_module_parallel_state_names`, populated with the distributed modules — eager inference modules stay unscoped), not the objects. Registration never overwrites the orchestrator's current global state (`init_parallel_state` only mutates the default global when it is unset). Module names being unique + distinct from `"base"` means no registry collisions; same-topology modules legitimately share one cached state object (same groups).
   - Cache robustness: `init_parallel_state()` caches states by topology in the module-level `_PARALLEL_STATE_CACHE` and only (re)establishes the default `_PARALLEL_STATE` global when it is `None` — on both the build path and the cache-hit path (a same-topology hit no longer asserts the global is set, since tests reset `_PARALLEL_STATE = None` at teardown without clearing the cache).
   - Verified: uniform SP4 across ViT + text-encoder + LLM (`qwen3_0.6b/modules_train_visual_instruction_tuning.yaml` + `--accelerator.ulysses_size 4`) trains cleanly (loss decreases, no shape mismatch). A regression here surfaces as a wrong-ranks all-to-all (`split_with_sizes` mismatch or delayed loss NaN).

7f. **DDP must materialize and load meta-init weights itself, before the wrap**
   - `model.accelerator.init_device` defaults to `"meta"` and only `fsdp_mode == "fsdp2"` is asserted to use it, so a `ddp` config reaches `build_parallelize_model()` with an empty model. DDP registers gradient hooks and broadcasts rank0's parameters, but it materializes nothing and loads nothing, and `BaseTrainer` has no load step of its own — so `parallelize_model_ddp()` owns that pass, exactly as `parallelize_model_fsdp2()` does. Both call `_materialize_and_load_weights()`, which is the single place where the choice between random init, an HF snapshot and a checkpoint resume is made; a new dp mode owes the same call. Omit it and DDP's constructor dies on `Tensor.item() cannot be called on meta tensors`.
   - The gate is `param.is_meta`, not `init_device`. The flag states an intent the model builder is free to ignore — `tests/data/*` construct their model eagerly and leave the flag at its `meta` default — and materializing a model that already holds real weights discards them, or raises outright on a plain `nn.Module` with no `init_weights`. A model built under `cuda` arrives with weights already loaded (`empty_init=False` in `veomni/models/auto.py`) and must be left alone. `parallelize_model_fsdp2()` keeps its call unconditional because `arguments_types.py` asserts `init_device == "meta"` for fsdp2, so a real model cannot reach it.
   - `init_device == "cpu"` is refused for `ddp`, by an assert in `AcceleratorConfig._validate_init_device()` alongside the one that pins fsdp2 to `meta` — not in `parallelize_model_ddp()`. Parse time is the right place: every rank fails together, before a model is built or a snapshot read. It never worked for the wrap: `device_ids=[local_rank]` has been passed since the first commit, and torch rejects that together with a CPU module (`torch/nn/parallel/distributed.py`), so rank0's CPU replica cannot be wrapped while every other rank builds empty and skips the load (`veomni/models/loader.py`). It was FSDP1's `sync_module_states` recipe (rank0 reads, the wrapper broadcasts), dropped in #756; fsdp2 replaced it with `broadcast_model_weights_from_rank0`.
   - Dropping `"cpu"` from the field's `Literal` is *not* what enforces this, and no `Literal` in the arguments layer enforces anything. The parser turns it into argparse `choices`, which covers the CLI only; a YAML value goes straight to the dataclass constructor via `_instantiate_recursive()` (`veomni/arguments/parser.py`), and annotations are never checked at runtime. Any config value that must actually be rejected needs an explicit assert in `__post_init__`.
   - Two other users of `"cpu"` are unaffected and must not be swept up in that: `build_foundation_model(init_device="cpu")` is a live public API (several tests build on CPU that way), and `materialize_device="cpu"` is how fsdp2 CPU offload reaches `load_model_weights()`. The `init_device` parameters in `veomni/models/module_utils.py` are fed by the latter, not by `model.accelerator.init_device`.
   - `should_skip_hf_weight_load` must be honoured here too: a distributed-checkpoint resume is about to overwrite every parameter, so reading the HF snapshot doubles peak memory, and the snapshot may not exist at all.
   - `broadcast_model_weights_from_rank0` is honoured here, and the `AcceleratorConfig._validate_init_device()` warning that used to call it fsdp2-only is gone. That warning was correct only while DDP loaded nothing at all; now that this path loads, the flag applies verbatim — `rank0_load_and_broadcast_weights()` broadcasts over the default (world) group from global rank0 (`dist.broadcast(..., src=0)`, no `group=`), and a DDP replica wants exactly that whole tensor. It defaults to `True`, so ignoring it would have pinned every DDP run to the every-rank-reads path while printing that it was being ignored.
   - After the load pass, `parallelize_model_ddp()` re-checks `param.is_meta` and raises with the offending parameter names. A loader that leaves one behind would otherwise surface inside DDP's constructor as `Tensor.item() cannot be called on meta tensors`, which names neither the parameter nor the cause. A test that stubs the loader must make the stub materialize, exactly as a real one does.
   - ExtraParallel is refused on the DDP path, keyed on **the model's plan** (`_has_extra_parallel_plan()`), not on `ParallelState.any_extra_parallel_enabled`. Only `parallelize_model_fsdp2()` applies the plan that shards expert weights, so DDP experts are whole and loading a sharded-expert config into them — previously prevented only by the meta crash — would silently produce full tensors. But the mesh alone does not identify such a model: a SeedOmni V2 sub-module's accelerator is `_deep_update(global, override)`, so a DDP vision tower inherits the backbone's ep dim while owning no experts, and refusing on the mesh would block it. The same predicate gates `ep_sharded_stream_load` in `_materialize_and_load_weights()`: a plan-less model skips the fast path with a log line (there was never one to take), while a model that *does* have a plan still lets the loader's `NotImplementedError` propagate, because that one means the checkpoint layout is unsupported — the distinction `tests/utils/test_moe_ep_sharded_load_matrix.py` pins for `nonmerged x ep_sharded`.
   - `weights_path` is one snapshot path or `None`, never a per-sub-module mapping. D2.2 added a `Mapping[str, str]` branch keyed on `model.named_children()`, for a planned D2.3 in which OmniTrainer wrapped the whole OmniModel once; D2.3 instead parallelizes one `ModuleRuntime` at a time (`module_runtime.py`), each with its own `args.model_path`, so the branch never acquired a caller and was removed. A future top-level wrap can restore it from that commit — but a sub-module that needs its own snapshot should get its own `build_parallelize_model()` call, which is what every caller does today.
   - A fully frozen module is materialized and loaded, then returned **unwrapped**: torch's DDP rejects a module with no trainable parameters, while the fsdp2 path accepts one. Seedream's `offline_cache` OE/ViT/VAE run this way.

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
    - Extra state is saved per-rank via `_EXTRA_STATE_FORMAT` — changing rank count requires checkpoint resharding.

17. **Checkpoint save/load requires all ranks to participate**
    - DCP operations are collective — all ranks must call save/load simultaneously.
    - Calling checkpoint operations from only rank 0 causes deadlocks.

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

22. **NPU (Ascend) code paths require guards**
    - NPU-specific code must be guarded with `is_torch_npu_available()` or `IS_NPU_AVAILABLE`.
    - NPU kernels live in `veomni/ops/kernels/{rms_norm,rotary}/npu.py` and `veomni/ops/platform/npu/` — they must not be imported on GPU-only environments.

23. **Device-agnostic code must use `veomni.utils.device` helpers**
   - Use `get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()` instead of direct `torch.cuda.*` calls.
   - Direct CUDA calls break NPU compatibility.

## Trainer Extensions

24. **Trainer callback lifecycle changes must cover composed trainers**
   - `TextDPOTrainer` and `DiTTrainer` compose a `BaseTrainer` and override `forward_backward_step()`; they do not inherit the base implementation.
   - Lifecycle work added only inside `BaseTrainer.forward_backward_step()` is skipped by these trainers. Update every supported override or reject the unsupported trainer explicitly.

25. **Module-level OpSlots are shared by every model instance**
   - Modeling modules expose `OpSlot` objects such as `veomni_causal_lm_loss` as globals. Policy/reference models in DPO can therefore use the same slot.
   - Temporary interception must use forward-scoped ownership and reference-counted dispatch. A closure bound to one model or callback can observe another model's forward and corrupt side-channel state.

26. **DCP full resume skips HF weight materialization only when DCP owns the model restore**
    - `BaseTrainer` combines the caller's permission with the generic resume check: `effective_skip = caller_skip and should_skip_hf_weight_load(load_path, lora_config)`. A caller veto (`False`) must short-circuit the generic helper; the default (`True`) preserves standalone trainer behavior. The value is forwarded through `build_parallelize_model` to `parallelize_model_fsdp2` / `parallelize_model_ddp`.
    - When the effective value is `True`, the model is materialized without an HF weight read and its parameters must be restored by DCP in `CheckpointerCallback.on_train_begin`.
    - Materialize through `_to_empty_preserving_nonpersistent_buffers()`, never bare `to_empty()`, on the random-init path as much as the resume path. `init_empty_weights()` patches `register_parameter` only, so a meta-built model holds *real* buffer values and `to_empty()` swaps every one for uninitialized memory. What restores them is narrower than it looks: DCP saves `state_dict()`, which omits `persistent=False`, and HF's `_init_weights` recomputes a rope table only for a module exposing `original_inv_freq` — which leaves Gemma3's per-layer-type `{type}_inv_freq`, its `embed_scale` and the Omni audio tower's sinusoidal `positional_embedding` with nothing behind them. A buffer built from a parameter is itself on meta, has no data to copy out of, and is skipped with a warning; no model registers one today. Note `veomni/models/module_utils.py` has the same unguarded pattern on the HF-load path.
    - SeedOmni V2 decides this per `ModuleRuntime` (`ModuleRuntime.skip_hf_weight_load`, consulted by `_build_parallelized_model`). A fully frozen module with a non-empty `state_dict()` gets NO checkpoint manager from `_init_checkpoint`, so nothing restores its persistent model state; it must veto the skip and load its released HF weights even during resume. The property is read after `_freeze_model_module`, so `has_trainable_parameters` already reflects freezing + LoRA.
    - A parameterless Omni module with an empty `state_dict()` has no persistent model state to restore and may allow the BaseTrainer resume fallback to skip HF loading.
    - LoRA/PEFT must not set the effective skip value (and the shared materialize helper raises if both are set): LoRA DCP is trainable-only and still needs the HF base from `model.model_path`.
    - After DCP load, `empty_cache()` is called to reduce first-step NCCL OOM risk from allocator fragmentation on near-OOM MoE jobs.

## Environment Reproducibility

27. **Exact uv synchronization removes separately installed overlays**
    - The MagiAttention SM90 CUTLASS overlay is installed by `scripts/kernel/install_magi_sm90.sh` after the locked GPU environment. Reinstall it after a later exact `uv sync` before running MagiAttention on SM90.
