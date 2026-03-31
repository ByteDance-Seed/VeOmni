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
   - Files under `veomni/models/transformers/*/generated/` are created by `python -m veomni.patchgen.run_codegen`.
   - Manual edits are silently overwritten on the next patchgen run.
   - To change generated behavior, edit the patch spec (`patch_spec.py`) or the modeling patch file (`modeling_*_patch.py`).

4. **Transformers version: dual-track (v4.57.3 / v5.2.0), new development targets v5**
   - VeOmni currently supports two transformers versions via uv conflicts:
     - **Default**: `transformers==4.57.3` (dependency group `transformers-stable`, active by default)
     - **Experimental**: `transformers==5.2.0` (optional extra `transformers5-exp`)
   - Switch: `uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --dev`
   - **All new code must target v5 APIs.** v4 compatibility code exists but will be removed.
   - Version branching uses `is_transformers_version_greater_or_equal_to()` from `veomni/utils/import_utils.py`.
   - Key v4→v5 API changes:
     - `AutoModelForVision2Seq` removed → use `AutoModelForImageTextToText`
     - `no_init_weights` moved from `transformers.modeling_utils` to `transformers.initialization`
     - Model `__init__.py` files select `generated.patched_modeling_*` (v5) vs upstream + `apply_*_patch()` (v4)
   - Patchgen regeneration must be done with the target transformers version installed.

## Distributed Training

VeOmni supports FSDP1 and FSDP2, but **FSDP1 is legacy and will be removed**. All new development must use FSDP2.

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
   - Process groups: `comm.py` sets `ulysses_sequence_parallel_group`, `context_parallel_group`, `unified_sequence_parallel_group` from the device mesh.

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

11. **`attention_mask` sum = token count for dynamic batching**
    - Dynamic batching (`DynamicBatchingSizeDataset`, `DynBszBuffer`) uses `attention_mask.sum()` as the length function.
    - With FA varlen, `attention_mask` is expected to be all-ones over packed length; boundaries come from `position_ids` and `cu_seq_lens`.
    - When SP is enabled, `attention_mask` must use `sp_pad_value=1` (asserted in `MainCollator.__post_init__`).

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
    - NPU patches live in `veomni/ops/npu_patch/` — they must not be imported on GPU-only environments.

22. **Device-agnostic code must use `veomni.utils.device` helpers**
    - Use `get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()` instead of direct `torch.cuda.*` calls.
    - Direct CUDA calls break NPU compatibility.
