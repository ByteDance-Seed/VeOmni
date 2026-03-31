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

4. **Transformers version >= 5.2.0**
   - VeOmni is based on HuggingFace transformers v5. Some internal APIs changed between v4 and v5 (e.g., `AutoModelForVision2Seq` removed, `no_init_weights` moved to `transformers.initialization`).
   - Model patches in `veomni/models/transformers/` must be compatible with transformers v5 APIs.

## Distributed Training

5. **FSDP2 `fully_shard()` requires torch >= 2.4**
   - The `fully_shard` composable API is conditionally imported. Code that unconditionally calls it will crash on older PyTorch versions.
   - Always check: `is_torch_version_greater_than("2.4")` before using FSDP2 APIs.

6. **FSDP wrap policy must match model structure**
   - The wrap policy in `build_parallelize_model()` determines which modules become FSDP units.
   - Wrong granularity causes: too coarse -> OOM, too fine -> excessive communication overhead.
   - Each model's `parallel_plan.py` defines the correct wrapping boundaries.

7. **Gradient clipping must use `veomni_clip_grad_norm`**
   - Standard `torch.nn.utils.clip_grad_norm_` does not handle FSDP sharded parameters correctly.
   - VeOmni provides `veomni.distributed.clip_grad_norm.veomni_clip_grad_norm` which handles both FSDP and FSDP2.

8. **Sequence parallel requires consistent attention splitting**
   - `veomni/distributed/sequence_parallel/ulysses.py` splits input sequences across ranks.
   - Attention masks, position IDs, and KV caches must be split/gathered consistently.
   - Gathering outputs before loss computation is mandatory — partial outputs produce incorrect loss.
   - Use `gather_outputs()` from `veomni.distributed.sequence_parallel`.

9. **MoE expert parallel requires matching device mesh**
   - Expert parallel device mesh must be compatible with the FSDP device mesh.
   - `veomni/distributed/moe/` handles expert assignment — changing mesh topology without updating MoE configuration causes routing failures.

## Data Pipeline

10. **Data collator must match model modality**
    - `MainCollator` in `veomni/data/data_collator.py` dispatches to modality-specific collation.
    - Using a text-only collator on multimodal data (or vice versa) causes silent shape mismatches or crashes.

11. **`IGNORE_INDEX` (-100) for loss masking**
    - Labels set to `IGNORE_INDEX` are excluded from loss computation.
    - Custom data transforms must preserve this convention. Using a different ignore value silently corrupts training.

12. **Flash attention kwargs must be precomputed**
    - `add_flash_attention_kwargs_from_position_ids()` computes `cu_seq_lens` and `max_length` from `position_ids`.
    - These must be added to the batch dict before the model forward pass. Recomputing inside each attention layer causes host-device sync, hurting performance.

13. **Dynamic batching preserves per-sample boundaries**
    - `veomni/data/dynamic_batching.py` packs multiple samples into a single sequence.
    - Position IDs and attention masks must correctly reflect sample boundaries. Breaking boundaries causes cross-contamination between samples.

## Checkpoint

14. **DCP checkpoint keys must match model state dict**
    - `veomni/checkpoint/dcp_checkpointer.py` uses PyTorch's DCP (`torch.distributed.checkpoint`).
    - Renaming model parameters or changing the model structure between save and load breaks checkpoint loading.
    - Extra state is saved per-rank via `_EXTRA_STATE_FORMAT` — changing rank count requires checkpoint resharding.

15. **Checkpoint save/load requires all ranks to participate**
    - DCP operations are collective — all ranks must call save/load simultaneously.
    - Calling checkpoint operations from only rank 0 causes deadlocks.

## Code Quality

16. **Ruff must pass before commit**
    - `make quality` runs `ruff check` and `ruff format --check`.
    - Pre-commit hooks enforce this automatically (`pre-commit run --all-files`).

17. **All comments and docstrings must be in English**
    - No Chinese or other non-English text in code comments. This is enforced by project convention.

18. **PR title must follow format: `[{modules}] {type}: {description}`**
    - Modules: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`, `trainer`, `agent`
    - Types: `feat`, `fix`, `refactor`, `chore`, `test`
    - CI checks PR titles automatically (`check_pr_title.yml`).

## Hardware

19. **NPU (Ascend) code paths require guards**
    - NPU-specific code must be guarded with `is_torch_npu_available()` or `IS_NPU_AVAILABLE`.
    - NPU patches live in `veomni/ops/npu_patch/` — they must not be imported on GPU-only environments.

20. **Device-agnostic code must use `veomni.utils.device` helpers**
    - Use `get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()` instead of direct `torch.cuda.*` calls.
    - Direct CUDA calls break NPU compatibility.
