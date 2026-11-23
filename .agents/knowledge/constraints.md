# Hard Constraints

Cross-cutting invariants checked before every code change. Keep entries concise (2–4 lines); deep dives for specific subsystems live in `.agents/knowledge/topics/` and `docs/design/` (see table at bottom).

## Model Loading, Registry & Ops

1. **Import-time registration**: `MODELING_REGISTRY`, `MODEL_CONFIG_REGISTRY`, and `MODEL_PROCESSOR_REGISTRY` (`veomni/models/registry.py`) populate when model `__init__.py` modules are imported. Never defer registrations into functions, and never import placeholder packages from `veomni/models/transformers/__init__.py`.
2. **`model_type` matches registry key**: Mismatches fail explicitly under the VeOmni backend (`MODELING_BACKEND=hf` selects upstream HF without VeOmni kernels).
3. **Single standalone `modeling_<m>.py`**: Every HF-derived model under `veomni/models/transformers/<m>/` ships one self-contained `modeling_<m>.py` with `# Upstream-Module:` and `# Base-Transformers-Version: 5.16.1` headers. Never split into `_gpu`/`_npu` files; dispatch via instance-local `VeomniOp` handles and gate rare structural NPU branches with `IS_NPU_AVAILABLE` (never `not IS_CUDA_AVAILABLE`, which breaks ROCm/MLU).
4. **Transformers pinned to `5.16.1`**: Do not add `is_transformers_version_greater_or_equal_to()` gates for versions `<= 5.16.1`.
5. **Instance-local `VeomniOp` handles & config ownership**: Construct `VeomniOp` handles from `self.config.ops_config` on model instances — never introduce mutable module-level dispatch callables or import `veomni.ops.kernels.<family>.<impl>` directly in modeling. Native HF config classes are never monkey-patched; use `VeOmniPretrainedConfig` / `as_veomni_config`. Attention-free containers override `_check_and_adjust_attn_implementation` to return the canonical selection without preloading kernels. See `docs/design/module_config.md` and `docs/design/op_selection.md`.

## Distributed Training (FSDP2, SP, EP, DDP)

VeOmni uses **FSDP2 exclusively**. Entry points: `veomni/distributed/parallel_state.py`, `veomni/distributed/torch_parallelize.py`, `veomni/distributed/parallel_plan.py`.

6. **FSDP2 & device mesh**: `init_parallel_state_from_config()` builds a global `DeviceMesh` (`pp`, `dp_replicate`, `dp_shard`, `cp`, `ulysses`, `tp`) plus ExtraParallel `[para × para_fsdp]` submeshes. `parallelize_model_fsdp2()` shards transformer blocks then root on `fsdp_mesh` (which fuses with SP as `dp_shard_sp`).
7. **Local `ParallelState` scoping**: Never store a `ParallelState` object on a runtime — resolve by name (`VeOmniModelRuntime.parallel_state`). Build scope lives inside `VeOmniModelRuntime.__init__`; run-time trainer ops wrap forward/postforward/backward per op in `use_parallel_state(...)`. `sequence_parallel/comm.py` has no group globals. See `.agents/knowledge/topics/distributed.md` and `docs/design/local_parallel_state.md`.
8. **Ulysses SP & gradient scaling**: SP uses all-to-all (`gather_seq_scatter_heads` / `gather_heads_scatter_seq`), not all-gather. Because FSDP2/DDP averages gradients over `dp_shard_sp`/`dp_sp`, SP backward must scale by `sp_size` (`ReduceLoss` or `_Gather(sum_grad=True)`) and clone/use out-of-place buffers before backward collectives. Always verify SP **gradients**, not only losses.
9. **DDP under SP & meta-init**: `parallelize_model_ddp()` must pass `broadcast_buffers=False` unconditionally, materialize/load meta params via `_materialize_and_load_weights()`, refuse ExtraParallel when `_has_extra_parallel_plan()` is true, and all-reduce grads over `fsdp_group` (`SUM` then divide, keyed by `requires_grad`) before clipping when `sp_size > 1`. See `.agents/knowledge/topics/distributed.md`.
10. **EP sharding & DCP**: `ParallelPlan` defines ExtraParallel sharding (`Shard(0)` on EP mesh); FSDP2 shards experts on `ep_fsdp` (`Shard(1)` by default, or `Shard(0)` via `expert_fsdp_shard_dim=0` with `_StridedShard` in DCP). Pass both `SpecInfo.placement.dim` and `SpecInfo.fsdp_shard_dim` when restoring expert weights or exporting LoRA adapters.
11. **Meta materialization preserves non-persistent buffers**: Always use `_to_empty_preserving_nonpersistent_buffers()`, never bare `to_empty()`, on both random-init and resume paths (`init_empty_weights()` leaves real buffer values that `to_empty()` would wipe). Custom init rules belong in `_init_weights` on parameter-bearing leaf modules.

## Data Pipeline

12. **`MainCollator` 3-stage pipeline & strict SP order**: `PrecomputePositionIDsCollator` → `PackingCollator` → `SequenceParallelCollator` (when SP enabled). SP collation order is load-bearing: **pad → slice batch tensors → compute FA kwargs on full `position_ids` → slice `position_ids` last**.
13. **`position_ids == 0` & `IGNORE_INDEX` (-100)**: `position_ids == 0` marks packed segment boundaries for FlashAttention varlen (must be precomputed in batch before forward to avoid host-device sync; 3D multimodal `position_ids` uses `[:, 0, :]`). `PackingCollator` sets `IGNORE_INDEX` on the first token of every non-first packed sample.
14. **Dynamic batching token count (`dyn_bsz_count_mode`)**: `"total"` uses `get_length_fn`; `"effective"` uses `get_length_by_labels_fn` while still enforcing the `micro_batch_size * max_seq_len` physical-token cap. With SP enabled, `attention_mask` requires `sp_pad_value=1`.
15. **Multimodal preprocessing split**: `process_sample_qwen_vl()` uses `chat_template.encode_messages()` with negative placeholder IDs (`veomni/utils/constants.py`), whereas `process_sample_qwen_omni()` uses `processor(...)` directly and derives masks from model token IDs. Mask keys are `{modality}_mask`.

## Checkpoint & Trainer

16. **Collective DCP & layout**: DCP save/load (`veomni/checkpoint/dcp_checkpointer.py`) is collective across all ranks; parameter keys must match `state_dict`. Per-model `extra_state` (`lr_scheduler`, etc.) is written per rank before DCP `.metadata` (completeness marker). Job cursor/RNG (`loader/rank_{N}.pt`, `extra_state/rank_{N}.pt`) is rank-local. See `docs/usage/checkpoint.md`.
17. **DCP full resume skips HF load**: Non-LoRA resume sets `should_skip_hf_weight_load=True` (LoRA/PEFT must keep it `False` to load HF base weights).
18. **Non-floating tensors in safetensors consolidation**: Keep `apply_dcp_consolidation_patch()` (`veomni/checkpoint/dcp_consolidation.py`) during PyTorch 2.9–2.11 upgrades (upstream `torch.finfo` crashes on integer/boolean buffers like `tid2eid`). Cast `save_dtype` only on floating tensors.
19. **Composed trainers**: `TextDPOTrainer` and `DiTTrainer` compose `BaseTrainer` and override `forward_backward_step()`. Any trainer lifecycle change added to `BaseTrainer.forward_backward_step()` must also be propagated to or explicitly rejected by composed overrides.
20. **Config validation**: YAML values bypass argparse `Literal` choices (`veomni/arguments/parser.py`). Enforce required config invariants with explicit checks in `__post_init__`.

## Hardware & Code Quality

21. **Device-agnostic code & backend guards**: Use `veomni.utils.device` helpers (`get_device_type()`, `get_torch_device()`, `synchronize()`, `empty_cache()`) instead of `torch.cuda.*`. Guard CUDA/NPU/MLU vendor code with `IS_NPU_AVAILABLE` / `IS_MLU_AVAILABLE` and device-type sets (e.g. `MOE_TRITON_DEVICE_TYPES`); never import vendor packages unconditionally. MagiAttention (`--extra magi`) is SM100+ only.
22. **Code quality & PR rules**: `make quality` (ruff check + format) must pass before every commit. All code comments and docstrings must be in English. PR titles must match `.github/workflows/check_pr_title.yml`.

## Deep-Dive Topic Constraints (Read On Demand)

| When modifying | Read |
|---|---|
| QAT, `torchao`/`qat`/`deepgemm`, `gated_mlp`/`moe_experts` activations, `torch.compile` single-tensor ops | `.agents/knowledge/topics/qat_and_ops.md` & `docs/usage/low_precision_training.md` |
| Distributed internals (`torch_parallelize.py`, `ParallelState`, DDP+SP, DeepGEMM FP8 all-gather, PLE/QSA, RoPE/meta-init) | `.agents/knowledge/topics/distributed.md` |
| FlexAttention / pinned FA4 (`4.0.0b16`) shim & callbacks | `.agents/knowledge/topics/flex_attention.md` & `docs/transformers_v5/veomni_attention_interface.md` |
| SeedOmni (`veomni/models/seed_omni/`, `veomni/trainer/omni/`) | `.agents/knowledge/veomni-seed-omni.md` |
| VLM/Omni metadata precompute hooks | `.agents/knowledge/multimodal_metadata.md` |
| CI test selection & launch plans | `.agents/knowledge/testing.md` |
