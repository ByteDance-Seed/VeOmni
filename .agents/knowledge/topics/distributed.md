# Distributed & Model-Init Traps

Implementation traps beyond `docs/design/local_parallel_state.md`, `docs/design/deepgemm_moe.md`, and `docs/design/qwen4_exp_ple_2d_parallelism.md`. Read before modifying `veomni/distributed/`, `veomni/models/checkpoint/`, or the models below.

## SP Collectives & Gradient Scaling

- `veomni/distributed/sequence_parallel/comm.py` has no group globals/setters; getters resolve from `get_parallel_state()`. Meshless SP is unsupported (tests build a real state via `_init_parallel_state`).
- Because FSDP2/DDP averages gradients over `dp_shard_sp`/`dp_sp`, replicated-loss SP callers must use `_Gather(sum_grad=True)` (`gather_outputs()` default) or `grad_scale=True, sum_grad=False`; plain `sum_grad=False` silently scales grads by `1/sp_size` while losses still match. Always verify SP **gradients**, not just losses.
- Never reduce an incoming backward gradient in place without owning the buffer (`.contiguous()` does not copy contiguous aliased grads): `_Gather.backward` uses out-of-place `reduce_scatter*` on NCCL and clones before `dist.all_reduce` on other backends (`tests/parallel/ulysses/test_all_gather.py`).
- Keep gathered SP halos connected to every rank's autograd graph (zero-valued dependency when locally unused) so backward collective counts stay symmetric.
- Multi-model trainers (`TextDPOTrainer`) must wrap each model's forward/reduce/backward in its own `use_parallel_state(...)` — never `(loss_a + loss_b).backward()` across different SP groups under one ambient state. Cache `Callback.parallel_state` at construction.

## DDP Compatibility (`torch_parallelize.py` / `clip_grad_norm.py`)

- When `dp_mode == "ddp"` and `sp_size > 1`, `veomni_clip_grad_norm()` must all-reduce `p.grad` over `fsdp_group` (`dp_sp`) via `SUM` then divide (never `ReduceOp.AVG`, unsupported on NPU), keyed on `requires_grad` (zero-filling missing grads) so ranks cannot deadlock. Convert any DTensor norm via `full_tensor()` before `.item()`.
- Pass `broadcast_buffers=False` unconditionally (matches FSDP2 and avoids corrupting per-rank dynamic RoPE `inv_freq`).
- `parallelize_model_ddp()` calls `_materialize_and_load_weights()` gated on `param.is_meta` (not `init_device`), re-checks `param.is_meta` after load, and rejects `init_device == "cpu"` at parse time (`AcceleratorConfig._validate_init_device()`).
- Gate DDP ExtraParallel refusal and `extra_parallel_stream_load` on `_has_extra_parallel_plan()` (not `ParallelState.any_extra_parallel_enabled`) so plan-less SeedOmni DDP towers can inherit a global `ep` mesh while planned models still surface `NotImplementedError` for unsupported layouts (`tests/utils/test_extra_parallel.py`).

## EP / DeepGEMM FP8 All-Gather / Qwen4-Exp

- **Whole-expert FSDP (`expert_fsdp_shard_dim=0`)**: when EP and FSDP split the same dim, DCP must use `_StridedShard(dim, split_factor=ep_size)` on the FSDP axis (two plain `Shard(0)` placements silently permute global expert IDs). Pass both `SpecInfo.placement.dim` and `SpecInfo.fsdp_shard_dim` to `restore_extra_parallel_dim` (including LoRA export).
- **`fsdp_config.extra_parallel_only`**: applies the ExtraParallel plan and skips the FSDP2 wrap (`_finish_without_fsdp_wrap()`), keeping unnamed params whole. Requires `mixed_precision.enable=False` and `<para>_fsdp_size == 1` (rejects offload / `torch.compile`). `clip_grad_norm` averages replicated grads and divides non-persistent plan-sharded grads (and `LoraSharedExperts` EP-summed grads by `world_size`); DCP restores plain tensors along the ExtraParallel dim alone; pairs with `dim0_rowwise_companion` / `convert_rows`.
- **DeepGEMM FP8 (`distributed/fsdp2/fp8_all_gather.py`)**: CUDA-only, one GPU per process (pinned `5081f85` caches handles without a device key; `GroupedLayout.from_counts()` enforces this). Sync activation offload must skip `FP8ExpertWeight` containers. Compact block scales use a separate all-gather (Torch 2.11 nonzero-dim FSDP extension copy-out reshapes each output to shard shape). Align shards to 128×128, preserve BF16 rounding before quantizing FP32 master shards, install hooks after materialization and before FSDP lazy init, and transport payloads as `uint8` views.
- **Qwen4-Exp**: persistent 2D ExtraParallel (`ple`) requires a full-world 2D mesh (no HSDP/CPU offload). QSA indices address global full-resolution K/V (`-1` sentinel; never rebase by rank); packed block boundaries come only from global `cu_seq_lens_q`.
- **Async activation offload (`async_offload.py`)**: offload must wrap the checkpoint boundary — selected modules must own checkpointing inside `__call__` (like `GradientCheckpointingLayer`); calling checkpoint outside nests offload saved-tensor hooks inside checkpoint hooks and can suppress recomputation (tests must assert backward reruns the block). FSDP parent wrap targets are not necessarily offload targets: restrict auto-discovery via `_activation_offload_target_classes` and avoid selecting both a parent and its checkpointed children.

## Meta Materialization, `_init_weights`, & RoPE Checkpointing

- Always materialize meta models via `_to_empty_preserving_nonpersistent_buffers()`, never bare `to_empty()` (random-init, resume, and `veomni/models/checkpoint/weights.py`), so non-persistent buffers (e.g. Gemma3 `{type}_inv_freq`/`embed_scale`, Omni audio sinusoidal `positional_embedding`) are preserved.
- Put custom initialization rules in `_init_weights` on parameter-bearing leaf modules (Transformers 5.16 can skip parameterless custom containers).
- Under FSDP2 BF16 mixed precision + non-reentrant activation checkpointing, fixed vision RoPE cos/sin tables can change dtype between forward and recompute: eager/Liger RoPE adapters normalize fixed saved tables to `q.dtype` (`tests/ops/rope/test_rope.py`), while models needing identical FP32 constants (e.g. HunyuanVideo 1.5 `RotaryTables`) isolate them from FSDP input casting.
