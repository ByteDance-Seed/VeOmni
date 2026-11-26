# Test Wiring & CI Selection

Read before adding or modifying tests. For manifest syntax and local runner commands, also see `docs/testing.md`, `docs/design/ci_case_scheduling.md`, and `scripts/ci/accelerator/README.md`.

## CI Selects Tests by Explicit Enumeration

A passing local test does **not** mean CI runs it. GPU and NPU unit/e2e jobs dispatch via `scripts/ci/accelerator/runner.py` reading `scripts/ci/accelerator/suites/gpu.json` and `scripts/ci/accelerator/suites/npu.json`. Never add per-file pytest lines or manual device quotas to workflow YAML.

| Path | How CI picks it up |
|---|---|
| `tests/ci/`, `tests/arguments/`, `tests/data/`, `tests/checkpoints/`, `tests/seed_omni/` | Whole directory in both GPU & NPU manifests (`suite: unit`) |
| `tests/ops/` | Whole directory in both unit manifests, excluding `base/test_op_documentation.py` (device-API workflow) |
| `tests/models/` | Whole directory in both unit manifests, excluding `transformers/test_generated_call_site_signatures.py` (device-API workflow); GPU also excludes `test_omni_fsdp.py` |
| `tests/parallel/context_parallel/` | Whole directory in GPU unit manifest; NPU lists only `test_deepseek_v4_cp.py` |
| `tests/e2e/test_e2e_parallel.py`, `tests/distributed/test_fsdp_equivalence.py` | Both GPU & NPU manifests (`suite: e2e`) |
| `tests/models/test_omni_fsdp.py` | GPU `61-e2e` (`suite: e2e`); NPU runs it in models unit selection |
| **Everything else** | **Must be explicitly enumerated** in `gpu.json` and/or `npu.json` (otherwise invisible to CI) |

Check whether a path is already wired:
```bash
TEST_PATH=tests/trainer/test_my_new_thing.py
rg -n "$(basename "$TEST_PATH")" scripts/ci/accelerator/suites/*.json
rg -n "$(dirname  "$TEST_PATH")" scripts/ci/accelerator/suites/*.json
```

- **Device-API workflow** (no Torch, `pytest --noconftest`): runs architecture, op-doc, and call-site-signature tests plus `check_device_api_usage.py` over `tasks`, `veomni`, and `tests`.
- **Fail-fast fleet**: GPU unit runs on an L20-8 fleet (120 min budget, `-x` fail-fast). The runner cancels peers on first failure by default (`--keep-going` continues all).

## Decision Rule (Stop at First Match)

1. **Extend an already-enumerated test** (default — add a `pytest.param` row, zero manifest churn):
   - Model registration/build → `tests/models/base/test_auto_registry.py` (`_MODEL_CASES`)
   - Model eager/accelerator parity → matching family under `tests/models/transformers/` or `tests/models/diffusers/`
   - Host-device sync regressions → `tests/models/transformers/test_model_forward_no_implicit_sync.py`
   - MoE checkpoint conversion → `tests/models/base/test_checkpoint_tensor_converter.py`
   - VLM trainer / freeze-ViT → `tests/trainer/test_vlm_trainer.py`
   - VLM/Omni dummy forward → `tests/distributed/test_dummy_forward.py`
   - `torch.compile` / grad checkpointing → `tests/distributed/test_torch_compile.py`, `tests/distributed/test_gradient_checkpointing.py`
   - Ulysses SP → `tests/parallel/ulysses/test_ulysses.py` and siblings
   - Checkpoint / DCP / DiT resume → `tests/checkpoints/test_{checkpoint_callback,trainer_saveload,dcp_checkpointer,dit_resume_align}.py`
   - Weight load / ExtraParallel stream load & clip / FLOPs → `tests/utils/test_{rank0_load_and_broadcast_weights,extra_parallel,count_flops}.py`
   - Optimizer (Muon / VLM groups / SwapAdamW) → `tests/optim/test_{muon_*,optimizer_vlm_param_groups,swap_adamw*}.py`
   - LoRA / MoE-LoRA → enumerated `tests/lora/test_*.py`
   - E2E parallel parity → `tests/e2e/test_e2e_parallel.py`
2. **New self-contained test in an auto-collected directory** → place under `tests/{models,ops,data,arguments,seed_omni,checkpoints,ci}/` (auto-collected in both manifests; no manifest edit).
3. **New file elsewhere** → fold into an existing enumerated file first; otherwise add to `gpu.json` (and `npu.json` if device-portable) and note wall-time cost in the PR.
4. **No new test** → valid for pure refactors with existing coverage, docs/config/`.agents/`-only edits, hardware CI lacks (multi-node, SM90+ on SM89 runners), or lint-only checks. State the reason explicitly in the PR's **Test** section.

## Launch Plans, CPU Tests & Distributed Rendezvous

- **Launch plans**: use `pytest.mark.launch_plan(provider=...)` for multi-card cases (same `nproc/world_size` as worker startup); unmarked defaults to 1 card (or CPU on CPU-only hosts). Pure CPU tests (parsing, preprocessing, eager refs, mocked backends) must declare an explicit CPU launch plan (0 cards; mark modules only if every case is CPU-only).
- **CPU isolation**: CPU-planned workers hide devices (restoring `DefaultDeviceType`, optimizer foreach, `torch_npu` DataLoader/Inductor hooks), and the model-test autouse fixture blocks FlashAttention preloading on CPU plans/hosts (local MLU hosts bypass this CI fixture). Plain-tensor checkpoint exports must not query the accelerator's current device unless materializing a DTensor. CPU launcher subprocess regressions must isolate device env vars before importing torch; CPU-only `spawn` workers using checkpoint recomputation must also set `torch.utils.checkpoint.DefaultDeviceType` to `"cpu"` inside the child (parent plugin setting does not propagate through `spawn`, so Ascend hosts can otherwise initialize hidden NPUs on AMP queries).
- **CPU model & eager fixtures**: select eager for **all** consumed ops via `make_eager_ops_config()` (setting only attention to eager leaks fused RMSNorm/MLP/GDN defaults). On composite models, set root eager policy before building reference or candidate. Use config-owned `sdpa` for FP32 accelerator parity when the default attention backend needs half precision. Scope `MODELING_BACKEND=veomni` with a restoring fixture. Standalone leaf fixtures must adapt HF configs (`get_veomni_config_class` / `as_veomni_config`).
- **Rendezvous**: use `tests.tools.launch_utils.spawn_with_rendezvous` for `mp.spawn` workers (keeps Store listener alive through startup and reaps workers before releasing) or the `torchrun` helper (`--standalone`). Never rely on `find_free_port()` probe-close-bind races, and never retry whole training/checkpoint runs to work around a bind race.

## NPU CI Runtime Notes

NPU workflows run `uv sync --extra npu` on Atlas A2 (non-arch35) runners, so image-baked packages are not on the test path:
- `triton-ascend`: installed explicitly after `uv sync` — gates vendored Ascend Triton kernels (`causal_conv1d`, `chunk_gated_delta_rule` `triton`, MoE clamped SwiGLU).
- `fla_npu`: image venv only — `chunk_gated_delta_rule_implementation=fla_npu` is untestable in CI.
- `flash-linear-attention`: `gpu` extra only — treat `fla` backends as GPU-only.
- `is_arch35()` kernels stay unreachable on Atlas A2 runners.
- **NPU test guards & portability**: `check_device_api_usage.py` skips `tests/ops/` and does not scan NPU literals — manually audit changed tests for hard-coded backend strings. Before allocating on NPU, check both `IS_NPU_AVAILABLE` and `get_torch_device().is_available()` (or `torch.npu.is_available()`).

## New-Model Coverage Policy & Numerical Pitfalls

Follow `docs/transformers_v5/testing_new_model.md` for new models: registry/build, eager fwd/bwd parity, one supported accelerator training parity check, and E2E alignment — ordinary dense `(SP1,EP1)` + `(SP2,EP1)`, ordinary MoE `(SP1,EP1)` + `(SP2,EP2)` on 4 ranks (Qwen3-MoE keeps full 8-rank matrix; GPT-OSS covers `(1,1)/(2,1)/(2,2)` for its Quack variant). Keep alignment assertions enabled (compare DP-mean loss + grad norm from identical initial checkpoint/data/global batch).

Key testing pitfalls:
- **Compiled backward**: can donate saved buffers and reject `retain_graph=True`; use independent leaf tensors for reference vs candidate and differentiate once.
- **Valid op names in stubs**: `VeomniOp` stubs must expose `.impl` and use legal implementation names (validation runs before the stub).
- **Uninitialized `torch.empty` params**: standalone HF layers (e.g. DeepSeek-V4 `sinks`, `position_bias`) rely on `PreTrainedModel.post_init()`; initialize them before broadcasts (poisoning allocations with NaN in CPU tests catches misses).
- **CPU numerical flakes**: reproduce in a fresh process (PyTorch 2.11 CPU FP32 `torch.exp` can miscompute on its first parallel call; QSA CPU mock promotes only `exp` to FP64).
- **Ascend Diffusers RMSNorm CPU substitute**: preserve upstream FP32 promotion (`F.rms_norm` on BF16 input + FP32 weight outputs BF16 and fails FP32 Linear without autocast).
- **API removals/migrations**: validate merged with latest target branch (PR CI tests the merge commit).
- **Cheapest tier first**: CPU UT > 1-card > multi-card > E2E. Unwired test files are local reproducers, not CI coverage.
- **CPU-only hosts**: `make quality`, `for d in veomni tasks tests; do python tests/special_sanity/check_device_api_usage.py -d "$d"; done`, and CPU pytest subsets run without accelerators; fused CUDA/Triton kernels and multi-GPU/FSDP2/SP/EP tests require real hardware.
