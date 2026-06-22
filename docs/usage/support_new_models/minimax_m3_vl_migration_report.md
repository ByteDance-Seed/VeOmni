# MiniMax M3 VL Migration Report

Date: 2026-06-22

## Status

This PR now contains a MiniMax M3 VL patchgen-generated intake for VeOmni. It does not claim full production completion yet, because the 59-shard public checkpoint payload load and multi-card SP/EP alignment are still outstanding. Public checkpoint index coverage is verified against the generated model `state_dict()` keys, safetensors shard-header metadata verifies converted key/shape coverage across all 59 public shards without downloading tensor payloads, and single-card Ascend NPU generated-model toy SFT loss now passes.

Community RFC:

- RFC issue: `https://github.com/ByteDance-Seed/VeOmni/issues/852`

Upstream PR:

- PR: `https://github.com/ByteDance-Seed/VeOmni/pull/846`
- Branch: `codex/minimax-m3-vl-slice`
- Latest audited code head: `f15248f40d9f8498ae6749dc2410096f56a8d9e0`; subsequent commits in this report update validation evidence only.
- GitHub status after the validation-evidence pushes: CLA is `success`; upstream GitHub Actions are still `action_required`, so maintainers must approve fork-workflow execution before CI jobs/logs exist.

## Delivered Files

- `veomni/models/transformers/minimax_m3_vl/configuration_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.diff`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.diff`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`
- `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml`
- `scripts/multimodal/verify_minimax_m3_vl_checkpoint_index.py`
- `scripts/multimodal/train_minimax_m3_vl.sh`
- `scripts/multimodal/run_minimax_m3_vl_npu_loss.py`
- `scripts/multimodal/run_minimax_m3_vl_npu_loss_root.sh`
- `scripts/multimodal/verify_minimax_m3_vl_precision_parity.py`
- `scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py`
- `docs/examples/minimax_m3_vl.md`
- `docs/usage/support_new_models/minimax_m3_vl_data_module_design.md`
- `docs/usage/support_new_models/minimax_m3_vl_hyperparams_loading_report.md`
- `docs/usage/support_new_models/minimax_m3_vl_model_migration_design.md`
- `docs/usage/support_new_models/minimax_m3_vl_reduced_layer_sft_report.md`
- `docs/usage/support_new_models/minimax_m3_vl_precision_parity_guide.md`
- `tests/data/multimodal/test_minimax_m3_vl_data_transform.py`
- `tests/data/test_mm_metadata.py`
- `tests/models/test_models_patch.py`
- `tests/models/test_vlm_trainer.py`
- `tests/models/test_model_forward_no_implicit_sync.py`
- `tests/models/test_checkpoint_tensor_converter.py`
- `tests/distributed/test_dummy_forward.py`
- `tests/e2e/test_e2e_parallel.py`

## Bagel Reference Deliverable Check

Reference branch inspected: `https://github.com/Crystal-jiang/VeOmni/tree/fp/bagel`.

Latest two commits:

- `fa06c1e feat: support bagel on npu`
- `6a7c226 WIP: support bagel`

Observed Bagel-style new model deliverables:

- config: `configs/multimodal/omni/bagel.yaml`
- scripts: `scripts/multimodal/convert_bagel.py`
- docs: `docs/examples/bagel.md`
- data: `veomni/data/multimodal/bagel_transform.py`, `bagel_collator.py`, later `bagel_dataset.py`
- model config/modeling/processing under `veomni/models/seed_omni/.../bagel_*`
- tests: `tests/models/test_bagel_training.py`

MiniMax mapping in this PR:

- config/docs/modeling/data/test entries are present.
- shell launch handoff is present at `scripts/multimodal/train_minimax_m3_vl.sh`.
- Real checkpoint index conversion covers language/MoE keys, sparse-attention indexer keys, the MoE router correction buffer, and the VLM projector split (`multi_modal_projector` + `patch_merge_mlp`), but full public checkpoint tensor-payload loading has not been executed yet.

## VeOmni New-Model Completion Contract

Reading `.agents/skills/veomni-new-model/SKILL.md` and the Bagel reference commits, a new model migration should be treated as complete only when these deliverables exist and the corresponding gates pass:

| Area | Expected VeOmni path pattern | MiniMax M3 VL status |
|---|---|---|
| RFC / community alignment | GitHub issue or RFC before expanding scope | Present: `https://github.com/ByteDance-Seed/VeOmni/issues/852` |
| Model Python | `veomni/models/transformers/<model>/__init__.py`, config, patchgen configs, generated GPU/NPU files, `parallel_plan.py`, converter if checkpoint layout needs it | Present for patchgen slice |
| Data Python | `veomni/data/data_transform.py` and/or `veomni/data/multimodal/<model>*.py`; trainer glue if visual module lookup differs | Present: transform/template and VLM trainer visual-module lookup |
| Config | `configs/multimodal/<model>/<model>.yaml` for VLM models | Present: `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml` |
| Script | `scripts/multimodal/<model>.sh` or a conversion/training helper | Present: `scripts/multimodal/train_minimax_m3_vl.sh`, NPU loss evidence runners, toy precision parity, and real checkpoint payload parity tooling |
| Docs | `docs/examples/<model>.md` plus support-new-model evidence reports | Present, with the MiniMax-only `transformers>=5.12.0` environment documented |
| Tests | registry, patchgen hook, data/trainer, converter, dummy-forward, e2e where applicable | Partial: light tests pass; no-sync metadata and FSDP2 dummy-forward gates are wired; SP/EP e2e remains an explicit follow-up gate |
| Validation evidence | source facts, generated import, real or toy forward/backward, checkpoint load, trainer smoke, accelerator status | Partial: source facts, import, converter unit tests, public checkpoint index/state coverage, public shard-header shape coverage, remote real-tensor payload sample, CPU and single-card Ascend NPU HF-vs-VeOmni toy precision parity, toy/generated SFT smoke, and single-card Ascend NPU generated-model SFT smoke pass; full real-checkpoint forward parity and multi-card accelerator validation are not complete |

Assessment: the current MiniMax M3 VL PR is a useful patchgen-generated migration slice and has the right file families for a VeOmni new-model PR. It is not a completed production model migration yet, because completion still requires full public checkpoint forward parity, public-checkpoint trainer smoke, GPU parity reruns on target machines, and SP/EP e2e alignment on multi-card hardware. Synthetic image/video `VLMTrainer` forward/backward evidence, data-transform-level path/bytes video-container evidence, public checkpoint index/state coverage, public shard-header shape coverage, remote real-tensor payload sampling, CPU and single-card Ascend NPU HF-vs-VeOmni toy precision parity, real checkpoint full-forward tooling, and single-card Ascend NPU generated-model SFT loss evidence are present and documented below.

## Source Facts

Verified source facts:

- Hugging Face model: `MiniMaxAI/MiniMax-M3`
- Hugging Face sha: `051e8f961274fb4e18ac3b57991f13bffedde212`
- Last modified: `2026-06-16T05:18:24.000Z`
- `model_type`: `minimax_m3_vl`
- `architectures`: `["MiniMaxM3SparseForConditionalGeneration"]`
- processor class: `MiniMaxVLProcessor`
- ModelScope page: `https://modelscope.cn/models/MiniMax/MiniMax-M3`, HTTP 200
- `transformers/v5.9.0` MiniMax modeling: 404
- `transformers/v5.12.0` MiniMax modeling: 200

VeOmni global `transformers-stable` pin is unchanged. MiniMax example docs require a local `transformers>=5.12.0` environment.

Source recheck: HF sha and `lastModified` are unchanged, the public safetensors index is HTTP `200` with `23416` weight-map keys and `59` shard filenames. Projector index coverage is split across `multi_modal_projector.linear_{1,2}.{weight,bias}` and `patch_merge_mlp.linear_{1,2}.{weight,bias}`; the latter maps to transformers generated `merge_linear_{1,2}`.

## Implemented

- Config registry for `minimax_m3_vl`, `minimax_m3_vl_text`, and `minimax_m3_vl_vision`.
- Generated GPU and NPU modeling files from transformers v5.12.0 via patchgen.
- Local modeling gate: default `transformers==5.9.0` import remains healthy; MiniMax modeling raises a clear `transformers>=5.12.0` requirement.
- MiniMax config parser accepts nested `PretrainedConfig` text/vision sections as well as raw dicts, so upgraded transformers/cache paths do not crash in `_drop_nested_model_type()`.
- Parallel plan hook for multimodal and text-only MiniMax classes.
- Default position-id hook returning `None` so VeOmni uses default packed 1-D position ids.
- MiniMax `get_metadata_collate_func()` hook that precomputes image/video `grid_thw` lists in the collator, then routes them into the generated vision tower so MiniMax 3D RoPE does not call `grid_thw.tolist()` on the production path.
- MiniMax vision `dummy_forward()` and FSDP-symmetric model forward branch for text-only ranks in asymmetric multimodal batches.
- MiniMax multimodal data transform and chat template.
- MiniMax VLM example config and docs.
- MiniMax top-level causal-LM loss patch that unpacks VeOmni CE kernel tuple outputs into a trainer-consumable tensor loss while preserving bare transformers loss behavior.
- Runtime checkpoint converter for language tower, MoE, sparse-attention indexer, persistent router buffer, and VLM projector key/layout differences.
- MiniMax-specific e2e dummy VLM dataset, so the SP/EP alignment gate uses 1-D MiniMax position ids instead of Qwen mRoPE-style 3-D position ids.
- Targeted tests in:
  - `tests/models/test_model_registry.py`
  - `tests/models/test_models_patch.py`
  - `tests/models/test_vlm_trainer.py`
  - `tests/data/test_mm_metadata.py`
  - `tests/data/multimodal/test_minimax_m3_vl_data_transform.py`
  - `tests/models/test_model_forward_no_implicit_sync.py`
  - `tests/models/test_checkpoint_tensor_converter.py`
  - `tests/distributed/test_dummy_forward.py`
  - `tests/e2e/test_e2e_parallel.py` (MiniMax-specific dataset; xfail follow-up gate)

## Validation

Latest PR-head preflight after the ruff-format code commit (`f15248f`), repeated after the docs-evidence update:

```text
git diff --check

uv run --no-project --with ruff==0.13.2 ruff check tasks tests veomni docs
All checks passed!

uv run --no-project --with ruff==0.13.2 ruff format --check tasks tests veomni docs
468 files already formatted

uv run --no-project --with torch==2.7.1 --with transformers==5.12.0 --with psutil \
  --with pytest --with safetensors pytest tests/models/test_checkpoint_tensor_converter.py -k MiniMax -q
10 passed, 51 deselected in 5.22s

python3 scripts/ci/check_doc_task_paths.py
All task script paths in docs shell blocks exist.

python3 tests/special_sanity/check_device_api_usage.py -d tasks
python3 tests/special_sanity/check_device_api_usage.py -d veomni
python3 tests/special_sanity/check_device_api_usage.py -d tests
all checked files reported success or intentional whitelist skips
```

MiniMax targeted patchgen drift preflight at the same head:

```text
PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg \
  --with transformers==5.12.0 --with torch==2.7.1 \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config \
  -o /tmp/veomni_minimax_patchgen_check/gpu --diff

PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg \
  --with transformers==5.12.0 --with torch==2.7.1 --with psutil \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config \
  -o /tmp/veomni_minimax_patchgen_check/npu --diff

diff -u veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py \
  /tmp/veomni_minimax_patchgen_check/gpu/patched_modeling_minimax_m3_vl_gpu.py
diff -u veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.diff \
  /tmp/veomni_minimax_patchgen_check/gpu/patched_modeling_minimax_m3_vl_gpu.diff
diff -u veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py \
  /tmp/veomni_minimax_patchgen_check/npu/patched_modeling_minimax_m3_vl_npu.py
diff -u veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.diff \
  /tmp/veomni_minimax_patchgen_check/npu/patched_modeling_minimax_m3_vl_npu.diff
```

Both MiniMax patchgen commands succeeded and all four `diff -u` comparisons returned no differences, so the checked-in GPU/NPU generated `.py` and `.diff` files match the MiniMax patchgen configs byte-for-byte. The full CI command `uv run --extra gpu --dev patchgen --check` was also attempted locally, but this machine stalled during project dependency sync before patchgen execution; the targeted MiniMax regeneration above is the actionable local drift evidence for this PR head.

Static checks for this update:

```text
python3 -m py_compile \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py \
  veomni/data/dummy_dataset.py \
  tests/models/test_model_forward_no_implicit_sync.py \
  tests/models/test_models_patch.py \
  tests/models/test_vlm_trainer.py \
  tests/data/test_mm_metadata.py \
  tests/data/multimodal/test_minimax_m3_vl_data_transform.py \
  tests/distributed/test_dummy_forward.py \
  tests/e2e/test_e2e_parallel.py

git diff --check

ruff check veomni/models/transformers/minimax_m3_vl \
  veomni/data/dummy_dataset.py \
  tests/models/test_model_forward_no_implicit_sync.py \
  tests/models/test_models_patch.py \
  tests/models/test_vlm_trainer.py \
  tests/data/test_mm_metadata.py \
  tests/data/multimodal/test_minimax_m3_vl_data_transform.py \
  tests/distributed/test_dummy_forward.py \
  tests/e2e/test_e2e_parallel.py

All checks passed.
```

Additional local checks for the latest patchgen/docstring commit:

```text
python3 -m py_compile \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py \
  veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py \
  veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py

ruff check \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py \
  veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py \
  veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py \
  veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py

pytest tests/models/test_models_patch.py -k minimax_m3_vl -q

2 passed, 12 deselected in 7.03s
```

MiniMax toy build docstring audit with `transformers==5.12.0`: `build_foundation_model()` returned `MiniMaxM3SparseForConditionalGeneration` and `docstring_error_present False`, verifying HF `auto_docstring` no longer reports undocumented top-level `image_grid_thw` / `video_grid_thw`.

Default transformers 5.9 targeted tests:

```text
pytest tests/models/test_model_registry.py tests/models/test_checkpoint_tensor_converter.py \
  -k "minimax_m3_vl or MiniMax" -q

..........                                                               [100%]
10 passed, 55 deselected in 4.93s
```

Latest MiniMax registry targeted tests after the nested `PretrainedConfig` review fix:

```text
transformers==5.9.0: pytest tests/models/test_model_registry.py -k minimax_m3_vl -q
2 passed, 4 deselected in 11.57s

transformers==5.12.0: pytest tests/models/test_model_registry.py -k minimax_m3_vl -q
2 passed, 4 deselected in 5.06s
```

MiniMax checkpoint converter targeted test:

```text
pytest tests/models/test_checkpoint_tensor_converter.py -k MiniMax -q

10 passed, 51 deselected in 4.99s
```

This now includes `test_public_projector_index_mapping_covers_merge_weights_from_patch_merge_mlp`, which checks that public `patch_merge_mlp.linear_{1,2}` maps to generated `model.multi_modal_projector.merge_linear_{1,2}` and closes the projector index coverage without synthesizing weights. It also includes sparse-attention indexer mapping and `block_sparse_moe.e_score_correction_bias` mapping into the persistent router buffer.

Real HF public index and shard-header metadata coverage check:

```text
curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 180 \
  -o /tmp/minimax_m3_model.safetensors.index.json \
  https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/model.safetensors.index.json

PYTHONPATH=$PWD uv run --no-project --python 3.11 \
  --with torch==2.7.1 --with transformers==5.12.0 \
  --with packaging --with psutil --with rich --with numpy --with pillow \
  --with requests --with safetensors --with tqdm \
  scripts/multimodal/verify_minimax_m3_vl_checkpoint_index.py \
  --config-path MiniMaxAI/MiniMax-M3 \
  --index-json /tmp/minimax_m3_model.safetensors.index.json \
  --verify-shard-metadata \
  --metadata-cache-dir /tmp/minimax_m3_safetensors_headers \
  --fail-on-unexpected \
  --json-output /tmp/minimax_m3_vl_checkpoint_metadata_report.json

public_weight_map_keys: 23416
converted_index_keys: 1582
public_safetensors_metadata_keys: 23416
converted_metadata_keys: 1582
safetensors_shards_read: 59
safetensors_header_bytes_read: 3440088
model_parameter_keys: 1525
model_persistent_buffer_keys: 57
model_state_keys: 1582
missing_state_key_count: 0
unexpected_index_key_count: 0
missing_metadata_key_count: 0
unexpected_metadata_key_count: 0
shape_mismatch_count: 0
dtype_mismatch_groups: {"F32->BF16": 115}
missing_projector_keys: []
checkpoint_values_downloaded: false
full_checkpoint_load_executed: false
```

Artifact:

- [minimax_m3_vl_checkpoint_index_report.json](./artifacts/minimax_m3_vl_checkpoint_index_report.json)
- [minimax_m3_vl_checkpoint_metadata_report.json](./artifacts/minimax_m3_vl_checkpoint_metadata_report.json)

`converted_index_keys` is lower than the public index key count because the converter collapses source expert shards and `gate_proj` / `up_proj` pairs into generated fused parameters. The verifier intentionally excludes non-persistent rotary runtime buffers and compares against `state_dict()` keys. With `--verify-shard-metadata`, it also reads only the 59 safetensors headers through `/resolve/main` Range requests, proving converted key and shape coverage without downloading the 869 GB tensor payloads. The 115 dtype differences are all `F32->BF16` router/gate-state differences and are reported rather than treated as layout failures because VeOmni checkpoint dispatch casts tensors to the target parameter or buffer dtype during load.

Local transformers 5.12 generated-model tests:

```text
pytest tests/models/test_model_registry.py tests/models/test_models_patch.py \
  tests/models/test_model_forward_no_implicit_sync.py -k minimax_m3_vl -q

4 passed, 2 skipped, 34 deselected in 6.47s
```

Skip reason: the two runtime no-sync/equivalence cases require CUDA. Passing cases include static generated-hook checks and a mixed image+video generated-model loss/backward fixture.

Direct MiniMax no-sync readiness test:

```text
pytest tests/models/test_model_forward_no_implicit_sync.py -k minimax_m3_vl -q -rs

1 passed, 2 skipped, 18 deselected in 5.18s
```

Skip reasons: the two runtime equivalence/sync checks require CUDA; the static readiness case passes and confirms MiniMax generated hooks are wired.

VLM trainer freeze/visual-module and MiniMax trainer-glue forward/backward tests:

```text
pytest tests/models/test_vlm_trainer.py -k minimax_m3_vl -q

6 passed, 12 deselected, 3 warnings in 20.39s
```

The trainer-glue case builds MiniMax data transform and collator through `VLMTrainer._build_data_transform()` / `VLMTrainer._build_collate_fn()`, then runs `BaseTrainer.forward_backward_step()` on a toy image+video micro-batch. It uses transformers v5.12.0 `MiniMaxM3VLImageProcessor` / `MiniMaxM3VLVideoProcessor` on synthetic PIL image/frame inputs, with a local tokenizer/replacement shim, so the tensor shapes and placeholder counts come from the real MiniMax image/video processor classes. It caught and fixed the MiniMax top-level loss tuple issue under VeOmni ops binding.

The init smoke case parameterizes image-only, video-only, and mixed image+video fixtures through `VLMTrainer.__init__()`. It builds the MiniMax model, model assets, data transform, dataset, dataloader, collator, optimizer, LR scheduler, training context, and callbacks; then it pulls one dataloader micro-batch and executes `BaseTrainer.forward_backward_step()`, `optimizer.step()`, and `lr_scheduler.step()`. This init smoke now avoids monkeypatching media fetch: image-only uses a local JPEG path, video-only uses a local MP4 path, and mixed uses a local JPEG path plus raw MP4 bytes. This covers the trainer dataloader/optimizer lifecycle and local media-container path locally, but it still does not replace public-checkpoint or multi-card torchrun evidence.

MiniMax data transform test:

```text
pytest tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q

6 passed in 17.42s
```

The added end-to-end data-chain case feeds `process_sample_minimax_m3_vl()` output through `MainCollator(metadata_collate_func=model.get_metadata_collate_func())` and then runs a toy generated MiniMax image+video loss/backward step. The file also includes VeOmni video-fetch cases that do not monkeypatch `fetch_videos_metadata()`: one passes pre-decoded PIL frames through VeOmni's default video processing path, and a parameterized case writes a tiny MP4 then feeds both local path and raw bytes containers through the torchcodec/PyAV decode path before transformers v5.12.0 `MiniMaxM3VLVideoProcessor`, producing real `pixel_values_videos` / `video_grid_thw`. These cases prove the MiniMax transform, collator metadata hook, local str/bytes container decode, visual scatter masks, projector, and language loss can work in local generated-model paths. They still do not replace public-checkpoint or multi-card evidence.

Default transformers 5.9 compatibility for the same test file:

```text
pytest tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q -rs

2 passed, 4 skipped in 14.66s
```

Skip reasons: the real MiniMax video-processor, local video-container, and generated-model data-chain cases are gated on `transformers>=5.12.0`; the pure transform/template tests remain valid under VeOmni's default dependency pin.

MiniMax metadata collator hook test:

```text
pytest tests/data/test_mm_metadata.py -q

5 passed in 5.99s
```

MiniMax metadata + data transform combined test:

```text
pytest tests/data/test_mm_metadata.py tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q

11 passed in 17.13s
```

MiniMax model + VLM trainer combined generated-path test:

```text
pytest tests/models/test_models_patch.py tests/models/test_vlm_trainer.py -k minimax_m3_vl -q

8 passed, 24 deselected, 3 warnings in 20.16s
```

Default transformers 5.9 compatibility for MiniMax data/trainer-gated cases:

```text
pytest tests/data/multimodal/test_minimax_m3_vl_data_transform.py tests/models/test_vlm_trainer.py \
  -k minimax_m3_vl -q -rs

2 passed, 10 skipped, 12 deselected in 6.03s
```

FSDP2 asymmetric dummy-forward test:

```text
pytest tests/distributed/test_dummy_forward.py -k minimax_m3_vl -q -rs

1 skipped, 5 deselected in 6.19s
```

Skip reason: the current local machine did not expose the required two distributed devices. The MiniMax case is no longer xfail; it is enabled when the distributed device requirement is satisfied.

MiniMax e2e dummy dataset readiness:

```text
build_dummy_dataset("minimax_m3_vl", size=2, max_seq_len=512)

position_ids: (512,), [0, 1, 2] ... [509, 510, 511]
pixel_values: (4, 1176)
pixel_values_videos: (4, 1176)
image_grid_thw: [[1, 2, 2]]
video_grid_thw: [[1, 2, 2]]
image_mask_sum: 1
video_mask_sum: 1
visual_labels: [-100, -100]
```

This replaces the previous reuse of the Qwen3-VL dummy dataset for the MiniMax e2e case. The e2e test still remains xfail until the SP/EP metric alignment is run on suitable multi-card hardware.

MiniMax e2e collection check:

```text
pytest tests/e2e/test_e2e_parallel.py::test_minimax_m3_vl_parallel_align --collect-only -q

tests/e2e/test_e2e_parallel.py::test_minimax_m3_vl_parallel_align[minimax_m3_vl-./tests/toy_config/minimax_m3_vl_toy/config.json-True-0.1-0.1-None]
1 test collected in 6.21s
```

Direct CPU dummy-forward smoke:

```text
vision (1, 16, 16) projected (4, 32)
```

Generated-model reduced SFT smoke:

```json
{
  "first_loss": 5.57344913482666,
  "last_loss": 4.035123348236084,
  "steps": 8,
  "model_class": "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration"
}
```

Artifact:

- [generated_model_loss_log.json](./artifacts/minimax_m3_vl_sft_smoke/generated_model_loss_log.json)

Ascend NPU generated-model reduced SFT smoke:

```json
{
  "passed": true,
  "first_loss": 5.531774044036865,
  "last_loss": 4.8606367111206055,
  "steps": 8,
  "device": "npu:0",
  "model_class": "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_npu.MiniMaxM3SparseForConditionalGeneration",
  "torch_npu_version": "2.10.0",
  "transformers_version": "5.12.0"
}
```

Artifacts:

- [npu_generated_model_loss_log.json](./artifacts/minimax_m3_vl_npu_loss_smoke/npu_generated_model_loss_log.json)
- [npu_runtime_probe.json](./artifacts/minimax_m3_vl_npu_loss_smoke/npu_runtime_probe.json)
- [loss_curve.svg](./artifacts/minimax_m3_vl_npu_loss_smoke/loss_curve.svg)

HF reference vs VeOmni generated toy precision parity:

```json
{
  "passed": true,
  "num_checks": 37,
  "device": "cpu / npu:0",
  "reference": "transformers.models.minimax_m3_vl.modeling_minimax_m3_vl.MiniMaxM3SparseForConditionalGeneration",
  "candidate": "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration"
}
```

This parity gate uses one deterministic mixed image+video toy batch and the same randomly initialized state dict for both models. It checks forward loss/logits/projected image/video hidden states, MoE router logits/top-k weights/selected experts, attention mask and position-id inputs, VeOmni multimodal metadata contract, key gradients, and one AdamW parameter-update delta. CPU parity passes with `torch==2.7.1` and `transformers==5.12.0`; single-card Ascend NPU parity passes in `quay.io/ascend/vllm-ascend:v0.20.2rc1` with `torch_npu==2.10.0`, `transformers==5.12.0`, and NPU tolerances `forward=5e-4`, `grad=1e-3`, `param=1e-3`. GPU, real-checkpoint, and multi-card reruns are documented in the guide.

Artifacts:

- [toy_hf_veomni_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity.json)
- [toy_hf_veomni_parity_npu.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_npu.json)
- [minimax_m3_vl_precision_parity_guide.md](./minimax_m3_vl_precision_parity_guide.md)

Real checkpoint payload parity tooling:

- `scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py`
- [real_checkpoint_payload_remote_sample.json](./artifacts/minimax_m3_vl_precision_parity/real_checkpoint_payload_remote_sample.json)
- [toy_checkpoint_forward_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_checkpoint_forward_parity.json)

The script reads local public safetensors payloads or remote safetensors tensor byte ranges, applies `MiniMaxM3VLCheckpointTensorConverter` to real tensors, compares converted key/shape/dtype metadata against the VeOmni generated model state, and records value fingerprints for sampled payload evidence. The remote sample reads 11 real tensor payloads from Hugging Face `model-00001/00003/00026/00059-of-00059.safetensors` via HTTP Range (`payload_bytes_read=55808`), maps language dense/sparse attention norms, a MoE correction bias, vision tower tensors, multi-modal projector bias, and patch-merge projector bias into generated VeOmni state keys, and passes with `shape_mismatch_count=0`, `missing_model_key_count=0`, and only the expected `F32->BF16` runtime cast for the correction bias. With `--mode forward --confirm-full-load`, the same script now runs sequentially: it loads the complete local checkpoint into the upstream transformers reference, records fixed-prompt logits/top-k/greedy baseline, releases the HF model, then streams converted public checkpoint tensors into the VeOmni generated model without materializing a full converted tensor dict. The toy checkpoint forward smoke now runs a mixed image+video prompt and passes with `streaming_model_load=true`, strict missing/unexpected key counts of `0`, `forward.logits`, `forward.image_hidden_states`, and `forward.video_hidden_states` max diff `0.0`, equal top-k ids, and equal greedy ids. Full 869 GB payload forward parity is still not claimed in this report.

## Known Blockers

1. **Full public checkpoint payload load.** Public config/preprocessor metadata loads from HF, ModelScope raw config/preprocessor files are byte-identical to HF, and the HF `model.safetensors.index.json` has `23416` weight-map keys. Index-level conversion now covers all generated model `state_dict()` keys: `1525` parameters plus `57` persistent buffers, with `missing_state_key_count=0`, `unexpected_index_key_count=0`, and no missing projector keys. Safetensors header metadata has also been read for all `59` public shards, with `shape_mismatch_count=0` after conversion. CPU/NPU toy HF-vs-VeOmni precision parity passes, and remote real-tensor payload sampling passes for selected tensors, but full checkpoint loading still cannot be claimed complete until the 869 GB tensor payloads are downloaded and loaded through the converter, followed by logits/top-k/greedy parity.
2. **Full real multimodal trainer fixture.** The data path, trainer hooks, VeOmni pre-decoded-frame video fetch, local path/bytes MP4 decode through PyAV fallback, a single-process real-image/video-processor trainer-glue forward/backward fixture, and CPU `VLMTrainer.__init__()` image-only/video-only/mixed dataloader+optimizer smoke over local JPEG/MP4 path/bytes inputs are present. Remaining trainer evidence still needs public-checkpoint full trainer smoke and multi-card torchrun coverage.
3. **SP/EP e2e alignment.** `tests/e2e/test_e2e_parallel.py` now uses a MiniMax-specific dummy VLM dataset with 1-D position ids and MiniMax image/video masks, but the alignment case remains xfail until multimodal SP/EP behavior is validated on hardware.
4. **NPU kernels and distributed NPU.** Single-card Ascend 910B3 HF-vs-VeOmni toy precision parity and generated-model toy SFT loss both pass in the `quay.io/ascend/vllm-ascend:v0.20.2rc1` container with `torch_npu==2.10.0`, but no Ascend-specific RMSNorm/RoPE/attention/MSA kernel replacement or multi-card NPU SP/EP/FSDP2 result is claimed in this PR.
