# Kimi K3 multimodal training on Ascend NPUs

Kimi K3 is a hybrid multimodal MoE: 93 transformer layers mixing 68 KDA
(linear-attention) layers with 25 MLA layers, a latent-space MoE (experts run
on a compressed latent), and a MoonViT3d vision tower whose merged patches are
injected through a single `media_pad` placeholder per image. This recipe runs
K3 end-to-end on 8×Ascend 910B3 through the standard VeOmni VLM trainer —
`train.sh tasks/train_vlm.py <config>` — with no task-specific scripts.

Verified on 8×910B3 (60 GB HBM each):

- 30-step config-driven smoke: `rc=0`, loss `12.99 → 6.23`, model + trainer
  state checkpoints saved (`stage-7 parity record`).
- 300-step full-parameter run (2.69 B stage-6 width): loss `13.60 → 4.50`,
  **239/239 parameters with live gradients** at steps 0/150/299, 1.24 s/step,
  14.65 GiB peak HBM.

## 1. Model package

```
veomni/models/transformers/kimi_k3/
├── source/                     # vendored upstream (modeling + configuration)
├── kimi_k3_gpu_patch_gen_config.py    # patch source of truth (shared fns)
├── kimi_k3_npu_patch_gen_config.py    # NPU registrations (imports gpu fns)
└── generated/                  # make patchgen output (+ .diff sidecars)
```

Patches are formalized through patchgen — never edit `generated/` by hand:

```bash
make patchgen          # regen all 33 configs (incl. both kimi_k3 variants)
make check-patchgen    # CI gate: generated files up to date
```

K3-specific operators: `chunk_kda` (AscendC via fla-npu), `situ_and_mul` /
fused MoE (NPU triton-ascend), `rms_norm` (NPU), chunked cross-entropy —
all selected through the `ops_implementation` block of the training config.

## 2. Environment

Base image:

```
quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni
```

Container must use the **ascend runtime + privileged + ASCEND_VISIBLE_DEVICES
+ the three management devices** (`davinci_manager` / `devmm_svm` /
`hisi_hdc`). A `runc` container with raw `/dev/davinci0-7` mounts enumerates
devices fine but every aclnn op dies with `SpaceRegistry is nullptr` — see the
full recipe and pitfall list in `k3_workspace/K3_ENV_SETUP.md`.

Pip extras (inside the container):

```bash
pip install transformers==5.16.1
pip install -e patchgen-pkg --no-build-isolation
pip install ruff==0.13.2 pytest expecttest rich
# KDA AscendC kernels: build fla-npu locally, setup.py installs the OPP pack
pip install /path/to/fla-npu --no-build-isolation
# KDA wrapper helpers; --no-deps: pyproject pins triton-ascend==3.2.0
pip install -e /path/to/triton-ascend-kernels --no-deps --no-build-isolation
```

Dependency check:

```bash
python3 -c "from fla_npu.ops.ascendc import chunk_kda_fwd, chunk_kda_bwd; \
from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda_fwd as t; print('KDA deps OK')"
```

## 3. Data

One JSONL record per sample (image path + caption; `width`/`height` let the
transform render the exact media header):

```json
{"image": "/data/coco2017/val2017/000000039769.jpg", "width": 640, "height": 480,
 "caption": "A cat sitting on a mattress."}
```

Build it from COCO captions with the in-repo converter:

```bash
python scripts/multimodal/convert_data/coco_to_conversations.py \
  --coco-root /data/coco2017 --split val2017 \
  --output /data/k3_coco_val2017.jsonl
```

Wire it through a data manifest (`configs/multimodal/data/k3_coco.yaml`):

```yaml
sources:
- /data/k3_coco_val2017.jsonl
names:
- kimi_k3_coco
schedule:
- schedule_type: const
  weights: [1.0]
```

The `kimi_k3` transform is registered by `model_type` and renders the
single-placeholder media template
`<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|> {caption}`,
supervising only the caption + EOS.

## 4. Training config

`configs/multimodal/kimi_k3/kimi_k3_stage6_ascendc.yaml` — the verified smoke
recipe (30 steps, 2.69 B stage-6 width, random init):

| Block | Keys | Notes |
|---|---|---|
| model | `config_path: tests/toy_config/kimi_k3_stage6` | architecture; swap for a real-width config to scale up |
| | `model_path: null` | no weights → random init (to_empty-safe re-init is patched) |
| | `tokenizer_path: <kimi_k3_hf repo>` | tokenizer + AutoProcessor (auto_map) |
| model.ops_implementation | six knobs | `attn: veomni_flash_attention_2_with_sp`, `rms_norm: npu`, `ce: chunk_loss`, `moe: fused_npu`, `chunk_kda: ascendc`, `situ: npu` |
| model.accelerator | `ulysses_size: 2`, `ep_size: 2`, fsdp2, `init_device: meta` | SP=2 (text-sharded) + EP=2 + FSDP2 |
| data | `train_path` (manifest), `conversation`, `max_seq_len: 1024` | dynamic batching budgets **expanded** vision tokens (see §6) |
| train | `max_steps: 30`, micro 1, global 8 | smoke scale; lr 1e-3 cosine |

## 5. Launch

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_stage6_ascendc.yaml
```

Expected on 8×910B3: step-1 loss ≈ 13 (random init over the 163 840 vocab),
~1.55 s/it steady state, checkpoints under `k3_stage6_trainer_smoke/checkpoints/`
(both the distributed model checkpoint and the resumable trainer state).

## 6. K3-specific integration contracts

These are the four contracts that make the standard trainer work for K3; all
are model-side (patchgen) so the framework stays untouched except for two
small generic hooks:

1. **Vision-redundant SP** — under `ulysses_size > 1` the text tensors are
   sequence-sharded, but `pixel_values` / `grid_thws` stay **full on every SP
   rank** (each rank runs the whole vision tower). The model hook
   `get_extra_collate_infos` exempts them from sp-slicing — the opposite of
   the qwen convention (patch-sharded ViT).
2. **Expanded-length budgeting** — K3 expands one `media_pad` per image to
   `t·h·w / (mh·mw)` merged features *inside the model* (occupation table), so
   a plain `attention_mask` count would pack ~30 images into a 1024-token
   budget. The hook `get_dyn_bsz_length_fn` returns a picklable
   `functools.partial` of the module-level `kimi_k3_expanded_length`, and
   `data_loader` accepts an optional `dyn_bsz_length_fn` override.
3. **Labels shift is SP-aware** — `chunk_loss` uses the SP collator's
   pre-shifted labels when `sp_enabled` and shifts internally otherwise; both
   the K3 and stock paths align with no extra handling.
4. **Selective recompute under EP** — `KimiDecoderLayer` inherits transformers'
   `GradientCheckpointingLayer`, but under EP whole-layer checkpointing would
   replay the MoE alltoall collectives inside backward and desynchronize them
   (all-NaN gradients, observed at stage 6). K3 therefore checkpoints **only
   the attention stack** and keeps MoE activations live.

## 7. Verification gates

```bash
make quality          # ruff check + format
make check-patchgen   # 33 configs up to date
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_stage6_ascendc.yaml
```

Full-parameter evidence (300-step run, 239/239 live gradients, per-op kernel
bindings) is recorded in `k3_workspace/k3_p2_experiment_record.md` and
`k3_workspace/K3_VERIFICATION_REPORT.md`.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `SpaceRegistry is nullptr` / `561103` | container recipe — must be ascend runtime + privileged, not runc + raw davinci mounts |
| `tiling function does not exist` | fla-npu OPP pack not installed (§2) |
| `ImportError: chunk_kda_bwd` | fla-npu too old (mirror ships 1.0.0); install the local build |
| OOM at CE with many images per pack | `dyn_bsz_length_fn` hook not engaged — check the collate table log at startup |
| GPU migration | op-coverage analysis in `k3_workspace/K3_GPU_BASELINE_ANALYSIS.md` (config-only switch; 4 eager fallback gaps) |
