# Kimi K3 training guide

Kimi K3 is a hybrid multimodal MoE: 93 transformer layers mixing 69 KDA
(linear-attention) layers with 24 full-attention (MLA) layers, a latent-space MoE (experts run
on a compressed latent), and a MoonViT3d vision tower whose merged patches are
injected through a single `media_pad` placeholder per image.

This guide runs K3 end-to-end on Ascend NPUs through the standard VeOmni VLM
trainer — `bash train.sh tasks/train_vlm.py <config>` — with no
task-specific scripts. Verified on 8×910B3 with a 30-step full-parameter
run: loss `13.5 → 9.3`, ~17 s/step, distributed checkpoint and resumable
trainer state saved. The model package also carries GPU code paths; a
verified GPU recipe will follow.

## Install dependencies

Use the NPU image published at [quay.io/ascend/veomni](https://quay.io/repository/ascend/veomni):

```
quay.io/ascend/veomni:v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni
```

The container must run with the ascend runtime, privileged mode, and the
three management devices (`davinci_manager` / `devmm_svm` / `hisi_hdc`) in
addition to `ASCEND_VISIBLE_DEVICES`. A plain runc container with raw
`/dev/davinci*` mounts enumerates devices but every aclnn op fails with
`SpaceRegistry is nullptr`. See
[Ascend Docker usage](../hardware_support/AscendDockerUsage/overview.md) for details.

K3-specific operators need extra packages (inside the container):

```bash
pip install transformers==5.16.1
pip install -e patchgen-pkg --no-build-isolation
pip install ruff==0.13.2 pytest expecttest rich
# KDA AscendC kernels (chunked KDA fwd/bwd): local build of fla-npu; its
# setup.py also builds and installs the AscendC OPP custom-op pack (needs
# the CANN env - the image above already exports it)
git clone https://github.com/flashserve/flash-linear-attention-npu.git fla-npu
pip install ./fla-npu --no-build-isolation
# KDA helper ops (autocast guards, l2norm, chunk indices, triton reference
# path); --no-deps: pyproject pins triton-ascend==3.2.0
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
pip install -e ./triton-ascend-kernels --no-deps --no-build-isolation
```

Dependency check:

```bash
python3 -c "from fla_npu.ops.ascendc import chunk_kda_fwd, chunk_kda_bwd; \
from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda_fwd as t; print('KDA deps OK')"
```

## Prepare dataset

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
- /path/to/k3_coco_val2017.jsonl
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

## Prepare the tokenizer

The recipe needs a HF-format Kimi K3 tokenizer + processor directory. If you
have a hub repo, use the in-repo downloader:

```bash
python3 scripts/download_hf_model.py --repo_id <kimi-k3-repo> --local_dir ./kimi_k3_hf
```

The recipe trains from random init (`model_path: null`), so only the
tokenizer is required.

## Start training on NPU

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_vl_ascendc.yaml \
    --model.tokenizer_path ./kimi_k3_hf \
    --data.train_path configs/multimodal/data/k3_coco.yaml
```

Expected on 8×910B3: step-1 loss ≈ 13 (random init over the 163 840 vocab),
~17 s/step steady state, checkpoints under `kimi_k3_sft/checkpoints/`
(both the distributed model checkpoint and the resumable trainer state).

## Scale up

The full native architecture — 93 layers / 896 experts, everything else
native — lives in `configs/model_configs/kimi_k3/kimi_k3.json`. The recipe
reduces it to a single 8-card node with two overrides; the KDA /
full-attention layer pattern is rebuilt automatically for the new depth:

```yaml
model:
  config_path: configs/model_configs/kimi_k3/kimi_k3.json
  model_config:
    text_config:
      num_hidden_layers: 10
      num_experts: 16
```

Drop the `model_config` block for the full layout (multi-node), or set
`model_path` to a released checkpoint to fine-tune real weights.

## Training config notes

| Block | Keys | Notes |
|---|---|---|
| model | `config_path: configs/model_configs/kimi_k3/kimi_k3.json` | full native architecture (93 layers / 896 experts) |
| | `model_config.text_config`: `num_hidden_layers: 10`, `num_experts: 16` | single-node reduction; see [Scale up](#scale-up) |
| | `model_path: null` | random init (to_empty-safe re-init is patched) |
| | `tokenizer_path` | HF-format tokenizer + AutoProcessor (auto_map) |
| model.ops_implementation | six knobs | `attn: veomni_flash_attention_2`, `rms_norm: torch_npu`, `ce: chunk_loss`, `moe: fused_npu`, `chunk_kda: ascendc`, `situ: triton` |
| model.accelerator | `ulysses_size: 2`, `ep_size: 2`, fsdp2, `init_device: meta` | SP=2 (text-sharded) + EP=2 + FSDP2 |
| data | `train_path` (manifest), `conversation`, `max_seq_len: 1024` | dynamic batching budgets **expanded** vision tokens (see below) |
| train | `max_steps: 30`, micro 1, global 8 | smoke scale; lr 1e-3 cosine |

K3-specific behavior to know when writing configs or custom data pipelines:

1. **Vision-redundant SP** — under `ulysses_size > 1` the text tensors are
   sequence-sharded, but `pixel_values` / `grid_thws` stay **full on every SP
   rank** (each rank runs the whole vision tower); the model hook
   `get_extra_collate_infos` exempts them from sp-slicing. This is the
   opposite of the qwen convention (patch-sharded ViT).
2. **Expanded-length budgeting** — K3 expands one `media_pad` per image to
   `t·h·w / (mh·mw)` merged features *inside the model*, so a plain
   `attention_mask` count would overpack (~30 images into a 1024-token
   budget). The hook `get_dyn_bsz_length_fn` supplies the expanded length to
   the dynamic packer.
3. **Labels shift is SP-aware** — `chunk_loss` consumes the SP collator's
   pre-shifted labels when SP is on and shifts internally otherwise; both
   paths align with no extra handling.
4. **Selective recompute under EP** — under expert parallelism whole-layer
   checkpointing would replay the MoE alltoall collectives inside backward
   and desynchronize them (all-NaN gradients); K3 therefore checkpoints only
   the attention stack and keeps MoE activations live.

Model code is patchgen-managed — never edit files under `generated/` by
hand:

```bash
make patchgen          # regenerate patchgen configs
make check-patchgen    # CI gate: generated files up to date
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `SpaceRegistry is nullptr` / `561103` | container must use ascend runtime + privileged mode, not runc + raw davinci mounts |
| `tiling function does not exist` | fla-npu OPP pack not installed |
| `ImportError: chunk_kda_bwd` | fla-npu too old; install the local build |
| OOM at CE with many images per pack | expanded-length hook not engaged — check the collate table log at startup |
