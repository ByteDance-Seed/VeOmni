# Kimi K3 training guide

Kimi K3 is a hybrid multimodal MoE: 93 transformer layers mixing 69 KDA
(linear-attention) layers with 24 full-attention (MLA) layers, a latent-space MoE (experts run
on a compressed latent), and a MoonViT3d vision tower whose merged patches are
injected through a single `media_pad` placeholder per image.

This guide runs K3 end-to-end on Ascend NPUs through the standard VeOmni VLM
trainer — `bash train.sh tasks/train_vlm.py <config>` — with no
task-specific scripts. Verified on 8×910B3 with a 300-step full-parameter
run (resumed from a step-10 distributed checkpoint): loss `13.7 → 4.4`,
~15 s/step, distributed checkpoint and resumable trainer state saved.
The model package also runs the same trainer on NVIDIA GPUs, with verified
training parity against the NPU stacks; see
[Start training on GPU](#start-training-on-gpu).

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
~15 s/step steady state, checkpoints under `kimi_k3_sft/checkpoints/`
(both the distributed model checkpoint and the resumable trainer state).

## Start training on GPU

The model package runs the same trainer on NVIDIA GPUs. Install the GPU
extra (Python >= 3.10, glibc >= 2.28, CUDA 12.x / 13.x):

```bash
uv sync --extra gpu          # or: pip install -e ".[gpu]"
```

The GPU recipe `configs/multimodal/kimi_k3/kimi_k3_vl.yaml` pins the
official-default baseline the NPU fused stack compares against: flash-attn
FA2 attention (through the sequence-parallel adapter), eager RMSNorm /
cross-entropy, the official `fla` chunk_kda, and the SituGLU eager expert
loop (expert-parallel aware; fused GPU MoE rows may follow). Launch it
exactly like the NPU recipe:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_vl.yaml \
    --model.tokenizer_path ./kimi_k3_hf \
    --data.train_path configs/multimodal/data/k3_coco.yaml
```

Expected step-1 loss ≈ 13 (same random init over the 163 840 vocab).
`ep_size: 2` routes the eager expert loop through the shared EP all-to-all —
it trades compute replication for less expert memory per rank.

### NPU↔GPU training parity

Training parity between the GPU eager recipe and both NPU stacks (eager and
fused) is verified with a fixed 200-step resume protocol. Both sides restart
from the *same* step-10 distributed checkpoint — a published snapshot at
ModelScope `wangyier7/kimi-k3-align-ckpt` (resume from its
`checkpoints/global_step_10`) — with identical data order, seeding,
batching and lr schedule:

```bash
--data.dataloader.shuffle false
--train.seed 1234
--train.enable_full_determinism true
--train.max_steps 200
--train.micro_batch_size 1  --train.global_batch_size 8
--train.checkpoint.load_path <...>/checkpoints/global_step_10
```

Per-step loss deviation over the 200 steps:

| pair | mean | median | max | loss @ step 200 |
|---|---|---|---|---|
| NPU fused vs GPU eager | 0.46% | 0.32% | 2.96% | 6.315 / 6.294 |
| NPU eager vs GPU eager | 0.45% | 0.34% | 2.97% | 6.304 / 6.294 |

The cross-machine deviation sits at the run-to-run variation the GPU
platform itself shows under this protocol (about 0.4% mean between repeated
runs), and the NPU side is bit-reproducible run over run on both stacks.
One asymmetry to know: at global batch 8 the NPU **eager** stack needs
`--model.accelerator.fsdp_config.offload true` to fit, while the fused
stack runs without offload.

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
 3. **Labels shift is SP-aware** — the SP collator pre-shifts labels and the
   non-SP path shifts inside the loss helper. The multimodal merge scatters
   labels position-wise onto text slots only, so under SP the forward
   **un-shifts before the merge** and applies exactly one causal shift to the
   *merged* labels before slicing them back into shards (slice-pad slots read
   `ignore_index`); shifting on the pre-merge row would strand the first
   post-image supervision on a placeholder slot that the merge drops.
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
