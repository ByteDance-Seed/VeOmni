# Kimi K3 training guide

Kimi K3 is a hybrid multimodal MoE: 93 transformer layers mixing 69 KDA
(linear-attention) layers with 24 full-attention (MLA) layers, a latent-space MoE (experts run
on a compressed latent), and a MoonViT3d vision tower whose merged patches are
injected through pre-expanded `media_pad` placeholders (one per merged vision
token, materialized by the data transform).

This guide runs K3 end-to-end on Ascend NPUs through the standard VeOmni VLM
trainer — `bash train.sh tasks/train_vlm.py <config>` — with no
task-specific scripts. Verified on 8×910B3: a fresh run starts at loss
≈ 13.6, and a 200-step full-parameter resume from a published step-10
distributed checkpoint converges 9.2 → 6.9 on the fused stack,
bit-reproducible run over run and in verified training parity with the
GPU stacks (see the parity section below). The resume protocol itself
exercises both the distributed checkpoint and the resumable trainer
state.
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

Both packages stay outside VeOmni's lockfile (same policy as the
`gated_delta_rule` `fla_npu` backend), so stock NPU CI images do not ship
them. With the fields unset the platform default resolves to `ascendc` on
NPU, which falls back to the eager reference when the packages are
missing; this recipe pins `chunk_kda_implementation: ascendc` — the
registry records `requires=("fla_npu", "triton_ascend_kernels")` on that
row, and an explicitly selected implementation never falls back: it
raises a `RuntimeError` at op selection instead. To run without the
packages, switch the recipe to `chunk_kda_implementation: eager`; the
AscendC contract tests skip with a `missing: ...` reason either way.
Install both packages on production NPU hosts as above before keeping
`ascendc`.

The `ascendc` row runs guarded by default: at first call it asserts the
installed `fla_npu` version (supported: `1.0.x`) and the integrity of the
canonical chunk-indices table the aclnn kernel consumes, and the varlen
router keeps known-bad shapes off the fused kernel (segments shorter than
4 tokens and calls above the 32-chunk budget run through the eager
reference instead — the fused kernel corrupts those). The full set of
escape-hatch knobs:

| Knob | Default | Effect |
|---|---|---|
| `VEOMNI_KDA_ASCENDC_GUARDED` | `1` | `0` disables the AscendC routing guards (raw fused calls; upstream-kernel debugging only) |
| `VEOMNI_KDA_DEBUG_FINITE_CHECK` | `0` | `1` re-checks finiteness of every fused call and falls back to the eager reference on corruption (bring-up debugging; one device-to-host sync per call) |
| `K3_ALLOW_DENSE_PACKED` | `0` | `1` re-enables the dense eager/sdpa MLA path for packed varlen batches (debug only: one causal mask across the packed row leaks across samples; the CPU packing tests use it) |
| `K3_KDA_HEAD_SHARD` | `1` | `0` selects the legacy full-sequence-gather KDA path under sequence parallel instead of the head-sharded ulysses all-to-all (also the cache / inference path) |

## Prepare dataset

One JSONL record per sample (image path + caption; `width`/`height` let the
transform render the exact media header):

```json
{"image": "/data/coco2017/val2017/000000039769.jpg", "width": 640, "height": 480, "caption": "A cat sitting on a mattress."}
```

Build it from COCO captions with the in-repo converter:

```bash
python scripts/multimodal/convert_data/coco_to_conversations.py \
  --coco-root /data/coco2017 --split val2017 \
  --output /data/k3_coco_val2017.jsonl
```

Point the data manifest at the converter output - edit
`configs/multimodal/data/k3_coco.yaml` to use the generated dataset:

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
pre-expanded media template
`<|media_begin|>image {w}x{h}<|media_content|><|media_pad|>...<|media_end|> {caption}`,
with one `<|media_pad|>` per merged vision token: `(h // mh) * (w // mw)`
from the preprocessed `grid_thws`, **independent of `t`** — the tower's
`tpool_patch_merger` pools (averages) the temporal axis instead of
concatenating it, so a `[2, 4, 4]` grid yields 4 pads, not 8. The
transform supervises only the caption + EOS. The
in-model merge is therefore length-preserving (it masked-scatters the
projected features onto the pad slots), which keeps the collator's packed
position_ids / cu_seq_lens and the SP label pre-shift valid; a
`max_seq_len` that would cut into the media span fails loudly instead of
desynchronizing `input_ids` from the published `grid_thws`.

## Prepare the tokenizer

The recipe needs a HF-format Kimi K3 tokenizer + processor directory. If you
have a hub repo, use the in-repo downloader:

```bash
python3 scripts/download_hf_model.py --repo_id <kimi-k3-repo> --local_dir ./kimi_k3_hf
```

The downloader appends the repository name to `--local_dir`, so the assets
land in `./kimi_k3_hf/<repo-name>` - use that directory as the
`tokenizer_path` in the launch commands below.

The recipe trains from random init (`model_path: null`), so no pretrained
weights are needed — but the tokenizer directory must still include the
image processor assets (`preprocessor_config.json`): the transform
preprocesses images through the processor built from the same path, and a
tokenizer-only directory leaves it unset.

## Start training on NPU

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_vl_ascendc.yaml \
    --model.tokenizer_path ./kimi_k3_hf/<repo-name> \
    --data.train_path configs/multimodal/data/k3_coco.yaml
```

Expected on 8×910B3: step-1 loss ≈ 13 (random init over the 163 840
vocab), checkpoints under `kimi_k3_sft/checkpoints/` (both the
distributed model checkpoint and the resumable trainer state).

## Start training on GPU

The model package runs the same trainer on NVIDIA GPUs. Install the GPU
extra on an x86_64 Linux host (Python 3.11 or 3.12, glibc >= 2.28, CUDA
12.x / 13.x) - the extra's `flash-attn` build, which the attention/standard
FA2 rows require on GPU (the text decoder and the vision tower both bind
them), is published for x86_64 only:

```bash
uv sync --extra gpu
source .venv/bin/activate
```

`uv` is the supported installer here: the GPU extra pins
`torch==2.11.0+cu130` and a wheel-only `deep-gemm` build through
`[tool.uv.sources]`, which `pip install -e ".[gpu]"` cannot resolve.

The GPU recipe `configs/multimodal/kimi_k3/kimi_k3_vl.yaml` pins the
official-default baseline the NPU fused stack compares against: flash-attn
FA2 attention (through the sequence-parallel adapter), eager RMSNorm /
cross-entropy, the vendored eager `chunk_kda_naive` reference (portable
and autograd-friendly — the same math the NPU eager row and the AscendC
parity tests run), and the SituGLU eager expert
loop (expert-parallel aware; fused GPU MoE rows may follow). The `fla`
row stays available as an optional selection for compatible
flash-linear-attention builds (version-asserted: the row feature-detects
the KDA gating parameters on the installed build and raises
`NotImplementedError` naming what is missing instead of silently dropping
them). Launch it exactly like the NPU recipe:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_vl.yaml \
    --model.tokenizer_path ./kimi_k3_hf/<repo-name> \
    --data.train_path configs/multimodal/data/k3_coco.yaml
```

Expected step-1 loss ≈ 13 (same random init over the 163 840 vocab).
`ep_size: 2` routes the eager expert loop through the shared EP all-to-all —
it trades compute replication for less expert memory per rank.

### NPU↔GPU training parity

Training parity between the GPU stack and both NPU stacks (eager and
fused) is verified with a fixed 200-step resume protocol. All three runs
restart from the *same* step-10 distributed checkpoint — the published
snapshot at ModelScope `wangyier7/kimi-k3-align-ckpt-1113` (resume from
its `checkpoints/global_step_10`) — with identical data order, seeding,
batching and lr schedule:

```bash
--data.dataloader.shuffle false
--train.seed 1234
--train.enable_full_determinism true
--train.max_steps 210   # global cap: resumes at 10, runs 200 new steps (11..210)
--train.micro_batch_size 1  --train.global_batch_size 8
--train.checkpoint.load_path <...>/checkpoints/global_step_10
```

The protocol pins the GPU side to the shared eager reference
(`--model.ops_implementation.chunk_kda_implementation eager` — the
vendored `chunk_kda_naive`, the same code the NPU eager stack executes),
anchoring both machines to one reference implementation. The dispatch is
test-pinned, so an installed fused library cannot silently contaminate
it: the version-asserted `fla` row requires the K3 gating parameters as
named parameters and refuses `flash-linear-attention 0.4.1` (which
carries them only through `**kwargs`, i.e. not at all) with a
`NotImplementedError` naming what is missing. The GPU recipe pins `eager`
to match this protocol; selecting the `fla` row instead requires a
flash-linear-attention build that exposes those K3 gating parameters as
named `chunk_kda` arguments (anything newer than the locked 0.4.1 that
added `lower_bound` / `safe_gate` / `transpose_state_layout` /
`state_v_first` / `use_beta_sigmoid_in_kernel`).

Per-step loss deviation over the 200 steps (absolute, at the trainer's
2-decimal print precision):

| pair | mean \|Δ\| | max \|Δ\| | steps > 0.05 | equal @ print | loss @ 210 |
|---|---|---|---|---|---|
| GPU eager vs NPU eager | 0.010 | 0.04 | 0 / 200 | 65 / 200 | 6.88 / 6.87 |
| GPU eager vs NPU fused | 0.011 | 0.21 | 2 / 200 | 72 / 200 | 6.88 / 6.88 |
| NPU fused vs NPU eager | 0.015 | 0.21 | 4 / 200 | 55 / 200 | 6.88 / 6.87 |

The cross-device gap (GPU eager vs NPU eager, max 0.04) is *tighter*
than the same-device kernel gap on the NPU itself (fused vs eager, max
0.21): the fused AscendC kernel's deviation from the reference is the
same order as cross-platform bf16 accumulation noise, and there is no
device-specific systematic error. The single 0.21 outlier (step 87) is
transient — the trajectories reconverge immediately and never drift
apart. 50-step block means confirm no drift over the horizon:

| steps | GPU eager | NPU fused | NPU eager |
|---|---|---|---|
| 11–60 | 10.229 | 10.233 | 10.227 |
| 61–110 | 8.843 | 8.841 | 8.845 |
| 111–160 | 6.970 | 6.971 | 6.973 |
| 161–210 | 6.173 | 6.174 | 6.172 |

Cumulative sum of the 200 per-step losses: GPU 1610.75 / NPU fused
1610.94 / NPU eager 1610.81 (fused = 100.012 % of GPU). The NPU side is
bit-reproducible run over run on both stacks. Wall-clock speeds are
deliberately not quoted: the current stacks run with allocator
workarounds (CPU offload on the eager stack,
`MULTI_STREAM_MEMORY_REUSE=0` on resume) that do not reflect final
performance.

Table provenance — NPU: CANN 9.1.0, torch_npu 2.10.0.post2,
`flash-linear-attention-npu-a2 26.10.0+main.dev3f4c016`; GPU: an offline
environment (torch 2.11.0+cu130, transformers 5.16.1, tokenizers 0.23.3),
KDA pinned to the eager reference. Cross-machine data identity is
enforced by a fingerprint protocol (dataset manifest hashes,
order-critical library pins, and a first-batch `input_ids` diff), so a
different tokenization or sample order cannot slip in unnoticed.

One asymmetry to know: at global batch 8 the NPU **eager** stack needs
`--model.accelerator.fsdp_config.offload true` to fit, while the fused
stack runs without offload (and, on resume, wants
`MULTI_STREAM_MEMORY_REUSE=0` so HCCL's pool allocations stay inside the
device budget).

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

Drop the `model_config` block for the full layout (multi-node). The same
applies when fine-tuning a released checkpoint: set `model_path` to the
checkpoint and remove the `model_config` overrides with it - the recipe's
reduced 10-layer / 16-expert shape will not match the full weights.

## Training config notes

| Block | Keys | Notes |
|---|---|---|
| model | `config_path: configs/model_configs/kimi_k3/kimi_k3.json` | full native architecture (93 layers / 896 experts) |
| | `model_config.text_config`: `num_hidden_layers: 10`, `num_experts: 16` | single-node reduction; see [Scale up](#scale-up) |
| | `model_path: null` | random init (to_empty-safe re-init is patched) |
| | `tokenizer_path` | HF-format tokenizer + AutoProcessor (auto_map) |
| model.ops_implementation | six knobs | `attn: veomni_flash_attention_2`, `rms_norm: torch_npu`, `ce: chunk_loss`, `moe: fused_npu`, `chunk_kda: ascendc`, `situ: triton` |
| model.accelerator | `ulysses_size: 2`, `ep_size: 2`, fsdp2, `init_device: meta` | SP=2 (text + vision sharded) + EP=2 + FSDP2 |
| data | `train_path` (manifest), `conversation`, `max_seq_len: 1024` | the transform pre-expands the media span, so plain token counts already budget vision tokens (see below) |
| train | `max_steps: 30`, micro 1, global 8 | smoke scale; lr 1e-3 cosine |

K3-specific behavior to know when writing configs or custom data pipelines:

1. **Vision-sharded SP** — under `ulysses_size > 1` the collator sp-slices
   `pixel_values` (pad_scale=4 keeps every 2x2 merge window whole), so each
   rank runs the vision tower on 1/sp of the token rows; `grid_thws` stays
   global, and the tower derives the global cu_seqlens / pos-emb / rope
   tables from it and sp-slices those to the local rows. The tower's
   attention/standard rows exchange the packed sequence internally
   (Ulysses), and the features are gathered back across the SP group
   before the image-atomic `tpool_patch_merger` (the gather's backward
   all-reduces the grads, so no cross-rank contribution is lost). This
   mirrors the qwen convention (patch-sharded ViT); the SP path requires a
   `veomni_*` attention selection — raw eager/sdpa/flash rows refuse the
   attention-internal exchange — which both recipes already pin.
2. **Pre-expanded media span** — the kimi_k3 data transform materializes
   one `<|media_pad|>` per merged vision token (`(h // mh) * (w // mw)`,
   independent of `t`: the temporal axis is pooled by `tpool_patch_merger`,
   not concatenated) *before* packing, so the in-model merge is
   length-preserving: it
   masked-scatters the projected features onto the pad slots and leaves
   the collator's packed position_ids / cu_seq_lens untouched. Token
   counts are therefore exact at packing time — no expanded-length hook,
   no in-model rebuild — and the merge rejects the legacy
   one-placeholder-per-image form under SP (`require_pre_expanded`),
   where the collator metadata assumes a length-preserving merge.
  3. **Labels shift is SP-aware** — the SP collator pre-shifts labels on
    the full row and the VeOmni CE consumes them as-is; the non-SP path
    shifts raw labels inside the loss helper. Exactly one causal shift
    happens on every path: the pre-expanded merge scatters labels 1:1
    (length-preserving), so the SP pre-shift survives it untouched, and
    the media span carries `ignore_index` (written by the transform),
    which keeps vision positions out of the loss either way.
4. **Selective recompute under EP** — under expert parallelism whole-layer
   checkpointing would replay the MoE alltoall collectives inside backward
   and desynchronize them (all-NaN gradients); K3 therefore checkpoints only
   the attention stack and keeps MoE activations live.
5. **FSDP2 only** — `fsdp_mode` must stay `fsdp2`. The first KDA layer's
   attention-res parameters (`layers.0.self_attention_res_{norm,proj}`)
   are architecturally unused — the block-residual chain starts with a
   zero-width state, so they never receive a gradient — and the plain
   `ddp` reducer aborts the second optimizer step over them (`Expected
   to have finished reduction in the prior iteration`). FSDP2's flat-
   parameter resharding is unaffected.

Model code is a standalone module — edit
`veomni/models/transformers/kimi_k3/modeling_kimi_k3.py` directly (the same
layout every VeOmni family uses; there is no generation step).
