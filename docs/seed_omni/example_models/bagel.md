# BAGEL-7B-MoT (SeedOmni V2)

End-to-end recipe for training and inferring **BAGEL-7B-MoT** as a SeedOmni V2
graph model. The upstream BAGEL checkpoint is split into five OmniModules: text
embedding / LM head, SigLIP-NaViT understanding tower, VAE codec, flow connector,
and Qwen2-MoT backbone.

All paths below assume the upstream BAGEL checkpoint lives at
`/mnt/hdfs/user_dir/veomni_omni/models/transformers/BAGEL-7B-MoT`. Adjust to your
own storage.

Config dir: `configs/seed_omni/Bagel/bagel_7b_mot/`.

| Module | Holds | Role |
|--------|-------|------|
| `bagel_text_encoder` | tokenizer, token embedding, LM head | text/template markers, text logits, image-marker embeddings |
| `bagel_siglip_navit` | SigLIP-NaViT tower + connector | user image context for understanding and edit prompts |
| `bagel_vae` | VAE encoder/decoder | assistant image training latents, edit context latents, generated latent decode |
| `bagel_flow_connector` | VAE↔LLM projections, timestep embedding | latent patch embedding, velocity prediction, denoise state |
| `bagel_qwen2_mot` | Qwen2-MoT decoder backbone | text AR and flow-denoise hidden states |

The omni config layout uses the same `base.yaml` for training and inference. The
training block references `train/modules_train.yaml` and `train/graph_train.yaml`; the
inference block maps each scenario to a separate generation graph.

| File | Role |
|------|------|
| `train/base.yaml` | Top-level launcher: model paths, accelerator, data, train, and `infer` block. |
| `train/modules_train.yaml` | Per-module training paths. `bagel_qwen2_mot` is the accelerated class with `flex_attention`. |
| `infer/modules_infer_eager.yaml` | Single-process inference: every module loads eager; MoT uses SDPA. |
| `infer/modules_infer_fsdp.yaml` | Distributed inference: every module uses FSDP2; MoT uses FlexAttention. |
| `train/graph_train.yaml` | Training DAG. |
| `infer/graph_infer_und.yaml` | Image/text understanding to text. |
| `infer/graph_infer_gen.yaml` | Text to image generation. |
| `infer/graph_infer_edit.yaml` | Text+image to image edit. |
| `data.yaml` | Weighted multisource data list. |

The V2 Bagel wiring currently exposes understanding, generation, and edit.

---

## 1. Convert the checkpoint

The converter reads upstream `llm_config.json`, `vit_config.json`,
`ema.safetensors`, and `ae.safetensors`, then writes one sub-checkpoint per
module:

```bash
python scripts/convert_model.py \
  --model_type bagel \
  --model_path /mnt/hdfs/user_dir/veomni_omni/models/transformers/BAGEL-7B-MoT \
  --output_dir /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT
```

The output root becomes `model.model_path` and `infer.model_path` in
`base.yaml`. It must contain:

```text
BAGEL-7B-MoT/
├── bagel_text_encoder/
├── bagel_siglip_navit/
├── bagel_vae/
├── bagel_flow_connector/
└── bagel_qwen2_mot/
```

`bagel_text_encoder` also stores tokenizer assets copied from the upstream
checkpoint; SigLIP-NaViT and VAE save their processors next to their weights.

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mixture:

```yaml
sources:
  - /mnt/hdfs/user_dir/dataset/imagenet1k_train
  - /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/data
  - /mnt/hdfs/veomni/datasets/sharegpt4v_cap_100k
  - /mnt/hdfs/user_dir/dataset/seed_edit_p23_multi_turn
names:
  - imagenet1k
  - tulu-3-sft-mixture
  - sharegpt4v_cap_100k
  - seed_edit_p23_multi_turn
schedule:
  - { schedule_type: const, weights: [0.4, 0.2, 0.2, 0.2] }
```

The Bagel CPU preprocessor routes images by role:

- user images become `BAGEL_SIGLIP_CONTEXT` for SigLIP-NaViT.
- assistant images become `BAGEL_VAE_CONTEXT` for VAE latent targets.

`ConversationItem.source` is the real producer/consumer branch identity.
`meta["source"]` is only an alignment hint for dummy/worker paths and should not
drive real routing.

---

## 3. Train

Training uses the accelerated Qwen2-MoT class from `train/modules_train.yaml`. The
default packed attention backend is FlexAttention.

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml
```

The training DAG is:

```text
bagel_text_encoder.encode          -> bagel_qwen2_mot
bagel_siglip_navit                 -> bagel_qwen2_mot
bagel_vae.encode                   -> bagel_flow_connector.embed_latent
bagel_flow_connector.embed_latent  -> bagel_qwen2_mot
bagel_qwen2_mot                    -> bagel_text_encoder.decode
bagel_qwen2_mot                    -> bagel_flow_connector.decode_velocity
```

`bagel_flow_connector.embed_latent` patchifies VAE latents, samples the training
noise/timestep, writes the velocity target, and projects noised latent patches
into LLM space. `bagel_flow_connector.decode_velocity` consumes the MoT hidden
states for those latent query positions and computes the flow velocity loss.

Quick smoke run:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml \
  --model.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --train.max_steps 10 \
  --train.global_batch_size 8 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false
```

### 3.1 Attention backends

`bagel_qwen2_mot` keeps one packed visibility metadata and materializes it as
Flex, Magi, or SDPA. Visibility is the same in all three:
`same_document & (causal | same_full_span) & ~foreign_noise_key`. Packed MoT
training does not use FlashAttention-2.

| Backend | Class | Config | Use |
|---------|-------|--------|-----|
| FlexAttention | accelerated | `train/modules_train.yaml`, `infer/modules_infer_fsdp.yaml` | Default packed training / FSDP inference |
| MagiAttention | accelerated | CLI override onto `bagel_qwen2_mot` | SM90+ fused training; see nfunc below |
| SDPA | eager | `infer/modules_infer_eager.yaml` | Single-process inference and the dense-mask oracle |

Leave the YAML on `flex_attention` unless you are opting into Magi. Enable Magi
from the CLI (the launcher rewrites it to `veomni_magi_attention_with_sp`):

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml \
  --model.model_config.modules.bagel_qwen2_mot.ops_implementation.attn_implementation magi_attention \
  --train.micro_batch_size 1
```

Magi requires physical batch size 1, `cp_size == 1`, and NVIDIA SM90 or newer.
Ulysses still works. On 8 ranks, `--model.accelerator.ulysses_size 4` gives
`dp_shard=2` and SP4. Packed Magi/Flex is the training (and FSDP prefill) path;
the denoise loop still uses FlashAttention-2. The Magi adapter contract is in
[`docs/transformers_v5/veomni_fused_attention.md`](../../transformers_v5/veomni_fused_attention.md).

#### SM90 nfunc vs dataset

On SM90, Magi uses a precompiled CUTLASS overlay. `nfunc` is baked at install
time: it is the HSTU interval count for the worst query in a sample, not the
Magi range-list length. The installer default is `1,3,5`. A later exact
`uv sync` removes the overlay, so rerun the installer before SM90 Magi runs.

Text-only and single-image gen samples are typically `nfunc=1`. Multi-turn
`seed_edit_p23_multi_turn` punches noise-key holes, so nfunc grows with the
number of assistant images (about `2G-1`). That mixture needs **nfunc ≥ 11**:

```bash
bash scripts/kernel/install_magi_sm90.sh --nfunc 1,3,5,7,9,11
```

If a sample's runtime nfunc is missing from the compiled matrix, the kernel
fails with `Compile-time kNFunc (...) must match runtime arbitrary_func_num (...)`.
SM100+ uses CUTE DSL/JIT and compiles unseen nfunc at runtime, so it does not
need this overlay matrix.

### 3.2 Offline VAE posterior cache (two stages)

Caching is not a special framework mode — it is a different `train_graph` plus a
different dataset type. A module opts in with `model_config.support_cache: true`,
and `OfflineEncodingMixin.derive_cache_mode`
(`veomni/models/seed_omni/mixins/offline_encoding_mixin.py`) turns
`train.train_type` into that module's cache mode.

**Stage 1 — produce the cache** (`train.train_type: offline_cache` ⇒ VAE cache
mode `encode_only`). `offline_cache/modules_train.yaml` declares only
`bagel_vae`, and the DAG is a single edge, so nothing else is built:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/offline_cache/base.yaml
```

```text
bagel_vae.offline_encode -> end
```

Posteriors are written to `train.offline_cache_dir`
(`outputs/bagel_vae_cached_dataset` by default), reading normal `seedomni` data.

**Stage 2 — train from the cache** (`train.train_type: train_with_cache` ⇒ VAE
cache mode `process_only`, which makes the VAE preprocessor return `None` and
skips CPU image prep entirely). `data.data_type` becomes `seedomni_cached` and
`data.train_path` points at the stage-1 output directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/with_cache/base.yaml
```

The DAG matches §3 except that `bagel_vae.online_process` replaces
`bagel_vae.encode` as the latent source — it rehydrates the cached posterior
instead of encoding pixels.

---

## 4. Inference

`tasks/omni/infer_omni.py` selects a generation graph with `--infer.infer_type`.
Use `infer/modules_infer_eager.yaml` for a single-process run or
`infer/modules_infer_fsdp.yaml` with `bash train.sh` for a torchrun/FSDP2 run.

### 4.1 Understanding

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml \
  --infer.infer_type infer_und \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/infer/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.image /path/to/image.jpg \
  --infer.prompt "Describe this image." \
  --infer.output_dir bagel_out
```

`infer_und` runs SigLIP-NaViT for the prompt image, inserts image marker
embeddings via the text encoder, and then uses Qwen2-MoT + text encoder AR
decode until `text_done`.

### 4.2 Text-to-image generation

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml \
  --infer.infer_type infer_gen \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/infer/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.prompt "A watercolor painting of a small cabin beside a lake." \
  --infer.output_dir bagel_out \
  --infer.generation_kwargs.num_timesteps 50 \
  --infer.generation_kwargs.timestep_shift 3.0
```

`infer_gen` runs prompt prefill, then loops:

```text
flow_connector.prepare_denoise_query
  -> text_encoder.encode_image_markers
  -> qwen2_mot.denoise_branch
  -> flow_connector.decode_velocity_from_hidden
  -> qwen2_mot.collect_velocity
  -> flow_connector.advance_denoise
```

When the denoise state emits `image_complete`, `bagel_vae.decode_generated`
decodes the final `BAGEL_GENERATED_LATENT`.

### 4.3 Image edit

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml \
  --infer.infer_type infer_edit \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/infer/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.image /path/to/source.jpg \
  --infer.prompt "Make it look like a snowy evening." \
  --infer.output_dir bagel_out
```

Edit first builds context from both image branches: VAE encodes the edit image
as `BAGEL_VAE_CONTEXT`, SigLIP-NaViT keeps the raw prompt image as visual
context, and `flow_connector.embed_context_latents` projects the VAE context into
the denoise prompt. The downstream denoise loop is shared with `infer_gen`.

---

## 5. Visualize the graphs

```bash
python scripts/visualize_omni_graph.py \
  configs/seed_omni/Bagel/bagel_7b_mot/train/base.yaml
# -> graphs/bagel_7b_mot_base/{training,infer_edit,infer_gen,infer_und}.mmd
```

The visualized graphs are generated from the same config loader path as training
and inference, so regenerate them after changing module entrypoints or graph
YAML.

---

## 6. Contract checks

The Bagel module and graph contracts cover carrier/source routing, generation
state transitions, packing/cache behavior, and graph config structure. Packed
MoT also compares Flex and Magi against eager SDPA on toy CE / MSE / gradients
in `tests/seed_omni/bagel/test_bagel_accel_align.py`. Magi cases skip unless the
SM90 CUTLASS overlay or SM100+ CUTE JIT backend is present.

```bash
.venv/bin/python -m pytest -q tests/seed_omni/bagel
```
