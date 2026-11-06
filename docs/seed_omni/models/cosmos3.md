# Cosmos3-Nano (SeedOmni)

SFT recipe for **Cosmos3-Nano**, a mixture-of-towers text-to-video diffusion
transformer, as a SeedOmni graph model. It follows the open-source
`cosmos-framework` `vision_sft_nano` recipe. This is an independent
implementation written from the released checkpoint layout: it copies no
`cosmos-framework` source and has no runtime dependency on it. Cosmos3-Nano is
released under OpenMDW-1.1; check its terms before redistributing the weights.

The checkpoint is split into four OmniModules:

| Module | Holds | Role |
|--------|-------|------|
| `cosmos3_text_encoder` | `embed_tokens`, `lm_head` (never trained), tokenizer | Tokenizes the prompt in the DataLoader worker and embeds it |
| `cosmos3_vae` | Wan2.2 `AutoencoderKLWan` | Cuts and resizes the clip in the worker, then encodes it to a normalized latent. Frozen, cacheable |
| `cosmos3_mot` | the 36 decoder blocks of both towers, the final norms, the 3-D mRoPE | Runs the prompt and the noised latent through the joint two-pass attention |
| `cosmos3_flow_connector` | `proj_in`, `proj_out`, `time_embedder` | Draws sigma / noise / clean-prefix mask, embeds the noised latent, returns the loss |

Both towers stay in `cosmos3_mot` because each block's two passes share one
attention softmax: the generation queries read the reasoner's keys of the same
block, so the blocks cannot be split between modules.

Config dir: `configs/seed_omni/Cosmos3/cosmos3_nano/`.

| File | Role |
|------|------|
| `train/online.yaml` | Online recipe: the VAE encodes inside the step. |
| `train/offline_embedding.yaml` | Stage 1 of 2: encode every clip once into a latent cache. |
| `train/offline_training.yaml` | Stage 2 of 2: train on that cache. |
| `train/modules_train.yaml` | Per-module settings shared by the two training recipes, including the freeze switch. |
| `train/modules_offline_embedding.yaml` | The VAE alone, for stage 1. |
| `train/graph_*.yaml` | One training DAG per recipe. |
| `infer/graph_infer.yaml` | Placeholder generation graph; see [What is not supported](#5-what-is-not-supported). |

---

## 1. Checkpoint

The release is a diffusers-style tree (`transformer/`, `vae/`, `text_tokenizer/`)
with no root `config.json`, so the converter is registered on the transformer's
`model_type` (`qwen3_vl_text`) and runs on that subfolder:

```bash
modelscope download --model nvidia/Cosmos3-Nano --local_dir pretrained_models/Cosmos3-Nano
python scripts/seed_omni/convert_model.py \
  --model_path pretrained_models/Cosmos3-Nano/transformer \
  --output_dir pretrained_models/Cosmos3-Nano-omni
```

Every transformer tensor is routed by a prefix table and loaded with
`strict=True`, so a key that no module owns, or a module parameter the
checkpoint lacks, stops the conversion. Two groups are dropped on purpose: the
Qwen3-VL vision encoder (`blocks.`, `merger.`, `patch_embed.`, ...), which this
recipe never runs, and the audio / action pathways, which Nano does not train.

## 2. The freeze switch

By default only the generation tower trains: the MoT parameters whose names
contain `moe_gen`, `add_*_proj`, `to_add_out` or `norm_added_*` (the recipe's
`keys_to_select`), plus the whole flow connector. The text encoder and the VAE
train nothing.

`train_joint_reasoner: true` releases the reasoner too. Set it on both
`cosmos3_mot` (the reasoner blocks) and `cosmos3_text_encoder` (`embed_tokens`)
in `modules_train.yaml`. `lm_head` stays frozen either way: there is no
autoregressive loss. Joint training needs `fsdp_mode: fsdp2`, because the
reasoner's final norm and the last block's reasoner side never reach the loss;
`freeze_model` rejects `ddp`.

## 3. Data

`data.source_name: cosmos3_sft_jsonl` reads the open-source SFT JSONL dump, one
row per clip:

| Field | Meaning |
|---|---|
| `vision_path` | Clip path. A relative path is joined to `mm_configs.data_dir`; an absolute path or a `://` URI is used as is. |
| `t2w_windows[0].caption_json` | Preferred prompt, serialized verbatim. |
| `t2w_windows[0].caption`, then the row's `caption` | Fallback prompts. A blank value falls through. |

```json
{"vision_path": "videos/clip_0001.mp4",
 "t2w_windows": [{"caption_json": "A bridge at sunset."}]}
```

Only `t2w_windows[0]` is read, and the clip is not cropped to that window.

Where each step of the data pipeline happens:

| Step | Where | Set by |
|---|---|---|
| CFG dropout: one prompt in ten becomes empty | dataset preprocessor, global `random` | `mm_configs.cfg_dropout_rate` |
| Decode every source frame | SeedOmni video loader | `mm_configs.fps: null`, no `max_frames` |
| Keep `duration * sample_fps` frames, aligned to `4k + 1` within `[min_frames, max_frames]`, as a fixed-stride prefix; pad short clips with the last frame | `cosmos3_vae` preprocessor | `cosmos3_vae.model_config` |
| Map to `[-1, 1]` and resize to a multiple of 32 (VAE stride 16 x patch 2) | `cosmos3_vae` preprocessor | — |
| Chat template with the generation prompt, truncate to 2048, append `[eos, <\|vision_start\|>]` | `cosmos3_text_encoder` preprocessor | `max_sequence_length` |

`dyn_bsz` prices a sample before batching from what the two preprocessors write:
the prompt's token count, plus the clip's latent patches (counted on the frames
before the resize) and three marker tokens. That is the reference packer's count,
so `data.max_seq_len: 45056` packs the same way. `cosmos3_text_encoder`'s
`price_prompt_by_chars: true` charges `ceil(len(prompt) / 4)` instead, the
estimate the DiT port used online; it exists only to replay that port's batches.

The frame choice, the price and the mRoPE rate (the source clip's) are the DiT
port's, so the two produce the same batches. The VAE preprocessor refuses a clip
the loader has thinned, since it picks frames on the source timeline.

## 4. Training

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Cosmos3/cosmos3_nano/train/online.yaml
```

Two-stage training runs the VAE once instead of every step:

```bash
# Stage 1: builds the VAE only and writes prompt text + latent per sample.
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Cosmos3/cosmos3_nano/train/offline_embedding.yaml
# Stage 2: the VAE stays on meta; prompts are tokenized as the cache is read.
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Cosmos3/cosmos3_nano/train/offline_training.yaml
```

The cache holds each sample's prompt after CFG dropout, so stage 2 sees the
dropout pattern stage 1 drew. The VAE frame settings decide what is in the cache,
which is why `modules_offline_embedding.yaml` repeats them.

The recipes keep `mixed_precision.enable: false`: weights are bf16 end to end,
like the reference's, and the sigma, timestep and target tensors the connector
passes between its nodes stay fp32 because nothing casts the graph inputs. With
mixed precision on, FSDP would cast them to `param_dtype`.

The sigma / noise / clean-prefix stream comes from a CPU generator in the flow
connector, seeded `seed + dp_rank`, and is checkpointed per rank with that
module, so a resumed run continues the same stream. The CFG-dropout draws use
the global `random` and restart on resume.

## 5. What is not supported

- **Generation.** There is no flow sampler. The launcher requires a generation
  graph for every model, so `infer/graph_infer.yaml` exists only to satisfy it;
  running it raises `NotImplementedError`.
- **Sequence parallelism.** The joint attention spans a whole sample, so every
  module runs with `ulysses_size: 1`; the MoT raises otherwise.
- **Joint training under DDP**, as above.
- **Several clips, or windows, per sample.** One prompt and one clip per sample.
- **Sparse-MoE feed-forwards and the autoregressive text loss.**
