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

There is no generation graph: the model is training-only for now, see
[What is not supported](#5-what-is-not-supported).

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
autoregressive loss. So do the reasoner parameters that never reach the loss,
because only the generation tower's output is decoded: the reasoner's final
`norm`, and in the last block `to_q`, `norm_q`, `to_out`,
`post_attention_layernorm` and `mlp` (its keys and values still feed the
generation pass). They would never get a gradient anyway, so freezing them
changes no update; it makes every trainable parameter reach the loss, which is
what DDP needs.

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
| CFG dropout: one prompt in ten becomes empty | dataset preprocessor, a per-sample draw | `mm_configs.cfg_dropout_rate`, `mm_configs.cfg_dropout_seed` |
| Decode every source frame | SeedOmni video loader | `mm_configs.fps: null`, no `max_frames` |
| Keep `duration * sample_fps` frames, aligned to `4k + 1` within `[min_frames, max_frames]`, as a fixed-stride prefix; pad short clips with the last frame | `cosmos3_vae` preprocessor | `cosmos3_vae.model_config` |
| Map to `[-1, 1]` and resize to a multiple of 32 (VAE stride 16 x patch 2) | `cosmos3_vae` preprocessor | — |
| Chat template with the generation prompt, truncate to 2048, append `[eos, <\|vision_start\|>]` | `cosmos3_text_encoder` preprocessor | `max_sequence_length` |

`dyn_bsz` prices a sample before batching from what the two preprocessors write:
the templated prompt's token count (before the two appended markers), plus the
clip's latent patches (counted on the frames before the resize) and three marker
tokens. That is the reference packer's count, `prompt + 1 + patches + 2`, so
`data.max_seq_len: 45056` packs the same way.

The frame choice and the mRoPE rate (the source clip's) are the DiT port's. The
VAE preprocessor refuses a clip the loader has thinned, since it picks frames on
the source timeline.

**Decode cost.** Because the VAE preprocessor picks frames on the source
timeline, the loader has to hand it every source frame: `mm_configs.max_frames`
cannot be used as a decode cap, since the shared loader thins a long clip by
`linspace`, which drops frames the prefix needs. Each DataLoader worker therefore
holds one whole decoded clip (`T x 3 x H x W` `uint8`) before the VAE
preprocessor keeps at most `max_frames` (117) of them: a 10 s, 30 fps, 720p clip
is 300 frames, about 830 MB. For a corpus of long or high-resolution clips, cut
the clips beforehand or budget worker memory for the longest one. An
out-of-memory there kills the worker during decoding, not in the VAE
preprocessor.

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

The cache holds each sample's prompt after CFG dropout. Whether a row is dropped
depends only on `cfg_dropout_seed`, its clip path and its prompt, so stage 1
drops the same prompts an online run with the same seed would, and so does a
resumed run. The same rows are dropped in every epoch; the recipes train one. The VAE frame settings decide what is in the cache,
which is why `modules_offline_embedding.yaml` repeats them.

The recipes keep `mixed_precision.enable: false`: weights are bf16 end to end,
like the reference's, and the sigma, timestep and target tensors the connector
passes between its nodes stay fp32 because nothing casts the graph inputs. With
mixed precision on, FSDP would cast them to `param_dtype`.

The sigma / noise / clean-prefix stream comes from a CPU generator in the flow
connector, seeded `seed + dp_rank`, and is checkpointed per rank with that
module, so a resumed run continues the same stream; see
[Module Sampling State](../mixins/module_rng_state.md).

## 5. What is not supported

- **Generation.** There is no flow sampler, so the recipes and the converted
  checkpoint declare no generation graph, and `OmniModel.generate` says so.
- **Sequence parallelism.** The joint attention spans a whole sample, so every
  module runs with `ulysses_size: 1`; the MoT raises otherwise.
- **Several clips, or windows, per sample.** One prompt and one clip per sample.
- **Sparse-MoE feed-forwards and the autoregressive text loss.**
