# DeepSeek-V4.1-Flash (SeedOmni)

Training recipe for **DeepSeek-V4.1-Flash** as a SeedOmni graph model. The
monolithic release checkpoint is split into five OmniModules wired as a
vision-language training DAG with N-gram (Engram) memory and DSpark multi-token
prediction (MTP):

```text
deepseek_v41_vision ──────────→ deepseek_v41_ngram ──┐
                                                      ├─→ deepseek_v41_llm → deepseek_v41_mtp → deepseek_v41_text_encoder.decode → end
deepseek_v41_text_encoder.encode ────────────────────┘
```

| Module | Holds | Role |
|--------|-------|------|
| `deepseek_v41_vision` | ViT + 2D patch merger + image/video processor | image / video frames → visual token spans |
| `deepseek_v41_text_encoder` | `embed_tokens` + untied `lm_head` + tokenizer | token ↔ embedding, CE loss head, DeepSeek V4.1 chat template |
| `deepseek_v41_ngram` | multi-head rolling-hash Engram tables | token n-gram lookup injected into configured LLM layers (frozen by default) |
| `deepseek_v41_llm` | 40-layer mHC + DSA sparse-attention + MoE backbone | `inputs_embeds → hidden_states`, DSA indexer KL loss, Engram fusion |
| `deepseek_v41_mtp` | 3-stage DSpark draft backbone | multi-token prediction (`dspark_block_size: 5`, `num_nextn_predict_layers: 3`) |

A 3-module training DAG without N-gram or MTP is also provided under `deepgemm/`:

```text
deepseek_v41_vision ─────────────┐
                                 ├─→ deepseek_v41_llm → deepseek_v41_text_encoder.decode → end
deepseek_v41_text_encoder.encode ┘
```

`model.model_path` and the `sources` entries in `data.yaml` are **placeholders**:
replace each with a local path or storage URL (`hdfs://` etc.) this job can read
before launching.

Config dir: `configs/seed_omni/DeepSeek/deepseek_v41_flash/`

| File | Role |
|------|------|
| `data.yaml` | Weighted multisource SFT mix (`sharegpt4v_cap_100k`, `llava_video`, `tulu-3-sft-mixture`). Shared by all recipes. |
| `graph_train.yaml` | Full 5-module training DAG (`vision → ngram`, `{ngram, text_encoder.encode} → llm → mtp → text_encoder.decode → end`). |
| `bf16/base.yaml` | Launcher for 128-GPU (256k context, CP=16) BF16 training (`fused_triton` MoE, `eager` linear, `liger_kernel` MLP). |
| `bf16/modules_train.yaml` | Per-module BF16 overrides (`vision`, `text_encoder`, `ngram`, `llm`, `mtp`). |
| `qat/base.yaml` | Launcher for 128-GPU (256k context, CP=16) QAT training (blockwise 32x32 FP8 QAT on linear/MLP/MoE + FP4 KV rounding). |
| `qat/modules_train.yaml` | Per-module QAT overrides. |
| `deepgemm/base.yaml` | Launcher for 128-GPU (256k context, CP=16) DeepGEMM FP8 training. |
| `deepgemm/modules_train.yaml` | Per-module DeepGEMM FP8 overrides (`moe_implementation: {name: deepgemm, kwargs: {dtype: fp8}}`, `linear` & `gated_mlp`: `deepgemm`). |
| `deepgemm/no_ngram_mtp.yaml` | Launcher for DeepGEMM FP8 training without `deepseek_v41_ngram` and `deepseek_v41_mtp`. |
| `deepgemm/modules_no_ngram_mtp.yaml` | Per-module overrides disabling Engram (`engram_layer_ids: []`) and DSpark (`dspark_block_size: 0`, `emit_draft_hiddens: false`). |
| `deepgemm/graph_no_ngram_mtp.yaml` | 3-module training DAG (`vision`, `text_encoder`, `llm`). |

---

## 1. Checkpoint

DeepSeek-V4.1-Flash registers an HF layout (`modules/deepseek_v41/hf_layout.py`),
so `model.model_path` can point at the **upstream release checkpoint directly**
without an offline split. On load, FP8 blockwise linear weights and packed FP4
routed-expert weights are dequantized on the fly to BF16 for training, with
streaming EP/embedding-sharded loading (`extra_parallel_stream_load: true`,
`broadcast_model_weights_from_rank0: false`). On HF export, trained weights are
requantized back to the release layout.

An optional offline split is available through the same layout:

```bash
python scripts/seed_omni/convert_model.py \
  --model_path /abs/path/to/DeepSeek-V4.1-Flash \
  --output_dir /abs/path/to/DeepSeek-V4.1-Flash-split
```

Either root works as `model.model_path`.

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry matches a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`):

```yaml
# configs/seed_omni/DeepSeek/deepseek_v41_flash/data.yaml
sources:
  - sharegpt4v_cap_100k
  - llava_video
  - tulu-3-sft-mixture
names:
  - sharegpt4v_cap_100k
  - llava_video
  - tulu-3-sft-mixture
schedule:
  - schedule_type: const
    weights: [0.4, 0.4, 0.2]
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

Replace every entry in `sources` with a readable local directory, file, or
`hdfs://` URL; leave `names` unchanged. Video decoding parameters live under
`data.mm_configs` (`use_audio_in_video: false`, `fps: 2.0`, `min_frames: 4`,
`max_frames: 8`). See [Data Format](../usage/data_format.md) for the row schema.

---

## 3. Train

All shipped recipes target 128 GPUs (16 nodes × 8 GPUs) at 256k context with
`cp_size: 16`, `ep_size: 32`, and `emb: 128`:

```bash
# BF16
NNODES=16 NODE_RANK=<rank> MASTER_ADDR=<host> bash train.sh \
  tasks/omni/train_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/bf16/base.yaml

# QAT (blockwise FP8 QAT + FP4 KV rounding)
NNODES=16 NODE_RANK=<rank> MASTER_ADDR=<host> bash train.sh \
  tasks/omni/train_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/qat/base.yaml

# DeepGEMM FP8 (full 5-module graph)
NNODES=16 NODE_RANK=<rank> MASTER_ADDR=<host> bash train.sh \
  tasks/omni/train_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/deepgemm/base.yaml

# DeepGEMM FP8 (without n-gram & MTP)
NNODES=16 NODE_RANK=<rank> MASTER_ADDR=<host> bash train.sh \
  tasks/omni/train_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/deepgemm/no_ngram_mtp.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — release or split checkpoint root.
- `--data.max_seq_len` — packed sequence length budget (`262144` by default).
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for smoke runs.

### Parallelism layout

- **Hybrid sequence parallelism (`ViT Ulysses SP + LLM CP`)**: `deepseek_v41_llm`
  and `deepseek_v41_mtp` use Context Parallelism (`cp_size: 16, ulysses_size: 1`),
  while `deepseek_v41_vision` uses Ulysses SP (`ulysses_size: 16, cp_size: 1`).
  Keeping both degrees equal preserves a uniform `dp_size` across all modules.
  `skip_sp_gather_between_modules: true` avoids redundant inter-module SP
  all-gathers along the sharded sequence.
- **Expert parallelism (`ep_size: 32`)**: Shards the routed MoE experts in
  `deepseek_v41_llm` and `deepseek_v41_mtp`; `ep_size` must divide both the world
  size and `n_routed_experts` / `dspark_n_routed_experts`.
- **Sharded embeddings (`emb: 128`, `extra_parallel_only: true`)**: Shards the
  large `embed_tokens` / `lm_head` (`deepseek_v41_text_encoder`) and frozen
  Engram tables (`deepseek_v41_ngram`) across all 128 ranks without wrapping them
  in redundant FSDP2 all-gathers.

---

## 4. Resume

Each save writes per-module DCP shards plus per-rank trainer state under
`<output_dir>/checkpoints/global_step_N/`. Resume by passing `load_path`:

```bash
NNODES=16 NODE_RANK=<rank> MASTER_ADDR=<host> bash train.sh \
  tasks/omni/train_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/deepgemm/base.yaml \
  --train.checkpoint.load_path outputs/deepseek_v41_flash_deepgemm_fp8_256k_cp16/checkpoints/global_step_500
```

---

## 5. Architecture & Training Notes

- **mHC & DSA sparse attention**: Each LLM decoder block applies multi-stream
  hyper-connections (`mhc_implementation: tilelang`) and DeepSeek Sparse
  Attention with an auxiliary indexer KL loss (`dsa_indexer_loss: true`,
  `dsa_indexer_implementation: tilelang`).
- **Muon optimizer**: `deepseek_v41_llm` and `deepseek_v41_mtp` default to
  `optimizer.type: muon` with `muon_expert_zero_comm: true` and per-head
  splitting on `self_attn.wq_b` / `indexer.wq_b`.
- **Training-only scope**: This SeedOmni integration covers multimodal training
  (image, video, and text SFT) and checkpoint load/export; no generation graph is
  shipped yet.
