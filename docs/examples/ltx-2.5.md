# LTX-2.5 SFT training

Supports offline audio/video supervised fine-tuning (SFT) for **LTX-2.5 22B-dev BF16**, including
text conditioning (T2AV) and first-frame conditioning (I2AV). It reuses VeOmni's
LTX-2.3 implementation, DiTTrainer, FSDP2/Ulysses and native LoRA checkpoints.
The default recipe and real-weight validation use LoRA.

## 1. Prepare weights

Download the following components from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

| Component | File within the model repository |
| --- | --- |
| Transformer | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| Gemma-4, tokenizer and text projections | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| Conv video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` |
| Audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` |

Use the LTX-specific Gemma-4 pack; generic Gemma weights and cached Gemma-3
features are incompatible. The loader follows the [official Gemma implementation](https://github.com/Lightricks/LTX-2/tree/9ec55f9f22798a3198d9c923856824821bc3317e/packages/ltx-core/src/ltx_core/text_encoders/gemma),
selecting the text model from metadata and building its tokenizer in memory.

```bash
export LTX25_ROOT=/path/to/LTX-2.5
python -m veomni.models.diffusers.ltx2_3.ltx_condition.ltx25 \
  --transformer_path "$LTX25_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --output_dir /path/to/models/LTX-2.5-veomni
```

This validates the DiT keys, shapes and dtypes, writes the model config, and links
the original weights without copying them. Validation uses a meta model with
eager ops and preserves the saved training configuration. Use a new output directory
and keep the source weights accessible on every training node.

## 2. Prepare data

Use the repository GPU/NPU environment (`transformers==5.16.1`) and install the
additional preprocessing dependencies:

```bash
python -m pip install opencv-python-headless pillow-heif
```

Provide JSON/CSV/JSONL records with `media_path` and `caption`; relative media paths
resolve from the dataset file. Clips must contain audio. Optional scene splitting
and video-description generation can follow the [LTX-2.3 guide](ltx-2.3.md);
skip these steps if you already have captioned clips.

```bash
python -m veomni.models.diffusers.ltx2_3.ltx_condition.preprocess_dataset preprocess \
  --dataset_file /path/to/clips/dataset.json \
  --checkpoint_path "$LTX25_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --gemma_model_path "$LTX25_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video_vae_path "$LTX25_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio_vae_path "$LTX25_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --precomputed_dir /path/to/clips/.precomputed-ltx25 \
  --max_sequence_length 1024 \
  --resolution_buckets "768x512x49" \
  --with_audio

python -m veomni.models.diffusers.ltx2_3.ltx_condition.preprocess_dataset save-parquet \
  --precomputed_dir /path/to/clips/.precomputed-ltx25 \
  --output_dir /path/to/ltx25-parquet \
  --pad_to_multiple_of 8 \
  --with_audio
```

Use a fresh cache directory when changing model versions or weights. Spatial sizes
must be divisible by 32 and frame counts must be `8n + 1`. Set `--pad_to_multiple_of`
to the intended data-parallel world size and check the packing summary for missing audio.

## 3. Train and resume

Set the three model paths and `data.train_path` in
[`ltx25_av_lora.yaml`](../../configs/dit/ltx25_av_lora.yaml), and adjust
`train.global_batch_size` for the data-parallel world size:

```bash
bash train.sh tasks/train_dit.py configs/dit/ltx25_av_lora.yaml
```

The recipe repeats the dataset and sets `data.train_sample: 16000` to support
2,000 steps at global batch size 8, including DP=1. Adjust the sample budget
when increasing the batch size or step count.

- `model.condition_model_cfg.first_frame_conditioning_p`: `0.0` for T2AV,
  `1.0` for I2AV, or `0.5` for mixed conditioning. Conditioned frames are clean,
  receive timestep zero and are excluded from video loss.
- Sampling/loss retain LTX-2.3 behavior: default `shifted_logit_normal`, video MSE
  plus audio MSE without extra weighting. The existing `uniform` mode remains available.
- The recipe saves DCP training state and LoRA adapters. Resume by setting
  `train.checkpoint.load_path` to the saved step directory, using the same base weights.

This recipe covers the BF16 dev model and Conv VAE. Quantized imports, diffusion
decoder, multishot, duration prediction and IC-LoRA are outside its scope.

## Smoke test

On 2026-10-13, the official 22B weights passed real-media preprocessing, training,
checkpoint resume and adapter reload on **2 × A800 80 GB** (A100 resource flavor).

| Item | Configuration / result |
| --- | --- |
| Input / runtime | Two Sintel clips with audio, `256×256×49`, 128 text tokens; PyTorch `2.11.0+cu130`, Transformers `5.16.1`, Diffusers `0.37.0` |
| Training | BF16 FSDP2, SP2/DP1, CPU offload, gradient checkpointing; LoRA rank/alpha 4, dropout 0; global/micro batch 1, seed 42 |
| Optimizer / attention | AdamW, LR `1e-4` → `1e-5`, weight decay `0.01`, clip 1; deterministic math SDPA |
| Updates | T2AV and I2AV each ran 4 training + 2 resumed + 1 adapter-reload updates: 14 total |
| Checks | Finite losses, finite nonzero gradients and LoRA updates; exact resumed loss/gradient/parameter matches; all 2,304 adapter tensors matched, relative L2 error `0.0` |

Loss is video MSE plus audio MSE. Resume continues from step 2; adapter reload
uses a fresh optimizer.

| Case | Training loss, steps 1–4 | Resumed loss, steps 3–4 | Adapter-reload loss |
| --- | --- | --- | --- |
| T2AV | 0.842705, 1.458746, 1.190715, 1.387144 | 1.190715, 1.387144 | 0.863181 |
| I2AV | 0.838313, 1.360976, 1.186303, 1.295889 | 1.186303, 1.295889 | 0.859341 |

### SP1/SP2 consistency

The [small-model harness](../../tests/parallel/ulysses/ltx25_fsdp_smoke.py) ran
24 updates on two A800 GPUs: T2AV/I2AV × divisible/odd lengths, three steps each
under SP1/DP2 and SP2/DP1. It uses random weights/inputs and FP32, comparing each
step with a same-weight, same-input unsharded reference. The 22B test above used SP2 only.

Representative T2AV divisible-length results (six decimals); gradient norms are
measured before clipping. Each step uses a new random batch, so these losses do
not describe pretrained-model convergence.

| Step | SP1 loss | SP2 loss | SP1 gradient norm | SP2 gradient norm |
| --- | --- | --- | --- | --- |
| 1 | 44.987442 | 44.987442 | 12.802435 | 12.802435 |
| 2 | 127.662865 | 127.662872 | 31.529470 | 31.529472 |
| 3 | 73.588562 | 73.588570 | 34.761826 | 34.761826 |

Maximum errors across all four cases and three steps, against the unsharded reference:

| Metric | SP1 / DP2 | SP2 / DP1 |
| --- | --- | --- |
| Absolute loss error | `0` | `1.5258789e-5` |
| Relative L2 raw-gradient error | `9.6185408e-8` | `4.3861401e-7` |

Loader verification additionally passed 43 focused CPU tests. Forty real-tokenizer
cases matched official/previous token IDs and masks exactly; reduced-model FP32/BF16
buffers and hidden states also matched exactly. Loader changes were checked
separately for equivalence after the GPU smoke.

### Reproduce the real-weight smoke

To reproduce the real-weight smoke, use two 80 GB GPUs, sufficient host memory for
CPU offload, at least 160 GiB free storage, `curl`, `ffmpeg` and the dependencies above:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LTX25_E2E_ROOT=/path/to/new-ltx25-smoke-output
python tests/parallel/ulysses/ltx25_e2e/run_e2e.py
```

The harness downloads 70.1 GB of publisher weights and the Sintel trailer using
pinned weight/media manifests. This manual test validates the bounded training
and checkpoint path; it does not establish generation quality, convergence,
throughput, full-parameter/multi-node behavior or external adapter interoperability.

Model and trainer reference: [Lightricks LTX-2](https://github.com/Lightricks/LTX-2).
