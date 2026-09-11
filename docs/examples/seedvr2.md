# SeedVR2-3B

SeedVR2 restores existing images and videos. This integration ports the
independently implemented NaDiT into `veomni/models/diffusers/seedvr2/` and preserves the
official Diffusers-derived video VAE as a separate, frozen component.

## Provenance and scope

- Source: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR),
  revision `e4de8c24441a67e1b7df56abea10645059bb1185`, Apache-2.0.
- Weights: [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B),
  revision `37255ff8cccfb01071b87f635a5948ca8d53117c`.
- Supported inference: official 3B architecture, fixed positive text embedding,
  one-step velocity sampling at t=1000, CFG=1; image or video input.
- Optional training: paired, supervised one-step restoration through the existing
  `DiTTrainer`. This is **not** a reproduction of the unpublished official
  adversarial post-training. The pinned source publishes inference, not its
  referenced training implementation.
- This integration was implemented from the official source, without consulting
  the earlier `feat/model-migration-skill` implementation.

The main model has 3,391,475,776 parameters. The external state dictionary maps
one-to-one by adding `dit.` to each key; no tensor transposes, drops, or random
replacement parameters are needed. The VAE retains its original state layout.

## Environment and weights

Use the VeOmni environment for your backend, including Diffusers and PyAV.
The validation environment used Python 3.11, PyTorch 2.10.0, torch-npu 2.10.0,
Transformers 5.9.0, Diffusers 0.37.0, and torchvision 0.25.0 on Ascend.
Source comparison additionally needs `rotary-embedding-torch==0.5.3`; normal
inference and training do not need Apex, FlashAttention, rotary-embedding-torch,
OmegaConf, or an installed SeedVR repository.

On Ascend, source the installed CANN `set_env.sh` before running Python.
Run all commands below from the VeOmni repository root.
On the migration host, the isolated validation interpreter is
`.agents_workspace/seedvr2-env/bin/python`. The frozen environment sync stalled
on dependency downloads; validation used a separately installed environment
with the core versions above, not a completed frozen sync. Neither the system
Python environment nor `uv.lock` was changed.

```bash
hf download ByteDance-Seed/SeedVR2-3B \
  --revision 37255ff8cccfb01071b87f635a5948ca8d53117c \
  --include seedvr2_ema_3b.pth ema_vae.pth pos_emb.pt neg_emb.pt README.md \
  --local-dir /home/share/h00943455/SeedVR2-3B
```

The DiT SHA256 must be
`6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b`.
The original weights remain unchanged. Conversion writes a new directory and
rejects a different DiT checksum:

```bash
python scripts/model_conversion/convert_seedvr2.py \
  --source /home/share/h00943455/SeedVR2-3B \
  --output /home/share/h00943455/SeedVR2-3B-VeOmni
```

The output contains sharded safetensors, model and condition configs, and
`conversion-report.json`. The condition config points to the original VAE and
embedding directory; keep that directory available, or update
`base_model_path` after moving it.

## Restore an image or video

```bash
python tasks/infer/infer_seedvr2.py \
  --weights /home/share/h00943455/SeedVR2-3B \
  --input input.mp4 --output restored.mp4 \
  --device npu:0 --dtype bfloat16 --height 720 --width 1280 --seed 666
```

Use image paths, such as `input.png` and `restored.png`, for a still image.
The CLI refuses to overwrite an existing output.

Height and width specify the **target area**, not forced output dimensions:
aspect ratio is preserved, followed by a center crop to multiples of 16.
Video frames are padded by repeating the last frame to length 4n+1, then the
restored output is trimmed to the original frame count. The VAE posterior is
sampled, as upstream does. The seed controls it and the DiT noise.
The VAE and DiT are offloaded between stages.

The CLI decodes all frames into memory. Start with short clips and reduce the
target area for memory-constrained runs. It preserves frame rate but does not
copy audio, variable-frame-rate timestamps, or other container metadata.
Optional upstream color correction is not included in the pinned source or
this integration.

## Supervised paired fine-tuning

Prepare a JSONL manifest of temporally and spatially aligned low/high-quality
media. Relative paths resolve from the manifest directory:

```json
{"video": "lq/clip.mp4", "target": "hq/clip.mp4"}
```

Both items must have matching frame counts, frame rates, and aspect ratios.
Images are also accepted. The preprocessor validates shapes and rates, but
cannot establish semantic alignment for you.

```bash
python scripts/model_conversion/prepare_seedvr2_pairs.py \
  --manifest pairs.jsonl --weights /home/share/h00943455/SeedVR2-3B \
  --output output/seedvr2_pairs --device npu:0 --height 720 --width 1280
```

The generated Parquet schema uses `latents` (HQ), `condition_latents` (LQ),
and `context` (fixed text embedding). Each value is a pickled CPU tensor,
matching the existing `dit_offline` transform. Only load trusted datasets.
Provide enough samples/shards for every data-parallel rank.

Edit paths in [the training config](../../configs/dit/seedvr2.yaml), then use
the existing `tasks/train_dit.py` entry point:

```bash
torchrun --standalone --nproc-per-node=2 tasks/train_dit.py configs/dit/seedvr2.yaml \
  --model.model_path=/home/share/h00943455/SeedVR2-3B-VeOmni \
  --model.config_path=/home/share/h00943455/SeedVR2-3B-VeOmni/config.json \
  --model.condition_model_path=/home/share/h00943455/SeedVR2-3B-VeOmni/condition
```

The example uses FSDP2 and a global batch of 16. Choose a batch divisible by
the data-parallel size and size the hardware for the full model and optimizer.
`init_device=meta` requires FSDP; for a single-device debug run, set the actual
accelerator (`npu` or `cuda`) as the initialization device.

Conditioning builds `[noise, LQ latent, ones mask]`, fixes t=1000, and uses
`noise - HQ latent` as the velocity target. The model returns a mean MSE loss
dictionary for the normal trainer. Trainable parameters belong to NaDiT; the
VAE and fixed text embedding are frozen. Standard VeOmni checkpoint callbacks
own checkpoint I/O; no SeedVR trainer is used. Exact stochastic training resume
is **not supported by this baseline**: the current global-state callback saves
CPU RNG but not accelerator RNG. A fresh-process resume loaded model/optimizer
state and the step cursor, but did not reproduce the uninterrupted next-step
loss. Sampling on CPU alone also failed to align the resumed RNG stream, so
that workaround is not included. Fixing the shared trainer's RNG lifecycle is
separate from this model port. Do not treat checkpoint reload as exact resume.

## Implementation boundaries

- Native LayerNorm/RMSNorm and segmented PyTorch SDPA replace CUDA-only Apex
  and FlashAttention. Upstream's BF16 attention casts remain explicit.
- RoPE evaluates only needed positions instead of allocating an unused
  1024x128x128 grid. Its persistent frequency keys and position convention are
  preserved.
- The upstream output modulation implicitly reuses the attention modulation
  cache. The port selects that same embedding slice explicitly, preserving the
  output while allowing non-reentrant gradient checkpoint recomputation with
  fresh per-block caches.
- Sequence/context parallelism is not implemented; nonlocal sequence execution
  raises an error. FSDP2 is a separate path.
- The eager path contains host synchronization and per-window loops. No fused
  kernel speedup, high-resolution throughput, CUDA numerical parity, or full
  3B training convergence is claimed.
- The 7B checkpoint and arbitrary sampling schedules are outside this baseline.

## Validation

The CI-enumerated registry test contains the SeedVR2 case:

```bash
pytest tests/models/test_model_registry.py -k seedvr2 -q
python scripts/validation/validate_seedvr2.py
python scripts/validation/compare_seedvr2_source.py --source /path/to/pinned/SeedVR
make quality
```

The scripts are manual numerical reproducers, not newly added CI jobs.
The source comparison uses the pinned upstream model with explicitly disclosed
native attention/normalization substitutes and a bounded, equivalent RoPE grid.
It does not execute the original CUDA kernels.

Measured on 2026-09-11:

| Check | Result |
|---|---|
| Official DiT weight import | 635/635 state tensors; SHA256 verified; strict loading |
| Official VAE weight import | Strict loading; 250,648,227 parameters |
| Converted full-weight functional check | All 635 tensors exactly equal; fixed-noise NPU output max error 0 |
| Tiny source comparison, FP32 outer computation | Output max error 0; gradient max error 3.73e-9; atol 1e-5, rtol 1e-4 |
| Tiny source comparison, CPU BF16 autocast | Output max error 0.0078125; atol/rtol 0.02 |
| Gradient checkpoint on/off | Gradient max error 9.31e-10; finite gradients and an optimizer update |
| Tiny HF save/reload | Outputs exactly equal |
| Full 3B NPU inference | Finite outputs for 1 and 5 frames, 32x48 RGB |
| Media CLI | PNG and MP4 restored and decoded back; 5 video frames at 12 fps |
| Standard DiTTrainer | 3 steps on 2 NPUs with FSDP2 and real PNG-derived VAE embeddings; checkpoints saved |
| Fresh-process stochastic training resume | Failed next-step equivalence; not supported/claimed |
| Static checks | Registry case passed; make quality passed with repository-pinned Ruff 0.13.1 |

The full-weight smoke inputs are deterministic synthetic fixtures encoded in
real media formats. They prove executable plumbing, not restoration quality on
natural video. The short training run proves integration, not convergence.
