# HunyuanVideo 1.5: native 480p T2V

This integration connects the native Tencent backbone and frozen conditions to
`tasks/train_dit.py`. It has CPU structural/numerical and single-NPU random-weight backbone validation;
**pretrained real-data training and accelerator checkpoint resume are not yet
validated**. Config-only model downloads do not satisfy those prerequisites.

## Implementation and boundary

- Source: [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5),
  revision `60783e704160023913bee78f0b470036d393d4dfa`; VeOmni base `24d1ddf`.
- Backbone: 480p T2V base, 32 latent channels, 65 concatenated input channels,
  54 double-stream blocks, 8,326,608,160 parameters.
- Native parameter names/layouts are retained. The production meta-model has
  1793 matching state entries. No converter or checkpoint rename is needed
  based on this schema comparison; actual pretrained files still require strict
  loading and output verification.
- Dense PyTorch SDPA replaces the source's CUDA attention imports. The native
  `flash` config label selects dense SDPA here. Sparse/SSTA and sequence
  parallelism are rejected; use `ulysses_size: 1`.
- Frozen native VAE, Qwen text encoder and Glyph ByT5 provide conditions.
  The Transformers encoders are reused without modifying their modeling code.
  T2V does not execute the vision encoder, but its backbone projection parameters
  remain present to preserve checkpoint compatibility.
- VAE posterior sampling, log-normal timestep sampling, shift 3, and velocity
  target `noise - latents` follow the source training path. The conditioning
  generator is captured by `DiTModelRuntime.extra_state()`.
- AdamW is an explicit VeOmni example choice, not a claim of reproducing the
  official Muon optimizer trajectory. Frozen text runs in fp16, VAE in fp32,
  and trainable inputs in bf16. Model mixed precision is owned by FSDP2.
- Native block activation checkpointing is supported. FSDP2 uses the generic
  runtime; no ExtraParallel plan is introduced. No multi-device claim is made.
- This initial path supports online training. I2V, distillation, super-resolution,
  sparse attention, condition caches, and a generation pipeline are outside its
  validated scope. Do not use `offline_training` or `offline_embedding`.

The native code retains Tencent's license and provenance under
`veomni/models/diffusers/hunyuanvideo15/`.

## Required local assets

Use the native repository's checkpoint layout, not a Diffusers-converted model:

```text
HunyuanVideo-1.5/
  config.json
  transformer/480p_t2v/     # original config and full safetensors weights
  vae/                     # original config and full weights
  text_encoder/
    llm/                   # official Qwen encoder, tokenizer and configs
    byt5-small/            # base ByT5 model and tokenizer
    Glyph-SDXL-v2/
      assets/color_idx.json
      assets/multilingual_10-lang_idx.json
      checkpoints/byt5_model.pt
```

No component is automatically downloaded by this integration. Missing components
fail explicitly. The native glyph loader uses the source's documented
`module.text_tower.encoder.` extraction and strict encoder loading.
Unquoted captions get the source's zero ByT5 states and zero validity mask;
quoted strings are encoded, never silently discarded.

## Prepare Tom-and-Jerry

Download the original
[dataset](https://huggingface.co/datasets/Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset),
including `captions.txt`, `videos.txt`, and the referenced videos. Then, from
the VeOmni repository root:

```bash
python scripts/dataset/prepare_tom_and_jerry.py \
  --dataset-root "$DATASET_ROOT" \
  --output "$PREPARED_DATA/train.jsonl"
```

The converter validates all caption/video pairs before writing, rejects missing,
escaping or duplicate paths, and refuses to overwrite a manifest. It records
source-list and output hashes in a `.manifest` sidecar. Videos are referenced by
absolute path; they are not copied. Recreate the JSONL after moving the dataset.

The example reuses `dit_online` and the existing Tom-and-Jerry preprocessor.
It selects 33 frames with the existing uniform sampling rules, aligns to 4n+1,
resizes/crops to 480 x 832, and normalizes to [-1,1]. The fixed frame limit
takes precedence over the fps budget; it does not restore motion absent from
the source video. Captions remain paired with their original videos.
These short clips are an integration workload, not a reproduction of a
long-video production training recipe.

## Training and resume

Activate the chosen VeOmni accelerator environment and its device runtime first.
Set `MODEL_ROOT`, `PREPARED_DATA`, `RUN_DIR`, and `NPROC_PER_NODE` for that
environment. Start with a topology that has enough memory for the full 8.33B model,
its optimizer, frozen encoders and video activations; the default one-process
command is not a promise that it fits one card. Dense attention is expensive at
this resolution. Device availability and memory must be checked before launch.

```bash
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  tasks/train_dit.py configs/dit/hunyuanvideo15_t2v.yaml \
  --model.model_path "$MODEL_ROOT/transformer/480p_t2v" \
  --model.condition_model_path "$MODEL_ROOT" \
  --data.train_path "$PREPARED_DATA/train.jsonl" \
  --train.global_batch_size "${NPROC_PER_NODE:-1}" \
  --train.checkpoint.output_dir "$RUN_DIR"
```

The example runs four steps and checkpoints every two steps. To check resume,
repeat the same command in a **new process**, adding:

```bash
--train.checkpoint.load_path "$RUN_DIR/checkpoints/global_step_2" \
--train.checkpoint.output_dir "$RESUMED_RUN_DIR"
```

Compare steps 3 and 4 against the uninterrupted run: batch identity, losses,
parameters, optimizer/scheduler, global step, rank-local data cursor and RNG.
Use [the checkpoint guide](../usage/checkpoint.md) for the runtime-owned layout.
A weight reload or a condition-generator unit test is not this trainer test.

## Verification and remaining evidence

Run the focused tests without downloading model weights. NPU cases run when an
accessible NPU is present and otherwise skip:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/models/test_hunyuanvideo15.py \
  tests/data/test_hunyuanvideo15_data.py \
  tests/models/test_model_registry.py -k hunyuanvideo15
make quality
```

For the optional source comparison, check out the pinned revision separately and
set `HUNYUANVIDEO15_SOURCE` to its root before the same test command. That test
loads the original modeling and attention function bodies, substitutes only
external launch/import plumbing, and uses the source's uncompiled dense
FlexAttention reference on CPU. No CUDA/SP execution is implied.

The focused suite passed 20 tests in 48.72 seconds, including the optional
source comparison and four NPU cases (fp32/bf16, activation checkpointing on/off).
Each NPU case checks CPU output parity, finite nonzero gradients, two AdamW
updates, and condition-generator RNG replay. CPU bf16 uses autocast for the
float32 timestep embedding. Nonzero test linear weights exercise attention and
gates; the native zero output initializer could otherwise hide numerical errors.
Production initialization is unchanged.

On Ascend910_9382, the full 8.33B/54-layer backbone also completed bf16 forward,
backward and two SGD updates with activation checkpointing. Peak allocated tensor
memory was 31.07 GiB. Latents were only [1,32,2,4,4], text had 8 tokens and ByT5
had 4 tokens; weights and conditions were random. This does not establish memory
requirements for 480p video, full AdamW, frozen encoders or FSDP2 training.
The opt-in test needs at least 40 GiB free on the selected device:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 HUNYUANVIDEO15_FULL_NPU=1 PYTHONPATH=. \
  python -m pytest tests/models/test_hunyuanvideo15.py -k full_backbone_npu -q -s
```

The validation environment used torch/torch-npu 2.10.0 and CANN 9.2.0 B020.
The test account initially lacked access to the device nodes; an authorized
process with their owning group resolved it. Check device-node permissions and
runtime initialization before treating an unavailable device as a model failure.

Evidence obtained during onboarding:

| Check | Result |
|---|---|
| Registry/config and current training YAML | Passed |
| Tiny model forward/backward and AdamW update | Passed, with and without activation checkpointing |
| Tiny strict model serialization/reload | Passed, including the VeOmni loader with a one-rank CPU Gloo group |
| Source tiny output / gradients | Max absolute error 0 / 1.46e-11, fp32, nonzero test weights; masked text and ByT5 included |
| NPU tiny CPU parity | Max output absolute error 1.70e-4 (fp32), 3.91e-3 (bf16); checkpointing on/off |
| NPU full backbone | Two bf16 SGD updates, finite nonzero gradients, changed parameters; 31.07 GiB peak allocated |
| Source production state schema | 1793 matching keys/shapes/dtypes on meta |
| Condition RNG/runtime hook | Next noise, timestep and target reproduce exactly |
| Native tiny VAE encode/reload | Passed; random initialization only |
| Real dataset preparation | 6092 aligned pairs; revision `fc9066b00cb690f44779c5202572b8e052b5cced2` |
| Real video decode/preprocess | Caption preserved; [1,3,33,480,832], range [-1,1] |
| Pretrained loading and pretrained numerical parity | Blocked: only configs/non-weight assets available |
| Real-data trainer run and fresh-process DCP equivalence | Blocked: missing weights; backbone NPU smoke checks do not cover this trainer path |
| Multi-device/kernel performance | Not evaluated |

Source comparison tolerances are declared before execution: output
`atol=2e-5, rtol=2e-5`; gradients `atol=2e-5, rtol=2e-4`.
Reported relative error uses `abs(reference).clamp_min(1e-8)` as denominator.
The small random-weight comparison does not prove pretrained full-model parity.

NPU-to-CPU output tolerances are fp32 `atol=2e-4, rtol=2e-3` and bf16
`atol=1e-2, rtol=5e-2`. NPU gradient finiteness and updates are checked;
full-model pretrained numerical parity and optimizer resume remain unverified.
