# Gemma 4

The `gemma4` and `gemma4_text` registrations provide generated modeling for
Gemma 4 E2B against Transformers 5.16.1. They retain per-layer embeddings,
shared KV states and the Hugging Face checkpoint layout. The multimodal
registration includes image, video and audio towers.

## Attention and loss

Choose attention through `OpsImplementationConfig`, for example
`attn_implementation="flex_attention"`. The normalized HF interface name is
`veomni_flex_attention`. Packed full and sliding masks use the public
`veomni.ops.mask` builders with `cu_seq_lens_q` boundaries. Packed attention
does not support supplied `past_key_values` and disables automatic cache
creation, including when `use_cache=True`; unpacked masks retain HF cache handling.
With `use_bidirectional_attention="all"`, full attention remains bidirectional
within each packed sample. Eager/SDPA/Flex sliding attention preserves HF's
inclusive, symmetric window on both sides of each token; document boundaries
still apply. FlashAttention 2/3/4 and the FA2/FA3 Hub variants retain their
native metadata path: the attention adapter receives the module's causal
setting, sliding window and cumulative sequence lengths. No custom mask
predicate is sent to Flash. The pinned HF native Flash conversion uses radius
`W - 1`, whereas eager/SDPA/Flex include distance `W`, where `W` is the effective
`text_config.sliding_window` after Gemma's bidirectional config conversion.
For example, a configured window of 32 becomes effective `W=17`: dense/Flex
include a token 17 positions away, while native Flash uses `(16, 16)` and
excludes it. Cross-backend numerical references must account for this boundary.
This integration retains each pinned HF backend's behavior.
Packed Flash inputs must be padding-free: `None`
and all-valid 2D masks are accepted, but masks containing padding raise an
error because HF unpadding would replace the supplied document boundaries.
Unpacked Flash inputs retain normal padding-mask handling.
Packed multimodal calls with `use_bidirectional_attention="vision"` and
`mm_token_type_ids` require eager/SDPA/FlexAttention. Flash backends reject
the custom vision-block visibility mask.

Each LM instance holds its own `VeomniOp("cross_entropy_loss", "standard", impl)`
as `veomni_ce`. Training calls its `loss_function` wrapper so channel-loss
callbacks can observe both eager and fused loss selections. An uncapped model
retains the selected loss implementation. Logit softcapping forces eager loss
on capped logits and materializes logits in memory. When labels are supplied,
capped models reject `return_log_probs=True`, including the top-k distillation
path, since the auxiliary helpers would reproject uncapped logits. Calls
without labels return capped logits.

Audio uses a boolean local mask independently of the LM attention backend.
Vision Ulysses is unsupported: use `ulysses_size=1` for models retaining the
vision tower. Such models reject Ulysses on text-only ranks too, before peers
could enter different collectives.

Under FSDP2, missing modalities participate through config-derived dummy
forwards in a fixed image, video, audio order, with live compute dtypes and
zero-valued projected graph dependencies.

## Gradient checkpointing

When `text_config.num_kv_shared_layers > 0`, use non-reentrant gradient
checkpointing. Reentrant recomputation captures detached shared KV states and
silently drops shared layers' gradient contributions to the producer projections.
The model rejects this combination when checkpointing is enabled, before training
or any checkpoint flags are changed. The restriction applies to the text and
multimodal model entry points, including direct `gradient_checkpointing_enable()`
calls.

The VeOmni default is supported:

```yaml
model:
  accelerator:
    gradient_checkpointing:
      enable: true
      enable_reentrant: false
```

For direct model calls, pass
`gradient_checkpointing_kwargs={"use_reentrant": False}`. Omitting the kwargs
argument also selects non-reentrant checkpointing in the pinned Transformers
5.16.1; an explicit empty dict or `{"use_reentrant": None}` inherits PyTorch's
reentrant default and is rejected with shared KV. Disabling checkpointing is
supported, as is reentrant checkpointing when `num_kv_shared_layers=0`.
Regression tests compare logits, loss and every parameter gradient of supported
text and multimodal models against the same model without checkpointing.

## VLM training data

`tasks/train_vlm.py` selects the registered `gemma4` transform from the model
type. Point `model.model_path` (or `model.tokenizer_path`) at a checkpoint with
its native `Gemma4Processor` assets and chat template. No VeOmni chat-template
override is needed. The model exposes picklable position and collation hooks;
positions start at zero for every document before packing.

The transform accepts HF-style `messages`, with inline media paths:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "image.png"}, {"type": "text", "text": "Describe this image."}]}, {"role": "assistant", "content": [{"type": "text", "text": "A red flower."}]}]}
```

It also accepts existing source-registered `conversations` preprocessors with
`images` / `videos` / `audios` sidecars in placeholder order. Video and audio
items use `type="video"` / `type="audio"` and matching path fields. Inputs may
mix modalities or contain text only. Assistant targets are text responses;
`loss_mask=0` on an assistant message excludes that turn from supervision.

HF performs patchification, media-token expansion and tokenization. Native
image/video/audio token IDs and `mm_token_type_ids` are retained. Assistant-only
labels are mapped through HF's actual replacement offsets, including the native
turn terminator, while prompt and modality tokens remain masked. Prefix-rewriting
templates and samples without a supervised assistant response are rejected.

The model-owned collate hook pads ragged audio time dimensions with zero features
and false masks. Video frames retain their patch coordinates and are packed as
independent single-frame vision rows, preserving their original temporal order;
their encoded features match the native multi-frame layout. Compressed video
paths/bytes require the existing FFmpeg/torchcodec decoding dependencies. Video
audio is not extracted implicitly; supply audio inputs explicitly.

Example data settings (with ordinary model/optimizer/training settings):

```yaml
data:
  train_path: data/gemma4_train.jsonl
  max_seq_len: 2048
  mm_configs:
    min_frames: 2
    max_frames: 2
    processor_kwargs:
      images_kwargs:
        max_soft_tokens: 70
      videos_kwargs:
        max_soft_tokens: 70
```

```bash
torchrun --standalone --nproc_per_node=2 tasks/train_vlm.py path/to/gemma4.yaml
```

Media budgets must fit `data.max_seq_len`; overlength multimodal samples raise
instead of truncating tokens away from their features. By default, video fetching
uses the native processor's frame count for both minimum and maximum frames.
Keep these equal when processing multiple videos in one sample, as the native
processor stacks their frame axes. Set `ulysses_size=1` and disable other sequence
parallelism for this VLM data path. `freeze_vit`, `freeze_audio_tower` and `vit_lr`
recognize Gemma4's tower layout; freezing the audio tower leaves its separate
projector trainable.

## Validation

CPU regressions cover output/loss/gradient parity, packed sample isolation,
channel-loss interception, audio visibility and dummy graph participation:

```bash
pytest tests/models/test_gemma4.py -q
pytest tests/models/test_gemma4_numerics.py -q
pytest tests/models/base/test_auto_registry.py -k gemma4 -q
pytest tests/data/test_gemma4_data.py tests/trainer/test_vlm_trainer.py -k gemma4 -q
```

CPU backend-dispatch tests exercise the real HF/VeOmni Flash adapters up to a
stubbed kernel boundary, checking causal/window and varlen argument forwarding.
Additional cases run the HF varlen dispatcher with stubbed vendor functions to
check that supplied document offsets reach the selected kernel unchanged.
They do not execute FlashAttention kernels. The companion numerical suite runs
real FA2 and Triton Flex checks on one CUDA GPU when their dependencies are
available. It compares all model gradients to HF using the same backend and
tests four attention calls sharing two K/V leaf pairs against an independent
FP32 attention reference. FA3/FA4 require separate eligible-hardware validation.

Actual asymmetric BF16 FSDP2 validation requires two CUDA GPUs:

```bash
pytest tests/distributed/test_dummy_forward.py -k gemma4 -q
```

The three cases cover image versus text, video/audio versus text, and split
modalities across ranks. They skip on CPU. Numerical tests alone do not establish
collective correctness or full-checkpoint memory requirements.

### Numerical reference procedure

Match visibility before comparing floating-point results. Use the independent
packed dense-mask reference for the selected backend's causal/window contract;
the CPU and CUDA endpoint tests isolate a single nonzero value exactly on the
bidirectional window boundary.

For full-model comparisons, initialize one master state and copy its weights
and buffers into HF and VeOmni. Preserve tied embeddings, inputs, labels,
position IDs, document offsets, precision, KV sharing and kernel options. A seed
alone does not make independently initialized BF16 and FP16 weights identical.
The committed GPU regressions use native HF with the same backend as reference,
rather than imposing one global per-parameter tolerance across different
low-precision kernels in an untrained four-layer stress model.
Real-kernel cases clear the compiler cache before and after each case and reject
the unfused-Flex warning. Shape variants in preceding tests can exhaust Dynamo's
recompile budget and change the arithmetic path. FA2 model controls use
`deterministic=True` so atomic backward reductions do not confound comparisons.

The test-only FP32 attention oracle explicitly promotes Q/K/V inside a MATH
context before casting the output back to the requested model precision. An
outer `sdpa_kernel(MATH)` around a VeOmni model is insufficient: masked
`veomni_sdpa` calls install their own efficient/MATH backend context. The oracle
bypasses that adapter. FP32 attention inside a low-precision model and an
entirely FP32 model are separate controls.

To investigate the historical nine standalone comparisons on one CUDA GPU:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -m tests.tools.diagnose_gemma4_numerics --output-dir /tmp/gemma4-numerics
```

The diagnostic saves every original fixture, retains the original strict output
and gradient assertions as reported metrics, and compares each case to native
HF, FP32 attention and full FP32 model controls. It replays all four attention
calls with shared K/V leaves and fixed upstream gradients. A separate
test-local intervention preserves real-kernel forward outputs exactly while
recomputing only attention VJPs in FP32, separating forward-trajectory and
backward arithmetic effects. These references and interventions do not alter
production modeling or attention implementations.

### A100 numerical diagnostic (2026-10-29)

On A100 SM80 with PyTorch 2.11.0+cu130, Transformers 5.16.1 and FA2 2.8.4,
all nine frozen fixtures matched native HF with the same backend: logits,
loss and every parameter gradient were bitwise equal. The isolated diagnostic
below retains the old cross-backend comparison as a reported metric; it does
not declare those comparisons passing or increase their 8% gradient threshold.
The model has four layers, two shared K/V pairs, hidden size 64, head dimension
256, Gemma attention scaling 1, and two packed documents.

| Dtype / backend | Visibility / length | Max logit difference vs low-precision SDPA | Worst parameter gradient relative L2 vs SDPA | Worst parameter gradient relative L2 with fixed forward and FP32 attention VJP |
| --- | --- | ---: | ---: | ---: |
| BF16 / Flex | causal / 128 | 0 | 1.87% | 2.58% |
| BF16 / Flex | causal / 512 | 0.149414 | 37.13% | 2.63% |
| BF16 / Flex | bidirectional / 128 | 0.011719 | 6.98% | 2.13% |
| BF16 / Flex | bidirectional / 512 | 0.087891 | 37.46% | 2.25% |
| BF16 / FA2 | causal / 128 | 0.044922 | 94.24% | 1.55% |
| BF16 / FA2 | causal / 512 | 0.099396 | 44.81% | 2.23% |
| BF16 / FA2 | bidirectional / 128 | 0.906250 | 210.75% | 1.81% |
| BF16 / FA2 | bidirectional / 512 | 0.816406 | 178.27% | 1.83% |
| FP16 / FA2 | causal / 512 | 0.014206 | 10.51% | 0.62% |

The two bidirectional FA2 comparisons use different visibility contracts:
the old SDPA reference includes the window endpoint that native Flash excludes.
Correcting visibility is necessary, but low-precision arithmetic differences
remain; matching masks alone does not establish cross-backend gradient parity.

For the remaining cases, same-backend HF equality and fixed-forward VJP controls
localize the large cross-backend gradient differences primarily to differing
forward trajectories amplified by this untrained low-precision model. Replaying
all four captured attention calls with shared K/V and identical upstream
gradients gave worst joint-gradient relative L2 of 0.31–0.74% against FP32 math.
The FP16 case reproduces the 10.51% layer-norm difference, while its fixed-forward
VJP control is 0.62%. Fully FP32 attention and fully FP32 models follow different
trajectories too; neither is an exact low-precision whole-model oracle.

The original standalone file also ran twelve wide-head variants before the
model cases and hit Dynamo's recompile limit, executing unfused Flex math.
With the same frozen causal-length-128 fixture, this warmup reproduced the
original maximum logit difference of 0.113769; clearing compiler state restored
the cold-run logits and every parameter gradient bitwise.
Consequently its four Flex failures cannot be treated as isolated compiled
Triton measurements. The committed regressions isolate compiler state, test
backend-specific visibility, and check native-HF model parity separately from
FP32 attention accuracy. These controls do not establish arbitrary cross-backend
BF16 equivalence, pretrained-model convergence, or FA3/FA4 correctness.
