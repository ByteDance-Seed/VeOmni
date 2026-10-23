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
predicate is sent to Flash. Packed Flash inputs must be padding-free: `None`
and all-valid 2D masks are accepted, but masks containing padding raise an
error because HF unpadding would replace the supplied document boundaries.
Unpacked Flash inputs retain normal padding-mask handling.

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

## Validation

CPU regressions cover output/loss/gradient parity, packed sample isolation,
channel-loss interception, audio visibility and dummy graph participation:

```bash
pytest tests/models/test_gemma4.py -q
pytest tests/models/base/test_auto_registry.py -k gemma4 -q
```

CPU backend-dispatch tests exercise the real HF/VeOmni Flash adapters up to a
stubbed kernel boundary, checking causal/window and varlen argument forwarding.
Additional cases run the HF varlen dispatcher with stubbed vendor functions to
check that supplied document offsets reach the selected kernel unchanged.
They do not execute FlashAttention kernels or establish numerical/gradient
parity on GPU. Real FA2/FA3/FA4 forward/backward validation remains pending.

Actual asymmetric BF16 FSDP2 validation requires two CUDA GPUs:

```bash
pytest tests/distributed/test_dummy_forward.py -k gemma4 -q
```

The three cases cover image versus text, video/audio versus text, and split
modalities across ranks. They skip on CPU. Local checks do not establish
collective correctness or full-checkpoint memory requirements.
