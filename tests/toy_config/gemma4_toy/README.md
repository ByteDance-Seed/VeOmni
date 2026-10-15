# Gemma 4 Toy Config

Based on [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json).

The fixture keeps Gemma 4's full/sliding attention alternation and Per-Layer
Embeddings while reducing the text model to two layers, hidden size 64, and a
128-token vocabulary. Small vision and audio towers remain present so tests
exercise the same top-level architecture as the official checkpoint.

The loss tests also enable `final_logit_softcapping=30.0` for both text and
multimodal models. Softcapping uses eager cross-entropy on the capped logits
even when a fused-linear loss is selected, since reprojecting hidden states
inside that loss would discard the cap. This fallback materializes the logits;
uncapped models retain the selected loss implementation. CPU tests compare
losses and gradients against an explicit softcapped-logit reference, using the
hardware-independent `chunk_loss` backend to exercise non-eager dispatch.

`return_log_probs=True` (including its top-k distillation path) is not supported
when logit softcapping is enabled: those fused helpers reproject uncapped logits.
The model raises a clear error for this combination rather than returning
incorrect auxiliary log-probabilities.

Audio attention builds its own boolean local mask, so the language model's
eager, SDPA or FlexAttention choice does not change audio visibility. Tests use
valid audio features (width `audio_config.subsampling_conv_channels[0]`) and
compare the mask, outputs and gradients against the HF SDPA reference.

Capped models bind eager loss on the model instance and call `loss_function`
with capped logits only. Channel-loss callbacks therefore retain their normal
interception point and report the same capped cross-entropy as training.

Gemma 4 vision Ulysses sequence parallelism is not implemented. Image, video
and direct vision calls reject it explicitly. A conditional-generation model
retaining a vision tower also rejects Ulysses on text-only ranks, preventing
peers from entering different collectives. Use `ulysses_size=1` for this model.

Under FSDP, the multimodal model visits the vision tower once for images, once
for videos, then the audio tower. Missing modalities run config-derived dummy
inputs and their projected features contribute a zero-valued graph dependency.
CPU tests cover all eight modality combinations, zero dummy gradients, output
parity and changing compute dtypes. These checks do not validate collectives.

Actual BF16 FSDP forward/backward coverage is in the existing GPU-CI module:

```bash
pytest tests/distributed/test_dummy_forward.py -k gemma4 -q
```

The three cases require two CUDA GPUs: image versus text, video/audio versus
text, and images versus video/audio on different ranks. They skip on CPU;
their hardware results must be reported separately from local unit tests.
