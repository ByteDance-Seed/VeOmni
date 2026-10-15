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
