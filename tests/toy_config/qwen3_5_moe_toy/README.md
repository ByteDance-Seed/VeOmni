# Qwen3.5-MoE toy config

Based on the Qwen3.5-35B-A3B model config. Four layers preserve three
linear-attention layers followed by one full-attention layer. Text hidden width
is 256, with four attention heads and two KV heads. Explicit head width and RoPE
settings are retained. There are 16 experts with top-k 2, expert intermediate
width 256 and shared-expert intermediate width 256.

The vision model uses two layers, hidden width 128, intermediate width 512 and
four heads. Its output width is 256 to match the text model. Vocabulary, media
token IDs and patch/merge settings are preserved for multimodal inputs. See the
[shared fixture guide](../README.md) for the small-vocabulary text-only fixture.
