# Qwen3.5 toy config

Based on the Qwen3.5-27B model and processor config. The text model uses hidden
width 256, intermediate width 512, four attention heads and two KV heads. Four
layers preserve three linear-attention layers followed by one full-attention
layer. Explicit attention head width and RoPE settings are retained.

The vision model uses two layers, hidden width 128, intermediate width 512 and
four heads. Its output width is 256 to match the text model. Vocabulary, media
token IDs, patch/merge settings and `preprocessor_config.json` are preserved for
processor integration. See the [shared fixture guide](../README.md) for the
small-vocabulary text-only fixture.
