# Qwen4-Exp Toy Config

Based on the internal `Qwen3.8-Flash-Next` Qwen4-Exp checkpoint config.

The hidden sizes, layer counts, expert counts, PLE vocabulary, QSA budget and
vision dimensions are reduced so registry, checkpoint-conversion and short
VLM-SFT forward tests can run locally. The toy retains one linear-attention
layer with PLE and one QSA layer.
