# Gemma 4 Toy Config

Based on [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json).

The fixture keeps Gemma 4's full/sliding attention alternation and Per-Layer
Embeddings while reducing the text model to two layers, hidden size 64, and a
128-token vocabulary. Small vision and audio towers remain present so tests
exercise the same top-level architecture as the official checkpoint.
