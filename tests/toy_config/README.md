# Toy model configs

Local tests and accelerator CI share the configs in this directory. Load the
canonical `<model>_toy/config.json` directly; fixture selection does not depend
on whether a test runs locally or in CI. Reuse an existing fixture when adding
a test. Add a distinct fixture only when its model or input contract differs,
and document that contract alongside it.

## Compact integration fixtures

The Llama 3.1, DeepSeek-V3, Qwen3, Qwen3-MoE, Qwen3.5 dense/MoE,
Qwen3-VL, Qwen3-VL-MoE and Qwen3-Omni fixtures use reduced dimensions for
checkpoint, LoRA, parity and distributed integration tests. Text hidden widths
are 256. Keep model-specific structure required by the tests:

- Qwen3-MoE has four MoE layers for per-layer expert-parallel slicing checks.
- Qwen3.5 has four layers: three linear-attention layers and one full-attention
  layer. Its explicit head and RoPE dimensions must remain compatible.
- DeepSeek-V3 retains both dense and MoE layers.
- Multimodal fixtures retain tokenizer/media token IDs, vocabularies, processor
  settings and RoPE dimensions. Vision/audio outputs must match text connector
  widths. Shrinking the vocabulary would invalidate real processor inputs.
- Qwen3-VL-MoE expert intermediate width must differ from hidden width, and twice
  that intermediate width must also differ from hidden width, so checkpoint
  conversion can identify weight layouts without guessing a transpose.

Standalone text fixtures use a vocabulary of 4096. The existing
`qwen3_5_fsdp_equiv_toy` and `qwen3_5_moe_fsdp_equiv_toy` fixtures supply the
small-vocabulary text contract used by synthetic FSDP/shared-prefix tests.
Their `model_type` is the full Qwen3.5 wrapper, but these tests use only text
inputs. Use `qwen3_5_toy` / `qwen3_5_moe_toy` for actual processor or multimodal
inputs, which require the original vocabulary and special token IDs. Both
variants belong to the shared fixture set and are available locally and in CI.

Keep backend choices, parallel world sizes, input shapes, numerical tolerances
and round-trip assertions in the tests. A smaller fixture must not implicitly
remove the model structure or kernel shapes a test intends to cover; tests for
specific kernel dimensions should declare those dimensions explicitly.
