# Qwen4-Exp hybrid sequence parallelism

This opt-in path combines Ulysses with CP for Qwen4-Exp. The model build gate currently permits U8 with CP2 or CP4 and requires `allow_hybrid_cp=true`; default behavior remains closed. QSA query heads must divide Ulysses. Cache decoding, explicit CP attention masks and nonzero training attention dropout are unsupported.

Input tokens and hidden states are split over the combined SP group. QSA exchanges sequence for heads inside Ulysses, then gathers KV across CP; queries remain CP-local and compact selected indices refer to the global sequence. Its CANN dispatch depends on the bounded QSA kernel in PR #1225. This is replicated global KV, not a ring attention implementation.

GDN exchanges over the combined SP group and runs the complete sequence with local heads. CP4 supports the 1K/3V head ratio by duplicating key heads and padding a fourth value slot, with autograd accumulating repeated contributions. This is **headwise GDN**, not context-state passing, and does not promise proportional recurrent activation savings as CP grows.

PLE exchanges only left token/convolution halos. Global packed boundaries reset n-gram history and convolution, including within halos and across rank boundaries. Non-SP GDN retains upstream varlen kernels or independent per-segment eager fallback. Decoder forwarding and multimodal position metadata retain upstream packing semantics.

## Validation scope

CPU/Gloo tests use real collectives and generated models, with deterministic CPU replacements only for GDN device kernels. U2CP2 and U2CP4 validate QSA/GDN/PLE outputs, input and parameter gradients, full text-model composition and non-reentrant checkpointing. PLE and GDN are also compared against independent example execution without EOS separators. The unchanged upstream packed full-model regression and compact QSA unit tests pass (17 tests total, with an import-only shim bypassing unrelated CUDA GEMM initialization).

Generated GPU/NPU source comes from Transformers 5.16.1 via patchgen. CPU tests do not qualify native NPU kernels, accelerator FSDP/EP, production U8 geometry, multimodal distributed training, throughput, DCP save or cold resume. Those remain separate integration gates. No state-pass, streaming-CP or TileLang experimental route is included.
