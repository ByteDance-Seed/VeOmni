# Qwen4-Exp hybrid sequence parallelism

This opt-in path combines Ulysses with CP for Qwen4-Exp. The model build gate currently permits U8 with CP2 or CP4 and requires `allow_hybrid_cp=true`; default behavior remains closed. QSA query heads must divide Ulysses. Cache decoding, explicit CP attention masks and nonzero training attention dropout are unsupported.

Input tokens and hidden states are split over the combined SP group. QSA exchanges sequence for heads inside Ulysses, then gathers KV across CP; queries remain CP-local and compact selected indices refer to the global sequence. Its CANN dispatch depends on the bounded QSA kernel in PR #1231. This is replicated global KV, not a ring attention implementation.

GDN exchanges over the combined SP group and runs the complete sequence with local heads. CP4 supports the 1K/3V head ratio by duplicating key heads and padding a fourth value slot, with autograd accumulating repeated contributions. This is **headwise GDN**, not context-state passing, and does not promise proportional recurrent activation savings as CP grows.

PLE exchanges only left token/convolution halos. Global packed boundaries reset n-gram history and convolution, including within halos and across rank boundaries. Non-SP GDN retains upstream varlen kernels or independent per-segment eager fallback. Decoder forwarding and multimodal position metadata retain upstream packing semantics.

## Validation scope

CPU/Gloo tests use real collectives and generated models, with deterministic CPU replacements only for GDN device kernels. U2CP2 and U2CP4 validate QSA/GDN/PLE outputs, input and parameter gradients, full text-model composition and non-reentrant checkpointing. PLE and GDN are also compared against independent example execution without EOS separators. The unchanged upstream packed full-model regression and compact QSA unit tests pass (17 tests total, with an import-only shim bypassing unrelated CUDA GEMM initialization).

Generated GPU/NPU source comes from Transformers 5.16.1 via patchgen. CPU tests do not qualify native NPU kernels, accelerator FSDP/EP, production U8 geometry, multimodal distributed training, throughput, DCP save or cold resume. Those remain separate integration gates. No state-pass, streaming-CP or TileLang experimental route is included.

## Current mesh ordering and regression gate

The SP mesh is CP-major and Ulysses-inner (`[CP, U]`). Ulysses exchanges
local sequence for heads, then CP KV gathering concatenates canonical
`[CP, U, local_tokens]` order without an axis transpose. QSA selections keep
global token coordinates.

`tests/parallel/ulysses/test_qwen4_exp_cp_gloo.py` launches U2CP2 and U2CP4
against both generated model variants. It checks packed boundaries, QSA/GQA,
PLE halos, GDN, the complete toy text model, checkpoint recomputation, and
all input/parameter gradients against the non-SP reference. GDN device
kernels use CPU references in this gate; it does not qualify native kernels.
The fused NPU QSA implementation matches #1231, including INT8 mask scatter.

## Packed metadata reuse

Each model forward resolves validated host sequence lengths once. Main QSA/PLE
boundaries and independently supplied GDN boundaries retain separate length
tuples. Boundary tensor identity and version checks prevent stale tuples from
being reused; inference tensors without version counters are resolved again.

QSA builds an integer-only layout once per distinct packed boundary content,
compression ratio, batch/sequence shape, SP rank/size and device within a
forward. The layout is shared across attention layers and their checkpoint
recomputation. A new model forward creates a new cache. It contains no projected
keys or differentiable activations, and the score, TopK and attention arithmetic
remain unchanged. Stable owner-major order preserves within-owner sample and
block ordering, including multi-batch inputs and empty owners.

Host row counts and a cached padding-crop index let compressed-row gathering
avoid repeated device-to-host count conversions. Existing tensor-count callers
retain their original interface. These changes reduce metadata construction and
host synchronization; they do not reduce the attention's arithmetic complexity
or guarantee any particular end-to-end speedup.

`tests/parallel/context_parallel/test_qwen4_exp_packed_layout.py` checks layout
ordering, cache invalidation, independent GDN boundaries and empty-row gradients.
The Gloo model gate uses two QSA layers and checks that layout reuse survives
non-reentrant checkpoint recomputation. An optional 32-NPU replay is provided in
`tests/parallel/context_parallel/qwen4_exp_packed_layout_native_probe.py`; it
requires the #1275 NPU-safe TopK follow-up in both baseline and candidate. It
measures selector timing and numerical agreement, not full-training throughput.
