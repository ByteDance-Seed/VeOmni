# Bagel NaViT data balance: implementation plan

Part 3 of issue #1074 builds on the shared scheduling API and Qwen consumer
in #1191 and #1194. This draft records the second consumer's contract;
runtime integration and its balancing on/off E2E results are pending.

## Completed baseline E2E

The existing five-module Bagel training graph passed two actual AdamW steps
with BF16/FSDP2 on four L20 GPUs in each layout: DP4/SP1, DP2/SP2 and DP1/SP4.
All layouts used the same seeded tiny checkpoint (shared hidden size 128,
8 query heads, 4 KV heads, head dimension 16) and 16 raw multi-turn
editing/generation samples. Each log contains all eight rank/step records:
finite positive text loss, flow loss and gradient norm, and identical loss
and module token dictionaries across ranks within that layout. All five
modules have positive token counts. Cross-layout numerical equality is not
asserted.

Native eager understanding, generation and editing FSMs also passed through
the real shared CPU preprocessors. Generation and editing each completed two
finite denoise updates, VAE decoding, image output and flow request-state
reset. Understanding produced nonempty text from a raw image.

These are untrained tiny-checkpoint execution checks, not pretrained-model
quality, convergence, data-balancing on/off equality or performance evidence.
The two 8-by-8 image artifacts below are actual saved FSM outputs, not
illustrations. They are identical for this seeded fixture; no meaningful
editing-quality claim is made.

Generation: ![tiny generation output](assets/bagel_e2e/tiny_generation.png)

Editing: ![tiny editing output](assets/bagel_e2e/tiny_edit.png)

## Consumer contract

The consumer is `modules/bagel/siglip_navit/accelerated/accelerated.py`.
Its preprocessor emits `token_lens`, `patchified_pixel_values`, and
`patchified_position_ids`. Lengths partition both patch tensors by image.
Declare per-image attention costs using token lengths and the module's
attention structure; preserve dummy carriers as participating items.

Meter full original sample lengths before scheduling. Resolve groups inside
the scoped hook. Cross-DP exchange precedes SP item assignment. NaViT's
contiguous per-image SP split needs a balanced item permutation, followed by
inverse permutation after SP gather. Restore cross-DP owners before scattering
embeddings into the original conversation. Rebuild cu_seqlens and max_seqlen
from the selected local image lengths. Keep invocation plans independent
through backward and checkpoint recomputation.

## Required validation

- Existing CI-enumerated distributed round-trip tests parameterized for the
  Bagel item layout, unequal token lengths, dummy items and inverse VJPs.
- Complete tiny Bagel OmniTrainer on/off runs, ordinary and packed where
  supported, with bit-identical loss and per-module consume_tokens.
- Assigned four-GPU DP/SP layouts; asymmetric multimodal forward/backward
  without collective desynchronization.
- Deliberately skewed image-count workloads: per-rank encoder wall time and
  MFU before/after, with warmup, measurement scope and host-load limitations.

The maintainer requested review of the mixin contract before wiring this
consumer. The draft remains incomplete until that feedback, implementation,
independent review and the validation above are recorded.
