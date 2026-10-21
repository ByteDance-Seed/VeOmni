# Bagel NaViT data balance: implementation plan

Part 3 of issue #1074 builds on the shared scheduling API and Qwen consumer
in #1191 and #1194. This draft records the second consumer's contract;
runtime integration and its E2E results are pending.

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
