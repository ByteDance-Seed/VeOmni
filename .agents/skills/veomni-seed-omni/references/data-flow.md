# Data Flow

Read this for changes to `conversation_list`, CPU preprocessing, dummy items,
raw data transforms, or request preprocessing.

## Current Pipeline

1. `seedomni_transform.py` emits one model-agnostic sample:
   `{"conversation_list": [ConversationItem, ...]}`; chat templates and
   tokenization run inside module preprocessors.
2. `SeedOmniCollator` groups per-sample dicts into
   `{"conversation_list": [[...], ...]}` plus aligned extra keys.
3. `SeedOmniCollator` runs the bound `OmniProcessor` (`OmniProcessor.from_config`):
   - Preprocessors run serially in module declaration order and mutate
     `ConversationItem.value` and `meta` in place.
4. Module `pre_forward` selects items by `type`, `role`, and `meta` tags and
   moves prepared CPU tensors to the module device.
5. Module forward/modeling runs GPU work and returns a dict.
6. `post_forward` scatters outputs back to carrier items and returns
   `{"conversation_list": conversation}` (and optionally the exact scalar
   `"_loss"` key).

Janus and Qwen3-VL packed training (`packed/graph_train.yaml`) are an
optional second path: after step 3 the text-encoder preprocessor also writes
packed tensors onto the collator batch dict. Packed graph nodes
(`pack_encode` / `pack_forward` / `pack_decode`) read those tensors and
`masked_scatter` embeddings onto `packed_features`, while folding dummy
FSDP-anchor images with `mean() * 0`.

## Preprocessor Contract

`ModulePreprocessorBase` lives in `veomni/models/seed_omni/modules/module_processing_base.py`.

Rules:

- Keep preprocessors picklable, weight-free, and CPU-only (using tokenizer,
  image/video/audio processor, config values, and special token IDs) so they
  can run inside DataLoader workers.
- Mutate `batch["conversation_list"]` in place (or write packed tensors onto
  `batch` for packed graphs).
- Shared by training and inference:
  - Training: run by `SeedOmniCollator` inside DataLoader workers.
  - Inference: run once by `OmniInferencer._preprocess_request` before the FSM.
- Use `inference=True` for inference-only behavior (vision modules skip FSDP
  dummy injection; text encoders append generation prompts when needed).

## Item Routing

Modules select items by `type`, `role`, and `meta` tags so the same data works
across module combinations:

- Route by the data layer's tags first (`role`, `meta[_IMG_TAG_KEY]` =
  `"und"` / `"gen"` / `"edit"`); the CPU preprocessor and GPU hooks of one
  module use the same `iter_desired_items(..., types=, roles=, meta=)` filter.
- A model family that splits one item into internal copies or phases tags them
  with its own `meta` key.
- A dummy placeholder has `is_dummy=True` and preserves the `role` and tags of
  the items it stands in for; preprocessors that select real inputs filter on
  `not item.is_dummy`.

## Training vs Inference

- Training keeps every active graph node participating by injecting real-shaped
  dummy items when a modality is absent from a micro-batch.
- Inference skips absent optional inputs.
- If a module generates a raw item mid-FSM during inference, the consuming
  module preprocesses that generated item on the fly in `generate`.
