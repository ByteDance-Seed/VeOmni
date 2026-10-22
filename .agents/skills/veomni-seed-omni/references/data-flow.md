# Data Flow

Read this for changes to `conversation_list`, CPU preprocessing, dummy items,
raw data transforms, or request preprocessing.

## Current Pipeline

1. `seedomni_transform.py` builds and preprocesses one sample:
   - Constructs `{"conversation_list": [ConversationItem, ...]}` from raw data
     and decoded media, then runs the bound `OmniProcessor` (built from active
     graph modules via `OmniProcessor.from_config` in `OmniTrainer._build_data_transform`).
   - Preprocessors run serially in module declaration order on each sample,
     mutating `ConversationItem.value` and `meta` (including `_omni_token_len`
     and `labels` for `dyn_bsz`, plus per-sample `is_dummy` anchors) in place.
2. `SeedOmniCollator` groups per-sample dicts (after optional `dyn_bsz` batching)
   into `{"conversation_list": [[...], ...]}` plus aligned extra keys.
3. Module `pre_forward` selects items by `type`, `role`, and `meta` tags and
   moves prepared CPU tensors to the module device.
4. Module forward/modeling runs GPU work and returns a dict.
5. `post_forward` scatters outputs back to carrier items and returns
   `{"conversation_list": conversation}` (and optionally the exact scalar
   `"_loss"` key).

Janus and Qwen3-VL packed training (`packed/graph_train.yaml`) are an
optional second path: during step 1 the text-encoder preprocessor writes
packed tensors onto the sample/batch dict. Packed graph nodes
(`pack_encode` / `pack_forward` / `pack_decode`) read those tensors and
`masked_scatter` embeddings onto `packed_features`, while folding dummy
FSDP-anchor images with `mean() * 0`.

## Preprocessor Contract

`ModulePreprocessorBase` lives in `veomni/models/seed_omni/modules/module_processing_base.py`
(see `docs/seed_omni/mixins/module_preprocessor.md`).

Rules:

- Implement `preprocess_conversations(conversation_list, inference=False, **kwargs)`
  to mutate `ConversationItem.value` and `meta` in place (or override `__call__`
  when writing packed batch tensors).
- Keep preprocessors picklable, weight-free, and CPU-only (using tokenizer,
  image/video/audio processor, config values, and special token IDs) so they
  can run inside DataLoader workers.
- Shared by training and inference:
  - Training: run per sample by `seedomni_transform` inside DataLoader workers.
  - Inference: run once by `OmniInferencer` via `OmniProcessor` before the FSM.
- Use `inference=True` for inference-only behavior (vision modules skip FSDP
  dummy injection; text encoders append generation prompts when needed); when
  `inference=False` and a modality is absent from a sample, append a per-sample
  `is_dummy=True` placeholder to that sample with `_OMNI_TOKEN_LEN` set in `meta`.

## Dynamic Batching (`dyn_bsz`) Contract

When adding or modifying a SeedOmni model or module, follow these three rules so
`train.dyn_bsz: true` (`count_mode: total` or `effective`) works out of the box:

1. **Write `_OMNI_TOKEN_LEN` onto every `ConversationItem` in CPU preprocessors**:
   - Import `_OMNI_TOKEN_LEN` from `veomni.models.seed_omni.utils.conversation`.
   - **Text encoders**: `TextEncoderChatTemplate.tokenize` and `merge_text_embeds`
     automatically set `item.meta[_OMNI_TOKEN_LEN]` and `item.meta["labels"]`.
     If a model overrides `tokenize` or `merge_text_embeds` (e.g. `BagelChatTemplate`),
     it must set `item.meta[_OMNI_TOKEN_LEN] = int(token_ids.numel())` and
     `item.meta["labels"]` (`IGNORE_INDEX = -100` for masked positions).
   - **Vision / audio / codec / VAE preprocessors (`ModulePreprocessorBase`)**:
     In `preprocess_conversations`, set `item.meta[_OMNI_TOKEN_LEN]` on **every**
     item (both real items and `is_dummy=True` placeholders) to the token count
     that item contributes to the module/backbone sequence.
2. **Inject `is_dummy=True` placeholders per sample, never per batch**:
   - Because `seedomni_transform` runs `OmniProcessor.preprocess(conversation_list)`
     on each sample individually at the Dataset stage **before** `dyn_bsz` packs
     samples into a micro-batch, `preprocess_conversations` only sees a single
     sample (`len(conversation_list) == 1`) during training.
   - Always check missing modalities **per sample** (`for sample in conversation_list:`)
     and append the `is_dummy=True` placeholder (with `_OMNI_TOKEN_LEN` set in `meta`)
     to that `sample` when `not inference`.
   - In the module's GPU `pre_forward`, keep all selected items (real + `is_dummy=True`)
     in the forward batch rather than dropping `is_dummy` items when another
     sample in the micro-batch has real media.
3. **How `total` and `effective` lengths are computed (`veomni/data/dataset.py`)**:
   - `get_length_fn(sample)` (`count_mode="total"` and `physical_token_cap`):
     sums `item.meta[_OMNI_TOKEN_LEN]` across **all** items in
     `sample["conversation_list"]`, **including** `is_dummy=True` items (since
     dummy items still execute the encoder forward and consume physical budget).
   - `get_length_by_labels_fn(sample)` (`count_mode="effective"`):
     skips `is_dummy=True` items; sums `(item.meta["labels"] != IGNORE_INDEX).sum()`
     for `type == "text"` items, `item.meta[_OMNI_TOKEN_LEN]` for
     `role == "assistant"` media items (generation targets), and `0` for
     `role == "user"` media items (conditioning inputs).

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
