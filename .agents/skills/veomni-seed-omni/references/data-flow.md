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
2. `SeedOmniCollator` groups preprocessed samples:
   - Input: list of per-sample dicts (after optional `dyn_bsz` batching).
   - Output: `{"conversation_list": [[...], ...]}` plus aligned extra keys.
   - No tensor stacking, sequence padding, or SP slicing happens here.
3. Module `pre_forward` becomes thin:
   - Select items by `type`, `role`, and `meta` tags.
   - Stack/move prepared CPU tensors to the module device.
5. Module forward/modeling runs GPU work and returns a dict.
6. `post_forward` scatters outputs back to carrier items and returns
   `{"conversation_list": conversation}` and/or a `_loss` dict of scalar
   `*_loss` values.

Janus and Qwen3-VL packed training (`packed/graph_train.yaml`) are an
optional second path: after step 3 the text-encoder preprocessor also writes
packed tensors onto the collator batch dict. Packed graph nodes
(`pack_encode` / `pack_forward` / `pack_decode`) read those tensors and
`masked_scatter` embeddings onto `packed_features` instead of walking
`conversation_list`. Dummy FSDP-anchor images stay off the packed sequence
and are folded with `mean() * 0`. Qwen3-VL additionally CPU-builds 3-row
M-RoPE and `visual_pos_mask`; DeepStack features still come from the GPU
vision tower.

## Preprocessor Contract

`ModulePreprocessorBase` lives in `veomni/models/seed_omni/modules/module_processing_base.py`.

Rules:

- It must be picklable and weight-free.
- It must use CPU-safe assets only: tokenizer, image processor, config values,
  special token IDs. Never store the `nn.Module`.
- It mutates `batch["conversation_list"]` in place. Packed Janus writes tensors
  onto the same `batch` dict instead of walking items.
- It must not allocate CUDA tensors.
- It is shared by training and inference:
  - Training: run by `seedomni_transform` inside DataLoader workers.
  - Inference: run once by `OmniInferencer._preprocess_request` before the FSM.
- User images arrive in a different form on each path: training items carry
  the `(3, H, W) uint8` tensor from `veomni/data/seed_omni/image_utils.py`
  `fetch_images`, inference items carry the PIL image from
  `OmniProcessor._normalize_images`. HF image processors take both; a
  hand-written one must too, and needs a test that feeds it `fetch_images`
  output (see `tests/seed_omni/modules/deepseek_v41/test_deepseek_v41_vision_processing.py`).
  Text-only training configs never reach this path, so a run on them proves
  nothing about it.
- Use `inference=True` for inference-only behavior:
  - Vision modules skip FSDP dummy injection.
  - Text encoders append generation prompts when needed.

## Item Routing

Items carry no module ownership. Which module takes an item is decided by
`type` / `role` / `meta` tags alone, so the same data works under any
combination of modules.

- Route by the data layer's tags first (`role`, `meta[_IMG_TAG_KEY]` =
  `"und"` / `"gen"` / `"edit"`); the CPU preprocessor and the GPU hooks of one
  module use the same `iter_desired_items(..., types=, roles=, meta=)` filter.
- A model family that splits one item into internal copies or phases (BAGEL's
  SigLIP / VAE context copies, flow phases) tags them with its own `meta` key.
- A dummy placeholder is `is_dummy=True` and keeps the `role` and tags of the
  items it stands in for; read `item.is_dummy`, never a sentinel role.
- Preprocessors that select *real* inputs skip `is_dummy` items.

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
     that item contributes to the module/backbone sequence (e.g. merged ViT
     tokens `(t * h * w) // (spatial_merge_size ** 2)`, downsampled grid +
     delimiters, or VAE latent patch tokens).
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
     sample in the micro-batch has real media. If independently preprocessed
     samples can have different spatial shapes before `torch.stack`, pad them to
     a common shape in `pre_forward` (e.g. `BagelVAE._pad_to_batch_size`).
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

## Training vs Inference

- Training must keep every active graph node participating. Missing modality
  batches usually need real-shaped dummy items produced by the CPU preprocessor.
- Inference can skip absent optional inputs. A text-only request should not
  inject image dummies.
- Mid-FSM generated items are the main exception to pre-FSM preprocessing. If a
  module generates a raw item during inference, the consuming module may still
  need an on-the-fly preprocessing path in `generate`.

## Do Not

- Do not tokenize or apply chat templates in the global data transform.
- Do not stack variable-size images in the collator.
- Do not rerun request chat-template/image preprocessing inside `generate`
  after the inference CPU preprocessor already handled user inputs.
- Do not add framework-level modality routers when module-owned preprocessing is
  sufficient.
