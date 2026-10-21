# Data Flow

Read this for changes to `conversation_list`, CPU preprocessing, dummy items,
raw data transforms, or request preprocessing.

## Current Pipeline

1. `seedomni_transform.py` emits one sample:
   - `{"conversation_list": [ConversationItem, ...]}`
   - It should stay model-agnostic. Do not put model chat templates or tokenizer
     logic here.
2. `SeedOmniCollator` groups samples:
   - Input: list of per-sample dicts.
   - Output: `{"conversation_list": [[...], ...]}` plus aligned extra keys.
   - No tensor stacking, sequence padding, or SP slicing happens here.
3. `SeedOmniCollator` always runs the bound `OmniProcessor`:
   - Built from active graph modules (`OmniProcessor.from_config`).
   - Preprocessors run serially in module declaration order.
   - They mutate `ConversationItem.value` and `meta` in place.
4. Module `pre_forward` becomes thin:
   - Select items by `type`, `role`, and `meta` tags.
   - Stack/move prepared CPU tensors to the module device.
5. Module forward/modeling runs GPU work and returns a dict.
6. `post_forward` scatters outputs back to carrier items and returns
   `{"conversation_list": conversation}` or scalar `*_loss` values.

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
  - Training: run by `SeedOmniCollator` inside DataLoader workers.
  - Inference: run once by `OmniInferencer._preprocess_request` before the FSM.
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
