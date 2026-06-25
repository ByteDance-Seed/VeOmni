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
3. `SeedOmniCollator` runs ordered `CPUPreprocessor`s:
   - The trainer collects them from active graph modules.
   - They run serially in module declaration order.
   - They mutate `ConversationItem.value`, `source`, and `meta` in place.
4. Module `pre_forward` becomes thin:
   - Select items by `type`, `role`, and usually `source`.
   - Stack/move prepared CPU tensors to the module device.
5. Module forward/modeling runs GPU work and returns a dict.
6. `post_forward` scatters outputs back to carrier items and returns
   `{"conversation_list": conversation}` or scalar `*_loss` values.

## CPUPreprocessor Contract

`CPUPreprocessor` lives in `veomni/models/seed_omni/mixins/modulemixin.py`.

Rules:

- It must be picklable and weight-free.
- It must use CPU-safe assets only: tokenizer, image processor, config values,
  special token IDs. Never store the `nn.Module`.
- It mutates `conversation_list` in place.
- It must not allocate CUDA tensors.
- It is shared by training and inference:
  - Training: run by `SeedOmniCollator` inside DataLoader workers.
  - Inference: run once by `OmniInferencer._preprocess_request` before the FSM.
- Use `inference=True` for inference-only behavior:
  - Vision modules skip FSDP dummy injection.
  - Text encoders append generation prompts when needed.

## Source Tagging

Prefer `ConversationItem.source` as branch identity.

- CPU preprocessors should tag real and dummy items with their module source.
- GPU hooks should filter with helpers such as `iter_desired_items(..., sources=[...])`.
- Avoid mixing `meta["source"]`, `role == "dummy"`, and `None` fallbacks unless
  the source producer is not yet available in a specific path.

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
