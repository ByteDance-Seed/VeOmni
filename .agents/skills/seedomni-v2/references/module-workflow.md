# Module Workflow

Use this workflow when adding or changing a SeedOmni V2 module.

## 1. Pick The Closest Live Example

Start from `references/janus-example-map.md`.

- Vision or image-understanding encoder: Janus SigLIP.
- Image codec: Janus VQVAE.
- Text encoder/tokenizer/chat template: Janus text encoder plus base text encoder.
- AR backbone without vocab layers: Janus Llama.
- Text-only LLM: Qwen3 LLM.
- VLM tower + LLM: Qwen3VL vision and LLM.

Do not start from a synthetic placeholder module unless no live shape exists.

## 2. Decide Ownership

Answer these before editing:

- Which module owns the tokenizer or processor asset?
- Which module creates or consumes each `ConversationItem.source`?
- Which module injects dummy items for training?
- Which graph call-sites use this module?
- Does the module need a CPU preprocessor?
- Does the module need a per-module parallel plan?

## 3. Implement Locally

Follow the existing folder shape:

- `configuration.py`: config fields, unique `model_type`.
- `modeling.py`: weights and the actual model forward.
- `modulemixin.py`: SeedOmni hooks and CPU preprocessor.
- `processing.py`: processor wrapper only when needed.
- `chat_template.py`: text encoder family-specific template only when needed.

Keep hooks explicit and readable. Prefer small helpers with behavior names over
dense comprehensions for carrier filtering or source routing.

## 4. Wire The Runtime

- Add or update graph edges in the appropriate config.
- Add module config in `modules_train.yaml` and, if needed, inference module
  config files.
- Update split checkpoint conversion if weights or assets come from a monolith.
- Register and re-export the module class.

## 5. Validate The Narrow Surface First

- Import/build the new module.
- Run graph visualization for the target `base.yaml`.
- Run a focused unit or smoke test.
- Run broader SeedOmni tests only after the narrow behavior is stable.
