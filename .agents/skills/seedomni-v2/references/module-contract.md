# Module Contract

Read this before editing `veomni/models/seed_omni/modules/**`.

## Shape

A SeedOmni V2 module is usually:

- A `configuration.py` with a unique `model_type`.
- A `modeling.py` concrete model class that multi-inherits:
  - family/module mixin first,
  - then the real HF/diffusers/torch model class.
- A `modulemixin.py` with SeedOmni hooks.
- Optional `processing.py` for module-owned processors.
- Optional `chat_template.py` for text encoders.

Use short filenames inside the module folder:

```text
modules/<family>/<submodule>/
├── configuration.py
├── modeling.py
├── modulemixin.py
└── processing.py
```

## Required Runtime Rules

- Methods return `dict[str, Any]`.
- Training graph call-sites must implement `forward` or the referenced method.
- Use `@pre_forward("<method>")` and `@post_forward("<method>")` when one module
  has multiple graph call-sites.
- Loss keys end in `_loss` and are scalar tensors after the correct reduction.
- Module outputs should flow by mutating/returning `conversation_list` unless
  the current source file explicitly uses another supported carrier.
- Tokenizers and processors are module-owned assets.
- Do not add a top-level tokenizer path.

## Common Module Shapes

| Shape | Current example |
|---|---|
| Vision encoder | `modules/janus/siglip/` |
| VQ codec with encode/decode | `modules/janus/vqvae/` |
| Text encoder and tokenizer owner | `modules/janus/text_encoder/`, `modules/base/text_encoder/` |
| AR backbone without vocab ownership | `modules/janus/llama/` |
| Text-only LLM module | `modules/qwen3/llm/` |
| Vision-language LLM module | `modules/qwen3vl/llm/` |

## Dummy And Missing Input Semantics

- Training: active nodes must run. Use CPU-preprocessor dummies or module
  dummy paths so FSDP/DDP collective order remains aligned.
- Inference: missing optional input may return the unchanged carrier or `{}` if
  graph routing supports it for that call-site.
- Backbones that consume dummy upstream outputs must preserve a gradient path
  when training requires it.

## Registration

Check current registry names before editing. Depending on branch layout, module
registration may be centralized in `veomni/models/seed_omni/modules/__init__.py`
and re-exported from `veomni/models/seed_omni/__init__.py`.

Rules:

- `config.model_type` must match the registry key.
- Registration must happen at import time.
- New family folders should re-export public classes from their `__init__.py`.
