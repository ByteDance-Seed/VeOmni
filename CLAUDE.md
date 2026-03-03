# PROJECT: VeOmni Vision Language Agent Training with Youmu Libero Datasets

## General Instructions

- Make clear and simple code design for ease of understanding. Good design is simple yet effective deisgn.
- Write docstring and inline comments for every method
- Write comprehensive tests for every function: unit tests, integration tests, smoke tests and more if you know.
- Only write absolutely necessary code, avoid redundant code, regularly look back to see whether we can remove some necessary code, keep codebase clean.
- Avoid fallbacks. That means avoiding things like try-catch. Fail early to discover issues at the shallow surface.
- Remember that python environment is managed by `uv` and defined by `pyproject.toml`. Use `. .venv/bin/activate` to actiavte proper python env.
    - Use `uv sync --extra gpu --extra dev --extra robotics` to update environment after every edit on `pyproject.toml`
- It is fine to extend Youmu implementation when necessary. Youmu is also our repo, just follow the similar practice: simple design, comprehensive tests and doc, avoid fallback.

## Key files

- veomni/models/transformers/qwen3_vl/modeling_qwen3_vl.py
    - including the model definition for target model class: Qwen3VLForConditionalGenerationAction
- tasks/omni/train_qwen_vl_libero.py
    - training loop script
