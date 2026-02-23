# Qwen3-MoE Experts Patching in Transformers v5

This note documents the current VeOmni strategy for `qwen3_moe` on `transformers>=5.0.0`.

## Why This Changed

`transformers` v5 introduced experts dispatch hooks via `use_experts_implementation` and `ALL_EXPERTS_FUNCTIONS`.
We originally explored registering VeOmni fused experts through that interface, but it added coupling with config-time validation and runtime dispatch behavior that made integration brittle.

The current strategy is simpler:
- patch `Qwen3MoeExperts` directly in generated modeling code;
- call `veomni.ops.fused_moe_forward(...)` explicitly inside the patched experts `forward`;
- keep `_moe_implementation` (`eager` or `fused`) as the runtime switch.

## Current Code Path

1. Patch spec:
`veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py`
- replaces `Qwen3MoeExperts`;
- keeps expert weights as `gate_proj`, `up_proj`, `down_proj`;
- routes fused execution through `fused_moe_forward`.

2. Generated modeling:
`veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py`
- used by `veomni/models/transformers/qwen3_moe/__init__.py` when `transformers>=5.0.0`.

3. Kernel patch bootstrap:
`veomni/ops/fused_moe/__init__.py`
- selects kernel backend for `fused_moe_forward`;
- does not register an experts implementation into `ALL_EXPERTS_FUNCTIONS`.

4. Model build:
`veomni/models/auto.py`
- sets `config._moe_implementation` before model creation;
- does not set `_experts_implementation`.

## Regeneration

```bash
source .venv/bin/activate
python -m veomni.patchgen.run_codegen \
  veomni.models.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config \
  -o veomni/models/transformers/qwen3_moe/generated \
  --diff
```

## Checkpoint Layout Limitation

This path does not do runtime remapping from legacy per-expert keys.

Expected expert tensor layout is merged-by-expert dimension tensors (for `gate_proj`, `up_proj`, `down_proj`), not scattered per-expert module keys like:
- `...experts.0.gate_proj.weight`
- `...experts.1.up_proj.weight`

If your checkpoint is in per-expert layout (including multi-file safetensors shards), convert it offline before loading.
