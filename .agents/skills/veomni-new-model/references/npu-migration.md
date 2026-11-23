# GPU-to-NPU Model Migration

Loaded from `/veomni-new-model` when adapting an existing VeOmni GPU model under `veomni/models/transformers/<model>/modeling_<model>.py` to Ascend NPU. Scope stays focused on model code, `__init__.py` registration, and model tests (no ad-hoc training configs, scripts, logs, or checkpoints).

Read `.agents/knowledge/constraints.md`, `veomni/ops/README.md`, `docs/design/op_selection.md`, and (for VLMs) `.agents/knowledge/multimodal_metadata.md`. Peer references: `qwen3/`, `seed_oss/`, `qwen3_vl/`, `qwen3_5/`, `gemma3/` (dense text/VLM worked example).

## 1. Audit Existing Model

1. Verify `transformers` matches `pyproject.toml`, and inspect `modeling_<model>.py`, `__init__.py`, and live `OP_REGISTRY` rows:
   ```bash
   rg -n "cuda|triton|flash_attention|flex_attention|liger|is_cuda|\.cuda\(" veomni/models/transformers/<model>
   rg -n "VeomniOp|ops_config|attn_implementation|RMSNorm|apply_rotary|cross_entropy" veomni/models/transformers/<model> veomni/ops/kernels
   ```
2. Produce a minimal file inventory before editing. Record an explicit **no-change** decision when `VeomniOp` runtime dispatch already covers NPU (true for most HF-derived models).
3. Audit device-neutral allocations (derive device from inputs/params, no `.cuda()`), dtypes, distributed APIs, and exact op semantics (e.g. Gemma 3 zero-init RMSNorm uses variant `"offset"` for `1 + weight` — do not add `1` again in modeling; verify soft-capping, label shifting, ignore index, and loss reduction).

| Operation | Typical GPU backend | Typical NPU choice |
|---|---|---|
| Attention | `flex_attention` / flash attention | `sdpa` or `eager` |
| RMSNorm | `liger_kernel` | `torch_npu` (when variant matches; unweighted RMSNorm on NPU is `eager` only) |
| RoPE | `liger_kernel` | `torch_npu` |
| Causal CE | `liger_kernel` | `chunk_loss` or `eager` |
| SwiGLU / MoE / Aux loss | fused GPU / `fused_triton` / `triton` | registered NPU backend (`fused_npu`) or `eager` |

## 2. Adapt `modeling_<model>.py` & Registration

- Keep a **single** `modeling_<model>.py` across CUDA, ROCm, NPU, MLU, and CPU — never create `_npu.py` forks or top-level `torch_npu` imports.
- Bind instance-local `VeomniOp` handles from `config.ops_config` at `__init__`. If a required `(op, variant, implementation)` row is missing, add it via `/veomni-new-op` with `NpuKernelRequirement()`.
- Gate unavoidable structural NPU differences (e.g. Ascend TopKV2 finite masks or host `cu_seqlens` placement) explicitly with `IS_NPU_AVAILABLE` or `tensor.device.type == "npu"` — never `not IS_CUDA_AVAILABLE`.
- Ensure `__init__.py` registers all supported `model_type`s (for VLMs: both multimodal wrapper and text backbone).

## 3. Multimodal Wrappers (If Applicable)

- Audit vision/audio towers, projector, placeholder masks, packed metadata, SP boundaries, and outer `ForConditionalGeneration.forward` independently of the text backbone.
- Compare source checkpoint keys against `model.named_parameters()` (e.g. Gemma 3 legacy `language_model.model.*` / `vision_tower.vision_model.*` / `multi_modal_projector.*` mapping to nested `model.*`; treat unexplained missing/unexpected keys as hard failures).
- Ensure text-only microbatches on distributed ranks run a minimal `dummy_forward` through every trainable modality component (attached with a zero-valued dependency) so FSDP/DDP reductions stay symmetric with real-media ranks.

## 4. Validate

```bash
python -m py_compile veomni/models/transformers/<model>/modeling_<model>.py
PYTHONPATH=. pytest --noconftest tests/models/transformers/test_generated_call_site_signatures.py
python -c "import veomni.models.transformers.<model>.modeling_<model>"
make quality
```

Extend the closest model test under `tests/models/` covering import, instance-local `VeomniOp` isolation across configs, eager parity, finite fwd/bwd gradients, checkpoint conversion, and multimodal/dummy-forward propagation as applicable.
