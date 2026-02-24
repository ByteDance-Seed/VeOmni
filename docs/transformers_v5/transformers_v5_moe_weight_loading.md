# Transformers v5 MoE Weight Loading

This note documents the current VeOmni behavior for MoE weights with `transformers>=5.0.0`.

## Current Design

### qwen3_moe modeling format

For qwen3_moe on transformers v5, patched modeling uses:
- `gate_up_proj`: `[E, 2*I, H]` (in `[gate, up]` order)
- `down_proj`: `[E, H, I]`

VeOmni currently replaces the whole `Qwen3MoeExperts` class in patchgen output and keeps `_moe_implementation`
(`eager` or `fused`) as runtime selection.

### fused path behavior and TODO

The fused MoE path still consumes split gate/up weights. In forward:
- `gate_weight = gate_up_proj[:, :I, :]`
- `up_weight = gate_up_proj[:, I:, :]`

Both slices are materialized with `.contiguous()` before calling `fused_moe_forward(...)`, because the current
group-gemm fused MoE implementation expects contiguous weight tensors.

TODO:
- Add a fused MoE op interface that accepts stacked `gate_up_proj` directly to remove runtime split and
  `.contiguous()` copies.

## Runtime Checkpoint Conversion Flow (qwen3_moe Example)

Runtime conversion is model-specific and incremental.

### 1) Registration

In `veomni/models/transformers/qwen3_moe/__init__.py`, for `transformers>=5.0.0`, qwen3_moe model classes are
registered with:
- `model_cls._create_checkpoint_tensor_converter = create_qwen3_moe_checkpoint_tensor_converter`

### 2) Loader hook point

In `veomni/models/module_utils.py`:
- loader resolves converter once via `get_hf_checkpoint_tensor_converter(model)`;
- each checkpoint tensor is passed through `maybe_convert_hf_checkpoint_tensor(...)`;
- converter can return:
  - a converted tensor (`HfConvertedCheckpointTensor`) to dispatch immediately, or
  - `None` to keep accumulating until enough source tensors are seen.

This is used in both regular loading and rank0-broadcast loading paths.

### 3) qwen3_moe converter behavior

In `veomni/models/transformers/qwen3_moe/checkpoint_tensor_converter.py`:
- handles keys matching per-expert regex:
  - `model.layers.{L}.mlp.experts.{E}.(gate_proj|up_proj|down_proj).weight`
- validates layer/expert indices and input tensor shapes against config (`num_hidden_layers`, `num_experts`,
  `hidden_size`, `moe_intermediate_size`);
- accumulates tensors per layer until all experts are available;
- emits:
  - `model.layers.{L}.mlp.experts.gate_up_proj` as `cat([stack(gate), stack(up)], dim=1)` -> `[E, 2*I, H]`
  - `model.layers.{L}.mlp.experts.down_proj` as `stack(down)` -> `[E, H, I]`
- drops per-layer accumulation buffers immediately after emission to reduce CPU memory.

Because conversion is incremental, per-expert tensors can arrive in arbitrary order and across multiple safetensor
files; emission happens when a layer has all required experts for a target tensor.

## SOP: Add Runtime Conversion for a New Model

Use this checklist when a model’s checkpoint tensor format differs from `transformers>=5` modeling layout.

1. Add a model-local converter implementation
- Create `<model>/checkpoint_tensor_converter.py`.
- Implement converter with:
  - `can_handle(name: str) -> bool`
  - `convert(name: str, tensor: torch.Tensor) -> HfConvertedCheckpointTensor | None`
- Keep logic self-contained for that model.

2. Validate aggressively from config
- Validate key patterns (layer/expert indices).
- Validate source tensor shapes against config dimensions.
- Raise clear errors on mismatch.

3. Emit v5-native target tensors
- Emit final state-dict names and shapes that exactly match model parameters in patched/generated modeling.
- Return `None` while accumulating partial inputs.

4. Register converter factory on model classes
- In model `__init__.py`, set `_create_checkpoint_tensor_converter` for relevant model classes.
- Guard registration with transformers version checks when behavior is v5-specific.

5. Verify both load paths
- Ensure behavior works for:
  - direct checkpoint loading
  - rank0 load + broadcast path
- Confirm no unexpected keys/missing keys for converted parameters.

6. Add unit tests with fake safetensors
- Cover:
  - in-order and out-of-order key arrival
  - multi-shard-style arrival patterns
  - invalid shape/index error paths
  - exact emitted names and tensor shapes

7. Document model-specific rules
- Update `docs/transformers_v5/` with:
  - source checkpoint key format
  - target modeling tensor format
  - conversion strategy and current limitations.

## Survey: Qwen MoE Weight Formats

Reference mapping from HF:
- https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/conversion_mapping.py

### qwen3_moe

- Sample checkpoint: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- HF safetensor expert layout (per-expert split keys):

```text
model.layers.0.mlp.experts.0.gate_proj.weight  [I, H]
model.layers.0.mlp.experts.0.up_proj.weight    [I, H]
model.layers.0.mlp.experts.0.down_proj.weight  [H, I]
```

- Transformers v5 modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- safetensor keys are per expert, while v5 expects merged expert tensors;
- for VeOmni qwen3_moe training, run offline merge first via `scripts/moe_ckpt_merge/moe_merge.py`.

Other Qwen3 family models with similar layout like qwen3_moe (i.e., per-expert split keys in safetensors):
- Qwen3 Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3 Omni: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct

### qwen3_vl_moe

- Sample checkpoint: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
- HF safetensor layout:

```text
model.language_model.layers.0.mlp.experts.gate_up_proj  [num_experts, H, 2 * I]
model.language_model.layers.0.mlp.experts.down_proj     [num_experts, I, H]
```

- Transformers v5 modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- v5 layout is transposed vs the safetensor dimension order for these tensors;
- tensor transpose/conversion is required before direct v5 loading.

### qwen3_5_moe

- Sample checkpoint: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- HF safetensor layout:

```text
model.language_model.layers.0.mlp.experts.gate_up_proj  [num_experts, 2 * I, H]
model.language_model.layers.0.mlp.experts.down_proj     [num_experts, H, I]
```

- Transformers v5 modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- no special remap/transpose needed for shape semantics.

## Future Work: Align with Transformers v5 Weight Formatting

To reduce integration friction and runtime overhead, we should converge toward v5-native MoE weight handling.

- For models whose safetensor layout is already close to Transformers v5 (for example, `qwen3_5_moe`), add fused-op support for v5-native MoE tensors directly.
  This avoids extra offline remapping and avoids runtime reshape/copy steps such as `.contiguous()` in expert forward paths.

- For models with layout mismatch (for example, `qwen3_moe`), we still need to choose one stable strategy:
  1. Offline remap to v5 format before training.
  2. Runtime remap during model loading.

  - Tradeoffs:
    1. Offline remap: lower runtime complexity and more predictable execution, but adds preprocessing burden and user error risk.
    2. Runtime remap: less user preprocessing and easier onboarding, but adds loader complexity and may introduce runtime variability.
