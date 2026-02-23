# Transformers v5 MoE Weight Loading

This note documents VeOmni MoE weight-loading expectations for `transformers>=5.0.0`.

## Background

Transformers v5 introduced expert-dispatch integration points (`use_experts_implementation` and `ALL_EXPERTS_FUNCTIONS`).

For VeOmni qwen3_moe transformers v5 path, we use a simpler path:
- patch experts behavior in generated modeling;
- call `veomni.ops.fused_moe_forward(...)` explicitly in the patched forward;
- keep `_moe_implementation` (`eager` or `fused`) as runtime selection.

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

## Qwen3Moe Handling in VeOmni

For qwen3_moe, VeOmni keeps split expert tensors in patched modeling:
- `gate_proj`
- `up_proj`
- `down_proj`

This differs from native Transformers v5 `gate_up_proj` layout.

Checkpoint loading behavior:
- VeOmni does not do runtime remapping from legacy per-expert keys;
- HuggingFace safetensor checkpoints commonly store expert weights in per-expert form.

To avoid loading/mapping issues, merge weights offline before training:
- `scripts/moe_ckpt_merge/moe_merge.py`

## VeOmni Fused MoE Op Interface

VeOmni fused MoE entrypoint:
- `veomni.ops.fused_moe.fused_moe_forward(...)`

Current signature:

```python
fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
)
```

Expected tensor interface:
- `hidden_states`: token-major hidden states used by experts, shape `[num_tokens, hidden_dim]`;
- `routing_weights`: router top-k probabilities, shape `[num_tokens, top_k]`;
- `selected_experts`: router top-k expert indices, shape `[num_tokens, top_k]`;
- `fc1_1_weight` (gate): shape `[num_experts, intermediate_dim, hidden_dim]`;
- `fc1_2_weight` (up): shape `[num_experts, intermediate_dim, hidden_dim]`;
- `fc2_weight` (down): shape `[num_experts, hidden_dim, intermediate_dim]`.

Important constraints:
- op expects split gate/up tensors (`fc1_1_weight` and `fc1_2_weight`), not a merged `gate_up_proj` tensor;
- needs to be `.contiguous()`.

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
