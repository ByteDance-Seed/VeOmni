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

### Transformers v4 (stable, `transformers==4.57.3`)

VeOmni keeps split expert tensors in patched modeling:
- `gate_proj` `[E, I, H]`
- `up_proj` `[E, I, H]`
- `down_proj` `[E, H, I]`

This differs from native Transformers v5 `gate_up_proj` layout.

Checkpoint loading behavior:
- VeOmni does not do runtime remapping from legacy per-expert keys;
- HuggingFace safetensor checkpoints commonly store expert weights in per-expert form.

To avoid loading/mapping issues, merge weights offline before training:
- `scripts/moe_ckpt_merge/moe_merge.py`

### Transformers v5 (`transformers>=5.0.0`)

VeOmni v5 patchgen modeling uses the native v5 fused expert layout:
- `gate_up_proj` `[E, 2*I, H]`
- `down_proj` `[E, H, I]`

See `veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py` for the patchgen config.

#### Loading (HF safetensors -> v5 modeling)

A runtime `CheckpointTensorConverter` (`veomni/models/transformers/qwen3_moe/checkpoint_tensor_converter.py`)
is registered on model classes when `transformers>=5.0.0`. It converts per-expert HF keys at load time:

```
HF per-expert:                             v5 fused:
  experts.{j}.gate_proj.weight [I, H]   ->   experts.gate_up_proj [E, 2*I, H]
  experts.{j}.up_proj.weight   [I, H]   ->     (merged via torch.cat)
  experts.{j}.down_proj.weight [H, I]   ->   experts.down_proj    [E, H, I]
```

This eliminates the need for offline `moe_merge.py` preprocessing.

#### Saving (v5 modeling -> checkpoint)

Training saves the model state dict as-is, producing v5 fused format:

```
model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
model.layers.{i}.mlp.experts.down_proj     [E, H, I]
```

This format can be loaded directly by v5 VeOmni (the converter's regex does not match
`gate_up_proj` keys so they pass through without conversion). However, it is **not**
compatible with v4 VeOmni, standard HF `from_pretrained()`, or inference engines
(vLLM/SGLang) which expect per-expert keys.

#### Offline reverse conversion (v5 fused -> per-expert HF)

To convert a v5-format checkpoint back to the standard HF per-expert format:

```bash
python scripts/moe_ckpt_merge/moe_split.py \
    --merge_hf_path <v5_checkpoint> \
    --split_hf_path <output_dir>
```

The script auto-detects the input format (v5 `gate_up_proj` or v4 separate `gate_proj`/`up_proj`)
and splits back to per-expert keys. The output is compatible with:
- v4 VeOmni (after running `moe_merge.py` if needed)
- v5 VeOmni (runtime converter handles per-expert keys)
- HuggingFace `from_pretrained()`
- Inference engines (vLLM, SGLang)

## VeOmni Fused MoE Op Interface

VeOmni fused MoE entrypoint:
- `veomni.ops.fused_moe.fused_moe_forward(...)`

Current signature supports both split and fused gate/up weights:

```python
fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,       # gate [E, I, H], or None if fc1_1_2_weight is provided
    fc1_2_weight: torch.Tensor,       # up   [E, I, H], or None if fc1_1_2_weight is provided
    fc2_weight: torch.Tensor,         # down [E, H, I]
    fc1_1_2_weight: torch.Tensor,     # fused gate_up [E, 2*I, H], optional
)
```

Expected tensor interface:
- `hidden_states`: token-major hidden states used by experts, shape `[num_tokens, hidden_dim]`;
- `routing_weights`: router top-k probabilities, shape `[num_tokens, top_k]`;
- `selected_experts`: router top-k expert indices, shape `[num_tokens, top_k]`;
- `fc1_1_weight` (gate): shape `[num_experts, intermediate_dim, hidden_dim]`;
- `fc1_2_weight` (up): shape `[num_experts, intermediate_dim, hidden_dim]`;
- `fc2_weight` (down): shape `[num_experts, hidden_dim, intermediate_dim]`;
- `fc1_1_2_weight` (fused gate_up): shape `[num_experts, 2 * intermediate_dim, hidden_dim]`, used by v5 path.

## Weight Format Compatibility Matrix

| Checkpoint Format | v4 VeOmni Load | v5 VeOmni Load | HF `from_pretrained()` | vLLM/SGLang |
|---|---|---|---|---|
| HF per-expert (original) | needs `moe_merge.py` | runtime converter | direct | direct |
| v4 merged (gate/up/down separate) | direct | needs re-merge with v5 format | needs `moe_split.py` | needs `moe_split.py` |
| v5 fused (gate_up_proj) | incompatible | direct | needs `moe_split.py` | needs `moe_split.py` |
