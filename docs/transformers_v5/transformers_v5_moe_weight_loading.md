# Transformers v5 MoE Weight Loading

This note documents VeOmni MoE weight-loading expectations under
`transformers==5.9.0` (the only supported transformers version).

## Background

Transformers v5 introduced expert-dispatch integration points (`use_experts_implementation` and `ALL_EXPERTS_FUNCTIONS`).

For VeOmni's qwen3_moe path, we use a simpler approach:
- patch experts behavior in patchgen-generated modeling;
- call `veomni.ops.fused_moe_forward(...)` explicitly in the patched forward;
- gate the call on a module-level `OpSlot("moe_experts", "standard")` whose
  `use_non_eager_impl` flag is bound from
  `OpsImplementationConfig.moe_implementation` by `_bind_veomni_ops` at
  model-build time.

## Survey: Qwen MoE Weight Formats

Reference mapping from HF:
- https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/conversion_mapping.py

### qwen3_moe

- Sample checkpoint: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- HF safetensor expert layout (per-expert split keys):

```text
model.layers.0.mlp.experts.0.gate_proj.weight  [I, H]
model.layers.0.mlp.experts.0.up_proj.weight    [I, H]
model.layers.0.mlp.experts.0.down_proj.weight  [H, I]
```

- VeOmni modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- safetensor keys are per expert, while VeOmni's patchgen modeling holds
  merged expert tensors;
- a runtime `CheckpointTensorConverter` performs the per-expert -> fused
  conversion on load — no offline preprocessing required.

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

- VeOmni modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- VeOmni's modeling layout is transposed vs the safetensor dimension order
  for these tensors;
- tensor transpose/conversion is required before direct loading.

### qwen3_5_moe

- Sample checkpoint: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- HF safetensor layout:

```text
model.language_model.layers.0.mlp.experts.gate_up_proj  [num_experts, 2 * I, H]
model.language_model.layers.0.mlp.experts.down_proj     [num_experts, H, I]
```

- VeOmni modeling layout:

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

Handling summary:
- no special remap/transpose needed for shape semantics.

## Qwen3Moe Handling in VeOmni

VeOmni's patchgen-generated modeling uses the native v5 fused expert layout:
- `gate_up_proj` `[E, 2*I, H]`
- `down_proj` `[E, H, I]`

See `veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py` for the patchgen config.

### Loading (HF safetensors -> VeOmni modeling)

A runtime `CheckpointTensorConverter`
(`veomni/models/transformers/qwen3_moe/checkpoint_tensor_converter.py`) is
registered on every patchgen-generated model class. It converts per-expert HF
keys at load time:

```
HF per-expert:                             VeOmni fused:
  experts.{j}.gate_proj.weight [I, H]   ->   experts.gate_up_proj [E, 2*I, H]
  experts.{j}.up_proj.weight   [I, H]   ->     (merged via torch.cat)
  experts.{j}.down_proj.weight [H, I]   ->   experts.down_proj    [E, H, I]
```

This eliminates the need for offline `moe_merge.py` preprocessing.

### Saving (VeOmni modeling -> checkpoint)

When `model.safetensors.index.json` from a per-expert HF checkpoint is used for
sharded HF export, each MoE model registers
``_convert_fqn_to_index_mapping`` on its modeling classes (next to
``_create_checkpoint_tensor_converter`` in ``__init__.py``). Conversion runs at
runtime after weight load (cached on the model) and again in
``save_hf_safetensor`` via ``resolve_fqn_to_index_mapping_for_save``. Without this
step, ``save_safetensor_utils`` would drop fused expert tensors because the raw
index still lists ``experts.{j}.gate_proj.weight`` keys.

Training saves the model state dict as-is, producing the fused VeOmni format:

```
model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
model.layers.{i}.mlp.experts.down_proj     [E, H, I]
```

This format can be loaded directly by VeOmni (the converter's regex does not
match `gate_up_proj` keys so they pass through without conversion). However,
it is **not** compatible with standard HF `from_pretrained()` or inference
engines (vLLM/SGLang) which expect per-expert keys.

### Offline reverse conversion (fused -> per-expert HF)

To convert a VeOmni-format checkpoint back to the standard HF per-expert
format:

```bash
python scripts/moe_ckpt_merge/moe_split.py \
    --merge_hf_path <fused_checkpoint> \
    --split_hf_path <output_dir>
```

The script auto-detects the input format (fused `gate_up_proj` or legacy separate
`gate_proj`/`up_proj`) and splits back to per-expert keys. The output is compatible with:
- VeOmni (runtime converter handles per-expert keys)
- HuggingFace `from_pretrained()`
- Inference engines (vLLM, SGLang)

## VeOmni Fused MoE Op Interface

VeOmni fused MoE entrypoint:
- `veomni.ops.kernels.moe.fused_moe_forward(...)`

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
- `fc1_1_2_weight` (fused gate_up): shape `[num_experts, 2 * intermediate_dim, hidden_dim]`, used by the fused path.

## Weight Format Compatibility Matrix

| Checkpoint Format | VeOmni Load | HF `from_pretrained()` | vLLM/SGLang |
|---|---|---|---|
| HF per-expert (original) | runtime converter | direct | direct |
| legacy merged (gate/up/down separate) | needs fused layout | needs `moe_split.py` | needs `moe_split.py` |
| VeOmni fused (`gate_up_proj`) | direct | needs `moe_split.py` | needs `moe_split.py` |
