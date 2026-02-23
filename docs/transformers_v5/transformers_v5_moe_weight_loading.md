# Transformers v5 MoE Weight Loading

This note documents VeOmni MoE weight-loading expectations for `transformers>=5.0.0`.

## Background

Transformers v5 introduced expert-dispatch integration points (`use_experts_implementation` and `ALL_EXPERTS_FUNCTIONS`).
For VeOmni qwen3_moe, we use a simpler path:
- patch experts behavior in generated modeling;
- call `veomni.ops.fused_moe_forward(...)` explicitly in the patched forward;
- keep `_moe_implementation` (`eager` or `fused`) as runtime selection.

## Qwen3Moe Handling

For qwen3_moe, VeOmni keeps split expert tensors:
- `gate_proj`
- `up_proj`
- `down_proj`

This differs from the native Transformers v5 merged `gate_up_proj` layout.

Checkpoint loading behavior:
- VeOmni does not do runtime remapping from legacy per-expert keys.
- HuggingFace safetensor checkpoints often store MoE expert weights in per-expert form (for example `...experts.0.gate_proj.weight`).

To avoid loading/mapping issues, merge weights offline before training:
- `scripts/moe_ckpt_merge/moe_merge.py`

## Surveys of MoE weight formatting


- qwen3_moe
  - Sample HF checkpoint link: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
  - expert safetensor format:
  ```
  # Per expert split weight
  # Gate/up: [I, H]
  # Down: [H, I]
  model.layers.0.mlp.experts.0.down_proj.weight	[H, I]
  model.layers.0.mlp.experts.0.gate_proj.weight	[I, H]
  model.layers.0.mlp.experts.0.up_proj.weight [I, H]
  ```
  - transformers v5 modeling format: [Link](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226-L227)
    ```
        # Merged stacked gate_up + merged down
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    ```
  - transformers 4.57.3 format [Link](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226-L227)
    ```
        # Module list
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    ```
  - Conclusion: safetensor format needs to be merged (and transposed) to match transformers v5 modeling format


- qwen3_vl_moe
  - Sample HF checkpoint link: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
  - expert safetensor format:
  ```
  # Merged MoE weight
  # gate_up: [num_experts, H, 2 * I]
  # down: [num_experts, I, H]
  model.language_model.layers.0.mlp.experts.gate_up_proj	[128, 2048, 1536]
  model.language_model.layers.0.mlp.experts.down_proj	[128, 768, 2048]
  ```
  - transformers v5 modeling format: [Link](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py#L88-L89)
    ```
        # Merged stacked gate_up + merged down
        # !!!NOTE!!!: The dimension (1,2) of gate_up_proj was transposed between 4.57 and v5 which causes
        # a mismatch to the safetensor format.
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
    ```
  - transformers 4.57.3 modeling format [Link](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py#L74-L75)
    ```
        # Merged stacked gate_up + merged down
        # !!!NOTE!!!: The dimension (1,2) of gate_up_proj was transposed between 4.57 and v5.
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
    ```
  - Conclusion: safetensor format needs to be transposed to match transformers v5 modeling format


- qwen3_5_moe
  - Sample HF checkpoint link: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
  - expert safetensor format:
  ```
  # Merged MoE weight
  # gate_up: [num_experts, 2 * I, H]
  # down: [num_experts, H, I]
  model.language_model.layers.0.mlp.experts.gate_up_proj	[512, 2048, 4096]
  model.language_model.layers.0.mlp.experts.down_proj	[512, 4096, 1024]
  ```
  - transformers v5 modeling format: [Link](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L819-L820)

  ```
  self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
  self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
  ```
  - no transformers 4.57 modeling (the model was added after v5)
  - Conclusion: No special handling needed. transformers v5 modeling tensor dimension matches safetensor format
