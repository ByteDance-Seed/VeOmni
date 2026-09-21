# Qwen3.5-MoE Toy Config

Based on [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json).

Changes from the original config to make it test-friendly:

- **num_hidden_layers**: 40 → 4 (kept as multiple of 4 to preserve the 3 linear_attention + 1 full_attention repeating pattern)
- **num_experts**: 256 → 16 (reduce memory and compute)
- **num_experts_per_tok**: 8 → 2 (match reduced expert count)
- **output_router_logits**: added as `false` (explicit default)

All other parameters (hidden_size, head_dim, moe_intermediate_size, shared_expert_intermediate_size, rope_parameters, vision_config, etc.) are kept identical to the original.
