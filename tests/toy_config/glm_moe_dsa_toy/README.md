# GLM-MoE-DSA Toy Config

Based on the `GlmMoeDsaConfig` defaults (GLM-5 family, MLA + Dynamic Sparse Attention).
See `transformers.models.glm_moe_dsa.configuration_glm_moe_dsa` for the upstream reference;
the doc string points at [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5).

Changes from the upstream defaults to make it test-friendly:

- **vocab_size**: 154880 → 32256 (the test fixture caps random token ids at 32000).
- **hidden_size**: 6144 → 256.
- **intermediate_size**: 12288 → 512.
- **moe_intermediate_size**: 2048 → 128.
- **num_hidden_layers**: 78 → 4 (kept ≥ 4 so `mlp_layer_types` covers both
  `dense` and `sparse` — the default is 3 dense + rest sparse, so 4 layers
  exercises one of each).
- **num_attention_heads / num_key_value_heads**: 64 → 8.
- **q_lora_rank**: 2048 → 128.
- **kv_lora_rank**: 512 → 64.
- **qk_rope_head_dim**: 64 → 32.
- **qk_nope_head_dim**: 192 → 64.
- **v_head_dim**: 256 → 64.
- **n_routed_experts**: 256 → 8.
- **num_experts_per_tok**: 8 → 2.
- **index_topk**: 2048 → 16 (must be ≤ test seq_len of 32 to be meaningful).
- **index_head_dim**: 128 → 64.
- **index_n_heads**: 32 → 4.
- **max_position_embeddings**: 202752 → 4096.
- **rope_parameters**: explicit `{rope_type: "default", rope_theta: 10000.0}`.

`n_group` / `topk_group` are kept at 1 / 1 (the default), so the group-mask
branch in the router is a no-op for the toy. `routed_scaling_factor`,
`norm_topk_prob`, `attention_bias` and `attention_dropout` follow the
upstream defaults.
