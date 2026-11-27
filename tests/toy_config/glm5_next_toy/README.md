# GLM-5.3-Flash (`glm5_next`) Toy Config

Based on `Glm5NextConfig` / `Glm5NextTextConfig` defaults (hybrid KDA + DSA, mHC, MoE, vision tower).
Upstream reference: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

Changes from the upstream defaults for test friendliness:

- Smaller vocab / hidden / layers / experts / vision depth.
- `qk_rope_head_dim=0` (NoPE DSA contract).
- `index_topk` divisible by `index_kpool`.
- Explicit `layer_types` / `mlp_layer_types` covering both KDA and DSA plus dense/sparse MoE.
- Architecture is `Glm5NextForConditionalGeneration` so text SFT and EP/SP e2e can call `lm_head` + loss.
