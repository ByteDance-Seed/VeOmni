# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for LLaDA-MoE (Large Language Diffusion MoE, mask predictor).

``inclusionAI/LLaDA-MoE-7B-A1B-Instruct`` ships a custom ``LLaDAMoEModelLM`` whose
Transformer block is **architecturally identical to Qwen3-MoE** (qk-layernorm
attention + RoPE, ``mlp.gate`` softmax router over per-expert SwiGLU experts,
``norm_topk_prob=False``, every layer MoE, no shared experts). The only forward-pass
difference is that LLaDA-MoE is a **mask predictor** (bidirectional / diffusion LM),
not autoregressive.

This is a **standalone** config (the fields mirror Qwen3-MoE, but it does not subclass
``Qwen3MoeConfig``): the checkpoint converter emits a ``config.json`` with Qwen3-MoE-named
fields + ``model_type = "llada_moe"`` so the loader routes to the VeOmni LLaDA-MoE
(bidirectional) modeling. ``mask_token_id`` is the extra field the dLLM training recipe
needs (the reserved ``[MASK]`` token used by the forward masking process).
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class LLaDAMoEConfig(PreTrainedConfig):
    r"""Qwen3-MoE-compatible config for LLaDA-MoE mask-predictor models.

    Mirrors every Qwen3-MoE field (so a Qwen3-MoE-layout checkpoint loads unchanged) and
    adds ``mask_token_id``. ``model_type = "llada_moe"`` routes to the VeOmni LLaDA-MoE modeling.

    decoder_sparse_step (`int`, *optional*, defaults to 1):
        The frequency of the MoE layer.
    mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
        Indicate which layers use the dense MLP rather than the sparse MoE block.
    """

    model_type = "llada_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_experts": "num_local_experts",
    }

    # Default tensor parallel plan for base model `LLaDAMoE`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    # Expert-only EP plan: only shards MoE experts, not attention.
    # Attention is left unsharded — FSDP2 handles attention weight distribution.
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    # Reserved [MASK] token id used by the dLLM forward (masking) process.
    # Default matches LLaDA-MoE (``<|mdm_mask|>`` = 156895); kept on the config so the
    # collator / loss op can read it without a separate plumb-through.
    mask_token_id: int = 156895

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        super().__post_init__(**kwargs)


__all__ = ["LLaDAMoEConfig"]
