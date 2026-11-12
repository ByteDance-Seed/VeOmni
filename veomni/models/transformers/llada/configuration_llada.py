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
"""Configuration for LLaDA (Large Language Diffusion with mAsking).

LLaDA's upstream checkpoint (``GSAI-ML/LLaDA-8B-Base``) ships an OLMo-style
``LLaDAModelLM`` whose Transformer block is mathematically identical to Llama
(RMSNorm pre-norm, separate Q/K/V projections, MHA + RoPE with ``rotate_half``,
SwiGLU MLP, no biases, no weight tying). The only architectural difference that
matters for the forward pass is that LLaDA removes the causal mask — it is a
*mask predictor* (bidirectional / diffusion LM), not an autoregressive LM.

This is a **standalone** config (the fields mirror Llama, but it does not subclass
``LlamaConfig``): the checkpoint converter emits a ``config.json`` with Llama-named
fields + ``model_type = "llada"`` so the loader routes to the VeOmni LLaDA modeling.
``mask_token_id`` is the extra field the dLLM training recipe needs (the reserved
``[MASK]`` token used by the forward masking process).
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils.type_validators import interval

from veomni.models.configuration_base import VeOmniPretrainedConfig


@strict
class LLaDAConfig(VeOmniPretrainedConfig, PreTrainedConfig):
    r"""Llama-compatible config for LLaDA mask-predictor models.

    Mirrors every Llama field (so a Llama-layout checkpoint loads unchanged) and adds
    ``mask_token_id``. ``model_type = "llada"`` routes to the VeOmni LLaDA modeling.
    """

    model_type = "llada"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LLaDAModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 126464
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    # Reserved [MASK] token id used by the dLLM forward (masking) process.
    # Default matches LLaDA-8B (126336); kept on the config so the collator / loss op
    # can read it without a separate plumb-through.
    mask_token_id: int = 126336

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


__all__ = ["LLaDAConfig"]
