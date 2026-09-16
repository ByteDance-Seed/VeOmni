# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Config for :class:`DeepseekV41Llm` -- the 40-layer CED backbone.

Written out rather than subclassed from
:class:`~veomni.models.transformers.deepseek_v4.configuration_deepseek_v4.DeepseekV4Config`.
The reason is the attention schedule, and it is worth stating because the G1 design
doc assumed the opposite.

V4 describes its schedule as a list of *layer type names* -- ``"sliding_attention"``
/ ``"compressed_sparse_attention"`` / ``"heavily_compressed_attention"`` -- with the
actual compression rate looked up per type from a two-entry ``compress_rates``
dict. Its ``__post_init__`` accepts a legacy ``compress_ratios`` int list only
through ``_COMPRESS_RATIO_TO_LAYER_TYPE = {0: sliding, 4: CSA, 128: HCA}``.

V4.1's ``compress_ratios`` are ``0``, ``2`` and ``1``. None of the three is in that
map, so feeding V4.1's config to ``DeepseekV4Config`` raises ``KeyError: 2`` before
anything is built. The mismatch is not cosmetic: ratio 1 is a *plain projection*
with no pooling gate at all (see ``Compressor.forward`` in the released
``inference/model.py``), which is a third kind of layer that V4's two-name
taxonomy has no slot for. Squeezing it in by renaming would leave a layer called
"heavily compressed" that compresses nothing.

So the schedule is kept in V4.1's own vocabulary -- per-layer integer ratios --
and the classes that read it are local. The pieces that do *not* depend on the
schedule (mHC math, the MoE experts and their fp4 QAT hooks, the RMS norms) are
still imported from ``deepseek_v4``; see ``modeling.py``.

Two MoE fields are read differently by the two implementations, so the V4 router
is subclassed rather than imported outright. V4 renormalises the top-k weights
unconditionally; V4.1 does it only when ``norm_topk_prob and num_experts_per_tok
> 1``. And V4 has no ``gate_temp`` at all, which is the same as a hardcoded 1.0.
"""

from typing import List, Optional

from transformers import PretrainedConfig


class DeepseekV41LlmConfig(PretrainedConfig):
    """The backbone's shape, taken verbatim from the checkpoint's ``text_config``.

    ``compress_ratios`` has one entry per layer **including the DSpark draft
    stages**, so it is ``num_hidden_layers + num_nextn_predict_layers`` long. That
    is the released layout and the reason the field is not derived from
    ``num_hidden_layers`` here: truncating it would silently give the draft stages
    a compressed-KV branch they do not have.
    """

    model_type = "deepseek_v41_llm"

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        # MoE
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        gate_temp: float = 1.0,
        expert_dtype: str = "fp4",
        # Attention schedule
        sliding_window: int = 128,
        compress_ratios: Optional[List[int]] = None,
        compress_rope_theta: float = 160000.0,
        kv_source_layer_ids: Optional[List[int]] = None,
        index_source_layer_ids: Optional[List[int]] = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        # mHC
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Engram connection (the tables themselves live in ``deepseek_v41_ngram``)
        engram_layer_ids: Optional[List[int]] = None,
        engram_max_ngram_size: int = 4,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        # DSpark
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: Optional[List[int]] = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        # Multimodal
        image_token_id: int = 129264,
        enable_vision_routing_bias: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        }

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        # Absent from the released config.json, so it always resolves to the reference
        # ModelArgs default. Declared anyway: a checkpoint that ships a non-unit value
        # would otherwise be dropped on load and change routing silently.
        self.gate_temp = gate_temp
        self.expert_dtype = expert_dtype

        self.sliding_window = sliding_window
        self.compress_ratios = list(compress_ratios) if compress_ratios is not None else self._default_ratios()
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = list(kv_source_layer_ids) if kv_source_layer_ids is not None else [2, 8, 14, 20]
        self.index_source_layer_ids = (
            list(index_source_layer_ids) if index_source_layer_ids is not None else [2, 8, 14, 20, 24, 28, 32, 36]
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.engram_layer_ids = list(engram_layer_ids) if engram_layer_ids is not None else [1, 14]
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            list(dspark_target_layer_ids) if dspark_target_layer_ids is not None else [37, 38, 39]
        )
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok

        self.image_token_id = image_token_id
        self.enable_vision_routing_bias = enable_vision_routing_bias
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.validate_layer_schedule()

    @staticmethod
    def _default_ratios() -> List[int]:
        """The released 43-entry schedule: 2 sliding, 18 at ratio 2, 20 at ratio 1, 3 draft."""
        return [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]

    # ── Derived schedule ──────────────────────────────────────────────────────

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def num_total_layers(self) -> int:
        """Backbone plus draft stages -- the length ``compress_ratios`` must have."""
        return self.num_hidden_layers + self.num_nextn_predict_layers

    @property
    def intermediate_size(self) -> int:
        """Shared-expert width. Named as ``DeepseekV4MLP`` reads it, since that is
        the class this module imports for the shared expert."""
        return self.moe_intermediate_size

    @property
    def num_local_experts(self) -> int:
        """Alias ``DeepseekV4Experts`` reads for the routed-expert count."""
        return self.n_routed_experts

    def moe_config_for_layer(self, layer_idx: int) -> tuple:
        """``(n_routed, n_activated)`` -- the draft stages have their own, smaller MoE."""
        if layer_idx < self.num_hidden_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_num_experts_per_tok or self.num_experts_per_tok,
        )

    def validate_layer_schedule(self) -> None:
        """Refuse a schedule whose pieces disagree with each other.

        Each of these is a silent failure rather than a crash if left unchecked: a
        short ``compress_ratios`` makes the draft stages index off the end (or, worse,
        read a backbone layer's ratio), and a KV source that does not compress
        produces an indexer with nothing to index while still training its weights.
        """
        if len(self.compress_ratios) != self.num_total_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries but the model has "
                f"{self.num_hidden_layers} backbone + {self.num_nextn_predict_layers} draft layers "
                f"= {self.num_total_layers}. The released schedule lists every layer, draft stages included."
            )
        for layer_idx in self.kv_source_layer_ids:
            if not 0 <= layer_idx < self.num_hidden_layers:
                raise ValueError(f"kv_source_layer_ids contains out-of-range layer {layer_idx}.")
            if self.compress_ratios[layer_idx] == 0:
                raise ValueError(
                    f"layer {layer_idx} is a KV source but its compress_ratio is 0 (sliding window only), "
                    "so it has no compressed KV to publish to the layers that read it."
                )
        for layer_idx in self.index_source_layer_ids:
            if not 0 <= layer_idx < self.num_hidden_layers:
                raise ValueError(f"index_source_layer_ids contains out-of-range layer {layer_idx}.")
        for layer_idx in self.engram_layer_ids:
            if not 0 <= layer_idx < self.num_hidden_layers:
                raise ValueError(f"engram_layer_ids contains out-of-range layer {layer_idx}.")
        for layer_idx in self.dspark_target_layer_ids:
            if not 0 <= layer_idx < self.num_hidden_layers:
                raise ValueError(
                    f"dspark_target_layer_ids contains layer {layer_idx}, which is not a backbone layer: "
                    "the draft stages read the backbone's residual stream, not their own."
                )
        if self.head_dim <= self.qk_rope_head_dim:
            raise ValueError(f"head_dim={self.head_dim} must exceed qk_rope_head_dim={self.qk_rope_head_dim}.")
        if self.num_attention_heads % self.o_groups:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by o_groups={self.o_groups}: "
                "the output projection is block-diagonal over head groups."
            )


__all__ = ["DeepseekV41LlmConfig"]
