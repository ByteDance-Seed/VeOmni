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

from typing import Any, Dict, List, Optional

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
        weight_quantization: Optional[Dict[str, Any]] = None,
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
        dsa_indexer_loss: bool = False,
        dsa_indexer_loss_coef: float = 1.0,
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
        # Training-only. None trains every backbone layer; an int freezes
        # ``layers[n:]`` so an 8xH20 SFT can keep the first two. Absent from the
        # released config, same as ``dsa_indexer_loss``.
        n_trainable_layers: Optional[int] = None,
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
        # The released `quantization_config`, verbatim, or None when this
        # module should allocate bf16 Parameters and dequant at load. Two
        # things make this a separate field rather than that name:
        #
        # `expert_dtype` above is about QAT -- which format training should
        # fake-quantize the routed experts to, in bf16 arithmetic. This one is
        # about storage: whether the live module keeps the quantized GEMMs.
        # Training YAML sets this to null; the files on disk stay quantized.
        #
        # And `quantization_config` is a reserved name. transformers reads it
        # off the config in `get_hf_quantizer`, and `quant_method: "fp8"` is a
        # method it supports, so it would install its own fp8 quantizer,
        # substitute its own Linear and take over weight loading.
        self.weight_quantization = weight_quantization

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
        # Training-only, with no field in the released config. The indexer emits
        # indices, so nothing reaches its weights through the attention output;
        # this auxiliary KL is the only gradient they ever get. Off by default
        # because it costs a second kernel pass, on for any run that means to
        # train the indexer rather than inherit it.
        self.dsa_indexer_loss = dsa_indexer_loss
        self.dsa_indexer_loss_coef = dsa_indexer_loss_coef

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
        self.n_trainable_layers = n_trainable_layers
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

    @property
    def mlp_bias(self) -> bool:
        """``DeepseekV4MLP`` passes this to its three projections. Constant rather
        than a field: no released V4.1 expert carries a bias, and making it
        settable would offer a shape the checkpoint cannot fill."""
        return False

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
        if not 0 <= self.candidate_source_layer_id < self.num_hidden_layers:
            raise ValueError(
                f"candidate_source_layer_id={self.candidate_source_layer_id} is outside "
                f"[0, {self.num_hidden_layers}). Shrinking the stack is not just num_hidden_layers: "
                "the six layer-index lists have to move with it. See shrink_layer_schedule."
            )
        if self.candidate_source_layer_id > 0 and self.candidate_source_layer_id not in self.kv_source_layer_ids:
            raise ValueError(
                f"candidate_source_layer_id={self.candidate_source_layer_id} starts the CED decoder "
                "and must also be a KV source, matching released L20."
            )
        if self.head_dim <= self.qk_rope_head_dim:
            raise ValueError(f"head_dim={self.head_dim} must exceed qk_rope_head_dim={self.qk_rope_head_dim}.")
        if self.num_attention_heads % self.o_groups:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by o_groups={self.o_groups}: "
                "the output projection is block-diagonal over head groups."
            )
        if self.n_trainable_layers is not None and not 0 <= self.n_trainable_layers <= self.num_hidden_layers:
            raise ValueError(
                f"n_trainable_layers={self.n_trainable_layers} is outside [0, {self.num_hidden_layers}]. "
                "None trains the whole stack; 0 freezes it."
            )


RELEASED_NUM_HIDDEN_LAYERS = 40
RELEASED_ENCODER_LAYERS = 20
RELEASED_KV_SOURCE_LAYER_IDS = [2, 8, 14, 20]
RELEASED_INDEX_SOURCE_LAYER_IDS = [2, 8, 14, 20, 24, 28, 32, 36]
RELEASED_CANDIDATE_SOURCE_LAYER_ID = 20
RELEASED_ENGRAM_LAYER_IDS = [1, 14]
RELEASED_DSPARK_TARGET_LAYER_IDS = [37, 38, 39]


def _even_pick(values: List[int], k: int) -> List[int]:
    """``k`` unique entries spaced across ``values``, keeping the endpoints when k>1."""
    if k <= 0:
        return []
    if k >= len(values):
        return list(values)
    if k == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    picked: List[int] = []
    seen = set()
    for i in range(k):
        idx = int(round(i * last / (k - 1)))
        idx = min(max(idx, 0), last)
        while idx in seen and idx < last:
            idx += 1
        while idx in seen and idx > 0:
            idx -= 1
        seen.add(idx)
        picked.append(values[idx])
    return picked


def shrink_layer_schedule(n_layers: int, n_draft: int = 3) -> Dict[str, Any]:
    """Remap the released 40-layer CED lists onto ``n_layers`` backbone layers.

    Changing only ``num_hidden_layers`` leaves ``kv_source_layer_ids`` etc. pointing
    at layers that no longer exist. The rules are the Step 5 ones: encoder and
    decoder each keep a half; each half keeps a KV source and the decoder's first
    layer is also the candidate source; Engram keeps one layer per half; DSpark
    targets the new last three layers.
    """
    if n_layers < 2:
        raise ValueError(f"shrinking the CED backbone needs at least 2 layers, got {n_layers}")
    if n_draft < 0:
        raise ValueError(f"n_draft must be >= 0, got {n_draft}")
    if n_layers == RELEASED_NUM_HIDDEN_LAYERS:
        return {
            "num_hidden_layers": RELEASED_NUM_HIDDEN_LAYERS,
            "num_nextn_predict_layers": n_draft,
            "compress_ratios": DeepseekV41LlmConfig._default_ratios()[: RELEASED_NUM_HIDDEN_LAYERS + n_draft],
            "kv_source_layer_ids": list(RELEASED_KV_SOURCE_LAYER_IDS),
            "index_source_layer_ids": list(RELEASED_INDEX_SOURCE_LAYER_IDS),
            "candidate_source_layer_id": RELEASED_CANDIDATE_SOURCE_LAYER_ID,
            "engram_layer_ids": list(RELEASED_ENGRAM_LAYER_IDS),
            "dspark_target_layer_ids": list(RELEASED_DSPARK_TARGET_LAYER_IDS),
        }

    n_enc = n_layers // 2
    n_dec = n_layers - n_enc
    released_ratios = DeepseekV41LlmConfig._default_ratios()
    encoder_kv = [idx for idx in RELEASED_KV_SOURCE_LAYER_IDS if idx < RELEASED_ENCODER_LAYERS]
    encoder_pool = list(range(RELEASED_ENCODER_LAYERS))
    if n_enc == 1:
        enc_keep = [encoder_kv[0]]
    else:
        enc_keep = _even_pick(encoder_pool, n_enc)
        if not any(idx in encoder_kv for idx in enc_keep):
            enc_keep[-1] = encoder_kv[-1]
            enc_keep = sorted(set(enc_keep))
            drop_from = [idx for idx in enc_keep if idx not in encoder_kv]
            while len(enc_keep) > n_enc and drop_from:
                enc_keep.remove(drop_from.pop(0))

    decoder_rest = [
        idx
        for idx in range(RELEASED_ENCODER_LAYERS, RELEASED_NUM_HIDDEN_LAYERS)
        if idx != RELEASED_CANDIDATE_SOURCE_LAYER_ID
    ]
    dec_keep = [RELEASED_CANDIDATE_SOURCE_LAYER_ID] + _even_pick(decoder_rest, n_dec - 1)
    kept = enc_keep + dec_keep
    remap = {orig: new for new, orig in enumerate(kept)}

    kv_ids = sorted({remap[idx] for idx in RELEASED_KV_SOURCE_LAYER_IDS if idx in remap})
    if n_enc not in kv_ids:
        kv_ids = sorted(kv_ids + [n_enc])
    index_ids = sorted({remap[idx] for idx in RELEASED_INDEX_SOURCE_LAYER_IDS if idx in remap} | set(kv_ids))

    enc_engram = next((remap[idx] for idx in RELEASED_ENGRAM_LAYER_IDS if idx in remap and remap[idx] < n_enc), 0)
    dec_engram = n_layers - 1
    engram_ids = sorted({enc_engram, dec_engram})

    n_dspark = min(3, n_layers)
    dspark_ids = list(range(n_layers - n_dspark, n_layers))
    return {
        "num_hidden_layers": n_layers,
        "num_nextn_predict_layers": n_draft,
        "compress_ratios": [released_ratios[idx] for idx in kept] + [0] * n_draft,
        "kv_source_layer_ids": kv_ids,
        "index_source_layer_ids": index_ids,
        "candidate_source_layer_id": n_enc,
        "engram_layer_ids": engram_ids,
        "dspark_target_layer_ids": dspark_ids,
    }


__all__ = ["DeepseekV41LlmConfig", "shrink_layer_schedule"]
