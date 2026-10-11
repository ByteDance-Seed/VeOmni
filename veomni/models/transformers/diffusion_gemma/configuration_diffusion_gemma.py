# Copyright 2026 the HuggingFace Team. All rights reserved.
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

"""DiffusionGemma configuration, vendored from transformers and extended in VeOmni.

Transformers 5.16.1 already ships ``diffusion_gemma``; this package overrides the
HuggingFace class so the canvas-training fields round-trip with the checkpoint.
"""

from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaTextConfig(PreTrainedConfig):
    r"""
    use_bidirectional_attention (`str`, *optional*):
        Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
        attend bidirectionally while text tokens use causal attention. When set to `"all"`,
        all tokens use bidirectional attention.
    num_global_key_value_heads (`int`, *optional*):
        Number of key-value heads for global (full) attention layers. If `None`, defaults
        to `num_key_value_heads`.
    global_head_dim (`int`, defaults to 512):
        Dimension of each attention head in global (full) attention layers.
    attn_ffpa (`bool`, defaults to `False`):
        VeOmni extension. Run the global (full) attention layers on FFPA instead of sdpa.
        Those layers are `global_head_dim` wide, and no fused sdpa kernel accepts a head
        that wide, so sdpa falls back to unfused math there. Off by default because it
        needs the optional `ffpa-attn` package, which is installed out of band and is not
        in the lockfile; with the package missing the flag raises rather than downgrading
        to sdpa, so a run never reports one kernel's cost under the other's name. Being a
        config field it also round-trips into a saved checkpoint, which means a checkpoint
        saved with it set will refuse to run where the package is absent. It costs
        `torch.compile(fullgraph=True)` too, since FFPA disables Dynamo around its
        autograd entry. See `ffpa_attention.py`.
    attn_dg_prefix (`bool`, defaults to `False`):
        VeOmni extension. Decoder attention packs the encoder cache into
        ``H_prefix || current_canvas`` and runs the DG_PREFIX Triton kernel.
        Full layers gather every visible encoder key; sliding layers gather the
        last ``W-1`` visible keys. Padding, the clean current canvas, and future
        canvases are dropped physically rather than masked. CUDA only; sequence
        parallel is rejected. See `dg_prefix.py`.
    top_k_experts (`int`, *optional*):
        Number of experts activated per token in MoE layers.
    moe_intermediate_size (`int`, *optional*):
        Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
    """

    model_type = "diffusion_gemma_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.experts.gate_up_proj": "packed_colwise",
        "layers.*.experts.down_proj": "rowwise",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        # EP plan for google/gemma-4-26B-A4B-it: do not tp in attention (num_global_key_value_heads=2 too small to partition)
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.router": "ep_router",
        "layers.*.experts.gate_up_proj": "grouped_gemm",
        "layers.*.experts.down_proj": "grouped_gemm",
        "layers.*.experts": "moe_tp_experts",
    }

    base_model_pp_plan = {
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping = 30.0
    use_bidirectional_attention: Literal["all", "vision"] | None = None
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    attn_ffpa: bool = False
    attn_dg_prefix: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention == "all":
            self.is_causal = False
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        if self.layer_types is None:
            sliding_window_pattern = 6  # by default 5:1
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
            logger.warning(
                f"Last layer must use `full_attention`, but got `{last_layer_type}`. Forcing last layer to `full_attention`."
            )
            self.layer_types[-1] = "full_attention"

        default_rope_params: dict[Literal["full_attention", "sliding_attention"] : dict[str, Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaConfig(PreTrainedConfig):
    r"""
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    canvas_length (`int`, *optional*, defaults to 256):
        The size of the canvas or, in other words, the block length in block diffusion. Used to initialize an empty
        canvas.
    max_canvases (`int`, *optional*, defaults to 8):
        VeOmni extension. How many canvases of a training answer are reachable. The
        encoder always gets the clean blocks that fit; the transform draws one of
        them to denoise unless `compute_all_canvas_df_loss` is on. This caps both
        the tokenizer budget and how far into an answer training can see. Raising
        it lengthens the encoder context and therefore the cost per sample; answer
        tokens beyond `max_canvases * canvas_length` are dropped.
    canvas_start (`int`, *optional*, defaults to 0):
        VeOmni extension. 0-based offset into the answer at which the canvas grid begins.
        0 is the first token after the prompt (the default). `n` treats the first `n`
        answer tokens as already-committed encoder context and never denoises them; the
        drawn canvas is then `answer[n + kC : n + (k+1)C]`.
    self_conditioning_prob (`float`, *optional*, defaults to 0.5):
        VeOmni extension. Probability of enabling self-conditioning for a training
        sample. Packed batches that concatenate canvases along the sequence still draw
        one Bernoulli per canvas, not one bit for the padded row. The sampler feeds
        every denoising step the previous step's prediction, so a model finetuned with
        this at `0.0` would meet a signal at inference that it never saw in training.
        Costs one extra forward per step; `0.0` turns the second pass off entirely.
        Inference ignores this field -- the sampler always self-conditions.
    training_noise_eps (`float`, *optional*, defaults to 0.001):
        VeOmni extension. Lower bound of the per-sample corruption ratio `t ~ U(eps, 1)`
        the data transform draws: each canvas position is replaced by a uniformly random
        token with probability `t`. Keeps every training canvas at least slightly noisy.
    freeze_router (`bool`, *optional*, defaults to `False`):
        VeOmni extension. Freeze the MoE gate projection and shared router scale,
        matching NeMo's DiffusionGemma SFT recipe. The per-expert output scale stays
        trainable because NeMo folds it into the trainable expert down projection.
    encoder_ar_loss (`bool`, *optional*, defaults to `True`):
        VeOmni extension. Add the causal next-token loss over the encoder's own input to
        the canvas denoising loss, which is the objective of the paper's Eq. 13. The two
        are summed with no tunable weight between them; what balances them is the
        normalization, the encoder term dividing by its whole sequence length against the
        decoder term's one canvas. Without it the encoder is trained only through the
        gradient reaching it via the KV cache, and the autoregressive sampling the
        released weights keep would be free to drift. Turning it off buys memory back:
        the term projects the whole encoder sequence through the 262k-wide head, and
        inference never uses it.
    mask_prompt_ar_loss (`bool`, *optional*, defaults to `False`):
        VeOmni extension. When `encoder_ar_loss` is on, drop the prompt from that term
        and supervise only the clean canvases sitting in the encoder. `False` is Eq. 13
        (next-token over `P + K C`). `True` leaves the prompt in the KV cache but does
        not train the head to emit it; the first answer token is still predicted from
        the last prompt hidden state.
    mask_other_tail_eos (`bool`, *optional*, defaults to `True`):
        VeOmni extension. When `encoder_ar_loss` is on, keep only the first EOS of the
        trailing canvas fill as an AR target and drop the rest of that run. The stop
        signal is that first EOS; the suffix is EOS-to-EOS. Diffusion still supervises
        every EOS on the drawn last canvas. `False` is Eq. 13 over the whole suffix.
    compute_all_canvas_df_loss (`bool`, *optional*, defaults to `False`):
        VeOmni extension. `False` is Eq. 13: the transform draws one canvas and the
        diffusion term is that canvas's mean CE. `True` denoises every canvas that
        fits in the encoder and averages those means, so the sample's diffusion term
        is `(1/K) Σ_k mean_CE(canvas_k)`. The encoder AR term is still computed once
        over `P + K C`; the example still counts as one sample. The collator keeps
        the encoder as one row and stacks canvases as `(B, K, C)`. The model runs
        the encoder once and `canvas_batching` decides how the decoder consumes
        those K canvases; either way encoder activations stay at batch 1.
    canvas_batching (`str`, *optional*, defaults to `"serial"`):
        VeOmni extension. How the decoder consumes a stacked `(B, K, C)` canvas
        batch. `"serial"` runs one decoder pass per canvas over the shared encoder
        cache, which is what `compute_all_canvas_df_loss` shipped with.
        `"flat_batch"` folds the canvas axis into the batch and runs a single pass
        over `B * K` rows. `"flat_seq"` packs the canvases onto the sequence
        instead, one pass over `B` rows of `K * C` tokens.

        All three are the same computation, and the equivalence is pinned in
        `tests/models/test_diffusion_gemma_canvas_batching.py`. What differs is
        how often the work is launched and what the attention has to materialize.
        FSDP all-gathers and expert-parallel all-to-alls fire once per decoder
        pass, so `"serial"` pays them K times per micro-batch (and again for the
        self-conditioning pass), and the MoE routes one canvas of tokens at a time
        instead of K.

        The two packed modes trade differently against memory. `"flat_batch"`
        keeps a canvas in its own row, so visibility stays a key-side mask and any
        attention backend serves it, but the transient inside a layer grows with
        K and DG_PREFIX still materializes each canvas's visible prefix -- the KV
        it computes on is `K*P + K^2*C/2`. `"flat_seq"` makes visibility depend on
        which canvas a query sits in, so it needs either the hole kernel
        (`text_config.attn_dg_prefix`, which reads the cache in place at `P + K*C`)
        or a dense `K*C` by `P + K*C` mask on sdpa, which pays for the canvases it
        masks out. It also requires each canvas's visible cache to be one
        contiguous run, and refuses the batch rather than guessing if it is not.

        Ignored when the batch carries a single canvas, where all three are the
        same code path.

        `"flat_seq_pack"` is `"flat_seq"` with several *examples* in the one row
        as well: the collator concatenates their encoder tokens, marks the
        boundaries by restarting `position_ids`, and stacks every example's
        canvases on the canvas axis. The encoder then runs varlen attention over
        the segments rather than a padded rectangle, so a batch of ragged prompts
        costs the tokens it holds instead of the longest prompt times the batch
        size. Unlike the other three this is *not* the same computation as
        `"serial"` on the same collated batch -- it is the same computation as
        running those examples one at a time, which is what
        `tests/models/test_diffusion_gemma_packed_canvases.py` pins. Text only:
        the varlen kernels carry no image block boundaries, so the collator
        refuses image features rather than attending an image span as text.
    print_ar_df_norm (`bool`, *optional*, defaults to `False`):
        VeOmni extension. When true, each training micro-batch runs two backwards --
        encoder AR with ``retain_graph``, then diffusion -- and logs that term's
        param-grad L2 before clipping, under ``encoder_ar_grad_norm`` /
        ``diffusion_grad_norm``. The two grad buffers are added so the optimizer still
        steps on ``grad(AR + diffusion)``. Default false is a single backward of the
        sum, with no extra all-gather or sharded-grad clone.

        A diagnostic, not something to leave on. It refuses a step with more than one
        micro-batch rather than print a number it cannot stand behind: under FSDP2 an
        accumulated gradient behind an unsharded parameter is reachable by neither the
        stash nor ``zero_grad``, so the AR term would add onto it and report two
        micro-batches as one. It is also expensive -- ``retain_graph`` holds the first
        backward's saved tensors through the second and the AR gradients are cloned
        whole, which on the 26B recipe (80GB, ``ep_size: 8``, ``max_seq_len`` 1024) runs
        out of memory after the first step, with or without recompute.

    Example:

    ```python
    >>> from transformers import (
    >>>     DiffusionGemmaConfig,
    >>>     DiffusionGemmaModel,
    >>>     DiffusionGemmaTextConfig,
    >>>     Gemma4VisionConfig,
    >>> )

    >>> # Initializing a DiffusionGemma Text config.
    >>> text_config = DiffusionGemmaTextConfig()

    >>> # Initializing a Gemma 4 vision config (DiffusionGemma uses Gemma 4's vision block).
    >>> vision_config = Gemma4VisionConfig()

    >>> # Initializing a DiffusionGemma text config
    >>> configuration = DiffusionGemmaConfig(text_config, vision_config)

    >>> # Initializing a model from the configuration
    >>> model = DiffusionGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "diffusion_gemma"
    sub_configs = {
        "text_config": DiffusionGemmaTextConfig,
        "vision_config": AutoConfig,
    }

    text_config: DiffusionGemmaTextConfig | dict[str, Any] | None = None
    vision_config: PreTrainedConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    initializer_range: float | None = 0.02
    # Important: this model also ties the text encoder with the decoder. Setting this to `False` undoes all ties.
    tie_word_embeddings: bool = True
    canvas_length: int | None = 256
    max_canvases: int = 8
    canvas_start: int = 0
    self_conditioning_prob: float = 0.5
    training_noise_eps: float = 1e-3
    freeze_router: bool = False
    encoder_ar_loss: bool = True
    mask_prompt_ar_loss: bool = False
    mask_other_tail_eos: bool = True
    compute_all_canvas_df_loss: bool = False
    canvas_batching: Literal["serial", "flat_batch", "flat_seq", "flat_seq_pack"] = "serial"
    print_ar_df_norm: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = DiffusionGemmaTextConfig()
            logger.info("text_config is None. Using default DiffusionGemmaTextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = DiffusionGemmaTextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. DiffusionGemmaEncoderModel.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "gemma4_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["DiffusionGemmaTextConfig", "DiffusionGemmaConfig"]
