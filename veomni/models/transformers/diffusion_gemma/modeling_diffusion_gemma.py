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

"""DiffusionGemma modeling, vendored from transformers and extended in VeOmni.

Transformers 5.16.1 already ships ``diffusion_gemma``; this package overrides it
with the canvas-training, packing and kernel paths listed below. Do not convert
the file to patchgen until those extensions live upstream.

DiffusionGemma is an encoder-decoder discrete diffusion LM. The encoder is an
autoregressive prefill over the prompt that only exports its KV cache; the decoder
denoises a fixed-length canvas with bidirectional attention over ``[cache | canvas]``.
Encoder and decoder share every transformer weight (see ``_tied_weights_keys`` on
``DiffusionGemmaModel``) -- the encoder's only private tensors are the per-layer
``layer_scalar`` buffers, and the decoder additionally owns ``self_conditioning``.

VeOmni deviations from upstream:

* ``vision_tower`` / ``embed_vision`` are only built when ``config.vision_config`` is
  set, so text-only training can drop the vision tower by nulling it out. Images are an
  encoder-only input: ``DiffusionGemmaModel.forward`` and
  ``DiffusionGemmaForBlockDiffusion.forward`` name ``pixel_values`` /
  ``mm_token_type_ids`` / ``image_position_ids`` explicitly and route them to the
  encoder, because upstream's ``**kwargs`` also reaches the decoder, which has no
  vision path. ``generation_diffusion_gemma`` is not vendored (no ``generate``).
* Training support, absent upstream: ``supports_gradient_checkpointing`` is enabled,
  ``forward`` accepts ``labels``/``loss_weight`` and computes the masked diffusion
  loss, and the trainer-supplied ``use_cache`` kwarg is swallowed.
* Both transformer stacks are recomputable, where upstream marks the encoder as not
  checkpointable at all. ``SealableCache`` neutralizes the encoder's cache write on
  replay and ``_UnshardForRecompute`` gets the shared expert weights back in time for
  it; the encoder holds the whole context while the decoder holds one canvas, so
  leaving it un-recomputed is what dominates activation memory.
* ``DiffusionGemmaEncoderModel.forward`` now assigns the mask mapping it builds
  (upstream drops the return value, which silently discards the padding mask).
* Decoder attention can pack the encoder cache by layer type into
  ``H_prefix || current_canvas`` and call the DG_PREFIX Triton kernel
  (``config.attn_dg_prefix``). See ``dg_prefix.py``.
"""

import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Optional

import torch
import torch._dynamo.config as dynamo_config
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.func import grad_and_value, vmap
from transformers import initialization as init
from transformers import masking_utils
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import (
    bidirectional_mask_function,
    create_causal_mask,
    create_masks_for_generate,
    create_sliding_window_causal_mask,
    flex_attention_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.auto import AutoModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from transformers.utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from ....distributed.parallel_state import get_parallel_state
from ....ops import VeomniOp
from ....ops.kernels.moe_experts.standard.activation_gate import refuse_fused_moe_on_unsupported_activation
from ....utils import logging
from ....utils.constants import IGNORE_INDEX
from ...loss_utils.dllm import dllm_masked_cross_entropy
from ...modeling_base import VeOmniPreTrainedModel as PreTrainedModel
from . import align_nemo
from .configuration_diffusion_gemma import DiffusionGemmaConfig, DiffusionGemmaTextConfig
from .dg_hole import (
    canvas_visible_intervals,
    dense_hole_mask,
    hole_decoder_attention,
    packed_canvas_positions,
    should_use_hole_kernel,
    sliding_visible_intervals,
)
from .dg_prefix import (
    dg_prefix_decoder_attention,
    encoder_visible_from_decoder_mask,
    should_use_dg_prefix_decoder,
)
from .dg_sequence_parallel import (
    full_cache_length,
    gather_attention_heads,
    refuse_decoder_path_under_sequence_parallelism,
    sequence_parallel_preflight,
    shard_packed_row,
    sharded_ar_slice,
    sp_pad_tokens,
    sum_over_shards,
    ulysses_decoder_split,
)
from .dg_varlen import (
    block_diagonal_causal_mask,
    encoder_segments,
    should_use_varlen_encoder,
    varlen_encoder_attention,
)
from .ffpa_attention import ffpa_preflight, resolve_attention_interface


logger = logging.get_logger(__name__)


class DiffusionGemmaTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: DiffusionGemmaTextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: DiffusionGemmaTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DiffusionGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to addess compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class DiffusionGemmaClippableLinear(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)

        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


_FLEX_ATTENTION_IMPLS = ("flex_attention", "veomni_flex_attention")
_SDPA_ATTENTION_IMPLS = ("sdpa", "veomni_sdpa")


def flex_tile_cap(head_dim: int, shared_memory_limit: int) -> dict[str, int | bool] | None:
    """Shrink flex_attention's tiles when `head_dim` is too wide for the device's shared memory.

    flex_attention's Triton template keeps a query tile and double-buffered key/value tiles in
    shared memory, so its footprint grows with `head_dim`. Inductor's autotuner only considers
    tiles down to 64x32, which at DiffusionGemma's 512-wide global head needs ~196 KiB -- more
    than an A100's 163 KiB opt-in limit -- and the run dies with "No valid triton configs" the
    first time a global layer is compiled. 32x32 tiles fit, and even with the smaller tile the
    global layers stay well ahead of materializing their mask under sdpa.

    `FORCE_USE_FLEX_ATTENTION` comes along for the ride because Inductor otherwise swaps in a
    separate short-query kernel whenever a query is under 128 long -- which the encoder hits on
    short prompts. That kernel sizes its query tile from `q_len * queries_per_kv_head` instead
    of honoring the cap, so it needs even more shared memory than the tile we just rejected,
    and it asserts rather than falling back.

    Returns None when the autotuner has a config that fits, so it can pick freely.
    """
    smallest_autotuned = (64 + 4 * 32) * head_dim * torch.finfo(torch.bfloat16).bits // 8 + 4096
    if smallest_autotuned <= shared_memory_limit:
        return None
    return {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_M1": 32,
        "BLOCK_N1": 32,
        "BLOCK_M2": 32,
        "BLOCK_N2": 32,
        "FORCE_USE_FLEX_ATTENTION": True,
    }


@lru_cache(maxsize=None)
def flex_kernel_options_for(head_dim: int, device: torch.device) -> dict[str, int | bool] | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return flex_tile_cap(head_dim, torch.cuda.get_device_properties(index).shared_memory_per_block_optin)


def one_layout_for_flex(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Hand flex a single memory layout, because its compiled backward cannot hold two.

    q/k/v reach attention as ``view(batch, seq, heads, dim).transpose(1, 2)``, a stride-permuted
    view -- except on the layers that return them through ``past_key_values.update``, which
    concatenates and so answers contiguous. Both layouts arrive at the same ``flex_attention``
    frame. Dynamo does notice, and recompiles on a ``key`` stride guard, but the backward saved
    under one layout is then run against the other and aborts inside an inductor kernel on
    ``assert_size_stride``, naming a ``primals_4`` that nothing connects back to this model.

    The copy is not free and not only on the encoder. ``q`` is always a view, so it is always
    copied; ``k`` and ``v`` are copied on every layer that does not return them through the cache.
    flex's backward holds the contiguous copy while autograd still holds the view's base, so both
    are alive at once -- about 33MiB each at 4096 tokens, 16 heads and ``head_dim`` 256 in bf16.
    With ``gradient_checkpointing`` on, one layer is live at a time and the 26B run's peak did not
    move; with it off, 30 layers are, and that has not been measured.
    """
    return tuple(t.contiguous() for t in tensors)


def widen_flex_recompile_budget() -> None:
    """Raise Dynamo's per-frame cache budget so flex's kernel keeps being compiled.

    ``flex_attention`` is compiled once per call signature, against a default budget of 8 cache
    entries for the frame. This model spends that before the first step is over: two head widths,
    each with its own ``kernel_options``, across the encoder's sliding and full masks and the
    decoder's, and any library that touches ``torch.get_num_threads()`` invalidates the entries
    from outside. Past the budget Dynamo stops compiling the frame and flex degrades, which
    ``refuse_uncompiled_flex`` turns into an error rather than a slower run.

    The 26B recipe was observed needing 11 entries over four steps, so 64 is roughly six times
    the only measurement there is. ``accumulated_recompile_limit`` has no measurement behind it
    at all: it counts every frame in the process rather than this one, and 512 is margin. Only
    raise both, so a caller that has deliberately set either -- a debug session hunting
    recompilation churn sets it *low* -- is not silently overruled.
    """
    previous = (dynamo_config.recompile_limit, dynamo_config.accumulated_recompile_limit)
    dynamo_config.recompile_limit = max(previous[0], 64)
    dynamo_config.accumulated_recompile_limit = max(previous[1], 512)
    current = (dynamo_config.recompile_limit, dynamo_config.accumulated_recompile_limit)
    if current != previous:
        # Process-global and never restored, so say so: a run whose Dynamo budget changed
        # without a record cannot be read for step time (constraint 29 is the same rule).
        logger.info(
            "DiffusionGemma raised Dynamo's recompile budget for flex_attention: "
            f"recompile_limit {previous[0]} -> {current[0]}, "
            f"accumulated_recompile_limit {previous[1]} -> {current[1]}."
        )


def refuse_uncompiled_flex() -> None:
    """Turn flex's "called without torch.compile()" warning into an error.

    That warning is emitted at exactly one moment: flex has given up on the fused kernel and is
    about to materialize the whole ``(batch, heads, q, kv)`` score matrix instead. Everything
    downstream of it is wrong in the way constraint 28 is about -- under gradient checkpointing
    the recompute stops matching the forward and the step dies somewhere unrelated-looking, and
    without it the run simply reports the dense implementation's time and memory as flex's. One
    ``UserWarning`` per process is not enough to catch that in a log the size of a training run.

    ``_warn_once`` records the warning as shown only *after* ``warnings.warn`` returns, so
    promoting it to an error makes every occurrence raise rather than just the first. A caller
    that wants the unfused path on purpose can still take it under ``warnings.catch_warnings``.
    """
    warnings.filterwarnings(
        "error",
        message="flex_attention called without torch.compile",
        category=UserWarning,
    )


_compiled_block_mask_builder = masking_utils.create_block_mask


def _build_block_mask_eagerly(*args, **kwargs):
    # Override rather than default: `flex_attention_mask` passes `_compile` explicitly, so a
    # `functools.partial` keyword would lose to it. Defined once at module level rather than per
    # call so the patched attribute is a stable object -- a fresh closure each time would be a
    # fresh Dynamo guard, feeding the recompilation `widen_flex_recompile_budget` is there for.
    return _compiled_block_mask_builder(*args, **{**kwargs, "_compile": False})


@contextmanager
def uncompiled_block_masks(active: bool = True):
    """Build flex's ``BlockMask`` without the compiled builder, which is wrong past one shape.

    ``transformers.masking_utils.flex_attention_mask`` asks ``create_block_mask`` for
    ``_compile=True``, and torch answers that with ``torch.compile(create_block_mask)``.
    Dynamo makes the second shape it sees a dynamic specialization, and from there the sparsity
    comes back disagreeing with the same mask built alone. The forward does not notice: it was
    exact at every shape measured. flex's dK/dV kernel walks the transposed side of that
    sparsity, and it returns exactly zero -- so the step trains ``q_proj`` and leaves ``k_proj``
    and ``v_proj`` where they were, without raising.

    One shape per process is safe, which is why this stayed hidden. DiffusionGemma has four:
    the encoder's sliding and full layers run first and stay exact, and the decoder that follows
    -- a 16-long canvas against a 31-long cache where the encoder had 40 against 40 -- is the
    half that comes back zeroed. On the toy model that put 25 of 47 parameters more than 5% away
    from their sdpa gradients, the worst by 127%; with this context manager the worst is 6e-6.

    torch deprecates the flag and points at ``torch.compile(create_block_mask)``, which is the
    same automatic-dynamic path and wrong at *more* shapes rather than fewer -- 3 of 6 against
    the flag's 2. ``dynamic=False`` is correct but recompiles per shape and runs out the
    recompile limit within a handful of lengths. The uncompiled builder is correct at every
    shape measured, and adds under a millisecond per mask, against an attention call that takes
    longer than that. See ``scripts/bench_diffusion_gemma_block_mask.py``.

    ``active`` is what keeps this off the sdpa and eager paths, which build no ``BlockMask``:
    patching a process-global attribute on every one of their forwards would be visible to
    anything else in the process for no reason at all.
    """
    if not active:
        yield
        return

    previous_builder = masking_utils.create_block_mask
    masking_utils.create_block_mask = _build_block_mask_eagerly
    try:
        yield
    finally:
        masking_utils.create_block_mask = previous_builder


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class DiffusionGemmaEncoderTextAttention(nn.Module):
    """Attention layer for the diffusion model.

    This layer is just like `Gemma4TextAttention`, with one key differences:
    1. Removes shared KV cache logic, as it is unused in DiffusionGemma.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        super().__init__()
        self.is_causal = config.use_bidirectional_attention != "all"

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        packed_segments: tuple[torch.Tensor, int] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # The code in this function is adapted from Gemma4TextAttention. ** The modified parts are clearly indicated **
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CHANGED: removed `if self.is_kv_shared_layer` branch, kept the `else`
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        # CHANGED: removed the `if self.store_full_length_kv` branch

        # CHANGED: when the row holds several samples, the boundaries travel as
        # `cu_seqlens` and the backend is picked by head width rather than by
        # `_attn_implementation`. Taking this branch is not an optimisation: the
        # alternative for a packed row is a block-diagonal mask, which the caller
        # supplies through `attention_mask` when no kernel is available, and
        # which costs what constraint 32 measures. See `dg_varlen.py`.
        if packed_segments is not None:
            cu_seqlens, max_seqlen = packed_segments
            attn_output = varlen_encoder_attention(
                query_states,
                key_states,
                value_states,
                cu_seqlens,
                max_seqlen,
                head_dim=self.head_dim,
                sliding_window=self.sliding_window,
                sm_scale=self.scaling,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        # CHANGED: route the wide global heads to FFPA when `config.attn_ffpa` is set; this is
        # a no-op otherwise. See `ffpa_attention.py`.
        attention_interface: Callable = resolve_attention_interface(
            self.config, self.head_dim, eager_attention_forward
        )
        # CHANGED: cap the flex_attention tile size for wide heads; see `_flex_kernel_options`.
        if self.config._attn_implementation in _FLEX_ATTENTION_IMPLS:
            kwargs.setdefault("kernel_options", flex_kernel_options_for(self.head_dim, query_states.device))
            query_states, key_states, value_states = one_layout_for_flex(query_states, key_states, value_states)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _repeat_cache_onto_canvas_rows(
    keys: torch.Tensor, values: torch.Tensor, query_rows: int, canvas_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Line an encoder cache of ``B`` rows up with ``B * K`` decoder rows.

    ``canvas_groups`` is K, passed down from ``_decode_canvases_flat_batch`` rather than
    inferred from the two batch sizes. Inferring it would make any other shape mismatch
    -- a decoder batch that came apart from its encoder for an unrelated reason -- look
    like a fold and broadcast silently, and at the shipped ``micro_batch_size: 1`` a
    one-row cache divides into everything, so the check would never fire. Serial keeps
    the exact equality that ``torch.cat`` used to enforce for it.

    Only the concatenating path needs this; DG_PREFIX gathers from the shared row. The
    copies are the ones the serial loop also makes -- one per pass, each held for the
    backward -- so folding them together costs no extra memory beyond the transient.
    """
    cache_rows = keys.shape[0]
    if query_rows != cache_rows * canvas_groups:
        raise ValueError(
            f"The decoder has {query_rows} rows against an encoder cache of {cache_rows} and "
            f"{canvas_groups} canvases per cached sample."
        )
    if canvas_groups == 1:
        return keys, values
    return keys.repeat_interleave(canvas_groups, dim=0), values.repeat_interleave(canvas_groups, dim=0)


class DiffusionGemmaDecoderTextAttention(nn.Module):
    """Attention layer for the diffusion model.

    This layer is just like `Gemma4TextAttention`, with three key differences:
    1. Removes shared KV cache logic, as it is unused in DiffusionGemma.
    2. It doesn't update the KV cache in the forward pass. The KV cache here corresponds to the
       encoder's KV cache, which is passed in via `past_key_values` -- from the decoder's perspective, it can be seen
       as a read-only encoder KV cache.
    3. `self.is_causal` is set to `False`. `config.use_bidirectional_attention` only controls the
       encoder, not the decoder attention.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        super().__init__()
        self.is_causal = False  # In the decoder, attention is bidirectional!

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # The code in this function is adapted from Gemma4TextAttention. ** The modified parts are clearly indicated **
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CHANGED: removed `if self.is_kv_shared_layer` branch, kept the `else`
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # CHANGED: how many canvases of one cached sample this batch carries, sent down by
        # `_decode_canvases_flat_batch`. Popped unconditionally so it never reaches the attention
        # interface below.
        canvas_groups = kwargs.pop("canvas_groups", 1)
        # CHANGED: under `canvas_batching="flat_seq"` the K canvases share this row, and
        # this is the run of the cache each of them may read. Popped unconditionally so
        # it never reaches the attention interface below.
        canvas_interval = kwargs.pop("canvas_interval", None)
        if past_key_values is not None:
            # CHANGED: instead of calling `past_key_values.update()` which updates the KV cache in-place and returns
            # the full KV states, we first obtain the encoder cache contents, and then append the current KV states.
            encoder_key_states = past_key_values.layers[self.layer_idx].keys
            encoder_value_states = past_key_values.layers[self.layer_idx].values
            dropout = self.attention_dropout if self.training else 0.0
            # CHANGED: `flat_seq` packs the canvases onto this row, so visibility is a
            # function of the query and no key-side mask can express it. The hole kernel
            # takes it as an interval and leaves the cache where it is.
            if canvas_interval is not None:
                clean_lo, clean_hi = canvas_interval
                # CHANGED: under sequence parallelism the cache arrives as this rank's
                # slice of the row while the intervals are the row's, so the cache is
                # gathered back to full length here and the heads are split instead.
                # See `dg_sequence_parallel.ulysses_decoder_split`.
                sharded_heads = get_parallel_state().sp_enabled
                if sharded_heads:
                    (
                        query_states,
                        key_states,
                        value_states,
                        encoder_key_states,
                        encoder_value_states,
                    ) = ulysses_decoder_split(
                        query_states, key_states, value_states, encoder_key_states, encoder_value_states
                    )
                attn_output = hole_decoder_attention(
                    query_states,
                    encoder_key_states,
                    encoder_value_states,
                    key_states,
                    value_states,
                    clean_lo,
                    clean_hi,
                    sm_scale=self.scaling,
                )
                if sharded_heads:
                    attn_output = gather_attention_heads(attn_output)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), None
            # CHANGED: pack visible encoder keys (full vs last W-1, already in the
            # per-layer mask) plus the current canvas and run DG_PREFIX. Padding /
            # future canvases never enter the kernel.
            if should_use_dg_prefix_decoder(self.config, query_states, dropout):
                # CHANGED: `dg_prefix` re-derives the visible prefix from a mask that
                # spans the whole encoder row and gathers it out of the cache in one
                # piece; this rank holds a slice of that cache.
                #
                # No SP configuration reaches this today, and the conjunction that
                # keeps it out lives in two other files: arriving here needs
                # `canvas_interval` to be None, which under `flat_seq_pack` means
                # `should_use_hole_kernel` said no -- and it says no only when
                # `attn_dg_prefix` is off or the device is not CUDA, either of which
                # also makes the predicate above false. Kept rather than deleted
                # because that is an argument about three call sites, not a property
                # of this branch, and because it is the refusal that names the path
                # that does work. `dg_prefix.should_use_dg_prefix_decoder` used to
                # raise as well; that one was removed, since refusing inside the
                # predicate made this line unreachable for a second reason.
                refuse_decoder_path_under_sequence_parallelism(
                    "`attn_dg_prefix` on the decoder",
                    "it gathers each canvas's visible prefix out of the cache in one piece.",
                )
                encoder_visible = encoder_visible_from_decoder_mask(
                    attention_mask,
                    encoder_key_states.shape[2],
                    query_states.shape[0],
                    query_states.device,
                    sliding_window=self.sliding_window if attention_mask is None else None,
                )
                attn_output = dg_prefix_decoder_attention(
                    query_states,
                    key_states,
                    value_states,
                    encoder_key_states,
                    encoder_value_states,
                    encoder_visible,
                    sm_scale=self.scaling,
                    canvas_groups=canvas_groups,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
            # CHANGED: concatenating a sharded cache onto the canvas keys and masking the
            # pair gives shapes that line up over a fraction of the context.
            #
            # Unreachable under any configuration that passes preflight today, for the
            # same reason as the dg_prefix refusal above: preflight requires
            # `attn_dg_prefix: true`, which routes `flat_seq_pack` to the hole kernel
            # and returns before this branch. Kept as defence in depth.
            refuse_decoder_path_under_sequence_parallelism(
                "the decoder's concatenated cache path",
                "it appends the cache to the canvas keys and lets the attention backend mask the pair.",
            )
            # CHANGED: `canvas_batching="flat_batch"` hands the decoder K canvases per cached
            # sample while the cache keeps one row per sample. DG_PREFIX above gathers
            # each canvas's prefix from that shared row; concatenation cannot, so line
            # the rows up first.
            encoder_key_states, encoder_value_states = _repeat_cache_onto_canvas_rows(
                encoder_key_states, encoder_value_states, query_states.shape[0], canvas_groups
            )
            key_states = torch.cat([encoder_key_states, key_states], dim=2)
            value_states = torch.cat([encoder_value_states, value_states], dim=2)
        # CHANGED: removed the `if self.store_full_length_kv` branch

        # CHANGED: route the wide global heads to FFPA when `config.attn_ffpa` is set; this is
        # a no-op otherwise. See `ffpa_attention.py`. FFPA itself falls back to sdpa for the
        # decoder's 256-long canvas, so in practice this only fires on the encoder.
        attention_interface: Callable = resolve_attention_interface(
            self.config, self.head_dim, eager_attention_forward
        )
        # CHANGED: cap the flex_attention tile size for wide heads; see `_flex_kernel_options`.
        if self.config._attn_implementation in _FLEX_ATTENTION_IMPLS:
            kwargs.setdefault("kernel_options", flex_kernel_options_for(self.head_dim, query_states.device))
            query_states, key_states, value_states = one_layout_for_flex(query_states, key_states, value_states)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffusionGemmaText4MLP(nn.Module):
    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DiffusionGemmaTextRouter(nn.Module):
    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = DiffusionGemmaRMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        # TODO(joao): propagate fp32 to gemma4 and delete the modular overwrite in DiffusionGemma
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1, dtype=torch.float32)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class DiffusionGemmaTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    VeOmni deviation: upstream's ``@use_experts_implementation`` decorator is dropped. It
    replaces ``forward`` with a dispatch on ``config._experts_implementation`` that lands
    in HF's ``grouped_mm`` path, which bypasses VeOmni's fused MoE and, more importantly,
    assumes every rank holds all ``num_experts`` experts -- under EP it fails with
    "matrix batch sizes have to match". deepseek_v3's patch drops it for the same reason.
    """

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.hidden_activation = config.hidden_activation
        self.act_fn = ACT2FN[config.hidden_activation]
        moe_impl = getattr(getattr(config, "ops_config", None), "moe_implementation", None) or "eager"
        self.veomni_moe = VeomniOp("moe_experts", "standard", moe_impl)
        if self.veomni_moe.impl != "eager":
            refuse_fused_moe_on_unsupported_activation(config, self.veomni_moe.impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        # VeOmni addition: dispatch to the fused MoE kernel when `moe_implementation` is
        # not "eager". This is not just a speed knob -- it is what makes expert parallel
        # work at all. Under EP each rank holds only `E/ep` experts, while the router keeps
        # emitting global expert ids, so the eager loop below would index
        # `gate_up_proj[expert_idx]` out of range (or silently into the wrong expert).
        # `self.veomni_moe` owns the all-to-all and the global->local id mapping. Same
        # guard and the same stacked layout as deepseek_v3 / qwen3_moe.
        if self.veomni_moe.impl != "eager":
            unused = self.gate_up_proj.new_empty(0)
            return self.veomni_moe(
                hidden_states,
                top_k_weights.to(final_hidden_states.dtype),
                top_k_index,
                unused,
                unused,
                self.down_proj,
                self.gate_up_proj,
                num_experts=self.num_experts,
                activation=self.hidden_activation,
            )

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class SealableCache(DynamicCache):
    """A ``DynamicCache`` that stops accepting writes once its prefill is over.

    The encoder's forward is a side effect: it appends to the shared cache. Recompute
    replays that side effect, so a checkpointed encoder layer would append a second
    time and the replayed attention would read a cache twice as long as the original
    forward did. The seal is what makes the encoder recomputable: the model closes the
    cache after the prefill, and since recompute can only run afterwards, a write that
    arrives at a sealed cache is by definition a replay.

    A sealed ``update`` hands back exactly the states it was given, which is what the
    original call returned too -- but only because the encoder prefills into an *empty*
    cache, where there is nothing to concatenate. ``DiffusionGemmaEncoderTextModel``
    enforces that precondition before it lets recompute happen. Skipping the whole
    ``update`` also leaves ``cumulative_length`` alone, which matters because the
    sliding-window mask sizes are derived from it.

    Generation reuses one cache across canvases, so the seal is lifted at the start of
    every encoder forward rather than only at construction.

    Training additionally keeps the sliding layers untruncated. Eq. 13 puts every
    canvas in the encoder, and the decoder is only allowed to read the prefix in front
    of the draw; a sliding layer that stored the tail of the full sequence would hand
    the decoder exactly the tokens the mask is trying to hide. Generation still
    truncates -- the canvas always sits at the end of the growing context, which is
    the case the window is for.
    """

    sealed = False
    truncate_sliding = True

    def __init__(self, *args, truncate_sliding: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncate_sliding = truncate_sliding
        if truncate_sliding:
            return
        self.layers = [
            DynamicLayer() if isinstance(layer, DynamicSlidingWindowLayer) else layer for layer in self.layers
        ]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sealed:
            return key_states, value_states
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


class _UnshardForRecompute:
    """All-gathers a layer's nested FSDP units before its replay reads their parameters.

    FSDP2 unshards a unit from a hook on that unit's forward *outputs*, so it fires once
    a gradient reaches them -- long after the replay that has to produce those outputs.
    A layer whose parameters live in an enclosing unit gets away with it, because the
    enclosing unit's hook fires before anything inside it is replayed. The MoE experts
    are their own unit *nested inside* the layer, and under expert parallel the encoder
    and the decoder share that one unit: the decoder's backward reshards it, and the
    encoder's replay -- which comes later -- reads fp32 sharded weights and dies in the
    fused kernel's dtype assert.

    Unsharding here is the same call FSDP's own pre-backward makes, only early enough.
    What FSDP will not do is undo it: the unit's post-backward reshards only if it still
    has a gradient coming, and the one that prompted this may already have been reduced.
    A unit left unsharded is not merely wasted memory -- it leaves the *unsharded*
    parameter registered on the module, so ``named_parameters()`` hands the optimizer a
    tensor whose ``.grad`` is empty while the sharded one holds the real gradient. So
    every unshard is paired with a reshard queued for the end of the backward, which is
    the earliest point at which freeing the storage cannot pull it out from under a
    gradient formula that still saved it.
    """

    def __init__(self, module: nn.Module, inner: AbstractContextManager):
        self._module = module
        self._inner = inner

    def __enter__(self):
        for submodule in self._module.modules():
            if not isinstance(submodule, FSDPModule):
                continue
            group = submodule._get_fsdp_state()._fsdp_param_group
            if group is None or group.is_unsharded:
                continue
            submodule.unshard()
            torch.autograd.Variable._execution_engine.queue_callback(submodule.reshard)
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)


class DiffusionGemmaCacheAwareCheckpointLayer(nn.Module):
    """Gradient checkpointing layer that keeps ``past_key_values`` under recompute.

    ``transformers.modeling_layers.GradientCheckpointingLayer`` blanks out
    ``past_key_values`` whenever it recomputes, because for autoregressive models a
    cache is purely a decoding-time optimization. Both of DiffusionGemma's stacks need
    it during training instead: the decoder *reads* the encoder cache on every forward,
    so dropping it would hide the prompt from the canvas and silently train the wrong
    objective, and the encoder's replay has to see the same cache state its original
    forward did.

    The encoder additionally *writes* the cache, which recompute would replay; see
    ``SealableCache`` for how that write is neutralized.

    Requires ``use_reentrant=False``. Passing the cache through the closure is what keeps
    it alive under recompute, but a reentrant checkpoint only routes gradients back to the
    *positional* arguments, so the encoder cache -- and with it the encoder's only path to
    a gradient -- would be silently cut off.
    """

    gradient_checkpointing = False

    def _recompute_context_fn(self):
        """Hand the replay a context that all-gathers the parameters it is about to read.

        Composes with whatever context the trainer configured (see ``recompute_context_fn``
        in ``veomni.distributed.torch_parallelize``) instead of replacing it.
        """
        configured = getattr(self._gradient_checkpointing_func, "keywords", {}).get("context_fn")
        forward_context, recompute_context = configured() if configured is not None else (nullcontext(), nullcontext())
        return forward_context, _UnshardForRecompute(self, recompute_context)

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            if getattr(self._gradient_checkpointing_func, "keywords", {}).get("use_reentrant"):
                raise ValueError(
                    "DiffusionGemma requires non-reentrant gradient checkpointing: the decoder reads the "
                    "encoder KV cache through the recompute closure, and a reentrant checkpoint only "
                    "propagates gradients to positional arguments, which would leave the encoder without "
                    "any gradient at all. Set `train.gradient_checkpointing.enable_reentrant=false`."
                )
            return self._gradient_checkpointing_func(
                partial(super().__call__, **kwargs), *args, context_fn=self._recompute_context_fn
            )
        return super().__call__(*args, **kwargs)


class DiffusionGemmaEncoderTextLayer(DiffusionGemmaCacheAwareCheckpointLayer):
    """Encoder layer for the diffusion encoder.

    Identical to `Gemma4TextDecoderLayer` except that:
    1. It doesn't have the PLE code path
    2. Doesn't pipe `shared_kv_states` around

    Recomputable, unlike upstream, but only in combination with ``SealableCache``: the
    layer's forward appends to the shared cache, and a bare recompute would append a
    second time. Checkpointing this stack is worth the plumbing because the encoder is
    where the sequence length lives -- it spans the whole context while the decoder only
    spans one canvas -- so its activations dominate, and without recompute they are all
    held until backward.
    """

    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaEncoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Take hidden states before MLP here
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine mlp and moe outputs
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaDecoderTextLayer(DiffusionGemmaCacheAwareCheckpointLayer):
    """Decoder layer for the diffusion decoder.

    Identical to `Gemma4TextDecoderLayer` except that:
    1. Uses `DiffusionGemmaDecoderTextAttention`, which reads from the encoder KV cache without updating it
    2. It doesn't have the PLE code path
    3. Doesn't pipe `shared_kv_states` around
    """

    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaDecoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Take hidden states before MLP here
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine mlp and moe outputs
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaTextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class DiffusionGemmaMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: PreTrainedConfig,
        text_config: DiffusionGemmaTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = DiffusionGemmaRMSNorm(
            self.multimodal_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.
        Args:
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.
        Returns:
            A torch.Tensor of embeddings with shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


def expand_self_conditioning_mask(mask: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    """Broadcast a per-sample mask onto the decoder sequence.

    Unpacked batches have one canvas per row, so ``mask`` is ``(batch_size,)`` and
    becomes ``(batch_size, 1, 1)``. Packed batches concatenate whole canvases along
    the sequence, so ``mask`` is ``(batch_size * n_canvases,)`` and becomes
    ``(batch_size, sequence_length, 1)`` with every token of a canvas sharing one bit.
    """
    flags = mask.reshape(-1)
    n_samples = flags.numel()
    if n_samples == batch_size:
        return flags.view(batch_size, 1, 1)
    if batch_size < 1 or n_samples % batch_size != 0:
        raise ValueError(f"`self_conditioning_mask` has {n_samples} sample flags for batch {batch_size}.")
    canvases_per_row = n_samples // batch_size
    if sequence_length % canvases_per_row != 0:
        raise ValueError(
            f"`self_conditioning_mask` has {canvases_per_row} canvases per row but the "
            f"decoder sequence length {sequence_length} does not split evenly."
        )
    canvas_length = sequence_length // canvases_per_row
    return flags.view(batch_size, canvases_per_row).repeat_interleave(canvas_length, dim=-1).unsqueeze(-1)


class DiffusionGemmaSelfConditioning(nn.Module):
    """
    Self-conditioning module using a feed-forward block.

    Processes soft-embeddings from the previous denoising step, converted from the returned logits, into a
    self-conditioning signal that is added to the decoder's input embeddings. Uses Gemma4's Gated MLP structure,
    with pre/post rms norm.
    """

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.pre_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, inputs_embeds, self_conditioning_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            self_conditioning_signal: Soft-embeddings from previous denoising step
                of shape `(batch_size, canvas_length, hidden_size)`.

        Returns:
            Processed self-conditioning signal, same shape.
        """
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(self.act_fn(self.gate_proj(normed)) * self.up_proj(normed))
        combined = inputs_embeds + sc_signal
        return self.post_norm(combined)


class DiffusionGemmaPreTrainedModel(PreTrainedModel):
    config: DiffusionGemmaConfig
    base_model_prefix = "model"
    # Upstream is inference-only and disables this. Both stacks are recomputed, through
    # `DiffusionGemmaCacheAwareCheckpointLayer`; the encoder's cache write is neutralized
    # on replay by `SealableCache`.
    supports_gradient_checkpointing = True
    # ``DiffusionGemmaEncoderTextLayer`` is deliberately absent (upstream lists it).
    # Every encoder transformer parameter *is* the decoder's parameter object, so wrapping
    # both layer types makes FSDP2 build two units over one set of parameters. Sharding
    # and the tie both survive that, but the units reshard independently and backward dies
    # with "setStorage: ... out of bounds for storage of size 0" once the first unit frees
    # the all-gather buffer the second one still needs.
    #
    # Leaving them unwrapped is not free, and costs more than it looks like it should:
    # unwrapped does not mean unowned, it means the root unit takes them, and FSDP2 pins
    # the root unit's ``reshard_after_forward`` to False. Measured on the toy config at
    # ``dp_shard=2``, every encoder parameter is still unsharded after the forward pass and
    # the local parameter count is 1.92x what ``named_parameters()`` reports -- which is
    # also why the FSDP2 test cannot see this, since that view dedupes tied parameters onto
    # the decoder. Gradients are bit-identical to an unsharded reference, so the cost is
    # memory rather than correctness. The fix is to give each tied pair a single param
    # group via ``fully_shard([decoder.layers[i], encoder...layers[i]])``, the overload that
    # exists for shared weights; until then expect little sharding win on this stack.
    _no_split_modules = [
        "DiffusionGemmaDecoderTextLayer",
        # The vision tower is built by `AutoModel.from_config`, so its layers keep their
        # upstream Gemma 4 names. A name that matches nothing fails silently and drops the
        # whole tower into the root unit, which is the trap described above.
        "Gemma4VisionEncoderLayer",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = None  # override
    input_modalities = ("image", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DiffusionGemmaTextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

        elif isinstance(module, DiffusionGemmaTextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, DiffusionGemmaTextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, DiffusionGemmaTextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DiffusionGemmaDecoderTextLayer):
            init.ones_(module.layer_scalar)
        elif isinstance(module, DiffusionGemmaClippableLinear) and module.use_clipped_linears:
            init.constant_(module.input_min, -float("inf"))
            init.constant_(module.input_max, float("inf"))
            init.constant_(module.output_min, -float("inf"))
            init.constant_(module.output_max, float("inf"))
        # Gemma4 modules' classes won't be correctly expanded with modular, so we match the class name
        # Gemma4VisionPatchEmbedder
        elif module.__class__.__name__.endswith("VisionPatchEmbedder"):
            init.ones_(module.position_embedding_table)
        # Gemma4VisionRotaryEmbedding
        elif module.__class__.__name__.endswith("VisionRotaryEmbedding"):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
        # Gemma4VisionModel
        elif module.__class__.__name__.endswith("Gemma4VisionModel") and module.config.standardize:
            init.zeros_(module.std_bias)
            init.ones_(module.std_scale)


class DiffusionGemmaEncoderTextModel(DiffusionGemmaPreTrainedModel):
    config: DiffusionGemmaTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # DiffusionGemmaEncoder downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [DiffusionGemmaEncoderTextLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pack_samples: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pack_samples (`bool`, *optional*, defaults to `False`):
            VeOmni extension. Whether the single row in `input_ids` holds several samples end to
            end rather than one. When set, `position_ids` is required and must restart at 0 for
            each sample; those restarts are the segment boundaries, and attention is confined
            within them so no sample reads the ones packed in front of it. `attention_mask` and a
            non-empty `past_key_values` are refused, since neither can be reconciled with a row
            whose positions restart. On CUDA the boundaries go to the varlen kernels as
            `cu_seqlens` and no mask is built, which also *replaces* `attn_implementation` and
            `attn_ffpa` for this stack -- the backend is picked by head width instead, and
            constraint 32 measured that substitution as the faster one on a packed row. Off CUDA a
            block-diagonal mask is built, which only sdpa reads correctly. See `dg_varlen.py`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = SealableCache(config=self.config, truncate_sliding=not self.training)

        # Recompute replays this stack's cache write, so it has to run against a cache
        # that refuses the second one. A sealed `update` returns the states untouched,
        # which only reproduces what the original call returned when there was nothing
        # in front of them to concatenate -- so recompute requires an empty cache, which
        # is what a training forward always builds. Generation reuses one cache across
        # canvases and does not recompute, so it lands on the unsealed path below.
        if self.training and any(layer.gradient_checkpointing for layer in self.layers):
            if not isinstance(past_key_values, SealableCache):
                raise ValueError(
                    "Gradient checkpointing needs the encoder cache to be a `SealableCache`, got "
                    f"{type(past_key_values).__name__}. Recompute would otherwise append to the cache a "
                    "second time and the replayed attention would read twice the keys the original "
                    "forward did. Pass `past_key_values=None` and let the model build the cache."
                )
            if past_key_values.get_seq_length() != 0:
                raise ValueError(
                    "Gradient checkpointing needs the encoder to prefill into an empty cache, but it "
                    f"already holds {past_key_values.get_seq_length()} tokens. A sealed replay skips the "
                    "concatenation with what was already there, so it would attend to the new states "
                    "alone and the gradients would not match the forward pass."
                )
        if isinstance(past_key_values, SealableCache):
            past_key_values.sealed = False

        positions_were_supplied = position_ids is not None
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # CHANGED: a row holding several samples carries its boundaries in its own
        # positions, and all thirty layers are handed the one pair -- reading them
        # costs a host synchronisation, so it happens once rather than per layer.
        # Deriving them here rather than taking them as an argument is deliberate:
        # RoPE below is built from these same `position_ids`, so the boundaries the
        # attention boxes by and the phases the keys are rotated to cannot describe
        # different sequences.
        packed_segments = None
        if pack_samples:
            if not positions_were_supplied:
                raise ValueError(
                    "packing samples needs `position_ids` that restart at 0 for each sample, but none were "
                    "passed. The positions this stack builds for itself run straight through the row, which "
                    "describes one long sample -- the pack would attend across every boundary in it."
                )
            # Whatever a caller built, it was built against one sample per row:
            # `create_causal_mask` boxes by position and every sample's restart
            # at 0, so its notion of "the past" spans the whole pack. This path
            # replaces it rather than intersecting with it, which means silently
            # dropping a padding mask, a prebuilt mapping, or the bidirectional
            # vision blocks `use_bidirectional_attention: "vision"` puts in one --
            # and `DiffusionGemmaEncoderModel` always passes a mapping, so a
            # packed call arriving through the outer model would lose it. Refuse.
            if attention_mask is not None:
                raise ValueError(
                    "a packed encoder row builds its own block-diagonal visibility and cannot honour the "
                    f"`attention_mask` it was handed ({type(attention_mask).__name__}). Pack the real tokens "
                    "and let the restarts in `position_ids` carry the boundaries: padding has no place in a "
                    "packed row, and a mask built against one sample per row would let each sample read the "
                    "ones ahead of it. Bidirectional vision blocks are unsupported here for the same reason."
                )
            # `cu_seqlens` describes the new tokens only, so on the flash layers
            # a prefilled cache would simply never be read, and on the wide ones
            # the kernel's own shape check would refuse with a message about the
            # cache being *shorter*. Neither is worth inheriting.
            if past_key_values is not None and past_key_values.get_seq_length() != 0:
                raise ValueError(
                    "a packed encoder row must prefill into an empty cache, but the one passed already holds "
                    f"{past_key_values.get_seq_length()} tokens. The segment boundaries cover the row being "
                    "packed and say nothing about what is already cached, so those keys would go unread."
                )
            packed_segments = encoder_segments(position_ids)

        # It may already have been prepared by e.g. `generate`
        if packed_segments is not None:
            # CHANGED: the ordinary masks are not merely slower on a packed row,
            # they are wrong -- `create_causal_mask` boxes by position, and every
            # sample's positions start at 0 again, so each one would read all the
            # samples ahead of it in the row. Either the kernels take the
            # boundaries as data and no mask is built, or a block-diagonal one is,
            # and there is no third option that happens to work.
            cu_seqlens, _ = packed_segments
            total_tokens = inputs_embeds.shape[1]
            # Eval applies none of it, matching `_decode_canvases_flat_seq`.
            dropout = self.config.attention_dropout if self.training else 0.0
            if should_use_varlen_encoder(self.config, inputs_embeds.device, dropout):
                causal_mask_mapping = dict.fromkeys(self.unique_layer_types)
            else:
                packed_segments = None
                causal_mask_mapping = {
                    "full_attention": block_diagonal_causal_mask(cu_seqlens, total_tokens),
                    "sliding_attention": block_diagonal_causal_mask(
                        cu_seqlens, total_tokens, self.config.sliding_window
                    ),
                }
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            with uncompiled_block_masks(self.config._attn_implementation in _FLEX_ATTENTION_IMPLS):
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }

        # CHANGED: sequence parallelism starts here, after the boundaries have been read
        # and the backend chosen, and covers only the thirty layers below -- attention
        # gathers the row back to full length itself. The embedding above therefore ran
        # at full length on every rank and is thrown away outside this slice, which is
        # one lookup against thirty layers of activations and buys the guarantee that
        # `cu_seqlens` was derived from positions no rank had yet touched.
        if pack_samples and get_parallel_state().sp_enabled:
            if packed_segments is None:
                raise ValueError(
                    "sequence parallelism on a packed encoder row needs varlen segment "
                    "boundaries, but none were computed. Under SP the varlen path is "
                    "mandatory; a fallback that materializes a block-diagonal mask would "
                    "defeat the purpose."
                )
            # One device-to-host sync, once per forward: the layers under SP used to
            # re-read `cu_seqlens[-1]` themselves, which is sixty syncs under
            # checkpointing on a 30-layer stack. The product below is host-side, so
            # the layer check needs none.
            covered = int(packed_segments[0][-1])
            if covered != inputs_embeds.shape[1]:
                raise ValueError(
                    f"`cu_seqlens` covers {covered} tokens but the row holds {inputs_embeds.shape[1]}. "
                    "Under sequence parallelism the boundaries must describe the whole row, read "
                    "before it was sharded."
                )
            inputs_embeds, position_ids = shard_packed_row(inputs_embeds, position_ids)

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # decoder layers
        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                packed_segments=packed_segments,
                **kwargs,
            )

        # The prefill is over: from here on, any write can only come from a replay.
        if isinstance(past_key_values, SealableCache):
            past_key_values.sealed = True

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def get_block_sequence_ids_for_mask(mm_token_type_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    mm_token_type_ids = mm_token_type_ids.to(device)

    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    block_sequence_ids = torch.where(is_vision, vision_group_ids, -1)
    return block_sequence_ids


@auto_docstring(
    custom_intro="""
    The DiffusionGemma encoder model comprising a vision backbone and a language model, *without* a language modeling
    head. It is very similar to Gemma4Model, except that it doesn't support audio or video inputs, and always
    assumes the MoE code path in the inner layers.
    """
)
class DiffusionGemmaEncoderModel(DiffusionGemmaPreTrainedModel):
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False
    config: DiffusionGemmaConfig
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size

        self.language_model = DiffusionGemmaEncoderTextModel(config=config.text_config)
        # Text-only training drops the vision tower; upstream always builds it.
        if config.vision_config is not None:
            self.vision_tower = AutoModel.from_config(config.vision_config)
            self.embed_vision = DiffusionGemmaMultimodalEmbedder(config.vision_config, config.text_config)
        else:
            self.vision_tower = None
            self.embed_vision = None

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision model into language model space.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        # `Gemma4VisionModel` ends on `(hidden_states - std_bias) * std_scale`, and those are
        # *buffers*. FSDP2's mixed precision casts parameters but leaves buffers alone, so
        # under bf16 training that subtraction promotes the tower's output back to fp32 while
        # `embedding_projection.weight` is bf16, and the projection dies in the matmul.
        # Upstream never sees this because it runs the whole tower in one dtype.
        projection_dtype = self.embed_vision.embedding_projection.weight.dtype
        vision_outputs.pooler_output = self.embed_vision(inputs_embeds=last_hidden_state.to(projection_dtype))
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Obtains mask for multimodal placeholders (replaced by soft tokens) and hard text tokens.

        Masks will be obtained from `input_ids` or `inputs_embeds` as available and in that
        precedence order.

        Args:
            input_ids: A tensor containing the hard token IDs from the text tokenizer.
            inputs_embeds: A tensor containing the embeddings for all hard text tokens.

        Returns:
            image_mask
        """
        if input_ids is not None:
            special_image_mask = input_ids == self.config.image_token_id
        else:
            image_token_embeddings = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embeddings).all(-1)

        return special_image_mask

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        pack_samples: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        pack_samples (`bool`, *optional*, defaults to `False`):
            Treat `input_ids` as several samples packed into one row, with the boundaries carried by
            the restarts in `position_ids`. This is the layer that builds the mask mapping, so packing
            is where it stops building one: visibility becomes block diagonal and comes from the
            segments instead. See `DiffusionGemmaEncoderTextModel.forward`, which does the packing and
            refuses a mask outright -- the padding mask is validated and dropped here rather than
            there, because dropping a caller's mask silently is the failure this whole path is
            arranged to avoid.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pack_samples:
            # This has to be refused *here*, before the default below fills it in.
            # `DiffusionGemmaEncoderTextModel` refuses a missing one too, but never
            # sees it missing when reached through this class -- and the default is
            # an `arange` across the whole row, which has one restart and so reads
            # as a single sample. Every boundary would be attended across and the
            # answer would look plausible.
            if position_ids is None:
                raise ValueError(
                    "packing samples needs `position_ids` that restart at 0 for each sample, but none were "
                    "passed. The positions this class would build instead run straight across the row, "
                    "describing one long sample, and nothing downstream can tell that apart from a row "
                    "that really is one sample."
                )
            # A packed row is built out of real tokens, so its padding mask is all
            # ones and carries nothing -- but the collator emits one anyway, and
            # dropping it unexamined is how a genuinely padded row would get packed
            # and read across its own padding. Checked, then dropped.
            if attention_mask is not None:
                if isinstance(attention_mask, dict):
                    raise ValueError(
                        "a packed encoder row builds its own block-diagonal visibility and cannot honour a "
                        "mask mapping, which is already materialized against one sample per row. Pass the "
                        "padding mask itself, or nothing."
                    )
                if not bool(attention_mask.all()) and not sp_pad_tokens(attention_mask):
                    raise ValueError(
                        "a packed encoder row cannot carry padding, but its `attention_mask` has "
                        f"{int((attention_mask == 0).sum())} zeros. Packing exists so that the padding is "
                        "not there: pack the real tokens and let the restarts in `position_ids` carry the "
                        "boundaries. A padded row packed as if it were dense would read across the padding "
                        "of the sample in front of it."
                    )
                attention_mask = None
            # The vision blocks are bidirectional and live in the mask mapping this
            # path stops building, so they would go missing rather than be refused.
            # `should_use_varlen_encoder` catches `use_bidirectional_attention: all`
            # for the same reason; the production checkpoint's `vision` only shows up
            # once there are image tokens to make blocks out of.
            if pixel_values is not None or mm_token_type_ids is not None:
                raise ValueError(
                    "a packed encoder row cannot carry images: their bidirectional blocks live in the mask "
                    "mapping that packing replaces with per-sample segments, and neither varlen backend "
                    "implements a bidirectional block. Pack text-only samples."
                )

        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds)

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask_mapping = None
        if not pack_samples:
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Upstream discards this return value, which drops the padding mask.
                causal_mask_mapping = self.create_masks_for_generate(
                    config=self.config.get_text_config(),
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            pack_samples=pack_samples,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_per_layer_input_embeddings(self):
        return self.language_model.embed_tokens_per_layer

    def set_per_layer_input_embeddings(self, value):
        self.language_model.embed_tokens_per_layer = value

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> dict:
        # TODO(joao): this fn exists in a gemma4 class, but not in Gemma4Model. Move it there, and remove the modular
        # overwrite in DiffusionGemma. Also rewrite Gemma4Model to use this function.
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
        # Smaller Gemma models use a conventional casual attention mask
        if getattr(config.get_text_config(), "use_bidirectional_attention", None) == "vision":
            block_sequence_ids = torch.full([*inputs_embeds.size()[:-1]], -1, device=inputs_embeds.device)
            if mm_token_type_ids is not None:
                block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)

            mask_kwargs["block_sequence_ids"] = block_sequence_ids

        with uncompiled_block_masks(config._attn_implementation in _FLEX_ATTENTION_IMPLS):
            return create_masks_for_generate(**mask_kwargs)


def _cache_truncates_sliding(past_key_values: Cache) -> bool:
    return any(isinstance(layer, DynamicSlidingWindowLayer) for layer in getattr(past_key_values, "layers", ()))


def _sliding_encoder_keys(encoder_visible: torch.Tensor, sliding_window: int) -> torch.Tensor:
    """Last ``sliding_window - 1`` visible encoder positions per row.

    Matches ``DynamicSlidingWindowLayer``, which stores ``[:, :, -sliding_window + 1:]``.
    Applied as a mask on an untruncated cache so a batch whose rows hide different
    suffixes still shares one key axis.
    """
    encoder_len = encoder_visible.shape[-1]
    idx = torch.arange(encoder_len, device=encoder_visible.device)
    last = (encoder_visible.to(idx.dtype) * idx).amax(dim=-1)
    start = (last - (sliding_window - 1) + 1).clamp(min=0)
    return encoder_visible & (idx.unsqueeze(0) >= start.unsqueeze(1))


def _encoder_hides_a_suffix(encoder_visible: torch.Tensor) -> bool:
    """True if any row has a 0 after a 1 -- future canvases, not left padding."""
    seen_one = encoder_visible.cummax(dim=-1).values.bool()
    return bool((~encoder_visible & seen_one).any())


def _diffusion_decoder_block_mask(valid_keys: torch.Tensor, q_length: int):
    """Compress the decoder's key-validity mask into a flex-attention ``BlockMask``.

    The decoder's mask is awkward for every other backend but trivial for flex: every canvas
    position attends to the same key set, so it is a bidirectional mask restricted by a
    key-side padding mask and nothing else. As a ``BlockMask``, flex skips the all-invalid key
    blocks outright, and nothing of shape ``(batch, 1, canvas, cache + canvas)`` has to exist.

    Built through transformers' own ``flex_attention_mask`` rather than ``create_block_mask``
    directly, so the encoder and decoder halves go through one code path and inherit the same
    decisions about block-size padding -- but not about mask compilation, which
    ``uncompiled_block_masks`` takes back for the reason given there.
    """
    with uncompiled_block_masks():
        return flex_attention_mask(
            batch_size=valid_keys.shape[0],
            q_length=q_length,
            kv_length=valid_keys.shape[-1],
            mask_function=bidirectional_mask_function,
            attention_mask=valid_keys,
            device=valid_keys.device,
        )


class DiffusionGemmaDecoderModel(DiffusionGemmaPreTrainedModel):
    """
    Decoder model for DiffusionGemma.

    Processes canvas tokens with bidirectional self-attention and cross-attention to the encoder's KV cache.
    The decoder reads but does not update the KV cache. Excluding these differences, it is similar to
    `DiffusionGemmaEncoderTextModel`, and they share all weights they have in common.
    """

    config: DiffusionGemmaConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaDecoderTextLayer,
        "attentions": DiffusionGemmaDecoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.text_config = config.text_config
        self.padding_idx = config.text_config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.text_config.hidden_size,
            padding_idx=self.padding_idx,
            embed_scale=config.text_config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                DiffusionGemmaDecoderTextLayer(config.text_config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
        self.norm = DiffusionGemmaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config.text_config)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config.text_config)
        self.unique_layer_types = set(config.text_config.layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_embeddings: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        canvas_intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the
            self-conditioning embeddings. Packed training concatenates canvases along the
            sequence, so `canvas_length` here is `n_samples * config.canvas_length`.
        self_conditioning_embeddings (`torch.FloatTensor` of shape `(batch_size, canvas_length, hidden_size)`, *optional*):
            Precomputed soft token embeddings for self-conditioning. Training uses this
            memory-bounded form; generation supplies `self_conditioning_logits` instead.
        self_conditioning_mask (`torch.BoolTensor` of shape `(n_samples,)`, *optional*):
            Per-sample mask over `self_conditioning_logits`: samples set to `False` get a
            zeroed self-conditioning signal, as if no logits were passed for them. Equals
            `batch_size` when each row is one canvas, and `batch_size * n_canvases` when
            packing concatenates canvases along the sequence. Training draws one Bernoulli
            per sample, not per padded batch row.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        canvas_intervals (`dict[str, tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            Per-layer-type `(lo, hi)` pairs, each `(batch_size, n_canvases)` int32,
            giving the half-open run of the encoder cache each canvas may read. Set
            only by `canvas_batching: flat_seq` when the hole kernel is serving the
            attention; the sdpa path carries the same visibility in
            `decoder_attention_mask` instead.
        """
        if "use_cache" in kwargs:
            raise ValueError(
                "The decoder of DiffusionGemma always uses a cache, so it doesn't accept the `use_cache` argument"
            )

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # No signal means the all-zero self-conditioning vector used by the released
        # model and by the training objective. It is not equivalent to softmaxing
        # all-zero vocabulary logits, which would inject the vocabulary-mean embedding.
        if self_conditioning_logits is not None and self_conditioning_embeddings is not None:
            raise ValueError("Pass either `self_conditioning_logits` or `self_conditioning_embeddings`, not both.")
        if self_conditioning_embeddings is not None:
            if self_conditioning_embeddings.shape != inputs_embeds.shape:
                raise ValueError(
                    "`self_conditioning_embeddings` must match the decoder embeddings, got "
                    f"{tuple(self_conditioning_embeddings.shape)} and {tuple(inputs_embeds.shape)}."
                )
            soft_embeddings = self_conditioning_embeddings.to(inputs_embeds.dtype)
        elif self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        if self_conditioning_mask is not None and (
            self_conditioning_logits is not None or self_conditioning_embeddings is not None
        ):
            # One flag per sample, not per padded batch row: packing concatenates
            # canvases along the sequence with batch=1, and each canvas still gets
            # its own Bernoulli. See `expand_self_conditioning_mask`.
            mask = expand_self_conditioning_mask(
                self_conditioning_mask.to(soft_embeddings.dtype),
                soft_embeddings.shape[0],
                soft_embeddings.shape[1],
            )
            soft_embeddings = soft_embeddings * mask
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        # With no explicit positions, the decoder follows the physical cache. Training
        # supplies positions derived from each canvas's visible prefix in the outer
        # model, because its cache also contains masked current and future canvases.
        if decoder_position_ids is None:
            canvas_length = inputs_embeds.shape[1]
            cache_seq_length = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            decoder_position_ids = decoder_position_ids.unsqueeze(0)

        if not isinstance(mask_mapping := decoder_attention_mask, dict):
            mask_mapping = self.create_diffusion_decoder_attention_mask(
                config=self.text_config,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

        # Embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, decoder_position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.text_config.num_hidden_layers]):
            layer_type = self.text_config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=mask_mapping[layer_type],
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                # A sliding layer's window is measured against each canvas's own
                # visible prefix, so the two layer types get different intervals.
                canvas_interval=canvas_intervals[layer_type] if canvas_intervals else None,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # No past_key_values in the output: the decoder doesn't produce a KV cache
        return BaseModelOutput(last_hidden_state=hidden_states)

    @staticmethod
    def create_diffusion_decoder_attention_mask(
        config: DiffusionGemmaTextConfig,
        inputs_embeds: torch.Tensor,
        past_key_values: Cache,
        decoder_attention_mask: torch.Tensor | dict | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Creates the bidirectional attention mask for the decoder model.

        The decoder mask must have the length of the encoder kv cache plus the canvas being denoised, and it is
        bidirectional. The part of the attention mask corresponding to the encoder kv cache works like a usual
        bidirectional mask for an AR model -- it might be left or right padded. However, the part of the mask
        corresponding to the canvas is *always* set to 1.

        > [!TIP]
        > If `decoder_attention_mask` is manually set, be sure to follow the following practices:
        > 1. It has shape `(batch_size, sequence_length+canvas_length)`;
        > 2. The attention in the last `canvas_length` positions is set to 1s.

        A complex example:
        Let's consider a static-shaped KV cache with batch size = 2. One of the entries is left-padded, because
        it's shorter than the other. In our example, the canvas has a length of 4 tokens. Our cache has a length of 8
        tokens, and is pre-populated -- one of the sequences has 4 cached tokens, the other has 2 cached tokens
        (meaning that it has 2 left-padding tokens). Both sequences will have 4 empty positions in their cache.
        The produced attention mask corresponding to the encoder kv cache should be as follows

        indexing key: [batch_idx, canvas_idx]; shown dimension: kv attention
        [0, 0] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚
        [0, 1] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚
        [0, 2] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚
        [0, 3] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚
        [1, 0] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚
        [1, 1] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚
        [1, 2] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚
        [1, 3] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚

        In other words, the canvas will be able to attend to all non-padding and non-empty kv cache positions.
        To complete the attention mask, we add a bidirectional attention to the canvas tokens, resulting in the
        following final attention mask

        indexing key: [batch_idx, canvas_idx]; shown dimension: kv attention
        [0, 0] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [0, 1] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [0, 2] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [0, 3] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [1, 0] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [1, 1] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [1, 2] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■
        [1, 3] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■

        As a result, the canvas tokens for each batch index can attend to themselves, as well as to valid entries
        in the corresponding encoder kv cache.

        For more examples, see the tests for this function.

        Args:
            config (`DiffusionGemmaTextConfig`):
                The config used by the model.
            inputs_embeds (`torch.Tensor` of shape `(batch_size, canvas_length, hidden_dimension)`):
                The input embeddings used in the current forward pass. Only used to obtain the first two dimensions.
            past_key_values (`Cache`):
                The cache produced by the encoder part of the model.
            decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
                Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        """

        # NOTE: common mask utilities like `create_bidirectional_mask` are NOT used here, as they contain a few subtle
        # AR assumptions. Example: in sliding window mask preparation, we consider a KV with length
        # `sliding_window - 1 + query_length`, where we want `sliding_window + query_length`
        # (https://github.com/huggingface/transformers/blame/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py#L249)

        batch_size, canvas_length, _ = inputs_embeds.shape

        if past_key_values is None:
            raise ValueError(
                "`past_key_values` must be a `Cache` instance in `create_diffusion_decoder_attention_mask`."
            )
        if past_key_values.is_compileable and decoder_attention_mask is None:
            raise ValueError(
                "When `past_key_values` is a compileable cache, i.e. a static-shaped cache, `decoder_attention_mask` "
                "must be set."
            )
        # Shortcut: not compiling for sure AND no padding -> delegate mask creation to the inner functions by returning None
        #
        # VeOmni deviation: only take the shortcut when the attention backend can apply the
        # sliding window itself. Returning `None` hands that job to the backend, but `sdpa`
        # and `eager` swallow `sliding_window` in `**kwargs` and never look at it, and
        # `flex_attention` reads the window off the mask too, so for all three the window
        # exists *only* as long as the mask encodes it. Taking the shortcut would silently
        # give every sliding layer full attention over the encoder cache -- and only for
        # batches that happen to have no padding, so a padded batch and an unpadded one
        # would train different models. Inference never hits this because a static cache is
        # compileable and always materializes.
        backend_applies_sliding_window = config._attn_implementation not in (
            *_SDPA_ATTENTION_IMPLS,
            "eager",
            *_FLEX_ATTENTION_IMPLS,
        )
        has_sliding_layers = "sliding_attention" in (getattr(config, "layer_types", None) or ())
        if backend_applies_sliding_window or not has_sliding_layers:
            if decoder_attention_mask is None or (not past_key_values.is_compileable and decoder_attention_mask.all()):
                return {"full_attention": None, "sliding_attention": None}
        elif decoder_attention_mask is None:
            # Materializing below needs an explicit mask; all-ones is what the shortcut implied.
            decoder_attention_mask = torch.ones(
                (batch_size, past_key_values.get_seq_length() + canvas_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        # If we reach this point, we have padding and/or we may want to compile the forward pass. In either case, we
        # materialize the full mask.
        # - Full attention mask: built from the `decoder_attention_mask` input (if unset, then it's all 1s).
        # - Sliding attention mask: built from full attention mask, taking a slice of the attention mask based on the
        #   filled cache positions, plus the canvas attention
        valid_cache_tokens = past_key_values.get_seq_length()
        full_cache_kv_length = _decoder_cache_span(past_key_values)
        full_kv_length = full_cache_kv_length + canvas_length
        if decoder_attention_mask.shape != (batch_size, full_kv_length):
            raise ValueError(
                "When set, `decoder_attention_mask` must have the length = cache length + canvas length."
                f" Got `decoder_attention_mask` with length {decoder_attention_mask.shape[1]} "
                f"(!= {full_cache_kv_length} + {canvas_length})"
            )
        if (decoder_attention_mask.sum(dim=-1) > valid_cache_tokens + canvas_length).any():
            raise ValueError(
                "Your `decoder_attention_mask` has more 1s than there are cached + canvas tokens. "
                "There is one or more rows in the `decoder_attention_mask` with "
                f"{decoder_attention_mask.sum(dim=-1).max()} 1s, while there are at most "
                f"{valid_cache_tokens + canvas_length} tokens to be processed in each "
                "row. If you're using a static cache, don't forget to set empty positions to 0."
            )

        # Both masks are pure key-side masks: every canvas position sees the same keys, so
        # the query axis carries no information. Keep them 2D until the backend forces a
        # shape, since that is what lets flex compress them.
        full_valid_keys = decoder_attention_mask.bool()
        # Sliding window: generation truncates the sliding layers' cache to the last
        # `sliding_window` entries, so their key axis is a slice of the cache followed by
        # the whole canvas. Training keeps the full encoder history so a per-row suffix
        # of future canvases can be masked out; the window is then applied as a mask on
        # that full axis (the last `sliding_window - 1` *visible* encoder positions).
        encoder_visible = full_valid_keys[:, :full_cache_kv_length]
        if _cache_truncates_sliding(past_key_values):
            if _encoder_hides_a_suffix(encoder_visible):
                raise ValueError(
                    "The decoder mask hides encoder positions after a visible prefix, but the sliding "
                    "layers truncated the cache to its tail. Those hidden tokens are exactly what the "
                    "decoder would read. Pass `past_key_values=None` in training so SealableCache can "
                    "keep the full encoder history."
                )
            sliding_cache_is_full = valid_cache_tokens >= config.sliding_window
            if sliding_cache_is_full:
                # NOTE: currently, the compiled sliding window cache layer is 1 element longer than the non-compiled case.
                # This means that we technically have a slightly different implementation with compilable caches, where
                # the decoder sees one extra token.
                if past_key_values.is_compileable:
                    sliding_start_idx = valid_cache_tokens - config.sliding_window
                else:
                    sliding_start_idx = valid_cache_tokens - config.sliding_window + 1
                sliding_end_idx = valid_cache_tokens
            else:
                sliding_start_idx = 0
                if past_key_values.is_compileable:
                    sliding_end_idx = min(config.sliding_window, past_key_values.max_cache_len)
                else:
                    sliding_end_idx = valid_cache_tokens
            sliding_valid_keys = torch.nn.functional.pad(
                full_valid_keys[:, sliding_start_idx:sliding_end_idx], (0, canvas_length), value=True
            )
        else:
            sliding_valid_keys = torch.cat(
                [
                    _sliding_encoder_keys(encoder_visible, config.sliding_window),
                    full_valid_keys[:, full_cache_kv_length:],
                ],
                dim=-1,
            )

        if config._attn_implementation in _FLEX_ATTENTION_IMPLS:
            return {
                "full_attention": _diffusion_decoder_block_mask(full_valid_keys, canvas_length),
                "sliding_attention": _diffusion_decoder_block_mask(sliding_valid_keys, canvas_length),
            }

        # sdpa / eager need the mask materialized as 4D.
        def to_4d(valid_keys: torch.Tensor) -> torch.Tensor:
            # `expand` rather than a copy: the query axis is a broadcast, so the boolean mask
            # costs nothing beyond the 2D masks above.
            mask = valid_keys[:, None, None, :].expand(batch_size, 1, canvas_length, valid_keys.shape[-1])
            if config._attn_implementation != "eager":
                return mask
            # VeOmni deviation: `eager_attention_forward` *adds* the mask to the scores, so a
            # boolean mask adds +1.0 to the positions it means to keep and 0.0 to the ones it
            # means to drop -- padding stays fully visible and every score is off by one.
            # Upstream returns bool for every backend, so eager silently computes something
            # else than sdpa. Hand eager the additive mask it actually expects.
            min_value = torch.finfo(inputs_embeds.dtype).min
            return torch.zeros(mask.shape, dtype=inputs_embeds.dtype, device=mask.device).masked_fill_(
                ~mask, min_value
            )

        return {"full_attention": to_4d(full_valid_keys), "sliding_attention": to_4d(sliding_valid_keys)}


@auto_docstring
@dataclass
class DiffusionGemmaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden states at the output of the last layer of the encoder. Only set when `input_ids` is
        provided, e.g. to compute an autoregressive loss on the encoder during training.
    """

    encoder_last_hidden_state: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class DiffusionGemmaBlockDiffusionOutputWithPast(CausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Language modeling loss: `diffusion_loss + encoder_ar_loss`, the sum of Eq. 13.
    logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, config.text_config.vocab_size)`, *optional*):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        `None` when `return_logits=False` (the training path), which computes the diffusion
        term through a chunked LM head instead of materializing this tensor.
    diffusion_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        The canvas denoising term of `loss`, broken out for logging.
    encoder_ar_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        The encoder next-token term of `loss`, broken out for logging. Zero when
        `config.encoder_ar_loss` is off, rather than absent, so the logged series does
        not change shape with the setting.
    encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden states at the output of the last layer of the encoder. Only set when `input_ids` is
        provided; this is what the encoder autoregressive loss is computed from.
    self_conditioning_embeddings (`torch.FloatTensor` of shape `(batch_size, canvas_length, hidden_size)`, *optional*):
        Soft token embeddings produced for a training self-conditioning pass. Returned only when
        `return_self_conditioning_embeddings=True`; unlike logits, their memory does not scale with vocabulary size.
    """

    diffusion_loss: torch.FloatTensor | None = None
    encoder_ar_loss: torch.FloatTensor | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    self_conditioning_embeddings: torch.FloatTensor | None = None


def _canvas_mode(config) -> str:
    """How the decoder consumes the canvas axis, with packing folded away.

    `flat_seq_pack` differs from `flat_seq` in what the *collator* hands over -- several
    examples in the row instead of one -- not in how the decoder walks the canvases. So
    the decoder dispatch asks this rather than the raw field, and only the places that
    care whether samples are packed look at `pack_samples`.
    """
    return "flat_seq" if config.canvas_batching == "flat_seq_pack" else config.canvas_batching


def _packs_samples(config, pack_samples: bool) -> bool:
    """Whether this row holds several examples, from the config or the caller.

    `flat_seq_pack` is the packed mode, so a batch collated for it needs no second
    switch. Resolved through one helper because both `DiffusionGemmaForBlockDiffusion`
    and `DiffusionGemmaModel` branch on the answer and they must agree: the outer one
    segments the encoder AR term, the inner one segments the attention. Were only the
    inner one to notice, the AR term would quietly divide each sample's share by the
    number of samples packed with it -- a changed objective with no changed shape.
    """
    return bool(pack_samples) or config.canvas_batching == "flat_seq_pack"


@auto_docstring
class DiffusionGemmaModel(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma model consisting of an auto-regressive encoder (DiffusionGemmaEncoderModel, very similar to a
    Gemma4Model), and a diffusion decoder (DiffusionGemmaDecoderModel).

    NOTE: contrarily to most encoder-decoder models, where the encoder feeds its hidden states to the decoder, here the
    encoder only feeds its KV cache to the decoder. From the decoder's perspective, the KV cache is read-only.
    """

    # All weights in the text part of the encoder are present in the decoder. However, only the decoder has the
    # self-conditioning layers. At the time of writing, HF code assumes only weights can be tied.
    _tied_weights_keys = {
        "encoder.language_model.norm.weight": "decoder.norm.weight",
        # The lines below are equivalent to `"encoder.language_model.layers": "decoder.layers"`, but don't tie buffers
        # (see comment above).
        r"encoder.language_model.layers\.(?:[^.]+\.)*weight": r"decoder.layers\.(?:[^.]+\.)*weight",
        r"encoder.language_model.layers\.(?:[^.]+\.)*scale": r"decoder.layers\.(?:[^.]+\.)*scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*per_expert_scale": r"decoder.layers\.(?:[^.]+\.)*per_expert_scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*gate_up_proj": r"decoder.layers\.(?:[^.]+\.)*gate_up_proj",
        r"encoder.language_model.layers\.(?:[^.]+\.)*down_proj": r"decoder.layers\.(?:[^.]+\.)*down_proj",
        "encoder.language_model.embed_tokens.weight": "decoder.embed_tokens.weight",
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)

        # Registration order matters here, and is deliberately reversed from upstream.
        # ``nn.Module.named_parameters()`` deduplicates tied parameters down to whichever
        # name it reaches first, and VeOmni's weight loader matches checkpoint keys
        # against that deduplicated list. The checkpoint stores the shared tensors under
        # ``model.decoder.*`` (the encoder only contributes its ``layer_scalar`` buffers),
        # so the decoder has to be registered first for the canonical names to line up.
        # With the upstream order every shared tensor would be reported as an unexpected
        # checkpoint key and silently left at its initial value.
        self.decoder = DiffusionGemmaDecoderModel(config)
        self.encoder = DiffusionGemmaEncoderModel(config)

        # The decoder attends bidirectionally over `[encoder cache | canvas]`, which the
        # backend has to learn from an explicit mask. No FlashAttention can express that: the
        # `_upad_input` path reads any mask as a 2D padding mask over the query length,
        # indexes the `cache + canvas` long keys with it and gathers out of bounds, which
        # surfaces as an opaque device-side assert rather than an error. Reject it up front.
        #
        # A second, independent blocker: FA2 and FA3 both cap the head dimension at 256, and
        # this model's full-attention layers are 512 wide (`global_head_dim`; the sliding
        # layers are 256). So installing flash-attn could not cover the whole model even if
        # the mask were expressible -- the global layers would still need another backend.
        #
        # Asked of transformers rather than matched here, because the name this arrives
        # under is not the name the recipe wrote, and it is rewritten by *two* independent
        # mechanisms. `OpsImplementationConfig` rewrites `flash_attention_N` to
        # `veomni_flash_attention_N` whenever `MODELING_BACKEND=veomni`, which is
        # the default -- so a literal `== "flash_attention_2"` never fired for a real run.
        # And `_check_and_adjust_attn_implementation` rewrites the same names to
        # `kernels-community/flash-attn2` and friends when flash-attn is not importable but
        # `kernels` is, from inside `PreTrainedModel.__init__`, i.e. before this line. Those
        # names carry no `flash_attention` substring, so a substring test admits them.
        # `is_flash_attention_requested` tests for `flash`, which covers both rewrites and
        # the `paged|` prefix.
        #
        # Both configs are asked, because they can disagree: the hub-kernel rewrite writes
        # only the outer `_attn_implementation_internal`, while the attention layers read
        # `text_config`. Nothing in the flash family can serve this model, so there is no
        # name on either config worth admitting.
        if is_flash_attention_requested(config=config) or is_flash_attention_requested(config=config.text_config):
            raise ValueError(
                f"DiffusionGemma does not support FlashAttention, and `attn_implementation` is "
                f"{config._attn_implementation!r} on the model config and "
                f"{config.text_config._attn_implementation!r} on the text config: the decoder's "
                "bidirectional mask over the encoder KV cache plus the canvas cannot be expressed "
                "as the padding mask flash expects, and its full-attention layers are 512 wide, "
                "past FA2/FA3's 256 head dimension limit. Use `sdpa` or `flex_attention`."
            )

        # Sequence parallelism shards the encoder row, which only the packed mode has. The
        # other modes would not fail if asked -- they would replicate the micro-batch across
        # the SP ranks and agree on it, at `sp_size` times the cost. See `dg_sequence_parallel`.
        sequence_parallel_preflight(config)

        # DG_PREFIX takes the decoder over entirely and reads the per-layer decoder mask to
        # learn which encoder keys to pack, but under flex_attention that mask is a `BlockMask`
        # with no key-validity tensor to read. Refuse here rather than let the first backward
        # find it 26B parameters into the run, and rather than quietly dropping one of the two
        # flags -- either downgrade would have the run report the other's speed.
        if config._attn_implementation in _FLEX_ATTENTION_IMPLS:
            if getattr(config.text_config, "attn_dg_prefix", False):
                raise ValueError(
                    "attn_dg_prefix and flex_attention cannot both be on: the decoder kernel derives "
                    "its visible encoder keys from the decoder mask, which flex_attention passes as a "
                    "BlockMask. Set `attn_dg_prefix: false` to run the decoder on flex too, or keep "
                    "`attn_implementation: sdpa` to keep the kernel."
                )
            widen_flex_recompile_budget()
            refuse_uncompiled_flex()

        # Same reasoning, for the other opt-in kernel: refuse here rather than let the first
        # forward discover it, since a downgrade would have the run report sdpa's step time and
        # peak memory as FFPA's. See `ffpa_attention.py`.
        ffpa_preflight(config.text_config)

        self.post_init()
        if config.freeze_router:
            self.freeze_router_params()

    def freeze_router_params(self) -> None:
        """Freeze the gate parameters frozen by the NeMo DiffusionGemma recipe.

        ``per_expert_scale`` deliberately remains trainable. NeMo folds that scale
        into each expert's trainable down projection while adapting the checkpoint,
        so freezing it here would make the two parameterizations less equivalent.
        Encoder and decoder router parameters are tied, hence freezing the decoder's
        canonical parameters freezes both views.
        """
        for layer in self.decoder.layers:
            layer.router.proj.requires_grad_(False)
            layer.router.scale.requires_grad_(False)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_embeddings: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        pack_samples: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Uncached token IDs for the prompt to be encoded as context for the canvas.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `dict`, *optional*):
            Mask for the input tokens.
        pixel_values (`torch.FloatTensor` of shape `(num_images, max_patches, patch_dim)`, *optional*):
            Image patches for every image in the batch, concatenated along the image dimension.
        mm_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token modality tag over the prompt (0 text, 1 image). Drives the bidirectional
            attention blocks the encoder gives to image spans.
        image_position_ids (`torch.LongTensor` of shape `(num_images, max_patches, 2)`, *optional*):
            Patch (x, y) coordinates, with `(-1, -1)` marking padding patches.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the
            self-conditioning embeddings. Packed training concatenates canvases along the
            sequence, so `canvas_length` here is `n_samples * config.canvas_length`.
        self_conditioning_embeddings (`torch.FloatTensor` of shape `(batch_size, canvas_length, hidden_size)`, *optional*):
            Precomputed soft token embeddings for self-conditioning. Mutually exclusive
            with `self_conditioning_logits`.
        self_conditioning_mask (`torch.BoolTensor` of shape `(n_samples,)`, *optional*):
            Per-sample mask over `self_conditioning_logits`: samples set to `False` get a
            zeroed self-conditioning signal, as if no logits were passed for them. Equals
            `batch_size` when each row is one canvas, and `batch_size * n_canvases` when
            packing concatenates canvases along the sequence. Training draws one Bernoulli
            per sample, not per padded batch row.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        pack_samples (`bool`, *optional*, defaults to `False`):
            Run several samples as one row rather than one row each, with the boundaries carried by
            the restarts in `position_ids`. Both stacks are affected and neither infers it from the
            other: the encoder replaces its mask with per-sample segments, and the decoder places
            each canvas after its visible prefix inside its own sample. Requires
            `canvas_batching: flat_seq`, which is the mode whose visibility is already per canvas.
            Named rather than left in `kwargs` because `kwargs` also reaches the decoder, which
            takes no such argument.
        """
        pack_samples = _packs_samples(self.config, pack_samples)
        if pack_samples and _canvas_mode(self.config) != "flat_seq":
            raise ValueError(
                f"packing samples needs `canvas_batching: flat_seq` or `flat_seq_pack`, got "
                f"{self.config.canvas_batching!r}. The other two modes give each canvas its own row "
                "and carry visibility as a key-side mask, which cannot say that this canvas reads one "
                "sample's context and that one reads another's."
            )
        if pack_samples and input_ids is None:
            raise ValueError(
                "packing samples needs the encoder row it is packing: the sample boundaries are read off "
                "`position_ids` as the encoder runs, and a decode against a prebuilt cache has no way to "
                "recover where one sample's context ends and the next begins."
            )

        # 1: Encode new prompt tokens into the KV cache. Images are an encoder-only input:
        # they are named explicitly rather than left in `**kwargs` because `kwargs` also
        # flows into the decoder below, which has no vision path and would choke on them.
        encoder_last_hidden_state = None
        if input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                pixel_values=pixel_values,
                mm_token_type_ids=mm_token_type_ids,
                image_position_ids=image_position_ids,
                pack_samples=pack_samples,
                **kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            encoder_last_hidden_state = encoder_outputs.last_hidden_state
        elif past_key_values is None:
            raise ValueError("Either `input_ids` or `past_key_values` must be provided.")

        # 2: Run decoder with bidirectional self-attention in the canvas, and cross-attention to the KV cache.
        # In other words, the decoder attends to all tokens, KV cache and canvas, by default.

        # 2.a.: Prepare inputs for the decoder
        # If the canvas is unset, randomly sample from the vocabulary with uniform distribution
        if decoder_input_ids is None:
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.text_config.vocab_size,
                size=(input_ids.shape[0], self.config.canvas_length),
                device=self.decoder.device,
            )

        # 2.b.: Run the decoder. A stacked ``(B, K, C)`` canvas batch is the all-canvas
        # training path: one encoder cache, and K canvases that ``config.canvas_batching``
        # consumes either as K passes (``serial``) or as one pass of ``B * K`` rows
        # (``flat_batch``). Either way a canvas is a single segment in its own row;
        # concatenating them along the sequence instead would break sliding attention.
        decoder_outputs = self._decode_canvases(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            packed_position_ids=position_ids if pack_samples else None,
            **kwargs,
        )

        return DiffusionGemmaModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )

    def _decode_canvases(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None,
        self_conditioning_logits: torch.FloatTensor | None,
        self_conditioning_embeddings: torch.FloatTensor | None,
        self_conditioning_mask: torch.BoolTensor | None,
        decoder_attention_mask: torch.Tensor | dict | None,
        decoder_position_ids: torch.LongTensor | None,
        packed_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        if (
            decoder_position_ids is None
            and packed_position_ids is None
            and isinstance(decoder_attention_mask, torch.Tensor)
        ):
            if past_key_values is None:
                raise ValueError("deriving decoder positions from its context mask requires the encoder cache")
            decoder_position_ids = _decoder_positions_from_context_mask(
                decoder_attention_mask, _decoder_cache_span(past_key_values), decoder_input_ids.shape[-1]
            )

        # CHANGED: every canvas here reads the encoder KV cache, and under sequence
        # parallelism this rank holds a slice of it. `flat_seq` is the dispatch that
        # was built for that -- see `dg_sequence_parallel` -- and the ways in that
        # bypass it are refused where they part company below.
        if decoder_input_ids.ndim == 2:
            refuse_decoder_path_under_sequence_parallelism(
                "one canvas per row",
                "it is the generation shape, which carries no intervals over the cache.",
            )
            if packed_position_ids is not None:
                raise ValueError(
                    "a packed encoder row needs its canvases stacked as `(1, num_canvases, canvas_length)`, "
                    f"got {tuple(decoder_input_ids.shape)}. One canvas per row is the generation shape, and "
                    "it carries nothing to say which packed sample the canvas belongs to."
                )
            return self.decoder(
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                self_conditioning_logits=self_conditioning_logits,
                self_conditioning_embeddings=self_conditioning_embeddings,
                self_conditioning_mask=self_conditioning_mask,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                **kwargs,
            )
        if decoder_input_ids.ndim != 3:
            raise ValueError(
                "`decoder_input_ids` must be `(batch, canvas_length)` or "
                f"`(batch, num_canvases, canvas_length)`, got {tuple(decoder_input_ids.shape)}."
            )
        if self.config.canvas_batching == "flat_batch":
            refuse_decoder_path_under_sequence_parallelism(
                "`canvas_batching: flat_batch`",
                "it repeats the cache onto the canvas rows and concatenates it.",
            )
            return self._decode_canvases_flat_batch(
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                self_conditioning_logits=self_conditioning_logits,
                self_conditioning_embeddings=self_conditioning_embeddings,
                self_conditioning_mask=self_conditioning_mask,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                **kwargs,
            )
        if _canvas_mode(self.config) == "flat_seq":
            return self._decode_canvases_flat_seq(
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                self_conditioning_logits=self_conditioning_logits,
                self_conditioning_embeddings=self_conditioning_embeddings,
                self_conditioning_mask=self_conditioning_mask,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                packed_position_ids=packed_position_ids,
                **kwargs,
            )
        refuse_decoder_path_under_sequence_parallelism(
            "`canvas_batching: serial`",
            "each canvas is decoded on its own against the whole cache.",
        )
        batch_size, num_canvases, _canvas_length = decoder_input_ids.shape
        hidden_states = []
        for index in range(num_canvases):
            mask_k = decoder_attention_mask
            if isinstance(decoder_attention_mask, torch.Tensor) and decoder_attention_mask.ndim == 3:
                mask_k = decoder_attention_mask[:, index]
            sc_logits_k = self_conditioning_logits
            if self_conditioning_logits is not None and self_conditioning_logits.ndim == 4:
                sc_logits_k = self_conditioning_logits[:, index]
            sc_embeddings_k = self_conditioning_embeddings
            if self_conditioning_embeddings is not None and self_conditioning_embeddings.ndim == 4:
                sc_embeddings_k = self_conditioning_embeddings[:, index]
            sc_mask_k = self_conditioning_mask
            if self_conditioning_mask is not None:
                flags = self_conditioning_mask.reshape(-1)
                if flags.numel() == batch_size * num_canvases:
                    sc_mask_k = flags.view(batch_size, num_canvases)[:, index]
                elif flags.numel() != batch_size:
                    raise ValueError(
                        f"`self_conditioning_mask` has {flags.numel()} flags for batch {batch_size} "
                        f"and {num_canvases} canvases."
                    )
            pos_k = decoder_position_ids
            if decoder_position_ids is not None and decoder_position_ids.ndim == 3:
                pos_k = decoder_position_ids[:, index]
            hidden_states.append(
                self.decoder(
                    decoder_input_ids=decoder_input_ids[:, index],
                    past_key_values=past_key_values,
                    self_conditioning_logits=sc_logits_k,
                    self_conditioning_embeddings=sc_embeddings_k,
                    self_conditioning_mask=sc_mask_k,
                    decoder_attention_mask=mask_k,
                    decoder_position_ids=pos_k,
                    **kwargs,
                ).last_hidden_state
            )
        return BaseModelOutput(last_hidden_state=torch.stack(hidden_states, dim=1))

    def _decode_canvases_flat_batch(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None,
        self_conditioning_logits: torch.FloatTensor | None,
        self_conditioning_embeddings: torch.FloatTensor | None,
        self_conditioning_mask: torch.BoolTensor | None,
        decoder_attention_mask: torch.Tensor | dict | None,
        decoder_position_ids: torch.LongTensor | None,
        **kwargs,
    ) -> BaseModelOutput:
        """The serial loop's K passes as one pass over ``B * K`` rows.

        Folding the canvas axis into the batch rather than the sequence is what keeps
        this equivalent to the loop: a canvas stays a single segment at the end of its
        own row and carries its own key-side mask, so sliding attention still measures
        its window against that row's visible prefix and no canvas can reach another.
        Concatenating canvases along the sequence instead would need a query-side mask,
        and on sdpa it would also pay for the masked-out canvases, which is what the
        kernel verl pairs with that layout exists to skip.

        The encoder cache is not folded. It stays at ``B`` rows while the queries become
        ``B * K``, and K reaches the attention layers as the explicit ``canvas_groups``
        kwarg below -- it is deliberately not inferred from the two batch sizes, which
        would make an unrelated mismatch look like a fold. DG_PREFIX then gathers each
        canvas's prefix out of the one shared copy.
        """
        batch_size, num_canvases, canvas_length = decoder_input_ids.shape
        if isinstance(decoder_attention_mask, dict):
            raise ValueError(
                "`canvas_batching='flat_batch'` needs the unbuilt decoder mask: a mask mapping is already "
                "materialized against the encoder's batch and cannot be folded onto the canvas rows."
            )

        outputs = self.decoder(
            decoder_input_ids=decoder_input_ids.reshape(batch_size * num_canvases, canvas_length),
            past_key_values=past_key_values,
            self_conditioning_logits=_fold_canvas_axis(
                self_conditioning_logits, batch_size, num_canvases, stacked_ndim=4
            ),
            self_conditioning_embeddings=_fold_canvas_axis(
                self_conditioning_embeddings, batch_size, num_canvases, stacked_ndim=4
            ),
            self_conditioning_mask=_fold_self_conditioning_flags(self_conditioning_mask, batch_size, num_canvases),
            decoder_attention_mask=_fold_canvas_axis(decoder_attention_mask, batch_size, num_canvases, stacked_ndim=3),
            decoder_position_ids=_fold_canvas_axis(decoder_position_ids, batch_size, num_canvases, stacked_ndim=3),
            canvas_groups=num_canvases,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        return BaseModelOutput(
            last_hidden_state=hidden_states.view(batch_size, num_canvases, canvas_length, hidden_states.shape[-1])
        )

    def _decode_canvases_flat_seq(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None,
        self_conditioning_logits: torch.FloatTensor | None,
        self_conditioning_embeddings: torch.FloatTensor | None,
        self_conditioning_mask: torch.BoolTensor | None,
        decoder_attention_mask: torch.Tensor | dict | None,
        decoder_position_ids: torch.LongTensor | None,
        packed_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        """The serial loop's K passes as one pass over ``B`` rows of ``K * C`` tokens.

        Packing along the sequence rather than the batch is what lets the encoder
        cache be read where it lies: the hole kernel takes each canvas's visible run
        as an interval, so the KV it computes on is ``P + K*C`` against DG_PREFIX's
        ``K*P + K^2*C/2``. The price is that visibility now depends on which canvas a
        query sits in, which is a query-side mask -- free in the kernel, and a dense
        ``K*C`` by ``P + K*C`` boolean on sdpa.

        Positions are per canvas, not running: canvas ``k`` starts after the clean
        prefix it may read, at ``prompt_length + k*C``. The hidden current and future
        clean canvases remain in the encoder cache for the AR term but must not shift
        the decoder's RoPE positions.

        ``packed_position_ids`` turns the row into several samples' canvases rather
        than one sample's. The absolute interval is rebased to the start of the sample
        it belongs to, so preceding packed samples do not shift its positions.
        """
        batch_size, num_canvases, canvas_length = decoder_input_ids.shape
        if isinstance(decoder_attention_mask, dict):
            raise ValueError(
                "`canvas_batching='flat_seq'` needs the unbuilt decoder mask: a mask mapping is already "
                "materialized against one canvas per row and cannot be repacked onto the sequence."
            )
        if past_key_values is None:
            raise ValueError(
                "`canvas_batching='flat_seq'` reads each canvas's visible cache as an interval, which "
                "needs the encoder cache to measure against."
            )
        if _cache_truncates_sliding(past_key_values):
            raise ValueError(
                "`canvas_batching='flat_seq'` needs the untruncated encoder history: the sliding layers "
                "hold only the cache's tail, which is not the run each canvas is entitled to. Pass "
                "`past_key_values=None` in training so SealableCache keeps it."
            )

        text_config = self.config.text_config
        # CHANGED: under sequence parallelism the cache is sharded along the row while
        # every interval below is measured in the row's coordinates, which is also
        # what the replicated decoder mask describes. The attention gathers the cache
        # back to that length before it reads them.
        cache_length = full_cache_length(past_key_values.get_seq_length())
        tokens = num_canvases * canvas_length
        device = decoder_input_ids.device

        encoder_visible = _flat_seq_encoder_visibility(
            decoder_attention_mask, batch_size, num_canvases, canvas_length, cache_length, device
        )
        full_lo, full_hi = canvas_visible_intervals(encoder_visible)
        sliding_lo, sliding_hi = sliding_visible_intervals(full_lo, full_hi, text_config.sliding_window)

        dropout = text_config.attention_dropout if self.training else 0.0
        if should_use_hole_kernel(text_config, device, dropout):
            canvas_intervals = {
                "full_attention": (full_lo, full_hi),
                "sliding_attention": (sliding_lo, sliding_hi),
            }
            mask_mapping = {"full_attention": None, "sliding_attention": None}
        else:
            canvas_intervals = None
            mask_mapping = {
                "full_attention": _flat_seq_sdpa_mask(
                    text_config, full_lo, full_hi, cache_length, canvas_length, self.dtype
                ),
                "sliding_attention": _flat_seq_sdpa_mask(
                    text_config, sliding_lo, sliding_hi, cache_length, canvas_length, self.dtype
                ),
            }

        if packed_position_ids is not None:
            if decoder_position_ids is not None:
                raise ValueError(
                    "a packed row derives its decoder positions from the samples its canvases read, so it "
                    "cannot also be given `decoder_position_ids`. Two answers to where a canvas sits is one "
                    "more than the computation has."
                )
            if batch_size != 1:
                raise ValueError(
                    f"packing puts every sample in one row, so the canvases arrive as (1, K, C), got "
                    f"{batch_size} rows. A second row would need its own segment boundaries."
                )
            # Re-derived rather than carried out of the encoder, which costs a second
            # host sync per step against threading it through the encoder's output
            # type. `canvas_owner_segments` refuses a run that straddles two samples,
            # which is the leak this mode has to catch and the reason the positions
            # are derived here rather than trusted from the collator.
            cu_seqlens, _ = encoder_segments(packed_position_ids)
            if int(cu_seqlens[-1]) != cache_length:
                raise ValueError(
                    f"the packed encoder row covers {int(cu_seqlens[-1])} tokens but the cache holds "
                    f"{cache_length}. The canvases' visible runs are measured against the cache, so the two "
                    "have to describe the same row."
                )
            decoder_position_ids = packed_canvas_positions(full_lo, full_hi, cu_seqlens, canvas_length)
        elif decoder_position_ids is None:
            block = torch.arange(cache_length, cache_length + canvas_length, device=device, dtype=torch.long)
            decoder_position_ids = block.repeat(num_canvases).unsqueeze(0)
        elif decoder_position_ids.ndim == 3:
            decoder_position_ids = decoder_position_ids.reshape(decoder_position_ids.shape[0], tokens)

        if self_conditioning_logits is not None and self_conditioning_logits.ndim == 4:
            self_conditioning_logits = self_conditioning_logits.reshape(
                batch_size, tokens, self_conditioning_logits.shape[-1]
            )
        if self_conditioning_embeddings is not None and self_conditioning_embeddings.ndim == 4:
            self_conditioning_embeddings = self_conditioning_embeddings.reshape(
                batch_size, tokens, self_conditioning_embeddings.shape[-1]
            )

        outputs = self.decoder(
            decoder_input_ids=decoder_input_ids.reshape(batch_size, tokens),
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=mask_mapping,
            decoder_position_ids=decoder_position_ids,
            canvas_intervals=canvas_intervals,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        return BaseModelOutput(
            last_hidden_state=hidden_states.view(batch_size, num_canvases, canvas_length, hidden_states.shape[-1])
        )


def _decoder_positions_from_context_mask(
    decoder_attention_mask: torch.Tensor,
    cache_length: int,
    canvas_length: int,
) -> torch.Tensor:
    """Place each canvas immediately after the encoder prefix visible to it.

    The mask includes both the encoder cache and an all-visible canvas suffix.
    A two-dimensional mask is shared by one canvas per row; a three-dimensional
    mask carries one visible prefix per canvas. The returned shape follows that
    convention: ``(B, C)`` or ``(B, K, C)``.
    """
    if decoder_attention_mask.ndim == 2:
        visible = decoder_attention_mask.unsqueeze(1)
        squeeze_canvas = True
    elif decoder_attention_mask.ndim == 3:
        visible = decoder_attention_mask
        squeeze_canvas = False
    else:
        raise ValueError(
            "deriving decoder positions needs a mask of shape (batch, cache + canvas) or "
            f"(batch, canvases, cache + canvas), got {tuple(decoder_attention_mask.shape)}"
        )

    expected_width = cache_length + canvas_length
    if visible.shape[-1] != expected_width:
        raise ValueError(
            f"Expected a decoder mask width of cache + canvas = {expected_width}, got {visible.shape[-1]}."
        )
    if not visible[..., cache_length:].bool().all():
        raise ValueError("The canvas half of the decoder mask must be all ones when deriving its positions.")

    encoder_visible = visible[..., :cache_length].bool()
    cache_positions = torch.arange(cache_length, device=visible.device)
    visible_hi = torch.where(encoder_visible, cache_positions, torch.full_like(cache_positions, -1)).amax(dim=-1) + 1
    block = torch.arange(canvas_length, device=visible.device, dtype=torch.long)
    positions = visible_hi.to(torch.long).unsqueeze(-1) + block
    return positions[:, 0] if squeeze_canvas else positions


def _decoder_cache_span(past_key_values: Cache) -> int:
    """Width occupied by encoder slots in a decoder mask."""
    if past_key_values.is_compileable:
        return past_key_values.max_cache_len
    return full_cache_length(past_key_values.get_seq_length())


def _flat_seq_encoder_visibility(
    decoder_attention_mask: torch.Tensor | None,
    batch_size: int,
    num_canvases: int,
    canvas_length: int,
    cache_length: int,
    device: torch.device,
) -> torch.Tensor:
    """The encoder half of a per-canvas decoder mask, as ``(B, K, cache_length)``.

    An absent mask means every canvas may read the whole cache, which is what the
    serial loop and ``flat_batch`` do with one -- keeping the three modes the same
    computation. The all-canvas collator always sends the real per-canvas mask, so
    that case is the single-canvas one.
    """
    if decoder_attention_mask is None:
        return torch.ones(batch_size, num_canvases, cache_length, dtype=torch.bool, device=device)
    if decoder_attention_mask.ndim != 3:
        raise ValueError(
            "`canvas_batching='flat_seq'` needs a per-canvas decoder mask of shape "
            f"(batch, canvases, cache + canvas), got {tuple(decoder_attention_mask.shape)}."
        )
    expected = (batch_size, num_canvases, cache_length + canvas_length)
    if tuple(decoder_attention_mask.shape) != expected:
        raise ValueError(f"Expected a decoder mask of {expected}, got {tuple(decoder_attention_mask.shape)}.")
    mask = decoder_attention_mask.bool()
    if not mask[..., cache_length:].all():
        raise ValueError(
            "The canvas half of the decoder mask must be all ones: a canvas always attends to itself, "
            "and `flat_seq` carries no way to hide part of one."
        )
    return mask[..., :cache_length]


def _flat_seq_sdpa_mask(
    text_config,
    lo: torch.Tensor,
    hi: torch.Tensor,
    cache_length: int,
    canvas_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The intervals as the 4D mask sdpa or eager attends through."""
    mask = dense_hole_mask(lo, hi, cache_length, canvas_length)
    if text_config._attn_implementation != "eager":
        return mask
    # `eager_attention_forward` adds the mask to the scores, so a boolean one would
    # add +1.0 where it means to keep. Same deviation as the key-side masks above.
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill_(~mask, torch.finfo(dtype).min)


def _fold_canvas_axis(
    tensor: torch.Tensor | None, batch_size: int, num_canvases: int, *, stacked_ndim: int
) -> torch.Tensor | None:
    """Fold a decoder input onto the flat-batch path's ``B * K`` rows.

    An input that carries a canvas axis reshapes onto it, sample-major, so row ``b * K +
    k`` is canvas k of sample b. One that does not is shared by every canvas of its
    sample -- the serial loop hands each pass the same tensor -- so it repeats onto them
    instead, in the same order.

    Either way it must arrive with one row per sample. A single row is not broadcast
    across samples: on the serial path such a tensor is refused (a decoder mask is
    checked against ``(batch_size, full_kv_length)``), and accepting it here would hand
    every sample the first one's mask or conditioning signal.
    """
    if tensor is None:
        return None
    if tensor.ndim == stacked_ndim:
        if tensor.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} decoder rows, got {tensor.shape[0]}.")
        if tensor.shape[1] != num_canvases:
            raise ValueError(f"Expected {num_canvases} canvases per row, got {tensor.shape[1]}.")
        return tensor.reshape(batch_size * num_canvases, *tensor.shape[2:])
    if tensor.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} decoder rows, got {tensor.shape[0]}.")
    return tensor.repeat_interleave(num_canvases, dim=0)


def _fold_self_conditioning_flags(
    self_conditioning_mask: torch.Tensor | None, batch_size: int, num_canvases: int
) -> torch.Tensor | None:
    """One Bernoulli flag per decoder row, from either a per-canvas or a per-row draw."""
    if self_conditioning_mask is None:
        return None
    flags = self_conditioning_mask.reshape(-1)
    if flags.numel() == batch_size * num_canvases:
        return flags
    if flags.numel() == batch_size:
        return flags.repeat_interleave(num_canvases)
    raise ValueError(
        f"`self_conditioning_mask` has {flags.numel()} flags for batch {batch_size} and {num_canvases} canvases."
    )


def _first_eos_token_id(eos_token_id: int | list[int] | None) -> int:
    """Official CanvasChunker uses a single EOS id (Gemma4 special_tokens.EOS = 1)."""
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else 1
    return int(eos_token_id) if eos_token_id is not None else 1


def _other_tail_eos_mask(input_ids: torch.Tensor, is_real: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """True on the trailing EOS run except its first token.

    The official last-canvas fill is a suffix of EOS. The stop signal is the first of
    that run; the rest is EOS-to-EOS. Left-padding sits in front of the real tokens, so
    the suffix of the row is the suffix of the answer.
    """
    trailing = ((input_ids == int(eos_token_id)) & is_real).flip(-1).cummin(dim=-1).values.flip(-1)
    first = trailing & (trailing.to(torch.long).cumsum(dim=-1) == 1)
    return trailing & ~first


def _segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Which packed segment each of ``total`` positions belongs to, as ``(total,)``."""
    starts = cu_seqlens[1:-1].to(torch.long)
    ids = torch.zeros(total, dtype=torch.long, device=cu_seqlens.device)
    ids[starts] = 1
    return ids.cumsum(0)


def _segmented_tail_eos_mask(
    input_ids: torch.Tensor, is_real: torch.Tensor, eos_token_id: int, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """``_other_tail_eos_mask`` per packed sample rather than per row.

    A packed row has one trailing run as far as ``flip().cummin()`` can see, and
    it belongs to the last sample; every earlier sample's fill would stay
    supervised as EOS-to-EOS. Each sample's own suffix is what the unpacked case
    means by the row's.

    Counting instead of scanning is what makes it segmented: a position is in its
    sample's trailing run when no non-EOS position follows it *before that
    sample's end*, which a suffix sum answers by subtraction. At one segment this
    is the same mask ``_other_tail_eos_mask`` builds, position for position.
    """
    total = input_ids.shape[1]
    device = input_ids.device
    segment = _segment_ids(cu_seqlens, total)
    ends = cu_seqlens[1:].to(torch.long)[segment]

    is_eos = (input_ids == int(eos_token_id)) & is_real
    # `suffix[t]` counts non-EOS positions in `[t, total)`, with a zero at `total`
    # so a segment ending at the row's end subtracts something.
    suffix = torch.zeros(total + 1, dtype=torch.long, device=device)
    suffix[:total] = (~is_eos[0]).long().flip(0).cumsum(0).flip(0)
    trailing = is_eos & (suffix[torch.arange(total, device=device)] - suffix[ends] == 0)

    # The first of each sample's run is the stop target and stays supervised.
    inclusive = trailing[0].long().cumsum(0)
    before = torch.cat([torch.zeros(1, dtype=torch.long, device=device), inclusive])[cu_seqlens[:-1].to(torch.long)][
        segment
    ]
    first = trailing & ((inclusive - before) == 1)
    return trailing & ~first


def encoder_ar_targets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | dict | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    encoder_prompt_mask: torch.Tensor | None = None,
    mask_prompt_ar_loss: bool = False,
    mask_other_tail_eos: bool = True,
    eos_token_id: int = 1,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels and per-token weights for the encoder's next-token loss.

    Eq. 13 supervises every position of the encoder's input, prompt included, so the
    labels are the input ids themselves. Two kinds of position have to drop out anyway.

    Padding, first: the collator left-pads, so a padded row's leading hidden states
    attend to nothing but themselves. Predicting *from* one of them is as meaningless as
    predicting one, which costs the first real token of every padded row -- position `t`
    is read off the hidden state at `t-1`.

    Image spans, second: their ids are a placeholder repeated once per soft token, and
    their embeddings come from the vision tower rather than the embedding table. Asking
    the head to predict them trains the model to emit the placeholder. The first text
    token *after* an image stays supervised: predicting it from the image is the point.

    `mask_prompt_ar_loss` is a third, optional cut: the prompt stays in the encoder
    cache but is not a next-token target, so only the clean canvases train the head.
    The first answer token is still supervised -- its predecessor is the last prompt
    position. Weights are rebuilt over whatever remains.

    `mask_other_tail_eos` (default true) is a fourth cut: keep the first EOS of the
    trailing fill as the stop target, and drop the rest of that run from this term.
    The canvas still supervises every EOS when that last block is drawn. False restores
    Eq. 13 over the whole suffix.

    Weights are `1/N_s` over the row's `N_s` supervised positions, matching how the
    canvas term is normalized, so summing the two gives Eq. 13 per sample and the
    trainer's per-sample reduction applies to both alike.

    `cu_seqlens` makes all of that per *sample* instead of per row, for a packed row
    holding several. Three of the steps above are the row's and none of them changes
    a shape when the row stops being one sample: the weights would divide each
    sample's share by however many it was packed with, the drop of position 0 would
    leave every later sample's first token predicted from the previous sample's last
    hidden state, and the trailing-EOS cut would find the last sample's run alone.
    At one segment the result is identical to passing nothing.
    """
    if isinstance(attention_mask, dict):
        raise ValueError(
            "The encoder next-token loss reads the 2D `attention_mask` to find the padding. A prepared "
            "mask mapping hides it, and taking the batch for unpadded would supervise pad positions and "
            "every padded row's first real token."
        )
    if cu_seqlens is not None and input_ids.shape[0] != 1:
        raise ValueError(
            f"a packed row is one row, so segmented AR targets need `input_ids` of (1, total_tokens), got "
            f"{tuple(input_ids.shape)}. A second row would need its own segment boundaries."
        )
    is_real = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
    supervised = is_real.clone()
    supervised[:, 1:] &= is_real[:, :-1]
    # Nothing precedes position 0, and the shift inside the loss drops it. Clearing it
    # here is what makes `supervised.sum(-1)` the count the loss will actually see.
    # In a packed row that goes for every sample's first position: what precedes it is
    # the previous sample, which it cannot see and must not be trained to predict from.
    supervised[:, 0] = False
    if cu_seqlens is not None:
        supervised[:, cu_seqlens[:-1].to(torch.long)] = False
    if mm_token_type_ids is not None:
        supervised &= mm_token_type_ids == 0
    if mask_prompt_ar_loss and encoder_prompt_mask is not None:
        supervised &= ~encoder_prompt_mask.bool()
    if mask_other_tail_eos:
        if cu_seqlens is None:
            supervised &= ~_other_tail_eos_mask(input_ids, is_real, eos_token_id)
        else:
            supervised &= ~_segmented_tail_eos_mask(input_ids, is_real, eos_token_id, cu_seqlens)

    labels = input_ids.masked_fill(~supervised, IGNORE_INDEX)
    if cu_seqlens is None:
        counts = supervised.sum(dim=-1, keepdim=True)
        weights = supervised.to(torch.float32) / counts.clamp(min=1)
        return labels, weights

    # Per sample, so a token's weight does not depend on who it was packed with.
    # The alternative -- one count for the row -- divides every sample's share by
    # the number of samples, which is a silent change to the objective rather
    # than to any shape.
    total = input_ids.shape[1]
    segment = _segment_ids(cu_seqlens, total)
    counts = torch.zeros(cu_seqlens.numel() - 1, dtype=torch.float32, device=input_ids.device)
    counts.index_add_(0, segment, supervised[0].to(torch.float32))
    weights = supervised.to(torch.float32) / counts.clamp(min=1)[segment]
    return labels, weights


class ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_forward: Callable,
        loss_kwargs_chunks: list[dict],
        chunk_size: int,
    ):
        if head_bias is not None:
            raise NotImplementedError("head_bias is not supported in ChunkLoss")

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
        for hidden_states_chunk, grad_inputs_chunk, loss_kwargs in zip(
            hidden_states_chunks, grad_inputs_chunks, loss_kwargs_chunks, strict=False
        ):
            (chunk_grad_input, chunk_grad_weight), (chunk_loss, _) = ChunkLoss.compute_loss(
                loss_forward, hidden_states_chunk, head_weight, head_bias, loss_kwargs
            )
            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None, None

    @staticmethod
    def compute_loss(
        loss_forward: Callable,
        chunk_input: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        chunk_loss_kwargs: dict,
    ):
        def _compute_single_chunk(c_input, h_weight, h_bias, kwargs):
            return loss_forward(c_input, h_weight, h_bias, **kwargs)

        with torch.enable_grad():
            chunk_input = chunk_input.detach().requires_grad_()
            head_weight = head_weight.detach().requires_grad_()
            if head_bias is not None:
                head_bias = head_bias.detach().requires_grad_()

            grad_and_value_fn = grad_and_value(_compute_single_chunk, argnums=(0, 1), has_aux=True)
            vmap_fn = vmap(grad_and_value_fn, in_dims=(0, None, None, 0))
            (grad_input, grad_weight), (chunk_loss, chunk_logits) = vmap_fn(
                chunk_input, head_weight, head_bias, chunk_loss_kwargs
            )

        return (grad_input, grad_weight.sum(dim=0)), (chunk_loss.sum(dim=0), chunk_logits)


def chunked_weighted_causal_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    softcapping: float | None = None,
    logit_elements_per_chunk: int = 2**26,
) -> torch.Tensor:
    """Weighted next-token cross-entropy, summed, without materializing the logits.

    The encoder sequence is up to `max_seq_len` long against a 262k vocabulary: one
    fp32 logit tensor for 4096 positions is 4 GiB, and the backward needs its gradient
    alongside. `ChunkLoss` walks the sequence a slice at a time, folding the head
    projection into the loss and keeping only the gradient w.r.t. the hidden states,
    which is `hidden_size` wide rather than `vocab_size`.

    The chunk is sized by how many logits it materializes rather than by how many
    positions, since that is what the memory follows; the default keeps each fp32 slice
    at 256 MiB. `chunk_loss_function` is not reused directly because it normalizes by a
    single batch-wide token count, and per-sample weights are what put this term on the
    same footing as the canvas term.
    """
    labels = labels[:, 1:]
    weights = weights[:, 1:]
    hidden_states = hidden_states[:, :-1, :].contiguous()
    if labels.shape != weights.shape or hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            "encoder hidden states, labels and weights must describe the same tokens, got "
            f"hidden={tuple(hidden_states.shape)}, labels={tuple(labels.shape)}, weights={tuple(weights.shape)}"
        )
    labels = labels.to(device=hidden_states.device)
    # Same contract as `chunked_weighted_diffusion_loss`: the CE is zero at IGNORE
    # positions, so a non-finite padding weight would otherwise become a NaN loss.
    weights = weights.to(device=hidden_states.device, dtype=torch.float32).masked_fill(labels == IGNORE_INDEX, 0.0)

    def loss_forward(hidden_chunk, weight, bias, labels, weights):
        del bias
        logits = torch.nn.functional.linear(hidden_chunk.reshape(-1, hidden_chunk.size(-1)), weight).float()
        if softcapping is not None:
            logits = torch.tanh(logits / softcapping) * softcapping
        per_token = torch.nn.functional.cross_entropy(
            logits, labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
        )
        # `ChunkLoss` asks for an aux output and `torch.func` will only unwrap tensors,
        # so hand it an empty one rather than the logits this exists to not materialize.
        return (per_token * weights.reshape(-1)).sum(), logits.new_empty(0)

    chunk_size = max(1, logit_elements_per_chunk // (hidden_states.shape[0] * head_weight.shape[0]))
    chunk_kwargs = [
        {"labels": label_chunk, "weights": weight_chunk}
        for label_chunk, weight_chunk in zip(
            torch.split(labels.contiguous(), chunk_size, dim=1),
            torch.split(weights.contiguous(), chunk_size, dim=1),
            strict=True,
        )
    ]
    return ChunkLoss.apply(hidden_states, head_weight, None, loss_forward, chunk_kwargs, chunk_size)


def chunked_weighted_diffusion_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    softcapping: float | None = None,
    logit_elements_per_chunk: int = 2**26,
) -> torch.Tensor:
    """Weighted in-place canvas CE without retaining a full-vocabulary logit tensor.

    A long packed row can hold thousands of canvas tokens against the 262k vocabulary.
    Materializing all fp32 logits may fit in the forward, then OOM in backward when
    autograd allocates a same-sized gradient. The objective is token-local, so flatten
    every leading canvas dimension and apply the same chunked head used by the encoder
    AR term. Each slice stays at roughly 256 MiB with the default chunk budget.

    ``labels`` and ``weights`` must share the leading shape of ``hidden_states`` (before
    flattening) and live on the same device. The ignored-position weight contract matches
    ``dllm_masked_cross_entropy``: those entries are never read, so this path zeros them
    before the weighted sum rather than relying on the collator.
    """
    if labels.shape != weights.shape or hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            "decoder hidden states, labels and weights must describe the same tokens, got "
            f"hidden={tuple(hidden_states.shape)}, labels={tuple(labels.shape)}, weights={tuple(weights.shape)}"
        )
    hidden_states = hidden_states.reshape(1, -1, hidden_states.shape[-1]).contiguous()
    labels = labels.reshape(1, -1).contiguous().to(device=hidden_states.device)
    # `dllm_masked_cross_entropy` picks the supervised positions out before it reads any
    # weight, so its contract leaves the weight at an ignored position undefined. This
    # form weighs every position instead and relies on the ignored CE being exactly zero,
    # which turns a non-finite padding weight into a NaN loss rather than a dropped one.
    # Hold the contract here so a future collator cannot make the two paths disagree.
    weights = weights.reshape(1, -1).contiguous().to(device=hidden_states.device, dtype=torch.float32)
    weights = weights.masked_fill(labels == IGNORE_INDEX, 0.0)

    def loss_forward(hidden_chunk, weight, bias, labels, weights):
        del bias
        logits = torch.nn.functional.linear(hidden_chunk.reshape(-1, hidden_chunk.size(-1)), weight).float()
        if softcapping is not None:
            logits = torch.tanh(logits / softcapping) * softcapping
        per_token = torch.nn.functional.cross_entropy(
            logits, labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
        )
        return (per_token * weights.reshape(-1)).sum(), logits.new_empty(0)

    chunk_size = max(1, logit_elements_per_chunk // head_weight.shape[0])
    chunk_kwargs = [
        {"labels": label_chunk, "weights": weight_chunk}
        for label_chunk, weight_chunk in zip(
            torch.split(labels, chunk_size, dim=1),
            torch.split(weights, chunk_size, dim=1),
            strict=True,
        )
    ]
    return ChunkLoss.apply(hidden_states, head_weight, None, loss_forward, chunk_kwargs, chunk_size)


def chunked_self_conditioning_embeddings(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    embed_scale: torch.Tensor,
    softcapping: float | None = None,
    logit_elements_per_chunk: int = 2**26,
) -> torch.Tensor:
    """Convert decoder states to exact soft embeddings without a full-vocabulary tensor.

    Self-conditioning needs ``softmax(lm_head(hidden)) @ embedding_weight``, not the
    logits themselves. For a 24k canvas and 262k vocabulary, materializing fp32 logits
    costs about 25.5 GiB per rank. Chunking over token positions keeps the same math
    while bounding that temporary to roughly 256 MiB by default; the returned tensor
    scales with hidden size instead of vocabulary size.
    """
    if hidden_states.shape[-1] != head_weight.shape[-1]:
        raise ValueError(f"hidden size {hidden_states.shape[-1]} does not match LM head {tuple(head_weight.shape)}.")
    if head_weight.shape != embedding_weight.shape:
        raise ValueError(
            f"LM head and embedding weights must have the same shape, got "
            f"{tuple(head_weight.shape)} and {tuple(embedding_weight.shape)}."
        )
    if logit_elements_per_chunk < 1:
        raise ValueError("`logit_elements_per_chunk` must be positive.")

    leading_shape = hidden_states.shape[:-1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    chunk_size = max(1, logit_elements_per_chunk // head_weight.shape[0])
    chunks = []
    for hidden_chunk in torch.split(flat_hidden, chunk_size, dim=0):
        logits = torch.nn.functional.linear(hidden_chunk, head_weight).float()
        if softcapping is not None:
            logits = torch.tanh(logits / softcapping) * softcapping
        probabilities = logits.softmax(dim=-1).to(embedding_weight.dtype)
        chunks.append(torch.matmul(probabilities, embedding_weight))

    embeddings = torch.cat(chunks, dim=0).reshape(*leading_shape, embedding_weight.shape[-1])
    return embeddings * embed_scale.to(device=embeddings.device, dtype=embeddings.dtype)


class DiffusionGemmaForBlockDiffusion(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma model for block diffusion. It calls `DiffusionGemmaModel` to obtains the hidden states for
    the input canvas, conditioned by a prompt KV cache. Using its LM Head and self-conditioning blocks, it converts
    those hidden states into logits to sample the next canvas, as well as the self-conditioning embeddings for the
    next block diffusion step.

    Upstream is inference-only. VeOmni drops `DiffusionGemmaGenerationMixin` in favour of
    the sampler in `generation.py`, and adds the training objective on top -- see
    `forward`. The corruption is uniform-state, not masked: positions are replaced with
    tokens drawn uniformly from the vocabulary, and there is no mask token or absorbing
    state anywhere in this model.
    """

    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)

        self.model = DiffusionGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

        # Initialize weights and apply final processing
        self.post_init()

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    def retie_shared_weights(self):
        """Re-point the encoder at the decoder's parameters. See `parallel_plan`.

        Called by VeOmni after weight loading (`to_empty` clones every tensor, which
        breaks sharing) and after expert sharding.
        """
        from .parallel_plan import retie_encoder_to_decoder

        return retie_encoder_to_decoder(self)

    def get_input_embeddings(self):
        # Deliberately the decoder side: it is the one holding the canonical (loaded)
        # tensor, and VeOmni ties `lm_head` to whatever this returns.
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
        self.model.encoder.language_model.set_input_embeddings(value)

    def encoder_ar_term(
        self,
        encoder_hidden: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        mm_token_type_ids: torch.LongTensor | None,
        encoder_prompt_mask: torch.Tensor | None,
        encoder_ar_row_weight: torch.Tensor | None,
        pack_samples: bool,
    ) -> torch.Tensor:
        """The causal next-token half of Eq. 13, over the encoder's own input.

        A method rather than inline in `forward` so the sequence-parallel equivalence
        test drives the arithmetic a training step drives. Under SP the canvas half
        cannot run at all -- the decoder needs a cache no rank holds in full -- so
        this term is the only one there is to compare, and reaching it through
        `forward` is not available.

        Returns a *sum*, per sample and unnormalized, which is what makes the sharded
        version equal to the unsharded one by addition alone.
        """
        # The same boundaries the encoder attended by, read off the same tensor, so
        # this term cannot end up describing a different row than the attention did.
        ar_cu_seqlens = encoder_segments(position_ids)[0] if pack_samples else None
        ar_labels, ar_weights = encoder_ar_targets(
            input_ids,
            attention_mask,
            mm_token_type_ids,
            encoder_prompt_mask=encoder_prompt_mask,
            mask_prompt_ar_loss=self.config.mask_prompt_ar_loss,
            mask_other_tail_eos=self.config.mask_other_tail_eos,
            eos_token_id=_first_eos_token_id(self.config.get_text_config().eos_token_id),
            cu_seqlens=ar_cu_seqlens,
        )
        if encoder_ar_row_weight is not None:
            scale = encoder_ar_row_weight.to(device=ar_weights.device, dtype=ar_weights.dtype).reshape(-1)
            if ar_cu_seqlens is None:
                ar_weights = ar_weights * scale.reshape(-1, 1)
            else:
                # A packed row has one row and several samples, so a per-row scale can
                # no longer say "this example, not that one" -- the axis it indexes is
                # gone. It has to arrive per sample.
                segments = ar_cu_seqlens.numel() - 1
                if scale.numel() != segments:
                    raise ValueError(
                        f"`encoder_ar_row_weight` scales the AR term per example, and a packed row holds "
                        f"{segments} of them, but {scale.numel()} weights were passed. One weight for the "
                        "row would scale every sample in it alike, which is the opposite of what this "
                        "argument is for."
                    )
                ar_weights = ar_weights * scale[_segment_ids(ar_cu_seqlens, input_ids.shape[1])]

        # CHANGED: the targets are built at full length -- every input they read is
        # unsharded -- and are cut to this rank's states here, after each sample's
        # per-token weights have been formed from the whole of that sample.
        if get_parallel_state().sp_enabled:
            encoder_hidden, ar_labels, ar_weights = sharded_ar_slice(encoder_hidden, ar_labels, ar_weights)
        loss = chunked_weighted_causal_loss(
            encoder_hidden,
            self.lm_head.weight,
            ar_labels,
            ar_weights.to(encoder_hidden.device),
            softcapping=self.final_logit_softcapping,
        )
        return sum_over_shards(loss)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_embeddings: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_weight: torch.Tensor | None = None,
        encoder_prompt_mask: torch.Tensor | None = None,
        encoder_ar_row_weight: torch.Tensor | None = None,
        num_supervised_examples: torch.Tensor | None = None,
        use_cache: bool | None = None,
        pack_samples: bool = False,
        return_logits: bool = True,
        return_self_conditioning_embeddings: bool = False,
        logit_elements_per_chunk: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaBlockDiffusionOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Uncached token IDs for the prompt to be encoded as context for the canvas.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `dict`, *optional*):
            Mask for the input tokens.
        pixel_values (`torch.FloatTensor` of shape `(num_images, max_patches, patch_dim)`, *optional*):
            Image patches for every image in the batch, concatenated along the image dimension.
            Images condition the canvas only through the encoder KV cache.
        mm_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token modality tag over the prompt (0 text, 1 image).
        image_position_ids (`torch.LongTensor` of shape `(num_images, max_patches, 2)`, *optional*):
            Patch (x, y) coordinates, with `(-1, -1)` marking padding patches.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the
            self-conditioning embeddings. Packed training concatenates canvases along the
            sequence, so `canvas_length` here is `n_samples * config.canvas_length`.
        self_conditioning_embeddings (`torch.FloatTensor` of shape `(batch_size, canvas_length, hidden_size)`, *optional*):
            Precomputed soft token embeddings for self-conditioning. Training uses this
            compact representation instead of retaining full-vocabulary logits.
        self_conditioning_mask (`torch.BoolTensor` of shape `(n_samples,)`, *optional*):
            Per-sample mask over `self_conditioning_logits`: samples set to `False` get a
            zeroed self-conditioning signal, as if no logits were passed for them. Equals
            `batch_size` when each row is one canvas, and `batch_size * n_canvases` when
            packing concatenates canvases along the sequence. Training draws one Bernoulli
            per sample, not per padded batch row.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        labels (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            Ground-truth token IDs at masked canvas positions and `-100` everywhere else. There is no label
            shift: the decoder predicts each canvas position in place. The encoder's own next-token loss
            needs no labels of its own -- it supervises `input_ids`, and `encoder_ar_targets` derives what
            to skip from `attention_mask` and `mm_token_type_ids`.
        loss_weight (`torch.Tensor` of shape `(batch_size, canvas_length)`, *optional*):
            Per-token weights `1/N_s` over the sample's `N_s` supervised canvas positions,
            produced by the data transform. Required with `labels`. Unlike absorbing-mask
            dLLM recipes there is no `1/t_s` factor: corruption here re-randomizes tokens
            rather than replacing them with an absorbing mask, and the loss covers every
            supervised position instead of only the corrupted ones, so there is no
            `t`-dependent count to divide out.
        encoder_prompt_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            1 on encoder tokens that belong to the prompt (including `canvas_start`
            tokens already committed into it). Used only when `config.mask_prompt_ar_loss`
            is on; ignored otherwise. Left-padded with 0 alongside `input_ids`.
        encoder_ar_row_weight (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Per-row scale for the encoder AR term. The collator sets this to 1 on the
            first canvas of an example and 0 on the extra rows created when
            `compute_all_canvas_df_loss` expands K canvases, so AR is not multiplied
            by K. Ignored when omitted (every row contributes).
        num_supervised_examples (`torch.Tensor`, *optional*):
            Collator bookkeeping for the trainer's per-example reduction. Accepted and
            ignored here so `**micro_batch` can pass the whole batch through.
        use_cache (`bool`, *optional*):
            Accepted and ignored. The decoder always reads the encoder cache and never writes one, but
            `BaseTrainer.forward_backward_step` passes `use_cache=False` unconditionally.
        pack_samples (`bool`, *optional*, defaults to `False`):
            Run several samples as one row, with the boundaries carried by the restarts in
            `position_ids`. Both terms of Eq. 13 come out as they would with the samples apart: the
            canvas term is per canvas already and is bit-identical, and the encoder AR term is
            measured per sample from the same segment boundaries the encoder attended by. Requires
            `encoder_ar_row_weight` to arrive per sample rather than per row, since packing removes
            the axis it indexes.
        return_logits (`bool`, *optional*, defaults to `True`):
            Whether to materialize decoder logits. Training sets this to false and computes its
            weighted diffusion loss through a chunked LM head, avoiding a full-vocabulary
            tensor and its equally large backward gradient for that term. The label-free
            training self-conditioning pass also sets this false and requests compact
            soft embeddings directly.
        return_self_conditioning_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to output the exact soft token embeddings needed by the next self-conditioning
            pass, computed in bounded token chunks without materializing full logits.
        logit_elements_per_chunk (`int`, *optional*):
            Override the chunked diffusion LM-head budget (elements of the temporary fp32
            logit slice). Defaults to the budget inside `chunked_weighted_diffusion_loss`.
            Tests lower it so the multi-chunk path is actually exercised.
        """
        del use_cache
        del num_supervised_examples
        if return_logits and return_self_conditioning_embeddings:
            raise ValueError("Choose either logits or self-conditioning embeddings as the forward output.")
        if not return_logits and labels is None and not return_self_conditioning_embeddings:
            raise ValueError("`return_logits: false` requires labels so the chunked training loss has a consumer.")

        pack_samples = _packs_samples(self.config, pack_samples)

        # 1: Call the model
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
            image_position_ids=image_position_ids,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            pack_samples=pack_samples,
            **kwargs,
        )

        # 2. Obtain the logits and apply logits softcapping
        logits = None
        returned_self_conditioning_embeddings = None
        if return_logits:
            logits = self.lm_head(model_outputs.last_hidden_state)
            logits = logits.to(torch.float32)
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        elif return_self_conditioning_embeddings:
            chunk_kwargs = {}
            if logit_elements_per_chunk is not None:
                chunk_kwargs["logit_elements_per_chunk"] = logit_elements_per_chunk
            # The caller runs this pass under `no_grad`, which is where VeOmni's
            # stop-gradient sits. `align_nemo` moves it to NeMo's position, one product
            # later, so the embedding table can still learn from the hint it supplies.
            build_embeddings = (
                align_nemo.nemo_aligned_self_conditioning_embeddings
                if align_nemo.self_cond_embed_grad_enabled()
                else chunked_self_conditioning_embeddings
            )
            returned_self_conditioning_embeddings = build_embeddings(
                model_outputs.last_hidden_state,
                self.lm_head.weight,
                self.model.decoder.embed_tokens.weight,
                self.model.decoder.embed_tokens.embed_scale,
                softcapping=self.final_logit_softcapping,
                **chunk_kwargs,
            )

        # 3. The two terms of Eq. 13, summed with no weight between them: masked diffusion
        # over the canvas, and causal next-token over the encoder's own input. Both are
        # *local* per-sample sums; normalizing by the global sample count is the trainer's
        # job (``dllm_mean_global_loss``), mirroring how foundation models hand raw losses
        # upwards.
        loss = diffusion_loss = encoder_ar_loss = None
        if labels is not None:
            if loss_weight is None:
                raise ValueError("`loss_weight` is required when `labels` is provided.")
            if logits is None:
                chunk_kwargs = {}
                if logit_elements_per_chunk is not None:
                    chunk_kwargs["logit_elements_per_chunk"] = logit_elements_per_chunk
                diffusion_loss = chunked_weighted_diffusion_loss(
                    model_outputs.last_hidden_state,
                    self.lm_head.weight,
                    labels,
                    loss_weight,
                    softcapping=self.final_logit_softcapping,
                    **chunk_kwargs,
                )
            else:
                diffusion_loss = dllm_masked_cross_entropy(
                    logits,
                    labels,
                    loss_weight,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            encoder_ar_loss = torch.zeros_like(diffusion_loss)
            if self.config.encoder_ar_loss and model_outputs.encoder_last_hidden_state is not None:
                encoder_ar_loss = self.encoder_ar_term(
                    model_outputs.encoder_last_hidden_state,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    encoder_prompt_mask=encoder_prompt_mask,
                    encoder_ar_row_weight=encoder_ar_row_weight,
                    pack_samples=pack_samples,
                )
            loss = diffusion_loss + encoder_ar_loss

        return DiffusionGemmaBlockDiffusionOutputWithPast(
            loss=loss,
            logits=logits,
            diffusion_loss=diffusion_loss,
            encoder_ar_loss=encoder_ar_loss,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            past_key_values=model_outputs.past_key_values,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
            self_conditioning_embeddings=returned_self_conditioning_embeddings,
        )


__all__ = [
    "DiffusionGemmaPreTrainedModel",
    "DiffusionGemmaModel",
    "DiffusionGemmaDecoderModel",
    "DiffusionGemmaEncoderModel",
    "DiffusionGemmaEncoderTextModel",
    "DiffusionGemmaForBlockDiffusion",
    "chunked_self_conditioning_embeddings",
]
