"""BAGEL Qwen2 MoT backbone."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    import torch_npu

from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.utils import ModelOutput

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads
from ......ops.kernels.attention import flash_attention_forward
from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, get_tail_output_item
from ..sources import BAGEL_FLOW_HIDDEN, BAGEL_FLOW_QUERY, BAGEL_FLOW_VELOCITY, BAGEL_VAE_CONTEXT
from .configuration import BagelQwen2MoTConfig
from .generation_state import MotCacheContext, MotGenerationState
from .processing import PackedConversation, PackedSpan, preprocess_mot_inputs


class InferenceMixin:
    """FSM ``generate`` / denoise-branch / velocity-collection — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`BagelQwen2MoT`'s bases: ``OmniPreTrainedModel`` ships a no-op
    ``reset_local_inference_state`` default (kept as a safety net for modules
    that don't need real inference state), and MRO resolves left-to-right —
    put second, that no-op would shadow the real one below.
    """

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional["NaiveCache"] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        is_gen = _check_packed_inference_mode(mode)
        call_kwargs: Dict[str, Any] = {
            "packed_query_sequence": packed_query_sequence,
            "query_lens": query_lens,
            "packed_query_position_ids": packed_query_position_ids,
            "packed_query_indexes": packed_query_indexes,
            "past_key_values": past_key_values,
            "key_values_lens": key_values_lens,
            "packed_key_value_indexes": packed_key_value_indexes,
            "update_past_key_values": update_past_key_values,
            "is_causal": is_causal,
            "mode": mode,
        }
        if is_gen:
            call_kwargs["packed_vae_token_indexes"] = packed_vae_token_indexes
            call_kwargs["packed_text_indexes"] = packed_text_indexes
        output = self.model._forward_packed_inference(**call_kwargs)
        return {
            "hidden_states": output.packed_query_sequence,
            "past_key_values": output.past_key_values,
        }

    def reset_local_inference_state(self) -> None:
        self._generation_state.reset()

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires conversation_list.")

        generation_kwargs = generation_kwargs or {}
        infer_mode = self._generation_state.update_infer_mode(generation_kwargs)
        if self._generation_state.main.cache is None or infer_mode == "gen":
            hidden_states = self._prefill_prompt(conversation_list, generation_kwargs)
        else:
            hidden_states = self._decode_next_token(conversation_list)

        if infer_mode != "gen":
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
                hidden_states = hidden_states.squeeze(0)
            if hidden_states.dim() != 2:
                raise ValueError(f"BAGEL Qwen2-MoT expected packed hidden states, got {tuple(hidden_states.shape)}.")
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=hidden_states[-1:].contiguous(),
                    role="assistant",
                )
            )
        return {"conversation_list": conversation_list}

    def denoise_branch(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT denoise_branch requires conversation_list.")

        self._generation_state.validate_cfg_request(generation_kwargs or {})
        self._generation_state.main.require_ready()
        tail = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_QUERY])
        if tail is None or not torch.is_tensor(tail.value):
            raise ValueError("BAGEL Qwen2-MoT denoise branch requires source='bagel_flow_query'.")

        query = tail.value
        if query.dim() == 3 and query.shape[0] == 1:
            query = query.squeeze(0)
        if query.dim() != 2:
            raise ValueError(f"BAGEL Qwen2-MoT denoise branch expects rank-2 query tensor, got {tuple(query.shape)}.")
        if int(query.shape[-1]) != int(self.config.hidden_size):
            raise ValueError(
                "BAGEL Qwen2-MoT denoise branch hidden-size mismatch: "
                f"got {query.shape[-1]}, expected {self.config.hidden_size}."
            )
        if int(query.shape[0]) < 3:
            raise ValueError("BAGEL Qwen2-MoT denoise query must include start/end marker embeddings.")

        inputs = self._generation_state.preprocess_parallel_denoise_inputs(
            query,
            generation_kwargs or {},
            timestep=tail.meta.get("timestep"),
            empty_cache_factory=self._new_empty_cache,
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.forward_inference(
            **inputs,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
        )

        tail.source = BAGEL_FLOW_HIDDEN
        tail.value = outputs["hidden_states"].to(device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation_list}

    def collect_velocity(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL Qwen2-MoT collect_velocity requires conversation_list.")

        self._generation_state.validate_cfg_request(generation_kwargs or {})
        tail = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_VELOCITY])
        if tail is None or not torch.is_tensor(tail.value):
            raise ValueError("BAGEL Qwen2-MoT velocity collection requires source='bagel_flow_velocity'.")

        velocity = tail.value
        if velocity.dim() == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        if velocity.dim() != 2:
            raise ValueError(
                f"BAGEL Qwen2-MoT velocity collection expects rank-2 velocity, got {tuple(velocity.shape)}."
            )

        tail.value = self._generation_state.collect_velocity(
            velocity,
            generation_kwargs or {},
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation_list}

    def _prefill_prompt(
        self,
        conversation_list: list[ConversationItem],
        generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        packed = preprocess_mot_inputs(
            [conversation_list],
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        if packed is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires at least one embedded text/image item.")

        state = self._generation_state
        main_context = state.main
        main_context.reset()
        main_context.ensure_empty(empty_cache_factory=self._new_empty_cache, device=self.device)
        outputs = None

        for span in packed.spans:
            if span.item.type == "text":
                # Text CFG branches start from the main prompt cache before
                # the current text span, so snapshot before pre-filling it.
                self._generation_state.cfg_text.snapshot(
                    cache=main_context.cache,
                    key_values_lens=main_context.key_values_lens,
                    packed_key_value_indexes=main_context.packed_key_value_indexes,
                    next_position_id=packed.packed_position_ids[span.start],
                    empty_cache_factory=self._new_empty_cache,
                    device=self.device,
                )
                self._prefill_text_cfg_contexts(span, packed, generation_kwargs=generation_kwargs)

            outputs = self._prefill_main_prompt_span(span, packed, main_context)

            if span.item.type == "image":
                # After image context is in the main cache, text CFG should
                # keep that visual context while dropping later text condition.
                state.cfg_text.snapshot(
                    cache=main_context.cache,
                    key_values_lens=main_context.key_values_lens,
                    packed_key_value_indexes=main_context.packed_key_value_indexes,
                    next_position_id=main_context.next_position_ids,
                    empty_cache_factory=self._new_empty_cache,
                    device=self.device,
                )

        if outputs is None:
            raise RuntimeError("BAGEL Qwen2-MoT prefill produced no outputs.")
        return outputs["hidden_states"]

    def _decode_next_token(self, conversation_list: list[ConversationItem]) -> torch.Tensor:
        main_context = self._generation_state.main
        main_context.require_ready()
        tail = conversation_list[-1]
        if tail.type != "output":
            raise ValueError(f"BAGEL Qwen2-MoT decode expects tail output item, got {tail.type!r}.")

        packed_query_sequence = tail.value
        if not torch.is_tensor(packed_query_sequence):
            raise ValueError("BAGEL Qwen2-MoT decode expects tail output.value to be an embedding tensor.")
        if packed_query_sequence.dim() == 3 and packed_query_sequence.shape[0] == 1:
            packed_query_sequence = packed_query_sequence.squeeze(0)
        if packed_query_sequence.dim() != 2:
            raise ValueError(
                f"BAGEL Qwen2-MoT expected tail output embedding rank 2, got {tuple(packed_query_sequence.shape)}."
            )
        packed_query_sequence = packed_query_sequence[-1:].contiguous().to(device=self.device, dtype=self.dtype)

        query_lens, packed_query_indexes, packed_position_ids = main_context.packed_query_args(
            1,
            device=self.device,
        )
        outputs = self.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=main_context.cache,
            key_values_lens=main_context.key_values_lens,
            packed_key_value_indexes=main_context.packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        main_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
        )

        return outputs["hidden_states"]

    def _prefill_main_prompt_span(
        self,
        span: PackedSpan,
        packed: PackedConversation,
        main_context: MotCacheContext,
    ) -> Any:
        span_end = span.start + span.length
        span_position_ids = packed.packed_position_ids[span.start : span_end]
        query_lens, packed_query_indexes, packed_position_ids = main_context.packed_query_args(
            span.length,
            device=self.device,
            position_ids=span_position_ids,
        )
        call_kwargs = {
            "packed_query_sequence": packed.packed_sequence[span.start : span_end],
            "query_lens": query_lens,
            "packed_query_position_ids": packed_position_ids,
            "packed_query_indexes": packed_query_indexes,
            "past_key_values": main_context.cache,
            "key_values_lens": main_context.key_values_lens,
            "packed_key_value_indexes": main_context.packed_key_value_indexes,
            "update_past_key_values": True,
            "is_causal": span.item.type == "text",
            "mode": "und",
        }
        if span.item.type == "output":
            if span.length < 3:
                raise ValueError("BAGEL Qwen2-MoT output query must include start/end marker embeddings.")
            # Runtime flow query output remains marker-wrapped: marker tokens
            # stay on the text path, while interior latent tokens use the gen expert.
            call_kwargs["is_causal"] = False
            call_kwargs["mode"] = "gen"
            call_kwargs["packed_text_indexes"] = torch.tensor([0, span.length - 1], device=self.device)
            call_kwargs["packed_vae_token_indexes"] = torch.arange(
                1,
                span.length - 1,
                device=self.device,
                dtype=torch.long,
            )
        elif span.item.type == "image" and span.item.source == BAGEL_VAE_CONTEXT:
            # Prompt/edit VAE context is now source-routed as an image carrier.
            # Surrounding vision marker rows stay on the text path while the
            # image span itself uses the generation expert.
            call_kwargs["is_causal"] = False
            call_kwargs["mode"] = "gen"
            if span.is_image_triplet:
                call_kwargs["packed_text_indexes"] = torch.tensor(
                    [0, span.length - 1],
                    device=self.device,
                    dtype=torch.long,
                )
                start = span.primary_start
                end = start + span.primary_length
            else:
                start = 0
                end = span.length
            call_kwargs["packed_vae_token_indexes"] = torch.arange(
                start,
                end,
                device=self.device,
                dtype=torch.long,
            )

        outputs = self.forward_inference(**call_kwargs)
        main_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
            next_position_ids=packed_position_ids.max().reshape(1) + 1,
        )
        return outputs

    def _prefill_text_cfg_contexts(
        self,
        span: PackedSpan,
        packed: PackedConversation,
        *,
        generation_kwargs: dict[str, Any],
    ) -> None:
        # Text-only image CFG is only needed when image guidance is active.
        if not self._generation_state.cfg_img_requested(generation_kwargs):
            return

        # Image CFG keeps text conditioning while excluding image conditioning,
        # so it needs an independent text prefill branch.
        cfg_img_context = self._generation_state.cfg_img
        cfg_img_context.ensure_empty(empty_cache_factory=self._new_empty_cache, device=self.device)
        query_lens, packed_query_indexes, packed_position_ids = cfg_img_context.packed_query_args(
            span.length,
            device=self.device,
        )
        span_end = span.start + span.length
        outputs = self.forward_inference(
            packed_query_sequence=packed.packed_sequence[span.start : span_end],
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=cfg_img_context.cache,
            key_values_lens=cfg_img_context.key_values_lens,
            packed_key_value_indexes=cfg_img_context.packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        cfg_img_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
        )

    def _new_empty_cache(self) -> Any:
        return NaiveCache(len(self.model.layers))


class BagelQwen2MoT(InferenceMixin, OmniPreTrainedModel):
    config_class = BagelQwen2MoTConfig
    base_model_prefix = "bagel_qwen2_mot"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["BagelQwen2MoTDecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__(config)
        self.model = BagelQwen2MoTBackbone(config)
        self._generation_state = MotGenerationState()
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        packed_und_token_indexes = torch.nonzero(packed_token_type_ids == 0, as_tuple=False).flatten()
        packed_gen_token_indexes = torch.nonzero(packed_token_type_ids == 1, as_tuple=False).flatten()
        output = self.model(
            packed_sequence=packed_sequence,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return {"hidden_states": output.packed_query_sequence}


class NaiveCache:
    """Official BAGEL packed KV cache."""

    def __init__(self, num_layers: int):
        self.key_cache = dict.fromkeys(range(num_layers))
        self.value_cache = dict.fromkeys(range(num_layers))

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor | None = None
    past_key_values: Optional[NaiveCache] = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _fold_zero_anchors(target: torch.Tensor, *anchors: torch.Tensor) -> torch.Tensor:
    # When a packed batch has no generation tokens, the MoT "gen" expert weights still
    # run on an empty slice and produce zero-sized outputs. Folding ``sum() * 0.0`` of
    # those outputs into ``target`` keeps the gen-expert parameters in the autograd graph
    # (with zero gradient) so FSDP/DP gradient reduction sees the same parameter set on
    # every rank regardless of which modalities a micro-batch contains.
    anchor = target.new_zeros(())
    has_anchor = False
    for value in anchors:
        if torch.is_tensor(value):
            anchor = anchor + value.sum() * 0.0
            has_anchor = True
    if not has_anchor:
        return target
    return target + anchor


def _check_packed_inference_mode(
    mode: str,
) -> bool:
    if mode == "und":
        return False
    if mode == "gen":
        return True
    raise ValueError(f"Unsupported BAGEL Qwen2 MoT inference mode: {mode!r}")


class BagelQwen2RotaryEmbedding(nn.Module):
    """Official-compatible Qwen2 RoPE for BAGEL parity."""

    def __init__(self, config: BagelQwen2MoTConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        self.attention_scaling = 1.0
        inv_freq, _ = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        module = super()._apply(fn, recurse=recurse)
        self.inv_freq = self.inv_freq.float()
        self.original_inv_freq = self.original_inv_freq.float()
        return module

    @staticmethod
    def compute_default_rope_parameters(
        config: BagelQwen2MoTConfig,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        base = getattr(config, "rope_theta", None)
        if base is None:
            base = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BagelQwen2MoTAttention(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
        has_und_tokens = int(packed_und_token_indexes.numel()) > 0
        has_gen_tokens = int(packed_gen_token_indexes.numel()) > 0

        query_states_und = self.q_proj(packed_sequence_und)
        query_states_gen = self.q_proj_moe_gen(packed_sequence_gen)
        key_states_und = self.k_proj(packed_sequence_und)
        key_states_gen = self.k_proj_moe_gen(packed_sequence_gen)
        value_states_und = self.v_proj(packed_sequence_und)
        value_states_gen = self.v_proj_moe_gen(packed_sequence_gen)
        packed_query_states[packed_und_token_indexes] = query_states_und
        packed_query_states[packed_gen_token_indexes] = query_states_gen
        packed_key_states[packed_und_token_indexes] = key_states_und
        packed_key_states[packed_gen_token_indexes] = key_states_gen
        packed_value_states[packed_und_token_indexes] = value_states_und
        packed_value_states[packed_gen_token_indexes] = value_states_gen
        if not has_und_tokens:
            packed_query_states = _fold_zero_anchors(packed_query_states, query_states_und)
            packed_key_states = _fold_zero_anchors(packed_key_states, key_states_und)
            packed_value_states = _fold_zero_anchors(packed_value_states, value_states_und)
        if not has_gen_tokens:
            packed_query_states = _fold_zero_anchors(packed_query_states, query_states_gen)
            packed_key_states = _fold_zero_anchors(packed_key_states, key_states_gen)
            packed_value_states = _fold_zero_anchors(packed_value_states, value_states_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)
        query_states_norm_und = self.q_norm(packed_query_states[packed_und_token_indexes])
        query_states_norm_gen = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])
        key_states_norm_und = self.k_norm(packed_key_states[packed_und_token_indexes])
        key_states_norm_gen = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])
        packed_query_states_[packed_und_token_indexes] = query_states_norm_und
        packed_query_states_[packed_gen_token_indexes] = query_states_norm_gen
        packed_key_states_[packed_und_token_indexes] = key_states_norm_und
        packed_key_states_[packed_gen_token_indexes] = key_states_norm_gen
        if not has_und_tokens:
            packed_query_states_ = _fold_zero_anchors(packed_query_states_, query_states_norm_und)
            packed_key_states_ = _fold_zero_anchors(packed_key_states_, key_states_norm_und)
        if not has_gen_tokens:
            packed_query_states_ = _fold_zero_anchors(packed_query_states_, query_states_norm_gen)
            packed_key_states_ = _fold_zero_anchors(packed_key_states_, key_states_norm_gen)

        packed_query_states_, packed_key_states_ = _apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_position_cos,
            packed_position_sin,
            unsqueeze_dim=1,
        )

        sequence_length = int(attention_mask.shape[-1])
        ps = get_parallel_state()
        if ps.sp_enabled:
            # Ulysses all-to-all gathers the active sample's complete packed sequence
            # while sharding Q and native-GQA K/V heads.
            packed_query_states_ = gather_seq_scatter_heads(
                packed_query_states_,
                seq_dim=0,
                head_dim=1,
                group=ps.ulysses_group,
                unpadded_dim_size=sequence_length,
            )
            packed_key_states_ = gather_seq_scatter_heads(
                packed_key_states_,
                seq_dim=0,
                head_dim=1,
                group=ps.ulysses_group,
                unpadded_dim_size=sequence_length,
            )
            packed_value_states = gather_seq_scatter_heads(
                packed_value_states,
                seq_dim=0,
                head_dim=1,
                group=ps.ulysses_group,
                unpadded_dim_size=sequence_length,
            )

        packed_attn_output = self._masked_dense_attention(
            packed_query_states_,
            packed_key_states_,
            packed_value_states,
            attention_mask=attention_mask,
        )
        if ps.sp_enabled:
            # Reverse the Ulysses sequence/head redistribution for the attention
            # output: gather heads and scatter the packed sequence.
            packed_attn_output = gather_heads_scatter_seq(
                packed_attn_output,
                seq_dim=0,
                head_dim=1,
                group=ps.ulysses_group,
            )
        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        attn_output_und = self.o_proj(packed_attn_output[packed_und_token_indexes])
        attn_output_gen = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])
        packed_attn_output_[packed_und_token_indexes] = attn_output_und
        packed_attn_output_[packed_gen_token_indexes] = attn_output_gen
        if not has_und_tokens:
            packed_attn_output_ = _fold_zero_anchors(packed_attn_output_, attn_output_und)
        if not has_gen_tokens:
            packed_attn_output_ = _fold_zero_anchors(packed_attn_output_, attn_output_gen)
        return packed_attn_output_

    def _masked_dense_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one native-GQA SDPA call to the active sample's complete packed sequence."""

        query = query_states.transpose(0, 1).unsqueeze(0)
        key = key_states.transpose(0, 1).unsqueeze(0)
        value = value_states.transpose(0, 1).unsqueeze(0)
        with sdpa_kernel(
            backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        ):
            output = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        return output.squeeze(0).transpose(0, 1).contiguous()

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        *,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(mode)
        if not is_gen:
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        else:
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)
            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)
            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim).to(torch.float32)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                packed_query_states[packed_vae_token_indexes]
            )
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                packed_key_states[packed_vae_token_indexes]
            )

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = _apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            if key_values_lens is None or packed_key_value_indexes is None:
                raise ValueError("key_values_lens and packed_key_value_indexes are required when cache is non-empty.")
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = total_key_value_tokens
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        if IS_CUDA_AVAILABLE:
            packed_attn_output, _ = flash_attention_forward(
                self,
                packed_query_states.transpose(0, 1).unsqueeze(0),
                merged_key_states.transpose(0, 1).unsqueeze(0),
                merged_value_states.transpose(0, 1).unsqueeze(0),
                attention_mask=None,
                dropout=0.0,
                is_causal=is_causal,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                # Inference owns its packed KV-cache layout and does not enter the
                # training-only module Ulysses redistribution.
                skip_ulysses=True,
            )
            packed_attn_output = packed_attn_output.squeeze(0)
        else:
            head_num = packed_query_states.shape[1]
            if is_causal:
                atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(packed_query_states.device)
                packed_attn_output = torch_npu.npu_fusion_attention(
                    packed_query_states,
                    merged_key_states,
                    merged_value_states,
                    head_num,
                    pse=None,
                    padding_mask=None,
                    atten_mask=atten_mask_npu,
                    scale=1.0 / math.sqrt(packed_query_states.shape[-1]),
                    keep_prob=1,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seq_lens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seq_lens_k[1:].cpu().numpy().tolist()),
                    sparse_mode=3,
                )[0]
            else:
                packed_attn_output = torch_npu.npu_fusion_attention(
                    packed_query_states,
                    merged_key_states,
                    merged_value_states,
                    head_num,
                    pse=None,
                    atten_mask=None,
                    scale=1.0 / math.sqrt(packed_query_states.shape[-1]),
                    keep_prob=1,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seq_lens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seq_lens_k[1:].cpu().numpy().tolist()),
                )[0]
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if not is_gen:
            packed_attn_output = self.o_proj(packed_attn_output)
        else:
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                packed_attn_output[packed_vae_token_indexes]
            )

        if update_past_key_values:
            if past_key_values is None:
                raise ValueError("past_key_values is required when update_past_key_values=True.")
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self._forward_packed_train(*args, **kwargs), None
        return self._forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTDecoderLayer(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig, layer_idx: int):
        super().__init__()
        self.self_attn = BagelQwen2MoTAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        has_und_tokens = int(packed_und_token_indexes.numel()) > 0
        has_gen_tokens = int(packed_gen_token_indexes.numel()) > 0
        normed_sequence_und = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        normed_sequence_gen = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_[packed_und_token_indexes] = normed_sequence_und
        packed_sequence_[packed_gen_token_indexes] = normed_sequence_gen
        if not has_und_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_und)
        if not has_gen_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_gen)

        packed_sequence_, _ = self.self_attn(
            packed_sequence=packed_sequence_,
            attention_mask=attention_mask,
            packed_position_cos=packed_position_cos,
            packed_position_sin=packed_position_sin,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        packed_sequence = residual + packed_sequence_

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        post_attn_und = self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        post_attn_gen = self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        mlp_und = self.mlp(post_attn_und)
        mlp_gen = self.mlp_moe_gen(post_attn_gen)
        packed_sequence_[packed_und_token_indexes] = mlp_und
        packed_sequence_[packed_gen_token_indexes] = mlp_gen
        if not has_und_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, post_attn_und, mlp_und)
        if not has_gen_tokens:
            packed_sequence_ = _fold_zero_anchors(packed_sequence_, post_attn_gen, mlp_gen)
        output = residual + packed_sequence_
        return output

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        *,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        is_gen = _check_packed_inference_mode(mode)
        residual = packed_query_sequence
        if not is_gen:
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        else:
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            total_key_value_tokens=total_key_value_tokens,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        if not is_gen:
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        else:
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self._forward_packed_train(*args, **kwargs), None
        return self._forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTBackbone(nn.Module):
    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [BagelQwen2MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelQwen2RotaryEmbedding(config=config)
        self.use_moe = "Mo" in config.layer_module

    def _forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_und_token_indexes: Optional[torch.Tensor] = None,
        packed_gen_token_indexes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        packed_position_cos = cos.squeeze(0)
        packed_position_sin = sin.squeeze(0)

        if self.use_moe:
            if packed_und_token_indexes is None:
                raise ValueError("packed_und_token_indexes is required for BAGEL MoT training.")
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
        else:
            packed_und_token_indexes = torch.arange(packed_sequence.shape[0], device=packed_sequence.device)
            packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                packed_sequence, _ = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    packed_sequence,
                    attention_mask,
                    packed_position_cos,
                    packed_position_sin,
                    packed_und_token_indexes,
                    packed_gen_token_indexes,
                )
            else:
                packed_sequence, _ = decoder_layer(
                    packed_sequence=packed_sequence,
                    attention_mask=attention_mask,
                    packed_position_cos=packed_position_cos,
                    packed_position_sin=packed_position_sin,
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            normed_sequence_und = self.norm(packed_sequence[packed_und_token_indexes])
            normed_sequence_gen = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            packed_sequence_[packed_und_token_indexes] = normed_sequence_und
            packed_sequence_[packed_gen_token_indexes] = normed_sequence_gen
            if int(packed_und_token_indexes.numel()) == 0:
                packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_und)
            if int(packed_gen_token_indexes.numel()) == 0:
                packed_sequence_ = _fold_zero_anchors(packed_sequence_, normed_sequence_gen)
            return packed_sequence_
        return self.norm(packed_sequence)

    def _forward_packed_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> BaseNavitOutputWithPast:
        is_gen = _check_packed_inference_mode(mode)
        query_device = packed_query_sequence.device
        packed_query_indexes = packed_query_indexes.to(device=query_device)
        packed_query_position_ids = packed_query_position_ids.to(device=query_device)
        if packed_key_value_indexes is not None:
            packed_key_value_indexes = packed_key_value_indexes.to(device=query_device)
        if packed_vae_token_indexes is not None:
            packed_vae_token_indexes = packed_vae_token_indexes.to(device=query_device)
        if packed_text_indexes is not None:
            packed_text_indexes = packed_text_indexes.to(device=query_device)
        if past_key_values is None:
            past_key_values = NaiveCache(len(self.layers))

        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        cache_has_values = past_key_values.key_cache[0] is not None
        if cache_has_values and key_values_lens is None:
            raise ValueError("key_values_lens is required when cache is non-empty.")
        effective_key_values_lens = key_values_lens + query_lens if cache_has_values else query_lens
        cu_seq_lens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
        cu_seq_lens_k = torch.nn.functional.pad(torch.cumsum(effective_key_values_lens, dim=0), (1, 0)).to(torch.int32)
        max_length_q = int(query_lens.max().item())
        max_length_k = int(effective_key_values_lens.max().item())
        total_key_value_tokens = int(effective_key_values_lens.sum().item())

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                total_key_value_tokens=total_key_value_tokens,
            )

        if not is_gen:
            packed_query_sequence = self.norm(packed_query_sequence)
        else:
            query_device = packed_query_sequence.device
            packed_text_indexes = packed_text_indexes.to(device=query_device)
            packed_vae_token_indexes = packed_vae_token_indexes.to(device=query_device)
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def forward(self, *args: Any, **kwargs: Any) -> BaseNavitOutputWithPast:
        if self.training:
            return BaseNavitOutputWithPast(packed_query_sequence=self._forward_packed_train(*args, **kwargs))
        return self._forward_packed_inference(*args, **kwargs)


__all__ = ["BaseNavitOutputWithPast", "BagelQwen2MoT", "InferenceMixin", "NaiveCache"]
