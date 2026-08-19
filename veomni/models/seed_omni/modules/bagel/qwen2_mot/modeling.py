"""BAGEL Qwen2 MoT backbone."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.utils import ModelOutput

from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, get_tail_output_item
from ..sources import BAGEL_FLOW_HIDDEN, BAGEL_FLOW_QUERY, BAGEL_FLOW_VELOCITY
from .configuration import BagelQwen2MoTConfig
from .generation_state import MotGenerationState
from .masking import build_mot_sdpa_mask
from .processing import PackedConversation, preprocess_mot_inputs


_FLASH_ATTENTION_2 = "veomni_flash_attention_2_with_sp"


@contextmanager
def _temporary_attention_implementation(
    config: BagelQwen2MoTConfig,
    implementation: Optional[str],
) -> Iterator[None]:
    if implementation is None:
        yield
        return

    previous = config._attn_implementation
    config._attn_implementation = implementation
    try:
        yield
    finally:
        config._attn_implementation = previous


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
        attention_implementation: Optional[str] = None,
        packed_attention_metadata: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            "packed_attention_metadata": packed_attention_metadata,
        }

        is_gen = _check_packed_inference_mode(mode)
        if is_gen:
            call_kwargs["packed_vae_token_indexes"] = packed_vae_token_indexes
            call_kwargs["packed_text_indexes"] = packed_text_indexes

        with _temporary_attention_implementation(self.config, attention_implementation):
            output = self.model.forward_packed_inference(**call_kwargs)

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
        # Generation/edit rebuilds all CFG prompt caches. Understanding reuses
        # the main cache and switches to one-token AR decode after prefill.
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

        # All active CFG branches share this denoise query. Stack them into one
        # packed FA call; caches are read-only during denoising.
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
            attention_implementation=_FLASH_ATTENTION_2,
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

        # Restore branch identity from the layout recorded by denoise_branch,
        # strip marker rows, and replace the stacked tensor with guided velocity.
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
        main_packed = preprocess_mot_inputs(
            [conversation_list],
            device=self.device,
            dtype=self.dtype,
            hidden_size=int(self.config.hidden_size),
        )
        if main_packed is None:
            raise ValueError("BAGEL Qwen2-MoT generate requires at least one embedded text/image item.")

        state = self._generation_state
        state.main.reset()
        state.cfg_text.reset()
        state.cfg_img.reset()

        # Text CFG reuses the latest valid prefix of the main prompt: its cache
        # boundary is before a text span and after an image span.
        cfg_text_slice, cfg_text_next_position_id = self._cfg_text_prefix_slice(main_packed)
        use_cfg_img = state.cfg_img_requested(generation_kwargs)

        # Image CFG keeps logical text spans only. Inspect the grouped spans so
        # marker tokens belonging to image triplets are not treated as text.
        cfg_img_items = []
        if use_cfg_img:
            cfg_img_items = [item for span in main_packed.spans if span.item.type == "text" for item in span.items]

        # Pack main and cfg_img as independent logical documents. The attention
        # metadata prevents cross-document attention inside the Flex kernel.
        packed = main_packed
        if cfg_img_items:
            packed = preprocess_mot_inputs(
                [conversation_list, cfg_img_items],
                device=self.device,
                dtype=self.dtype,
                hidden_size=int(self.config.hidden_size),
            )
            if packed is None:
                raise RuntimeError("BAGEL Qwen2-MoT CFG prompt packing produced no tokens.")

        main_length = sum(main_packed.sample_splits[0])
        total_length = int(packed.packed_sequence.shape[0])
        main_slice = slice(0, main_length)
        cfg_img_slice = slice(main_length, total_length)

        # This is a fresh-cache prefill: forward_inference allocates the cache,
        # while packed attention metadata selects FlexAttention.
        outputs = self.forward_packed_prefill(packed)
        packed_cache = outputs["past_key_values"]

        # Detach main and cfg_img from the temporary packed cache allocation.
        # cfg_text is a read-only prefix view of main and can share its storage.
        main_cache = self._slice_prefill_cache(packed_cache, main_slice, clone=True)
        cfg_text_cache = self._slice_prefill_cache(main_cache, cfg_text_slice, clone=False)

        state.main.install_cache(
            cache=main_cache,
            cache_len=main_length,
            next_position_id=self._next_position_id(packed.packed_position_ids[main_slice]),
            device=self.device,
        )
        state.cfg_text.install_cache(
            cache=cfg_text_cache,
            cache_len=cfg_text_slice.stop,
            next_position_id=cfg_text_next_position_id,
            device=self.device,
        )
        if use_cfg_img:
            cfg_img_cache = self._slice_prefill_cache(packed_cache, cfg_img_slice, clone=True)
            state.cfg_img.install_cache(
                cache=cfg_img_cache,
                cache_len=cfg_img_slice.stop - cfg_img_slice.start,
                next_position_id=self._next_position_id(packed.packed_position_ids[cfg_img_slice]),
                device=self.device,
            )

        # cfg_img is an auxiliary packed document; graph consumers should only
        # observe hidden states from the main prompt.
        hidden_states = outputs["hidden_states"]
        if int(hidden_states.shape[0]) != total_length:
            raise RuntimeError(
                "BAGEL Qwen2-MoT FlexAttention prefill returned an unexpected sequence length: "
                f"expected {total_length}, got {hidden_states.shape[0]}."
            )
        return hidden_states[main_slice]

    def forward_packed_prefill(self, packed: PackedConversation) -> dict[str, Any]:
        """Run one fresh-cache Flex prefill over all logical documents."""
        total_length = int(packed.packed_sequence.shape[0])
        document_lens = [sum(split_lens) for split_lens in packed.sample_splits]
        query_lens = torch.tensor(document_lens, device=self.device, dtype=torch.int32)

        # A packed prefill may contain both understanding and generation expert
        # tokens even though it executes as one model forward.
        packed_gen_token_indexes = torch.nonzero(
            packed.packed_token_type_ids == 1,
            as_tuple=False,
        ).flatten()
        mode = "gen" if int(packed_gen_token_indexes.numel()) > 0 else "und"
        call_kwargs: dict[str, Any] = {
            "packed_query_sequence": packed.packed_sequence,
            "query_lens": query_lens,
            "packed_query_position_ids": packed.packed_position_ids,
            "packed_query_indexes": torch.arange(total_length, device=self.device, dtype=torch.long),
            "update_past_key_values": True,
            "is_causal": False,
            "mode": mode,
            "packed_attention_metadata": packed.packed_attention_metadata,
        }
        if mode == "gen":
            call_kwargs["packed_text_indexes"] = torch.nonzero(
                packed.packed_token_type_ids == 0,
                as_tuple=False,
            ).flatten()
            call_kwargs["packed_vae_token_indexes"] = packed_gen_token_indexes
        return self.forward_inference(**call_kwargs)

    def _decode_next_token(self, conversation_list: list[ConversationItem]) -> torch.Tensor:
        # AR has one query token and an existing contiguous cache, so the
        # dedicated FlashAttention path is preferable to rebuilding a packed mask.
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
            attention_implementation=_FLASH_ATTENTION_2,
        )
        main_context.append_packed_query(
            cache=outputs["past_key_values"],
            query_lens=query_lens,
            device=self.device,
        )

        return outputs["hidden_states"]

    def _cfg_text_prefix_slice(self, packed: PackedConversation) -> tuple[slice, torch.Tensor]:
        # Moving across an image keeps the image in text-CFG conditioning;
        # encountering text moves the boundary to just before that text span.
        prefix_end = 0
        next_position_id = torch.zeros(1, device=self.device, dtype=torch.long)
        for span in packed.spans:
            span_end = span.start + span.length
            if span.item.type == "text":
                prefix_end = span.start
                next_position_id = packed.packed_position_ids[span.start].reshape(1)
            elif span.item.type == "image":
                prefix_end = span_end
                next_position_id = self._next_position_id(packed.packed_position_ids[:span_end])
        return slice(0, prefix_end), next_position_id

    def _slice_prefill_cache(self, packed_cache: Any, rows: slice, *, clone: bool) -> Any:
        # main/cfg_img outlive the temporary packed cache and own their storage.
        # cfg_text is an immutable prefix of main, so it may use cheap views.
        cache = self._new_empty_cache()
        if rows.stop <= rows.start:
            return cache
        for layer_idx in cache.key_cache:
            key = packed_cache.key_cache[layer_idx][rows]
            value = packed_cache.value_cache[layer_idx][rows]
            cache.key_cache[layer_idx] = key.clone() if clone else key
            cache.value_cache[layer_idx] = value.clone() if clone else value
        return cache

    def _next_position_id(self, position_ids: torch.Tensor) -> torch.Tensor:
        if int(position_ids.numel()) == 0:
            return torch.zeros(1, device=self.device, dtype=torch.long)
        return position_ids.max().reshape(1).to(device=self.device, dtype=torch.long) + 1

    def _new_empty_cache(self) -> Any:
        return NaiveCache(len(self.model.layers))


class BagelQwen2MoT(InferenceMixin, OmniPreTrainedModel):
    config_class = BagelQwen2MoTConfig
    base_model_prefix = "bagel_qwen2_mot"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["BagelQwen2MoTDecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _export_hf_checkpoint_with_weight_conversions = False

    def __init__(self, config: BagelQwen2MoTConfig):
        super().__init__(config)
        self.model = BagelQwen2MoTBackbone(
            config,
            attention_cls=type(self).attention_cls,
            mlp_cls=type(self).mlp_cls,
            rms_norm_cls=type(self).rms_norm_cls,
        )
        self._generation_state = MotGenerationState()
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_token_type_ids: torch.Tensor,
        packed_attention_metadata: torch.Tensor,
    ) -> Dict[str, Any]:
        packed_und_token_indexes = torch.nonzero(packed_token_type_ids == 0, as_tuple=False).flatten()
        packed_gen_token_indexes = torch.nonzero(packed_token_type_ids == 1, as_tuple=False).flatten()
        output = self.model(
            packed_sequence=packed_sequence,
            packed_position_ids=packed_position_ids,
            packed_attention_metadata=packed_attention_metadata,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return {"hidden_states": output.packed_query_sequence}


class NaiveCache:
    """Official BAGEL packed KV cache."""

    def __init__(self, num_layers: int):
        self.key_cache = dict.fromkeys(range(num_layers))
        self.value_cache = dict.fromkeys(range(num_layers))


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
    """Eager packed RoPE. Fused dispatch lives on the accelerated attention class."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _sdpa_context(device: torch.device):
    """Pin mem-efficient SDPA on CUDA, matching official dense-train Bagel.

    Official 2D-mask training forces ``SDPBackend.EFFICIENT_ATTENTION`` so the
    dispatcher cannot pick MATH (OOM) or a newer CUDNN kernel. CPU/NPU keep the
    default dispatcher because Efficient is CUDA-only.
    """
    if device.type == "cuda":
        return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
    return nullcontext()


def _repeat_kv_heads(packed_states: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Expand packed KV from ``[tokens, kv_heads, dim]`` to query heads.

    Official dense-train Bagel repeats KV before Efficient SDPA. The CUDA
    Efficient kernel does not accept ``enable_gqa`` with a 2D mask.
    """
    num_kv_heads = int(packed_states.shape[1])
    if num_kv_heads == num_query_heads:
        return packed_states
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "BAGEL Qwen2-MoT GQA requires query heads divisible by KV heads: "
            f"num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}."
        )
    groups = num_query_heads // num_kv_heads
    packed_states = packed_states[:, :, None, :].repeat(1, 1, groups, 1)
    return packed_states.reshape(-1, num_query_heads, packed_states.shape[-1])


def _sdpa_packed_attention(
    packed_query_states: torch.Tensor,
    packed_key_states: torch.Tensor,
    packed_value_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_k: Optional[torch.Tensor] = None,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Packed SDPA returning ``[tokens, heads, head_dim]``.

    The 2D-mask path pins Efficient SDPA on CUDA. The varlen / ``is_causal``
    path is left unpinned: official inference uses ``flash_attn_varlen``, not
    Efficient.
    """
    if packed_query_states.shape[0] == 0:
        return packed_query_states.new_zeros(packed_query_states.shape)

    if attention_mask is not None:
        if enable_gqa:
            num_query_heads = int(packed_query_states.shape[1])
            packed_key_states = _repeat_kv_heads(packed_key_states, num_query_heads)
            packed_value_states = _repeat_kv_heads(packed_value_states, num_query_heads)
        query = packed_query_states.transpose(0, 1).unsqueeze(0)
        key = packed_key_states.transpose(0, 1).unsqueeze(0)
        value = packed_value_states.transpose(0, 1).unsqueeze(0)
        with _sdpa_context(packed_query_states.device):
            output = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        return output.squeeze(0).transpose(0, 1).contiguous()

    if cu_seq_lens_q is None or cu_seq_lens_k is None:
        raise ValueError("BAGEL Qwen2-MoT SDPA without an attention mask requires cu_seq_lens_q/k.")

    outputs: list[torch.Tensor] = []
    for seq_idx in range(int(cu_seq_lens_q.shape[0]) - 1):
        query_start = int(cu_seq_lens_q[seq_idx].item())
        query_end = int(cu_seq_lens_q[seq_idx + 1].item())
        key_start = int(cu_seq_lens_k[seq_idx].item())
        key_end = int(cu_seq_lens_k[seq_idx + 1].item())
        if query_end <= query_start:
            continue
        query = packed_query_states[query_start:query_end].transpose(0, 1).unsqueeze(0)
        key = packed_key_states[key_start:key_end].transpose(0, 1).unsqueeze(0)
        value = packed_value_states[key_start:key_end].transpose(0, 1).unsqueeze(0)
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    if not outputs:
        return packed_query_states.new_zeros(packed_query_states.shape)
    return torch.cat(outputs, dim=0)


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


def _scatter_moe(
    packed: torch.Tensor,
    packed_und_token_indexes: torch.Tensor,
    packed_gen_token_indexes: torch.Tensor,
    und_value: torch.Tensor,
    gen_value: torch.Tensor,
    *,
    extra_und_anchors: tuple[torch.Tensor, ...] = (),
    extra_gen_anchors: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Write MoT und/gen expert outputs into a packed buffer and keep empty-route FSDP anchors."""
    packed[packed_und_token_indexes] = und_value
    packed[packed_gen_token_indexes] = gen_value
    if int(packed_und_token_indexes.numel()) == 0:
        packed = _fold_zero_anchors(packed, und_value, *extra_und_anchors)
    if int(packed_gen_token_indexes.numel()) == 0:
        packed = _fold_zero_anchors(packed, gen_value, *extra_gen_anchors)
    return packed


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
        # Snapshot the true fp32 table before ``fn`` can quantize it (``.to(bfloat16)``).
        # Casting first and then ``.float()`` keeps bf16 rounding, which is not official
        # mixed-precision train (buffers stay fp32) and is not bitwise with a fresh table.
        inv_freq = self.inv_freq.detach().to(dtype=torch.float32).clone()
        original_inv_freq = self.original_inv_freq.detach().to(dtype=torch.float32).clone()
        module = super()._apply(fn, recurse=recurse)
        device = self.inv_freq.device
        self.inv_freq = inv_freq.to(device=device)
        self.original_inv_freq = original_inv_freq.to(device=device)
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
    def __init__(
        self,
        config: BagelQwen2MoTConfig,
        layer_idx: int,
        *,
        rms_norm_cls: type[Qwen2RMSNorm] = Qwen2RMSNorm,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.query_size = self.num_heads * self.head_dim
        self.key_value_size = self.num_key_value_heads * self.head_dim
        self.qkv_split_sizes = (self.query_size, self.key_value_size, self.key_value_size)

        self._build_qkv_proj()
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm_moe_gen = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)

    def _build_qkv_proj(self) -> None:
        self.q_proj = nn.Linear(self.hidden_size, self.query_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.key_value_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.key_value_size, bias=True)
        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.query_size, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.key_value_size, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.key_value_size, bias=True)

    def _project_qkv(
        self,
        hidden: torch.Tensor,
        *,
        is_gen: bool,
        split: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_proj, key_proj, value_proj = (
            (self.q_proj_moe_gen, self.k_proj_moe_gen, self.v_proj_moe_gen)
            if is_gen
            else (self.q_proj, self.k_proj, self.v_proj)
        )
        query_states = query_proj(hidden)
        key_states = key_proj(hidden)
        value_states = value_proj(hidden)
        if split:
            return query_states, key_states, value_states
        return torch.cat((query_states, key_states, value_states), dim=-1)

    @staticmethod
    def build_attention_mask(packed_attention_metadata: torch.Tensor) -> torch.Tensor:
        return build_mot_sdpa_mask(packed_attention_metadata)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    def project_qkv(
        self,
        packed_sequence: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_qkv_states = _scatter_moe(
            packed_sequence.new_zeros((packed_sequence.shape[0], sum(self.qkv_split_sizes))),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self._project_qkv(packed_sequence[packed_und_token_indexes], is_gen=False),
            self._project_qkv(packed_sequence[packed_gen_token_indexes], is_gen=True),
        )
        packed_query_states, packed_key_states, packed_value_states = packed_qkv_states.split(
            self.qkv_split_sizes, dim=-1
        )
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = _scatter_moe(
            packed_query_states.new_zeros(packed_query_states.shape),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self.q_norm(packed_query_states[packed_und_token_indexes]),
            self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes]),
        )
        packed_key_states = _scatter_moe(
            packed_key_states.new_zeros(packed_key_states.shape),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self.k_norm(packed_key_states[packed_und_token_indexes]),
            self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes]),
        )
        packed_query_states, packed_key_states = self.apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_position_cos,
            packed_position_sin,
            unsqueeze_dim=1,
        )
        return packed_query_states, packed_key_states, packed_value_states

    def attend(
        self,
        packed_query_states: torch.Tensor,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        packed_attn_output = _sdpa_packed_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask=attention_mask,
            scale=self.head_dim**-0.5,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        return packed_attn_output.reshape(-1, self.num_heads * self.head_dim)

    def project_o(
        self,
        packed_attn_output: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        return _scatter_moe(
            packed_attn_output.new_zeros(packed_attn_output.shape),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self.o_proj(packed_attn_output[packed_und_token_indexes]),
            self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes]),
        )

    def forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        packed_query_states, packed_key_states, packed_value_states = self.project_qkv(
            packed_sequence,
            packed_und_token_indexes,
            packed_gen_token_indexes,
            packed_position_cos,
            packed_position_sin,
        )
        packed_attn_output = self.attend(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
        )
        return self.project_o(
            packed_attn_output,
            packed_und_token_indexes,
            packed_gen_token_indexes,
        )

    def project_qkv_inference(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project packed inference tokens through the matching MoT attention expert."""
        if not is_gen:
            packed_query_states, packed_key_states, packed_value_states = self._project_qkv(
                packed_query_sequence, is_gen=False, split=True
            )
            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
            return packed_query_states, packed_key_states, packed_value_states

        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
        packed_qkv_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], sum(self.qkv_split_sizes))
        )
        packed_qkv_states[packed_text_indexes] = self._project_qkv(
            packed_query_sequence[packed_text_indexes], is_gen=False
        )
        packed_qkv_states[packed_vae_token_indexes] = self._project_qkv(
            packed_query_sequence[packed_vae_token_indexes], is_gen=True
        )
        packed_query_states, packed_key_states, packed_value_states = packed_qkv_states.split(
            self.qkv_split_sizes, dim=-1
        )
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim).to(torch.float32)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
        packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_vae_token_indexes]
        )
        packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
        packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])
        return packed_query_states, packed_key_states, packed_value_states

    def merge_kv_cache(
        self,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        *,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache],
        key_values_lens: Optional[torch.Tensor],
        packed_key_value_indexes: Optional[torch.Tensor],
        total_key_value_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the current query span with this layer's packed KV cache."""
        if past_key_values is None or past_key_values.key_cache[self.layer_idx] is None:
            return packed_key_states, packed_value_states
        if key_values_lens is None or packed_key_value_indexes is None:
            raise ValueError("key_values_lens and packed_key_value_indexes are required when cache is non-empty.")

        past_key_states = past_key_values.key_cache[self.layer_idx]
        past_value_states = past_key_values.value_cache[self.layer_idx]
        merged_key_states = past_key_states.new_zeros(
            (total_key_value_tokens, self.num_key_value_heads, self.head_dim)
        )
        merged_value_states = past_key_states.new_zeros(
            (total_key_value_tokens, self.num_key_value_heads, self.head_dim)
        )
        merged_key_states[packed_query_indexes] = packed_key_states
        merged_key_states[packed_key_value_indexes] = past_key_states
        merged_value_states[packed_query_indexes] = packed_value_states
        merged_value_states[packed_key_value_indexes] = past_value_states
        return merged_key_states, merged_value_states

    def attend_inference(
        self,
        packed_query_states: torch.Tensor,
        merged_key_states: torch.Tensor,
        merged_value_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        del max_length_q, max_length_k
        return _sdpa_packed_attention(
            packed_query_states,
            merged_key_states,
            merged_value_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            scale=self.head_dim**-0.5,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )

    def project_o_inference(
        self,
        packed_attn_output: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Project packed attention outputs through the matching MoT output expert."""
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if not is_gen:
            return self.o_proj(packed_attn_output)

        packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_vae_token_indexes]
        )
        return packed_attn_output

    def forward_packed_inference(
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
        attention_mask: Optional[torch.Tensor] = None,
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
        packed_query_states, packed_key_states, packed_value_states = self.project_qkv_inference(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = self.apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        merged_key_states, merged_value_states = self.merge_kv_cache(
            packed_key_states,
            packed_value_states,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            total_key_value_tokens=total_key_value_tokens,
        )
        packed_attn_output = self.attend_inference(
            packed_query_states,
            merged_key_states,
            merged_value_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        packed_attn_output = self.project_o_inference(
            packed_attn_output,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )

        if update_past_key_values:
            if past_key_values is None:
                raise ValueError("past_key_values is required when update_past_key_values=True.")
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self.forward_packed_train(*args, **kwargs), None
        return self.forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BagelQwen2MoTConfig,
        layer_idx: int,
        *,
        attention_cls: type[BagelQwen2MoTAttention] = BagelQwen2MoTAttention,
        mlp_cls: type[Qwen2MLP] = Qwen2MLP,
        rms_norm_cls: type[Qwen2RMSNorm] = Qwen2RMSNorm,
    ):
        super().__init__()
        self.self_attn = attention_cls(config, layer_idx, rms_norm_cls=rms_norm_cls)
        self.mlp = mlp_cls(config)
        self.mlp_moe_gen = mlp_cls(config)
        self.input_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def _apply_moe_norm(
        self,
        packed_sequence: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
        und_norm: nn.Module,
        gen_norm: nn.Module,
    ) -> torch.Tensor:
        return _scatter_moe(
            packed_sequence.new_zeros(packed_sequence.shape),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            und_norm(packed_sequence[packed_und_token_indexes]),
            gen_norm(packed_sequence[packed_gen_token_indexes]),
        )

    def forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_position_cos: torch.Tensor,
        packed_position_sin: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence = self._apply_moe_norm(
            packed_sequence,
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self.input_layernorm,
            self.input_layernorm_moe_gen,
        )
        packed_sequence, _ = self.self_attn(
            packed_sequence=packed_sequence,
            attention_mask=attention_mask,
            packed_position_cos=packed_position_cos,
            packed_position_sin=packed_position_sin,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        packed_sequence = residual + packed_sequence

        residual = packed_sequence
        post_attn_und = self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        post_attn_gen = self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence = _scatter_moe(
            packed_sequence.new_zeros(packed_sequence.shape),
            packed_und_token_indexes,
            packed_gen_token_indexes,
            self.mlp(post_attn_und),
            self.mlp_moe_gen(post_attn_gen),
            extra_und_anchors=(post_attn_und,),
            extra_gen_anchors=(post_attn_gen,),
        )
        return residual + packed_sequence

    def _apply_inference_input_norm(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize inference tokens with the matching MoT expert."""
        if not is_gen:
            return self.input_layernorm(packed_query_sequence)

        normed_sequence = torch.zeros_like(packed_query_sequence)
        normed_sequence[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
        normed_sequence[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        )
        return normed_sequence

    def _apply_inference_mlp(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the inference MLP with text and VAE tokens routed independently."""
        if not is_gen:
            return self.mlp(self.post_attention_layernorm(packed_query_sequence))

        packed_text_query_sequence = self.post_attention_layernorm(packed_query_sequence[packed_text_indexes]).to(
            torch.bfloat16
        )
        packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(
            packed_query_sequence[packed_vae_token_indexes]
        ).to(torch.bfloat16)
        mlp_output = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
        mlp_output[packed_text_indexes] = self.mlp(packed_text_query_sequence)
        mlp_output[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
        return mlp_output

    def forward_packed_inference(
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
        attention_mask: Optional[torch.Tensor] = None,
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
        packed_query_sequence = self._apply_inference_input_norm(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
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
            attention_mask=attention_mask,
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
        packed_query_sequence = self._apply_inference_mlp(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        if self.training:
            return self.forward_packed_train(*args, **kwargs), None
        return self.forward_packed_inference(*args, **kwargs)


class BagelQwen2MoTBackbone(nn.Module):
    def __init__(
        self,
        config: BagelQwen2MoTConfig,
        *,
        attention_cls: type[BagelQwen2MoTAttention] = BagelQwen2MoTAttention,
        mlp_cls: type[Qwen2MLP] = Qwen2MLP,
        rms_norm_cls: type[Qwen2RMSNorm] = Qwen2RMSNorm,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [
                BagelQwen2MoTDecoderLayer(
                    config,
                    layer_idx,
                    attention_cls=attention_cls,
                    mlp_cls=mlp_cls,
                    rms_norm_cls=rms_norm_cls,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelQwen2RotaryEmbedding(config=config)
        self.use_moe = "Mo" in config.layer_module
        self.attention_cls = attention_cls

    def _build_inference_attention_mask(
        self,
        packed_query_sequence: torch.Tensor,
        packed_attention_metadata: Optional[torch.Tensor],
        *,
        cache_has_values: bool,
    ) -> Any:
        """Validate packed-prefill invariants and materialize its attention mask."""
        if packed_attention_metadata is None:
            return None
        if cache_has_values:
            raise ValueError("BAGEL packed prefill requires an empty KV cache.")

        expected_metadata_shape = (3, int(packed_query_sequence.shape[0]))
        if tuple(packed_attention_metadata.shape) != expected_metadata_shape:
            raise ValueError(
                "BAGEL packed prefill metadata must match the packed query sequence: "
                f"expected {expected_metadata_shape}, got {tuple(packed_attention_metadata.shape)}."
            )
        return self.attention_cls.build_attention_mask(
            packed_attention_metadata.to(device=packed_query_sequence.device)
        )

    def _apply_inference_final_norm(
        self,
        packed_query_sequence: torch.Tensor,
        *,
        is_gen: bool,
        packed_text_indexes: Optional[torch.Tensor],
        packed_vae_token_indexes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the final backbone norm with the matching MoT expert."""
        if not is_gen:
            return self.norm(packed_query_sequence)

        normed_sequence = torch.zeros_like(packed_query_sequence)
        normed_sequence[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
        normed_sequence[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
        return normed_sequence

    def forward_packed_train(
        self,
        packed_sequence: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_attention_metadata: torch.Tensor,
        packed_und_token_indexes: Optional[torch.Tensor] = None,
        packed_gen_token_indexes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = self.attention_cls.build_attention_mask(packed_attention_metadata)
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
            return _scatter_moe(
                packed_sequence.new_zeros(packed_sequence.shape),
                packed_und_token_indexes,
                packed_gen_token_indexes,
                self.norm(packed_sequence[packed_und_token_indexes]),
                self.norm_moe_gen(packed_sequence[packed_gen_token_indexes]),
            )
        return self.norm(packed_sequence)

    def forward_packed_inference(
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
        packed_attention_metadata: Optional[torch.Tensor] = None,
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
        attention_mask = self._build_inference_attention_mask(
            packed_query_sequence,
            packed_attention_metadata,
            cache_has_values=cache_has_values,
        )

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
                attention_mask=attention_mask,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                total_key_value_tokens=total_key_value_tokens,
            )

        packed_query_sequence = self._apply_inference_final_norm(
            packed_query_sequence,
            is_gen=is_gen,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def forward(self, *args: Any, **kwargs: Any) -> BaseNavitOutputWithPast:
        if self.training:
            return BaseNavitOutputWithPast(packed_query_sequence=self.forward_packed_train(*args, **kwargs))
        return self.forward_packed_inference(*args, **kwargs)


BagelQwen2MoT.attention_cls = BagelQwen2MoTAttention
BagelQwen2MoT.mlp_cls = Qwen2MLP
BagelQwen2MoT.rms_norm_cls = Qwen2RMSNorm


__all__ = ["BaseNavitOutputWithPast", "BagelQwen2MoT", "InferenceMixin", "NaiveCache"]
