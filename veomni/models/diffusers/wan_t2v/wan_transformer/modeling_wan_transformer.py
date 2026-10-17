from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as _WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)
from .....utils import logging
from .....utils.device import IS_NPU_AVAILABLE, get_device_type
from .configuration_wan_transformer import WanTransformer3DModelConfig


try:
    import torch_npu
except ImportError:
    torch_npu = None


logger = logging.get_logger(__name__)


def wan_eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=False
    )
    return attn_output.transpose(1, 2), None


class WanAttentionKernelModule:
    def __init__(self, config: SimpleNamespace, attn: WanAttention):
        target_dtype = _get_weight_dtype(attn.to_q)
        if target_dtype == torch.float32:
            target_dtype = torch.bfloat16
        self.config = SimpleNamespace(
            _attn_implementation=config._attn_implementation,
            _pre_quantization_dtype=target_dtype,
        )
        self.is_causal = False
        self.layer_idx = getattr(attn, "layer_idx", None)
        self._attn = attn

    def modules(self):
        return self._attn.modules()


def _get_wan_full_sequence_varlen_kwargs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int | None = None,
    key_length: int | None = None,
) -> dict[str, torch.Tensor | int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Wan flash-attention expects Q/K/V tensors with shape (batch, seq, heads, head_dim).")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            "Wan flash-attention expects matching Q/K/V batch sizes; "
            f"got {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
        )

    batch_size = query.shape[0]
    query_length = query.shape[1] if query_length is None else query_length
    key_length = key.shape[1] if key_length is None else key_length
    _q_device = "cpu" if get_device_type() == "npu" else query.device
    _k_device = "cpu" if get_device_type() == "npu" else key.device
    cu_seq_lens_q = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        device=_q_device,
        dtype=torch.int32,
    )
    cu_seq_lens_k = torch.arange(
        0,
        (batch_size + 1) * key_length,
        key_length,
        device=_k_device,
        dtype=torch.int32,
    )
    return {
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_k,
        "max_length_q": query_length,
        "max_length_k": key_length,
    }


def _get_weight_dtype(module):
    if hasattr(module, "weight"):
        return module.weight.dtype
    if hasattr(module, "base_layer"):
        return module.base_layer.weight.dtype
    raise AttributeError(f"Cannot determine weight dtype for {type(module)}")


def _assert_wan_flash_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn: WanAttention,
) -> None:
    tensor_dtypes = {
        "query": query.dtype,
        "key": key.dtype,
        "value": value.dtype,
    }
    weight_dtypes = {
        "to_q.weight": _get_weight_dtype(attn.to_q),
        "to_k.weight": _get_weight_dtype(attn.to_k),
        "to_v.weight": _get_weight_dtype(attn.to_v),
    }
    assert all(dtype == torch.bfloat16 for dtype in tensor_dtypes.values()), (
        f"Wan flash-attention expects bf16 Q/K/V tensors, got {tensor_dtypes}."
    )
    assert all(dtype == torch.bfloat16 for dtype in weight_dtypes.values()), (
        f"Wan flash-attention expects bf16 projection weights, got {weight_dtypes}."
    )


class WanSPAttnProcessor(WanAttnProcessor):
    def __init__(self, attn_implementation: str):
        self.attn_implementation = attn_implementation
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        super().__init__()

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        # I2V: the first part of encoder_hidden_states holds image context.
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if IS_NPU_AVAILABLE and torch_npu is not None and query.device.type == "npu":
            query = torch_npu.npu_rms_norm(query, attn.norm_q.weight, epsilon=attn.norm_q.eps)[0]
            key = torch_npu.npu_rms_norm(key, attn.norm_k.weight, epsilon=attn.norm_k.eps)[0]
        else:
            query = attn.norm_q(query)
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        use_sp = get_parallel_state().sp_enabled and not is_cross_attention
        padded_seq_len = None
        if use_sp:
            group = get_parallel_state().ulysses_group
            # Put sequence/heads first to use the existing all_to_all_single
            # path, which also works on Gloo builds without list all-to-all.
            query, key, value = (
                gather_seq_scatter_heads(tensor.movedim(0, 2), seq_dim=0, head_dim=1, group=group).movedim(2, 0)
                for tensor in (query, key, value)
            )
            if rotary_emb is None:
                raise ValueError("Wan sequence parallel self-attention requires full rotary position tables.")
            sequence_length = rotary_emb[0].shape[1]
            padded_seq_len = query.shape[1]
            # Padding is only for communication; it must not enter softmax.
            query, key, value = (tensor[:, :sequence_length] for tensor in (query, key, value))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            def apply_rotary_emb_npu(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                cos = freqs_cos[..., 0::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
                sin = freqs_sin[..., 1::2].to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
                x_float = hidden_states.to(torch.float32)
                x_out = torch_npu.npu_rotary_mul(x_float, cos, sin, rotary_mode="interleave")
                return x_out.to(hidden_states.dtype)

            _apply_rope = (
                apply_rotary_emb_npu
                if IS_NPU_AVAILABLE and torch_npu is not None and query.device.type == "npu"
                else apply_rotary_emb
            )
            query = _apply_rope(query, *rotary_emb)
            key = _apply_rope(key, *rotary_emb)

        attention_interface: Callable = wan_eager_attention_forward
        use_flash_attention = self.attn_implementation in {"flash_attention_2", "veomni_flash_attention_2_with_sp"}
        if use_flash_attention:
            _assert_wan_flash_attention_bf16(query, key, value, attn)
        if self.attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        kernel_module = WanAttentionKernelModule(self.config, attn)

        # I2V: additional cross-attention over image tokens (no Ulysses SP needed).
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            if use_flash_attention:
                assert key_img.dtype == torch.bfloat16 and value_img.dtype == torch.bfloat16, (
                    "Wan image flash-attention expects bf16 added K/V tensors, "
                    f"got key={key_img.dtype}, value={value_img.dtype}."
                )
            image_attention_kwargs = (
                _get_wan_full_sequence_varlen_kwargs(query, key_img, value_img) if use_flash_attention else {}
            )
            hidden_states_img = attention_interface(
                kernel_module,
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                attention_mask=None,
                dropout=0.0,
                is_causal=False,
                skip_ulysses=True,
                **image_attention_kwargs,
            )[0]
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        attention_kwargs = _get_wan_full_sequence_varlen_kwargs(query, key, value) if use_flash_attention else {}
        hidden_states_out = attention_interface(
            kernel_module,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_mask=attention_mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=True,
            **attention_kwargs,
        )[0]

        if use_sp:
            hidden_states_out = F.pad(hidden_states_out, (0, 0, 0, 0, 0, padded_seq_len - hidden_states_out.shape[1]))
            hidden_states_out = gather_heads_scatter_seq(
                hidden_states_out.movedim(0, 2), head_dim=1, seq_dim=0, group=get_parallel_state().ulysses_group
            ).movedim(2, 0)

        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.type_as(query)

        if hidden_states_img is not None:
            hidden_states_out = hidden_states_out + hidden_states_img

        hidden_states_out = hidden_states_out.to(_get_weight_dtype(attn.to_out[0]))
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out


def _wan_block_ckpt_wrapper(block):
    """Return a callable compatible with VeOmni's gradient checkpointing.

    VeOmni's ``_gradient_checkpointing_func`` passes an ``early_stop`` kwarg that
    diffusers' ``WanTransformerBlock.forward()`` does not accept; this wrapper
    absorbs and ignores it.
    """

    def custom_forward(*args, early_stop=None, **kwargs):
        return block(*args, **kwargs)

    return custom_forward


def WanTransformer3DModel_forward(
    self: _WanTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    image_latents: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    **kwargs,
):
    parallel_state = get_parallel_state()
    if parallel_state.cp_enabled:
        raise NotImplementedError(
            "WanTransformer3DModel supports Ulysses sequence parallelism, but not context parallelism; "
            "set model.accelerator.cp_size=1."
        )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    if num_channels != self.config.in_channels:
        mask_channels = self.config.in_channels - 2 * num_channels
        if num_channels != self.config.out_channels or mask_channels <= 0:
            raise ValueError("Wan noisy latent channels do not match the transformer configuration.")
        if image_latents is None or mask is None:
            raise ValueError("Wan I2V requires full image_latents and mask from the condition model.")
        if image_latents.shape != hidden_states.shape:
            raise ValueError(
                "Wan I2V image_latents must match the full noisy video shape; first-frame repetition is invalid."
            )
        expected_mask = (batch_size, mask_channels, num_frames, height, width)
        if tuple(mask.shape) != expected_mask:
            raise ValueError(f"Wan I2V mask shape must be {expected_mask}, got {tuple(mask.shape)}.")
        # Match WanImageToVideoPipeline: noisy latents, temporal mask, VAE condition.
        hidden_states = torch.cat([hidden_states, mask.to(hidden_states), image_latents.to(hidden_states)], dim=1)
        num_channels = hidden_states.shape[1]

    p_t, p_h, p_w = self.config.patch_size
    if num_frames % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"Wan latent dimensions must be divisible by patch_size {self.config.patch_size}, "
            f"got {(num_frames, height, width)}."
        )
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 1. Rotary position embeddings for the full sequence
    rotary_emb = self.rope(hidden_states)
    # 2. Patch embedding: (B, C, F, H, W) → (B, seq, inner_dim)
    hidden_states = self.patch_embedding(hidden_states)
    if IS_NPU_AVAILABLE and torch_npu is not None and hidden_states.device.type == "npu":
        B, C = hidden_states.shape[:2]
        S = hidden_states.shape[2:].numel()
        hidden_states = torch_npu.npu_confusion_transpose(hidden_states, [0, 2, 1], [B, C, S], transpose_first=False)
    else:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # 3. Condition embedding
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if parallel_state.sp_enabled:
        hidden_states = slice_input_tensor(hidden_states, dim=1, group=parallel_state.sp_group)

        if ts_seq_len is not None:
            timestep_proj = slice_input_tensor(timestep_proj, dim=1, group=parallel_state.sp_group)

        # Keep full positions: self-attention gathers the sequence and discards
        # communication padding before applying RoPE and computing attention.

    # Cast rotary_emb to compute dtype for checkpoint dtype consistency.
    # WanRotaryPosEmbed stores frequencies in float64/float32, but FSDP2
    # cast_forward_inputs casts checkpoint args to param_dtype during the
    # first forward while recomputation uses the original high-precision
    # tensors, causing CheckpointError with use_reentrant=False.
    rotary_emb = tuple(f.to(hidden_states.dtype) for f in rotary_emb)

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            # Wrap the block to absorb VeOmni's `early_stop` kwarg that
            # _gradient_checkpointing_func injects but diffusers blocks
            # do not accept.
            hidden_states = self._gradient_checkpointing_func(
                _wan_block_ckpt_wrapper(block),
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # SP: gather before output head – every rank holds the full sequence so
    # that the loss is identical across SP ranks.
    if parallel_state.sp_enabled:
        hidden_states = gather_outputs(
            hidden_states,
            gather_dim=1,
            padding_dim=1,
            unpad_dim_size=post_patch_num_frames * post_patch_height * post_patch_width,
            group=parallel_state.sp_group,
        )

    # 5. Output: norm → projection → unpatchify
    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    # Return a plain tensor to simplify the DiT wrapper's forward.
    return output


@dataclass
class WanModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class _WanTransformerInitShim(_WanTransformer3DModel):
    """Breaks the init chain from PreTrainedModel to diffusers WanTransformer3DModel.

    When PreTrainedModel.__init__ calls super().__init__(), the MRO would normally
    invoke diffusers WanTransformer3DModel.__init__() with no arguments (default
    config), building a large default model that wastes memory. This shim intercepts
    that call and only runs nn.Module.__init__(), so the actual model is built once
    in WanTransformer3DModel.__init__ with the correct config.
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class WanTransformer3DModel(PreTrainedModel, _WanTransformerInitShim):
    config_class = WanTransformer3DModelConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["WanTransformerBlock"]

    def __init__(self, config: WanTransformer3DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        # Remove VeOmni-specific kwargs before passing to the diffusers init.
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _WanTransformer3DModel.__init__(self, **config.to_diffuser_dict())
        self.config: WanTransformer3DModelConfig = config
        self.config.tie_word_embeddings = False

        sp_processor = WanSPAttnProcessor(attn_implementation=config._attn_implementation)
        for block in self.blocks:
            block.attn1.set_processor(sp_processor)
            block.attn2.set_processor(sp_processor)

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def forward(
        self,
        latents: torch.Tensor | list[torch.Tensor],
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states_image: torch.Tensor | None = None,
        image_latents: torch.Tensor | list[torch.Tensor] | None = None,
        mask: torch.Tensor | list[torch.Tensor] | None = None,
        loss_start_frame: int | list[int] = 0,
    ):
        if not isinstance(hidden_states, list):
            latents = [latents]
            hidden_states = [hidden_states]
            timestep = [timestep]
            encoder_hidden_states = [encoder_hidden_states]
            training_target = [training_target]
            if image_latents is not None:
                image_latents = [image_latents]
            if mask is not None:
                mask = [mask]

        if not isinstance(loss_start_frame, list):
            loss_start_frame = [loss_start_frame] * len(hidden_states)
        if len(loss_start_frame) != len(hidden_states):
            raise ValueError("loss_start_frame must have one value per sample.")

        per_sample_losses = []
        predictions = []
        for i, (hs, ts, enc_hs, target) in enumerate(
            zip(hidden_states, timestep, encoder_hidden_states, training_target)
        ):
            # Per-sample I2V kwargs
            img_lat = image_latents[i] if image_latents is not None else None
            msk = mask[i] if mask is not None else None

            # Call the SP-patched diffusers forward for each sample.
            prediction = _WanTransformer3DModel.forward(
                self,
                hidden_states=hs,
                timestep=ts,
                encoder_hidden_states=enc_hs,
                encoder_hidden_states_image=encoder_hidden_states_image,
                image_latents=img_lat,
                mask=msk,
            )
            predictions.append(prediction)

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Wan prediction and target shapes must match, got {prediction.shape} and {target.shape}."
                )

            start = loss_start_frame[i]
            if not 0 <= start < prediction.shape[2]:
                raise ValueError("loss_start_frame must leave at least one target frame.")
            per_sample_loss = F.mse_loss(
                prediction[:, :, start:].float(), target[:, :, start:].float(), reduction="none"
            )
            per_sample_loss = per_sample_loss.reshape(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)

        loss = torch.cat(per_sample_losses).mean()
        return WanModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _WanTransformer3DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return _WanTransformer3DModel.from_pretrained(path, **kwargs)


def apply_veomni_wan_transformer_patch() -> None:
    """Monkey-patch ``_WanTransformer3DModel.forward`` with Ulysses SP support.

    The patch is structurally identical to the original diffusers forward but
    inserts two SP operations:

    1. **Sequence slice** (with gradient scaling) after patchification – each SP
       rank processes a contiguous chunk of video tokens.
    2. **Sequence gather** before the output head – all ranks see the full output
       so that the loss is identical across SP ranks.

    Slicing only takes effect when BOTH ``ulysses_enabled`` is True AND an SP-aware
    attention implementation (``veomni_flash_attention_*_with_sp``) is configured.
    Without the SP attention processor the required AllToAll in self-attention
    would be absent, making sequence slicing incorrect.
    """
    _WanTransformer3DModel.forward = WanTransformer3DModel_forward
    _WanTransformer3DModel._supports_flash_attn = True
    _WanTransformer3DModel._supports_flash_attn_2 = True
    logger.info_rank0("Applied VeOmni SP patch to WanTransformer3DModel.forward.")

    from veomni.models.transformers.wan.device_patch import apply_veomni_wan_device_patch

    apply_veomni_wan_device_patch()
