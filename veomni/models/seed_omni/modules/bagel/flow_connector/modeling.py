"""BAGEL flow connector.

``embed_latent`` projects patchified VAE latent tokens, timestep embeddings,
and latent position embeddings to MoT hidden width. ``decode_velocity`` projects
MoT hidden states back to patch-latent velocity tokens. Carrier selection,
dummy alignment, and loss computation live in the SeedOmni module mixin.

The flow-matching FSM ``generate``-family endpoints (``embed_context_latents``
/ ``prepare_denoise_query`` / ``decode_velocity_from_hidden`` / ``advance_denoise``)
and their shared :class:`~.generation_state.FlowGenerationState` live here too —
pure inference over an already-loaded ``BagelFlowConnector`` needs no VeOmni
training-graph machinery. ``accelerated.py`` only carries the training-graph
pre/forward/post hooks (carrier selection, SP token slicing, dummy alignment).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, get_tail_output_item, is_dummy, iter_desired_items
from ..sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_VAE_CONTEXT,
)
from .configuration import BagelFlowConnectorConfig
from .generation_state import FlowGenerationState
from .processing import flattened_position_ids, preprocess_context_latent_embed, unpatchify_latent_tokens


SIGNAL_IMAGE_COMPLETE = "image_complete"


def select_vae_context_latent_items(
    conversation_list: list[list[ConversationItem]] | None,
) -> list[ConversationItem]:
    """Select per-sample VAE context latents shared by training and inference hooks."""
    if conversation_list is None:
        raise ValueError("BAGEL flow connector requires conversation_list to select VAE context latents.")

    # VAE preprocessing already makes BAGEL_VAE_CONTEXT per-sample by
    # appending a dummy carrier only for samples without real VAE context.
    # Reuse those existing carriers here; do not append another dummy.
    return list(iter_desired_items(conversation_list, types=["image"], sources=[BAGEL_VAE_CONTEXT]))


def scatter_flow_latent_embeds(
    embed_items: list[ConversationItem],
    embed_lengths: list[int],
    latent_embeds: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Write flow latent embeds back onto carrier items (training + inference)."""
    offset = 0
    for item, length in zip(embed_items, embed_lengths, strict=True):
        item.value = latent_embeds[offset : offset + length].to(device=device, dtype=dtype)
        offset += length

        # Hand the upstream VAE dummy carrier off as a flow dummy anchor;
        # downstream MoT should not treat it as another VAE image span.
        if is_dummy(item):
            item.type = "output"
            item.role = "dummy"
            item.source = "bagel_flow_connector"
            item.meta = {}

    if offset != int(latent_embeds.shape[0]):
        raise RuntimeError("BAGEL flow connector latent count mismatch during embed scatter.")


class InferenceMixin:
    """Flow-matching denoise FSM (``embed_context_latents`` / ``prepare_denoise_query``
    / ``decode_velocity_from_hidden`` / ``advance_denoise``) — HF ``GenerationMixin``
    analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`BagelFlowConnector`'s bases: ``OmniPreTrainedModel`` ships a no-op
    ``reset_local_inference_state`` default (kept as a safety net for modules
    that don't need real inference state), and MRO resolves left-to-right —
    put second, that no-op would shadow the real one below.
    """

    def reset_local_inference_state(self) -> None:
        self._generation_state.reset()

    def embed_context_latents(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}

        batched = [conversation_list]
        embed_items = [item for item in select_vae_context_latent_items(batched) if not is_dummy(item)]
        if not embed_items:
            return {"conversation_list": conversation_list}

        inputs, embed_lengths = preprocess_context_latent_embed(
            embed_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.embed_latent(**inputs)
        scatter_flow_latent_embeds(
            embed_items,
            embed_lengths,
            outputs["latent_embeds"],
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation_list}

    def prepare_denoise_query(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            raise ValueError("BAGEL flow inference requires conversation_list.")

        state = self._generation_state
        if not state.initialized:
            self._generation_state.initialize(
                generation_kwargs or {},
                resolution=int(getattr(self.config, "resolution", 1024)),
                patch_latent_dim=int(self.config.patch_latent_dim),
                device=self._vae2llm_device,
            )

        x_t = state.latents
        timestep = state.current_timestep()
        timestep_tokens = state.current_timestep_tokens()
        position_ids = flattened_position_ids(
            state.grid_shape,
            max_latent_size=int(self.config.max_latent_size),
            device=x_t.device,
        )
        outputs = self.embed_latent(
            latents=x_t,
            position_ids=position_ids,
            timesteps=timestep_tokens,
        )
        query = outputs["latent_embeds"].to(device=self.device, dtype=self.dtype)
        timestep_meta = timestep.detach().to(device=query.device, dtype=torch.float32)

        item = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_VELOCITY])
        if item is None:
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=query,
                    role="assistant",
                    source=BAGEL_FLOW_QUERY,
                    meta={"timestep": timestep_meta},
                )
            )
        else:
            item.type = "output"
            item.role = "assistant"
            item.source = BAGEL_FLOW_QUERY
            item.value = query
            item.meta = {"timestep": timestep_meta}
        return {"conversation_list": conversation_list}

    def decode_velocity_from_hidden(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if conversation_list is None:
            raise ValueError("BAGEL flow inference requires conversation_list.")

        item = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_HIDDEN])
        if item is None or not torch.is_tensor(item.value):
            raise ValueError("BAGEL flow decode_velocity requires source='bagel_flow_hidden'.")

        hidden = item.value
        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.dim() != 2:
            raise ValueError(f"BAGEL flow decode_velocity expected rank-2 hidden states, got {tuple(hidden.shape)}.")
        if int(hidden.shape[-1]) != int(self.config.hidden_size):
            raise ValueError(
                "BAGEL flow decode_velocity hidden-size mismatch: "
                f"got {hidden.shape[-1]}, expected {self.config.hidden_size}."
            )

        outputs = self.decode_velocity(hidden_states=hidden)
        velocity = outputs["velocity"]
        item.type = "output"
        item.role = "assistant"
        item.source = BAGEL_FLOW_VELOCITY
        item.value = velocity.to(device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation_list}

    def advance_denoise(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if conversation_list is None:
            raise ValueError("BAGEL flow inference requires conversation_list.")

        item = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_VELOCITY])
        if item is None or not torch.is_tensor(item.value):
            raise ValueError("BAGEL flow advance requires source='bagel_flow_velocity'.")

        velocity = item.value
        if velocity.dim() == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        complete = self._generation_state.advance(velocity)

        if complete:
            return self._emit_final_latent(conversation_list)
        item.meta.pop("timestep", None)
        return {"conversation_list": conversation_list}

    def _emit_final_latent(
        self,
        conversation_list: list[ConversationItem],
    ) -> dict[str, Any]:
        x_t = self._generation_state.latents
        item = get_tail_output_item(conversation_list, sources=[BAGEL_FLOW_VELOCITY])
        latent = unpatchify_latent_tokens(
            x_t,
            self._generation_state.grid_shape,
            z_channels=int(self.config.z_channels),
            latent_patch_size=int(self.config.latent_patch_size),
        )
        if item is None:
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=latent,
                    role="assistant",
                    source=BAGEL_GENERATED_LATENT,
                    meta={},
                )
            )
        else:
            item.type = "output"
            item.role = "assistant"
            item.source = BAGEL_GENERATED_LATENT
            item.value = latent.to(device=self.device, dtype=self.dtype)
            item.meta = {}

        self._generation_state.reset()
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: SIGNAL_IMAGE_COMPLETE}


class BagelFlowConnector(InferenceMixin, OmniPreTrainedModel):
    config_class = BagelFlowConnectorConfig
    base_model_prefix = "bagel_flow_connector"
    main_input_name = "hidden_states"
    _no_split_modules: list[str] = []
    _supports_sdpa = True

    def __init__(self, config: BagelFlowConnectorConfig) -> None:
        super().__init__(config)
        self.time_embedder = TimestepEmbedder(config.hidden_size, config.timestep_frequency_embedding_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)
        self._generation_state = FlowGenerationState()
        self.post_init()
        nn.init.constant_(self.llm2vae.weight, 0)
        nn.init.constant_(self.llm2vae.bias, 0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, PositionEmbedding):
            module.reset_parameters()
            return
        super()._init_weights(module)

    @property
    def _vae2llm_device(self) -> torch.device:
        return self.vae2llm.weight.device

    @property
    def _llm2vae_device(self) -> torch.device:
        return self.llm2vae.weight.device

    @property
    def _pos_embed_device(self) -> torch.device:
        return self.latent_pos_embed.pos_embed.device

    @property
    def _time_embedder_device(self) -> torch.device:
        return self.time_embedder.mlp[0].weight.device

    def embed_latent(
        self,
        latents: torch.Tensor,
        position_ids: torch.LongTensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latents = latents.to(device=self._vae2llm_device, dtype=self.vae2llm.weight.dtype)
        position_ids = position_ids.to(device=self._pos_embed_device, dtype=torch.long).reshape(-1)
        timesteps = timesteps.to(device=self._time_embedder_device, dtype=torch.float32).reshape(-1)
        if position_ids.numel() != latents.shape[0]:
            raise ValueError("position_ids must have one value per latent token.")

        latent_embeds = self.vae2llm(latents)
        time_embeds = self.time_embedder(timesteps)
        pos_embeds = self.latent_pos_embed(position_ids)
        return {
            "latent_embeds": latent_embeds
            + time_embeds.to(device=latent_embeds.device, dtype=latent_embeds.dtype)
            + pos_embeds.to(device=latent_embeds.device, dtype=latent_embeds.dtype)
        }

    def decode_velocity(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = hidden_states.to(device=self._llm2vae_device)
        return {"velocity": self.llm2vae(hidden_states)}


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        first_weight = self.mlp[0].weight
        return self.mlp(t_freq.to(device=first_weight.device, dtype=first_weight.dtype))


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        new_data = _get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        if hasattr(self.pos_embed.data, "to_local"):
            self.pos_embed.data.to_local().copy_(new_data)
        else:
            self.pos_embed.data.copy_(new_data)

    def _init_weights(self) -> None:
        self.reset_parameters()

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids.to(device=self.pos_embed.device)]


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return torch.from_numpy(_get_2d_sincos_pos_embed_from_grid(embed_dim, grid)).float()


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


__all__ = [
    "BagelFlowConnector",
    "BagelFlowConnectorConfig",
    "InferenceMixin",
    "SIGNAL_IMAGE_COMPLETE",
    "select_vae_context_latent_items",
    "scatter_flow_latent_embeds",
]
