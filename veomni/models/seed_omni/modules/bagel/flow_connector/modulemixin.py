"""SeedOmni V2 carrier hooks for BAGEL's flow connector."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import ModuleMixin, post_forward, pre_forward
from ..carrier_updates import append as carrier_append
from ..carrier_updates import materialize_carrier_updates, meta_patch, replace_fields
from ..sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_VAE_CONTEXT,
)
from .configuration import BagelFlowConnectorConfig
from .generation_state import FlowGenerationState
from .processing import (
    active_output_item,
    flattened_position_ids,
    flow_hidden_items,
    flow_latent_items,
    prepare_context_embed_latent_inputs,
    prepare_decode_velocity_inputs,
    prepare_embed_latent_inputs,
    scatter_latent_embeds,
    scatter_velocity,
    single_inference_conversation,
    unpatchify_latent_tokens,
)


SIGNAL_IMAGE_COMPLETE = "image_complete"


class BagelFlowConnectorModuleMixin(ModuleMixin):
    """Carrier hooks for latent embedding and velocity projection."""

    config: BagelFlowConnectorConfig

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._embed_items: list[ConversationItem] = []
        self._embed_lengths: list[int] = []
        self._decode_items: list[ConversationItem] = []
        self._decode_lengths: list[int] = []
        self._decode_target: torch.Tensor | None = None
        self._generation_state = FlowGenerationState()

    def reset_local_inference_state(self) -> None:
        self._generation_state.reset()

    def generate(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        phase = self._resolve_denoise_phase(conversation_list, generation_kwargs or {}, kwargs)
        if phase == "prepare_query":
            return self._prepare_latent_query(conversation_list, generation_kwargs or {})
        if phase == "decode_velocity":
            return self._decode_velocity_from_hidden(conversation_list)
        if phase == "advance":
            return self._advance_denoise(conversation_list)
        raise RuntimeError(f"Unsupported BAGEL flow connector inference phase: {phase!r}")

    def _resolve_denoise_phase(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
        generation_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> str:
        del generation_kwargs, kwargs
        state = self._generation_state
        if state.phase == "advance":
            conversation = single_inference_conversation(conversation_list)
            item = active_output_item(conversation, sources={BAGEL_FLOW_VELOCITY})
            if item is not None and not torch.is_tensor(item.value):
                return "prepare_query"
        return state.phase

    def _prepare_latent_query(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
        generation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        conversation = single_inference_conversation(conversation_list)
        state = self._generation_state
        if not state.initialized:
            self._init_denoise_state(generation_kwargs)
        if state.is_complete():
            return self._emit_final_latent(conversation_list, conversation)

        x_t = state.require_latents()
        timestep = state.current_timestep()
        timestep_tokens = state.current_timestep_tokens()
        position_ids = flattened_position_ids(
            state.require_latent_grid_shape(),
            max_latent_size=int(self.config.max_latent_size),
            device=x_t.device,
        )
        outputs = self.embed_latent(
            latents=x_t,
            position_ids=position_ids,
            timesteps=timestep_tokens,
        )
        query = outputs["latent_embeds"].to(device=self.device, dtype=self.dtype)
        item = active_output_item(conversation, sources={BAGEL_FLOW_VELOCITY})
        timestep_meta = timestep.detach().to(device=query.device, dtype=torch.float32)
        if item is None:
            materialize_carrier_updates(
                conversation,
                [
                    carrier_append(
                        conversation,
                        ConversationItem(
                            type="output",
                            value=query,
                            role="assistant",
                            source=BAGEL_FLOW_QUERY,
                            meta={"timestep": timestep_meta},
                        ),
                    )
                ],
            )
        else:
            materialize_carrier_updates(
                conversation,
                [
                    replace_fields(
                        item,
                        type="output",
                        role="assistant",
                        source=BAGEL_FLOW_QUERY,
                        value=query,
                        meta={"timestep": timestep_meta},
                    )
                ],
            )
        state.phase = "decode_velocity"
        return {"conversation_list": conversation_list}

    def _decode_velocity_from_hidden(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
    ) -> dict[str, Any]:
        conversation = single_inference_conversation(conversation_list)
        item = active_output_item(conversation, sources={BAGEL_FLOW_HIDDEN})
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
        state = self._generation_state
        outputs = self.decode_velocity(hidden_states=hidden)
        velocity = state.strip_query_markers(outputs["velocity"])
        materialize_carrier_updates(
            None,
            [
                replace_fields(
                    item,
                    type="output",
                    role="assistant",
                    source=BAGEL_FLOW_VELOCITY,
                    value=velocity.to(device=self.device, dtype=self.dtype),
                )
            ],
        )
        state.phase = "advance"
        return {"conversation_list": conversation_list}

    def _advance_denoise(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
    ) -> dict[str, Any]:
        conversation = single_inference_conversation(conversation_list)
        item = active_output_item(conversation, sources={BAGEL_FLOW_VELOCITY})
        if item is None or not torch.is_tensor(item.value):
            raise ValueError("BAGEL flow advance requires source='bagel_flow_velocity'.")
        state = self._generation_state
        velocity = item.value
        if velocity.dim() == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        complete = state.advance(velocity)
        if complete:
            return self._emit_final_latent(conversation_list, conversation)
        materialize_carrier_updates(None, [meta_patch(item, {}, remove=("timestep",))])
        state.phase = "prepare_query"
        return {"conversation_list": conversation_list}

    def _init_denoise_state(self, generation_kwargs: dict[str, Any]) -> None:
        self._generation_state.initialize(
            generation_kwargs,
            resolution=int(getattr(self.config, "resolution", 1024)),
            patch_latent_dim=int(self.config.patch_latent_dim),
            device=self.vae2llm.weight.device,
        )

    def _emit_final_latent(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
        conversation: list[ConversationItem],
    ) -> dict[str, Any]:
        state = self._generation_state
        x_t = state.require_latents()
        item = active_output_item(conversation, sources={BAGEL_FLOW_VELOCITY})
        latent = unpatchify_latent_tokens(
            x_t,
            state.require_latent_grid_shape(),
            z_channels=int(self.config.z_channels),
            latent_patch_size=int(self.config.latent_patch_size),
        )
        if item is None:
            materialize_carrier_updates(
                conversation,
                [
                    carrier_append(
                        conversation,
                        ConversationItem(
                            type="output",
                            value=latent,
                            role="assistant",
                            source=BAGEL_GENERATED_LATENT,
                            meta={},
                        ),
                    )
                ],
            )
        else:
            materialize_carrier_updates(
                conversation,
                [
                    replace_fields(
                        item,
                        type="output",
                        role="assistant",
                        source=BAGEL_GENERATED_LATENT,
                        value=latent.to(device=self.device, dtype=self.dtype),
                        meta={},
                    )
                ],
            )
        self.reset_local_inference_state()
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: SIGNAL_IMAGE_COMPLETE}

    def _embed_context_latents(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
    ) -> dict[str, Any]:
        batched = self._as_batched_conversation(conversation_list)
        embed_items = flow_latent_items(
            batched,
            z_channels=int(self.config.z_channels),
            sources={BAGEL_VAE_CONTEXT},
        )
        if not embed_items:
            embed_items = flow_latent_items(batched, z_channels=int(self.config.z_channels))
        if not embed_items:
            return {"conversation_list": conversation_list}
        inputs, embed_lengths = prepare_context_embed_latent_inputs(
            embed_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.embed_latent(**inputs)
        scatter_latent_embeds(
            embed_items,
            embed_lengths,
            outputs["latent_embeds"],
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation_list}

    def _as_batched_conversation(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
    ) -> list[list[ConversationItem]]:
        if conversation_list is None:
            return []
        if not conversation_list:
            return []
        first = conversation_list[0]
        if isinstance(first, ConversationItem):
            return [conversation_list]  # type: ignore[list-item]
        return conversation_list  # type: ignore[return-value]

    @pre_forward("embed_latent")
    def embed_latent_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        timestep_shift = float(kwargs.get("timestep_shift", 1.0))
        self._conversation_carrier = conversation_list
        self._embed_items = flow_latent_items(conversation_list, z_channels=int(self.config.z_channels))
        self._embed_lengths = []
        if not self._embed_items:
            return {"latents": None}

        # Training/module-tier embedding adds flow noising; generation context
        # calls BagelFlowConnector.embed_latent and uses the clean timestep=0 path.
        inputs, self._embed_lengths = prepare_embed_latent_inputs(
            self._embed_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
            timestep_shift=timestep_shift,
        )
        return inputs

    @post_forward("embed_latent")
    def embed_latent_post(self, latent_embeds: torch.Tensor, is_dummy: bool = False) -> dict[str, Any]:
        conversation = self._conversation_carrier
        embed_items = self._embed_items
        embed_lengths = self._embed_lengths
        self._conversation_carrier = None
        self._embed_items = []
        self._embed_lengths = []

        if is_dummy:
            if conversation is not None:
                materialize_carrier_updates(
                    conversation,
                    [
                        carrier_append(
                            sample,
                            ConversationItem(
                                type="output",
                                value=latent_embeds.squeeze(0),
                                role="dummy",
                                meta={"source": "bagel_flow_connector"},
                            ),
                        )
                        for sample in conversation
                    ],
                )
            return {"conversation_list": conversation}

        scatter_latent_embeds(embed_items, embed_lengths, latent_embeds, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation}

    @pre_forward("decode_velocity")
    def decode_velocity_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._decode_items = flow_hidden_items(
            conversation_list,
            hidden_size=int(self.config.hidden_size),
            sources={BAGEL_FLOW_HIDDEN},
        )
        if not self._decode_items:
            self._decode_items = flow_hidden_items(conversation_list, hidden_size=int(self.config.hidden_size))
        self._decode_lengths = []
        self._decode_target = None
        if not self._decode_items:
            return {"hidden_states": None}

        inputs, self._decode_lengths, self._decode_target = prepare_decode_velocity_inputs(
            self._decode_items,
            hidden_size=int(self.config.hidden_size),
            patch_latent_dim=int(self.config.patch_latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        return inputs

    @post_forward("decode_velocity")
    def decode_velocity_post(self, velocity: torch.Tensor, is_dummy: bool = False) -> dict[str, Any]:
        conversation = self._conversation_carrier
        decode_items = self._decode_items
        decode_lengths = self._decode_lengths
        target = self._decode_target
        self._conversation_carrier = None
        self._decode_items = []
        self._decode_lengths = []
        self._decode_target = None

        if is_dummy:
            return {"conversation_list": conversation, "_loss": velocity.sum() * 0.0}

        scatter_velocity(decode_items, decode_lengths, velocity, device=self.device, dtype=self.dtype)

        if target is None:
            return {"conversation_list": conversation, "_loss": velocity.sum() * 0.0}
        mse = (velocity - target.to(device=velocity.device, dtype=velocity.dtype)).square()
        token_count = torch.tensor(float(mse.shape[0]), device=mse.device, dtype=mse.dtype)
        return {"conversation_list": conversation, "_loss": mse.mean(dim=-1).sum() / token_count}

    def dummy_inputs(self, kind: str = "embed_latent") -> dict[str, torch.Tensor]:
        if kind == "decode_velocity":
            return {
                "hidden_states": torch.zeros(
                    1,
                    int(self.config.hidden_size),
                    device=self.device,
                    dtype=self.dtype,
                )
            }
        return {
            "latents": torch.zeros(
                1,
                int(self.config.patch_latent_dim),
                device=self.device,
                dtype=self.dtype,
            ),
            "position_ids": flattened_position_ids((1, 1), max_latent_size=int(self.config.max_latent_size)).to(
                device=self.device
            ),
            "timesteps": torch.zeros(1, device=self.device, dtype=torch.float32),
        }


__all__ = ["BagelFlowConnectorModuleMixin"]
