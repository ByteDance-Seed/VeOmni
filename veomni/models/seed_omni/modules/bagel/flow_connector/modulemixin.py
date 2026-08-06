"""SeedOmni V2 carrier hooks for BAGEL's flow connector."""

from __future__ import annotations

from typing import Any

import torch

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import _IMG_TAG_KEY, ConversationItem, get_tail_output_item, is_dummy, iter_desired_items
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
    flattened_position_ids,
    preprocess_context_latent_embed,
    preprocess_decode_velocity,
    preprocess_latent_embed,
    unpatchify_latent_tokens,
)


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


def slice_sp_token_inputs(method: str, **inputs: torch.Tensor) -> tuple[int, dict[str, torch.Tensor]]:
    ps = get_parallel_state()
    if ps.cp_size != 1:
        raise ValueError(f"BAGEL flow connector {method} supports Ulysses groups only; got cp_size={ps.cp_size}.")

    lengths = {name: int(tensor.shape[0]) for name, tensor in inputs.items()}
    if not lengths or next(iter(lengths.values())) == 0:
        raise ValueError(f"BAGEL flow connector {method} SP requires at least one token.")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"BAGEL flow connector {method} SP inputs must have matching token lengths: {lengths}.")

    full_length = next(iter(lengths.values()))
    local_inputs = {
        name: slice_input_tensor(
            sp_pad(tensor, dim=0, pad_value=0),
            dim=0,
            padding=False,
            group=ps.sp_group,
        )
        for name, tensor in inputs.items()
    }
    return full_length, local_inputs


def gather_sp_token_output(method: str, output: torch.Tensor, full_length: int | None) -> torch.Tensor:
    if full_length is None:
        raise RuntimeError(f"BAGEL flow connector {method} SP token length was not initialized.")
    output = gather_outputs(output, gather_dim=0, group=get_parallel_state().sp_group)
    return output.narrow(0, 0, full_length)


class TrainingMixin(TrainingModuleMixin):
    """Training-graph carrier hooks — depends on :class:`BagelFlowConnector` modeling APIs."""

    config: BagelFlowConnectorConfig
    device: torch.device
    dtype: torch.dtype

    def embed_latent(
        self,
        latents: torch.Tensor,
        position_ids: torch.LongTensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
        ...

    def decode_velocity(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._embed_items: list[ConversationItem] = []
        self._embed_lengths: list[int] = []
        self._sp_embed_length: int | None = None
        self._decode_target_groups: list[list[ConversationItem]] = []
        self._decode_lengths: list[int] = []
        self._decode_target: torch.Tensor | None = None
        self._sp_decode_length: int | None = None

    @pre_forward("embed_latent")
    def embed_latent_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._embed_lengths = []
        self._sp_embed_length = None

        self._embed_items = select_vae_context_latent_items(conversation_list)
        if not self._embed_items:
            raise ValueError("BAGEL flow connector requires per-sample VAE context carriers before embed_latent.")

        parts: list[dict[str, torch.Tensor]] = []
        meter_lengths: list[int] = []
        for item in self._embed_items:
            if is_dummy(item):
                inputs, lengths = preprocess_context_latent_embed(
                    [item],
                    config=self.config,
                    device=self.device,
                    dtype=self.dtype,
                )
                anchor = item.value.to(device=self.device, dtype=self.dtype).sum() * 0.0
                inputs["latents"] = inputs["latents"] + anchor
                parts.append(inputs)
                self._embed_lengths.extend(lengths)
                meter_lengths.extend(int(v) for v in lengths)
                continue

            tag = item.meta.get(_IMG_TAG_KEY)
            if tag == "gen":
                inputs, lengths = preprocess_latent_embed(
                    [item],
                    config=self.config,
                    device=self.device,
                    dtype=self.dtype,
                    timestep_shift=float(self.config.timestep_shift),
                )
            elif tag == "edit":
                inputs, lengths = preprocess_context_latent_embed(
                    [item],
                    config=self.config,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(
                    f"BAGEL flow connector training expects VAE image {_IMG_TAG_KEY} to be 'edit' or 'gen', got {tag!r}."
                )

            parts.append(inputs)
            self._embed_lengths.extend(lengths)
            meter_lengths.extend(int(v) for v in lengths)

        self.metric_meter_set_seqlens("embed_latent", meter_lengths)

        inputs = {
            "latents": torch.cat([part["latents"] for part in parts], dim=0),
            "position_ids": torch.cat([part["position_ids"] for part in parts], dim=0),
            "timesteps": torch.cat([part["timesteps"] for part in parts], dim=0),
        }
        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_embed_length = None
            return inputs

        self._sp_embed_length, inputs = slice_sp_token_inputs("embed_latent", **inputs)
        return inputs

    @post_forward("embed_latent")
    def embed_latent_post(self, latent_embeds: torch.Tensor) -> dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            latent_embeds = gather_sp_token_output("embed_latent", latent_embeds, self._sp_embed_length)

        conversation = self._conversation_carrier
        embed_items = self._embed_items
        embed_lengths = self._embed_lengths
        self._conversation_carrier = None
        self._embed_items = []
        self._embed_lengths = []
        self._sp_embed_length = None

        scatter_flow_latent_embeds(
            embed_items,
            embed_lengths,
            latent_embeds,
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation}

    @pre_forward("decode_velocity")
    def decode_velocity_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._decode_lengths = []
        self._decode_target = None
        self._sp_decode_length = None

        self._decode_target_groups = self._select_velocity_target_groups(conversation_list)

        inputs_parts: list[dict[str, torch.Tensor]] = []
        target_parts: list[torch.Tensor] = []
        for sample, group in zip(conversation_list, self._decode_target_groups, strict=True):
            if not group:
                # No velocity target for this sample; run an anchored dummy decode
                # so the decode head stays in the graph with zero loss.
                dummy = self._anchor_dummy_decode_velocity_inputs([sample])
                inputs_parts.append(dummy)
                self._decode_lengths.append(1)
                continue

            inputs, lengths, target = preprocess_decode_velocity(
                group,
                config=self.config,
                device=self.device,
                dtype=self.dtype,
            )
            inputs_parts.append(inputs)
            self._decode_lengths.extend(lengths)
            target_parts.append(target)
        if target_parts:
            self._decode_target = torch.cat(target_parts, dim=0)

        self.metric_meter_set_seqlens("decode_velocity", [int(v) for v in self._decode_lengths])
        inputs = {"hidden_states": torch.cat([part["hidden_states"] for part in inputs_parts], dim=0)}
        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_decode_length = None
            return inputs

        self._sp_decode_length, inputs = slice_sp_token_inputs("decode_velocity", **inputs)
        return inputs

    @post_forward("decode_velocity")
    def decode_velocity_post(self, velocity: torch.Tensor) -> dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            velocity = gather_sp_token_output("decode_velocity", velocity, self._sp_decode_length)

        conversation = self._conversation_carrier
        decode_target_groups = self._decode_target_groups
        decode_lengths = self._decode_lengths
        target = self._decode_target
        self._conversation_carrier = None
        self._decode_target_groups = []
        self._decode_lengths = []
        self._decode_target = None
        self._sp_decode_length = None

        real_velocity_parts = self._scatter_velocity(decode_target_groups, decode_lengths, velocity)
        loss = velocity.sum() * 0.0
        if target is not None and real_velocity_parts:
            real_velocity = torch.cat(real_velocity_parts, dim=0)
            mse = (real_velocity - target.to(device=velocity.device, dtype=velocity.dtype)).square()
            token_count = torch.tensor(float(mse.shape[0]), device=mse.device, dtype=mse.dtype)
            loss = loss + mse.mean(dim=-1).sum() / token_count
        return {"conversation_list": conversation, "_loss": loss}

    def _select_velocity_target_groups(
        self,
        conversation_list: list[list[ConversationItem]] | None,
    ) -> dict[str, torch.Tensor]:
        """Tie dummy flow loss to MoT hidden states without changing its value."""
        dummy = {
            "hidden_states": torch.zeros(
                1,
                int(self.config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
        }

        if conversation_list is None:
            return dummy

        anchor = None
        for item in iter_desired_items(
            conversation_list,
            types=["text", "image", "output"],
            roles=["user", "assistant"],
        ):
            value = item.value
            if not torch.is_tensor(value):
                continue
            if value.dim() == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.dim() == 2 and int(value.shape[-1]) == int(self.config.hidden_size):
                anchor = value.to(device=self.device, dtype=self.dtype).sum() * 0.0
                break
        if anchor is None:
            return dummy

        return {"hidden_states": dummy["hidden_states"] + anchor}

    def _select_velocity_target_groups(
        self,
        conversation_list: list[list[ConversationItem]] | None,
    ) -> list[list[ConversationItem]]:
        if conversation_list is None:
            raise ValueError("BAGEL flow connector requires conversation_list to select velocity targets.")

        target_groups: list[list[ConversationItem]] = []
        for sample in conversation_list:
            # Keep one target group per sample. An empty group means the hook
            # should run a sample-level dummy decode anchored to this sample's
            # MoT hidden states.
            sample_target_items: list[ConversationItem] = []
            for item in iter_desired_items([sample], types=["image"]):
                if not is_dummy(item) and torch.is_tensor(item.meta.get("flow_velocity_target")):
                    sample_target_items.append(item)
            target_groups.append(sample_target_items)
        return target_groups

    def _scatter_velocity(
        self,
        target_groups: list[list[ConversationItem]],
        decode_lengths: list[int],
        velocity: torch.Tensor,
    ) -> list[torch.Tensor]:
        real_velocity_parts: list[torch.Tensor] = []
        offset = 0
        length_iter = iter(decode_lengths)
        for group in target_groups:
            if not group:
                length = next(length_iter)
                offset += length
                continue
            for item in group:
                length = next(length_iter)
                span = velocity[offset : offset + length]
                offset += length
                item.value = span.to(device=self.device, dtype=self.dtype)
                if item.source == BAGEL_FLOW_HIDDEN:
                    item.source = BAGEL_FLOW_VELOCITY
                real_velocity_parts.append(span)
        if offset != int(velocity.shape[0]):
            raise RuntimeError("BAGEL flow connector token count mismatch during velocity scatter.")
        return real_velocity_parts


class InferenceMixin(InferenceModuleMixin):
    """Inference-graph hooks — depends on :class:`BagelFlowConnector` modeling APIs."""

    config: BagelFlowConnectorConfig
    device: torch.device
    dtype: torch.dtype
    _generation_state: FlowGenerationState

    def embed_latent(
        self,
        latents: torch.Tensor,
        position_ids: torch.LongTensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
        ...

    def decode_velocity(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._generation_state = FlowGenerationState()

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


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for BAGEL's flow connector."""

    config: BagelFlowConnectorConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config
        proj_n = int(cfg.patch_latent_dim) * int(cfg.hidden_size) * 2
        tokens = sum(seqlens)
        return 6 * proj_n * tokens / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin, MeterMixin):
    """Carrier hooks for latent embedding and velocity projection."""


__all__ = [
    "SIGNAL_IMAGE_COMPLETE",
    "VeOmniMixin",
]
