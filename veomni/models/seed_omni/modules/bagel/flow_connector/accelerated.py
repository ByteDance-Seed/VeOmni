"""SeedOmni V2 carrier hooks for BAGEL's flow connector — training-graph hooks only.

``embed_context_latents()`` / ``prepare_denoise_query()`` / ``decode_velocity_from_hidden()``
/ ``advance_denoise()`` and the shared :class:`~.generation_state.FlowGenerationState`
live on the native :class:`~.modeling.BagelFlowConnector` — this file only carries
the training pre/forward/post hooks (carrier selection, SP token slicing, dummy
alignment).
"""

from __future__ import annotations

from typing import Any

import torch

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import _IMG_TAG_KEY, ConversationItem, is_dummy, iter_desired_items
from ..sources import BAGEL_FLOW_HIDDEN, BAGEL_FLOW_VELOCITY
from .configuration import BagelFlowConnectorConfig
from .modeling import BagelFlowConnector, scatter_flow_latent_embeds, select_vae_context_latent_items
from .processing import preprocess_context_latent_embed, preprocess_decode_velocity, preprocess_latent_embed


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


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for BAGEL's flow connector."""

    config: BagelFlowConnectorConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config
        proj_n = int(cfg.patch_latent_dim) * int(cfg.hidden_size) * 2
        tokens = sum(seqlens)
        return 6 * proj_n * tokens / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """Carrier hooks for latent embedding and velocity projection.

    ``embed_context_latents()`` / ``prepare_denoise_query()`` /
    ``decode_velocity_from_hidden()`` / ``advance_denoise()`` and the shared
    flow-matching generation-FSM state already live on the native
    :class:`~.modeling.BagelFlowConnector` (via its own :class:`~.modeling.InferenceMixin`),
    so no ``InferenceMixin`` is needed here.
    """


class BagelFlowConnectorAccelerated(VeOmniMixin, BagelFlowConnector):
    pass


__all__ = ["BagelFlowConnectorAccelerated"]
