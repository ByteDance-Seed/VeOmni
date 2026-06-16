"""BAGEL training conversation packing.

This module translates V2 ``ConversationItem`` samples into BAGEL's internal
packed training carrier. Debug metadata from parity fixtures is optional input
annotation and does not replace the implementation packer.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ...conversation import ConversationItem
from .training_pack import BAGEL_DUMMY_ANCHORS_META_KEY, conversation_samples, dummy_anchors_from_conversation


def pack_training_conversation(
    text_encoder: Any,
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    kwargs: dict[str, Any] | None = None,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Pack V2 BAGEL training conversation items into a packed batch."""

    return BagelTrainingPacker(text_encoder, dtype=dtype).pack(conversation_list, kwargs or {})


class BagelTrainingPacker:
    def __init__(self, text_encoder: Any, *, dtype: torch.dtype | None = None) -> None:
        self.text_encoder = text_encoder
        self._dtype = dtype

    @property
    def device(self) -> torch.device:
        return self.text_encoder.device

    @property
    def dtype(self) -> torch.dtype:
        if isinstance(self._dtype, torch.dtype):
            return self._dtype
        for parameter in self.text_encoder.parameters():
            if parameter.dtype.is_floating_point:
                return parameter.dtype
        return getattr(self.text_encoder, "dtype", torch.float32)

    def pack(
        self,
        conversation_list: list[list[ConversationItem]] | list[ConversationItem],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        samples = conversation_samples(conversation_list)
        if not samples:
            raise ValueError("BAGEL training pack requires non-empty conversation_list.")

        eos = self.text_encoder._resolve_eos_token_id()
        start = self.text_encoder._resolve_start_token_id()
        image_start, image_end = self.text_encoder._image_boundary_token_ids()
        timestep_shift = float(kwargs.get("timestep_shift", 3.0))

        packed_text_ids: list[torch.Tensor] = []
        packed_text_indexes: list[torch.Tensor] = []
        packed_position_ids: list[torch.Tensor] = []
        packed_vit_embeds: list[torch.Tensor] = []
        packed_vit_token_indexes: list[torch.Tensor] = []
        packed_latents: list[torch.Tensor] = []
        packed_latent_position_ids: list[torch.Tensor] = []
        packed_vae_token_indexes: list[torch.Tensor] = []
        patchified_vae_latent_shapes: list[tuple[int, int]] = []
        ce_loss_indexes: list[torch.Tensor] = []
        packed_label_ids: list[torch.Tensor] = []
        mse_loss_indexes: list[torch.Tensor] = []
        nested_attention_masks: list[torch.Tensor] = []
        total_lengths: list[int] = []
        flow_timesteps: list[torch.Tensor] = []
        flow_noise: list[torch.Tensor] = []

        sequence_cursor = 0
        for sample in samples:
            sample_start = sequence_cursor
            sample_splits: list[int] = []
            sample_attn_modes: list[str] = []
            sample_position_cursor = 0

            for item in sample:
                if item.type == "text":
                    sequence_cursor, sample_position_cursor = self._append_text_item(
                        item,
                        sequence_cursor=sequence_cursor,
                        sample_position_cursor=sample_position_cursor,
                        start=start,
                        eos=eos,
                        packed_text_ids=packed_text_ids,
                        packed_text_indexes=packed_text_indexes,
                        packed_position_ids=packed_position_ids,
                        ce_loss_indexes=ce_loss_indexes,
                        packed_label_ids=packed_label_ids,
                        sample_splits=sample_splits,
                        sample_attn_modes=sample_attn_modes,
                    )
                    continue

                if item.type != "image":
                    continue
                if item.role == "user":
                    sequence_cursor, sample_position_cursor = self._append_image_understanding_item(
                        item,
                        sequence_cursor=sequence_cursor,
                        sample_position_cursor=sample_position_cursor,
                        image_start=image_start,
                        image_end=image_end,
                        packed_text_ids=packed_text_ids,
                        packed_text_indexes=packed_text_indexes,
                        packed_position_ids=packed_position_ids,
                        packed_vit_embeds=packed_vit_embeds,
                        packed_vit_token_indexes=packed_vit_token_indexes,
                        sample_splits=sample_splits,
                        sample_attn_modes=sample_attn_modes,
                    )
                    continue

                sequence_cursor, sample_position_cursor = self._append_image_generation_item(
                    item,
                    sequence_cursor=sequence_cursor,
                    sample_position_cursor=sample_position_cursor,
                    image_start=image_start,
                    image_end=image_end,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_latents=packed_latents,
                    packed_latent_position_ids=packed_latent_position_ids,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                    mse_loss_indexes=mse_loss_indexes,
                    flow_timesteps=flow_timesteps,
                    flow_noise=flow_noise,
                    sample_splits=sample_splits,
                    sample_attn_modes=sample_attn_modes,
                )

            total_lengths.append(sequence_cursor - sample_start)
            nested_attention_masks.append(
                _prepare_attention_mask_per_sample(sample_splits, sample_attn_modes).to(self.device)
            )

        batch = self._build_batch(
            sequence_length=sequence_cursor,
            packed_text_ids=packed_text_ids,
            packed_text_indexes=packed_text_indexes,
            packed_position_ids=packed_position_ids,
            nested_attention_masks=nested_attention_masks,
            total_lengths=total_lengths,
            packed_vit_embeds=packed_vit_embeds,
            packed_vit_token_indexes=packed_vit_token_indexes,
            ce_loss_indexes=ce_loss_indexes,
            packed_label_ids=packed_label_ids,
            packed_latents=packed_latents,
            packed_latent_position_ids=packed_latent_position_ids,
            packed_vae_token_indexes=packed_vae_token_indexes,
            patchified_vae_latent_shapes=patchified_vae_latent_shapes,
            mse_loss_indexes=mse_loss_indexes,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
            timestep_shift=timestep_shift,
        )
        dummy_anchors = dummy_anchors_from_conversation(conversation_list)
        if dummy_anchors:
            batch[BAGEL_DUMMY_ANCHORS_META_KEY] = dummy_anchors
        return batch

    def _append_text_item(
        self,
        item: ConversationItem,
        *,
        sequence_cursor: int,
        sample_position_cursor: int,
        start: int,
        eos: int,
        packed_text_ids: list[torch.Tensor],
        packed_text_indexes: list[torch.Tensor],
        packed_position_ids: list[torch.Tensor],
        ce_loss_indexes: list[torch.Tensor],
        packed_label_ids: list[torch.Tensor],
        sample_splits: list[int],
        sample_attn_modes: list[str],
    ) -> tuple[int, int]:
        text_ids = self._raw_training_text_ids(item)
        token_ids = (
            text_ids
            if item.meta.get("bagel_train_exact_text_ids")
            else torch.cat(
                (
                    torch.tensor([start], device=self.device, dtype=torch.long),
                    text_ids,
                    torch.tensor([eos], device=self.device, dtype=torch.long),
                )
            )
        )
        indexes = torch.arange(sequence_cursor, sequence_cursor + int(token_ids.numel()), device=self.device)
        packed_text_ids.append(token_ids)
        packed_text_indexes.append(indexes)
        position_ids = self._position_ids(item, sample_position_cursor, int(token_ids.numel()))
        packed_position_ids.append(position_ids)
        if item.role == "assistant":
            ce_loss_indexes.append(indexes[:-1])
            packed_label_ids.append(token_ids[1:].detach())
        sequence_cursor += int(token_ids.numel())
        sample_position_cursor = max(
            sample_position_cursor + int(token_ids.numel()),
            int(position_ids.max().item()) + 1,
        )
        sample_splits.append(int(token_ids.numel()))
        sample_attn_modes.append("causal")
        return sequence_cursor, sample_position_cursor

    def _append_image_understanding_item(
        self,
        item: ConversationItem,
        *,
        sequence_cursor: int,
        sample_position_cursor: int,
        image_start: int,
        image_end: int,
        packed_text_ids: list[torch.Tensor],
        packed_text_indexes: list[torch.Tensor],
        packed_position_ids: list[torch.Tensor],
        packed_vit_embeds: list[torch.Tensor],
        packed_vit_token_indexes: list[torch.Tensor],
        sample_splits: list[int],
        sample_attn_modes: list[str],
    ) -> tuple[int, int]:
        vit_embeds = item.meta.get("packed_vit_embeds")
        if not torch.is_tensor(vit_embeds):
            raise ValueError("BAGEL image understanding item was not prepared by SigLIP pre_forward.")
        vit_embeds = vit_embeds.to(device=self.device, dtype=self.dtype)
        if item.meta.get("bagel_train_exact_no_boundaries"):
            length = int(vit_embeds.shape[0])
            vit_indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=self.device, dtype=torch.long)
            packed_position_ids.append(self._position_ids(item, sample_position_cursor, length, full=True))
            packed_vit_embeds.append(vit_embeds)
            packed_vit_token_indexes.append(vit_indexes)
            sequence_cursor += length
            sample_splits.append(length)
            sample_attn_modes.append("causal")
            position_ids = packed_position_ids[-1]
            sample_position_cursor = max(sample_position_cursor + 1, int(position_ids.max().item()) + 1)
            return sequence_cursor, sample_position_cursor

        length = int(vit_embeds.shape[0]) + 2
        packed_text_ids.append(torch.tensor([image_start, image_end], device=self.device, dtype=torch.long))
        packed_text_indexes.append(torch.tensor([sequence_cursor, sequence_cursor + length - 1], device=self.device))
        packed_position_ids.append(torch.full((length,), sample_position_cursor, device=self.device, dtype=torch.long))
        vit_indexes = torch.arange(
            sequence_cursor + 1,
            sequence_cursor + 1 + int(vit_embeds.shape[0]),
            device=self.device,
            dtype=torch.long,
        )
        packed_vit_embeds.append(vit_embeds)
        packed_vit_token_indexes.append(vit_indexes)
        sequence_cursor += length
        sample_splits.append(length)
        sample_attn_modes.append("full")
        return sequence_cursor, sample_position_cursor + 1

    def _append_image_generation_item(
        self,
        item: ConversationItem,
        *,
        sequence_cursor: int,
        sample_position_cursor: int,
        image_start: int,
        image_end: int,
        packed_text_ids: list[torch.Tensor],
        packed_text_indexes: list[torch.Tensor],
        packed_position_ids: list[torch.Tensor],
        packed_latents: list[torch.Tensor],
        packed_latent_position_ids: list[torch.Tensor],
        packed_vae_token_indexes: list[torch.Tensor],
        patchified_vae_latent_shapes: list[tuple[int, int]],
        mse_loss_indexes: list[torch.Tensor],
        flow_timesteps: list[torch.Tensor],
        flow_noise: list[torch.Tensor],
        sample_splits: list[int],
        sample_attn_modes: list[str],
    ) -> tuple[int, int]:
        latent = item.meta.get("padded_latent")
        latent_shape = item.meta.get("patchified_vae_latent_shape")
        latent_pos = item.meta.get("packed_latent_position_ids")
        if not torch.is_tensor(latent) or not isinstance(latent_shape, tuple) or not torch.is_tensor(latent_pos):
            raise ValueError("BAGEL image generation target was not prepared by VAE pre_forward.")
        h, w = int(latent_shape[0]), int(latent_shape[1])
        clean_latents = self._patchified_clean_latents(latent, h, w)
        if item.meta.get("bagel_train_exact_no_boundaries"):
            length = int(clean_latents.shape[0])
            vae_indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=self.device, dtype=torch.long)
            packed_position_ids.append(self._position_ids(item, sample_position_cursor, length, full=True))
            packed_latents.append(latent.detach().to(device=self.device, dtype=self.dtype))
            packed_latent_position_ids.append(latent_pos.detach().to(device=self.device, dtype=torch.long).reshape(-1))
            packed_vae_token_indexes.append(vae_indexes)
            patchified_vae_latent_shapes.append((h, w))
            mse_loss_indexes.append(vae_indexes)
            self._append_flow_metadata(item, flow_timesteps, flow_noise)
            sequence_cursor += length
            sample_splits.append(length)
            sample_attn_modes.append("causal")
            position_ids = packed_position_ids[-1]
            sample_position_cursor = max(sample_position_cursor + length, int(position_ids.max().item()) + 1)
            return sequence_cursor, sample_position_cursor

        length = int(clean_latents.shape[0]) + 2
        packed_text_ids.append(torch.tensor([image_start, image_end], device=self.device, dtype=torch.long))
        packed_text_indexes.append(torch.tensor([sequence_cursor, sequence_cursor + length - 1], device=self.device))
        packed_position_ids.append(
            torch.arange(sample_position_cursor, sample_position_cursor + length, device=self.device)
        )
        vae_indexes = torch.arange(
            sequence_cursor + 1,
            sequence_cursor + 1 + int(clean_latents.shape[0]),
            device=self.device,
            dtype=torch.long,
        )
        packed_latents.append(latent.detach().to(device=self.device, dtype=self.dtype))
        packed_latent_position_ids.append(latent_pos.detach().to(device=self.device, dtype=torch.long).reshape(-1))
        packed_vae_token_indexes.append(vae_indexes)
        patchified_vae_latent_shapes.append((h, w))
        mse_loss_indexes.append(vae_indexes)
        sequence_cursor += length
        sample_splits.append(length)
        sample_attn_modes.append("noise")
        return sequence_cursor, sample_position_cursor + length

    def _build_batch(
        self,
        *,
        sequence_length: int,
        packed_text_ids: list[torch.Tensor],
        packed_text_indexes: list[torch.Tensor],
        packed_position_ids: list[torch.Tensor],
        nested_attention_masks: list[torch.Tensor],
        total_lengths: list[int],
        packed_vit_embeds: list[torch.Tensor],
        packed_vit_token_indexes: list[torch.Tensor],
        ce_loss_indexes: list[torch.Tensor],
        packed_label_ids: list[torch.Tensor],
        packed_latents: list[torch.Tensor],
        packed_latent_position_ids: list[torch.Tensor],
        packed_vae_token_indexes: list[torch.Tensor],
        patchified_vae_latent_shapes: list[tuple[int, int]],
        mse_loss_indexes: list[torch.Tensor],
        flow_timesteps: list[torch.Tensor],
        flow_noise: list[torch.Tensor],
        timestep_shift: float,
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "sequence_length": sequence_length,
            "packed_text_ids": torch.cat(packed_text_ids).to(device=self.device, dtype=torch.long),
            "packed_text_indexes": torch.cat(packed_text_indexes).to(device=self.device, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids).to(device=self.device, dtype=torch.long),
            "nested_attention_masks": nested_attention_masks,
            "sample_lens": total_lengths,
            "cu_seqlens": torch.nn.functional.pad(
                torch.cumsum(torch.tensor(total_lengths, device=self.device, dtype=torch.int32), dim=0),
                (1, 0),
            ),
        }
        if packed_vit_embeds:
            batch["packed_vit_embeds"] = torch.cat(packed_vit_embeds, dim=0)
            batch["packed_vit_token_indexes"] = torch.cat(packed_vit_token_indexes).to(
                device=self.device, dtype=torch.long
            )
        if ce_loss_indexes:
            ce_mask = torch.zeros(sequence_length, device=self.device, dtype=torch.bool)
            ce_mask[torch.cat(ce_loss_indexes).to(device=self.device, dtype=torch.long)] = True
            batch["ce_loss_indexes"] = ce_mask
            batch["packed_label_ids"] = torch.cat(packed_label_ids).to(device=self.device, dtype=torch.long)
        if packed_latents:
            self._add_latent_fields(
                batch,
                packed_latents=packed_latents,
                packed_latent_position_ids=packed_latent_position_ids,
                packed_vae_token_indexes=packed_vae_token_indexes,
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                mse_loss_indexes=mse_loss_indexes,
                flow_timesteps=flow_timesteps,
                flow_noise=flow_noise,
                timestep_shift=timestep_shift,
                sequence_length=sequence_length,
            )
        return batch

    def _add_latent_fields(
        self,
        batch: dict[str, Any],
        *,
        packed_latents: list[torch.Tensor],
        packed_latent_position_ids: list[torch.Tensor],
        packed_vae_token_indexes: list[torch.Tensor],
        patchified_vae_latent_shapes: list[tuple[int, int]],
        mse_loss_indexes: list[torch.Tensor],
        flow_timesteps: list[torch.Tensor],
        flow_noise: list[torch.Tensor],
        timestep_shift: float,
        sequence_length: int,
    ) -> None:
        first_shape = patchified_vae_latent_shapes[0]
        if any(shape != first_shape for shape in patchified_vae_latent_shapes):
            raise ValueError("BAGEL training pack requires uniform VAE latent shapes within a batch.")
        packed_clean = self._patchify_padded_latents(packed_latents, first_shape)
        raw_timesteps = (
            torch.cat(flow_timesteps).to(device=self.device, dtype=torch.float32)
            if flow_timesteps
            else torch.randn(int(packed_clean.shape[0]), device=self.device, dtype=torch.float32)
        )
        fixed_noise = (
            torch.cat(flow_noise, dim=0).to(device=self.device, dtype=packed_clean.dtype)
            if flow_noise
            else torch.randn_like(packed_clean, device=self.device)
        )
        batch.update(
            {
                "padded_latent": torch.cat(packed_latents, dim=0),
                "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
                "packed_latent_position_ids": torch.cat(packed_latent_position_ids).to(device=self.device),
                "packed_vae_token_indexes": torch.cat(packed_vae_token_indexes).to(
                    device=self.device, dtype=torch.long
                ),
                "packed_timesteps": raw_timesteps,
                "shifted_timesteps": _shifted_timesteps(raw_timesteps, timestep_shift),
                "fixed_noise": fixed_noise,
            }
        )
        mse_mask = torch.zeros(sequence_length, device=self.device, dtype=torch.bool)
        mse_mask[torch.cat(mse_loss_indexes).to(device=self.device, dtype=torch.long)] = True
        batch["mse_loss_indexes"] = mse_mask

    def _raw_training_text_ids(self, item: ConversationItem) -> torch.Tensor:
        value = item.value
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        if not isinstance(value, str):
            raise TypeError(f"BAGEL raw training text must be str or token tensor, got {type(value).__name__}.")
        tokenizer = getattr(self.text_encoder, "_tokenizer", None)
        if tokenizer is None:
            raise ValueError("BAGEL tokenizer is required for raw training text.")
        return torch.tensor(tokenizer.encode(value, add_special_tokens=False), device=self.device, dtype=torch.long)

    def _position_ids(
        self,
        item: ConversationItem,
        start: int,
        length: int,
        *,
        full: bool = False,
    ) -> torch.Tensor:
        exact_position_ids = item.meta.get("bagel_train_position_ids")
        if torch.is_tensor(exact_position_ids):
            return exact_position_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        if full:
            return torch.full((length,), start, device=self.device, dtype=torch.long)
        return torch.arange(start, start + length, device=self.device, dtype=torch.long)

    def _patchified_clean_latents(self, latent: torch.Tensor, h: int, w: int) -> torch.Tensor:
        patch_h = int(latent.shape[-2]) // h
        patch_w = int(latent.shape[-1]) // w
        return (
            latent.detach()
            .to(device=self.device)
            .reshape(1, int(latent.shape[1]), h, patch_h, w, patch_w)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(0, 2)
            .flatten(1, 3)
        )

    def _patchify_padded_latents(
        self,
        packed_latents: list[torch.Tensor],
        patchified_shape: tuple[int, int],
    ) -> torch.Tensor:
        h, w = patchified_shape
        padded = torch.cat(packed_latents, dim=0).to(device=self.device, dtype=self.dtype)
        patch_h = int(padded.shape[2]) // h
        patch_w = int(padded.shape[3]) // w
        return (
            padded.reshape(padded.shape[0], padded.shape[1], h, patch_h, w, patch_w)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(0, 2)
            .flatten(1, 3)
        )

    def _append_flow_metadata(
        self,
        item: ConversationItem,
        flow_timesteps: list[torch.Tensor],
        flow_noise: list[torch.Tensor],
    ) -> None:
        item_timesteps = item.meta.get("flow_timesteps")
        item_noise = item.meta.get("flow_noise")
        if torch.is_tensor(item_timesteps):
            flow_timesteps.append(item_timesteps.detach().to(device=self.device, dtype=torch.float32).reshape(-1))
        if torch.is_tensor(item_noise):
            flow_noise.append(item_noise.detach().to(device=self.device, dtype=self.dtype))


def _prepare_attention_mask_per_sample(split_lens: Iterable[int], attn_modes: Iterable[str]) -> torch.Tensor:
    split_lens = list(split_lens)
    attn_modes = list(attn_modes)
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool)
    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode not in {"causal", "full", "noise"}:
            raise ValueError(f"Unsupported BAGEL training attention mode: {mode!r}")
        if mode == "causal":
            attention_mask[cursor : cursor + length, cursor : cursor + length] = torch.ones(
                (length, length), dtype=torch.bool
            ).tril()
        else:
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        attention_mask[cursor : cursor + length, :cursor] = True
        cursor += length

    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "noise":
            attention_mask[:, cursor : cursor + length] = False
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        cursor += length
    return torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(~attention_mask, float("-inf"))


def _shifted_timesteps(timesteps: torch.Tensor, timestep_shift: float) -> torch.Tensor:
    values = torch.sigmoid(timesteps)
    return timestep_shift * values / (1 + (timestep_shift - 1) * values)


__all__ = ["BagelTrainingPacker", "pack_training_conversation"]
