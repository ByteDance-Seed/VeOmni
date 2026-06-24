"""Internal generation state for BAGEL Qwen2-MoT inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class MotCacheContext:
    name: str
    _cache: Any | None = None
    _key_values_lens: torch.Tensor | None = None
    _packed_key_value_indexes: torch.Tensor | None = None
    _next_position_ids: torch.Tensor | None = None

    # ── Cache state ─────────────────────────────────────────────

    def reset(self) -> None:
        self._cache = None
        self._key_values_lens = None
        self._packed_key_value_indexes = None
        self._next_position_ids = None

    @property
    def cache(self) -> Any | None:
        return self._cache

    @property
    def key_values_lens(self) -> torch.Tensor | None:
        return self._key_values_lens

    @property
    def packed_key_value_indexes(self) -> torch.Tensor | None:
        return self._packed_key_value_indexes

    @property
    def next_position_ids(self) -> torch.Tensor | None:
        return self._next_position_ids

    # ── Cache mutation ──────────────────────────────────

    def _set_cache_state(
        self,
        *,
        cache: Any,
        key_values_lens: torch.Tensor | None,
        packed_key_value_indexes: torch.Tensor | None,
        next_position_ids: torch.Tensor | None,
    ) -> None:
        self._cache = cache
        self._key_values_lens = key_values_lens
        self._packed_key_value_indexes = packed_key_value_indexes
        self._next_position_ids = next_position_ids

    def snapshot(
        self,
        *,
        cache: Any,
        key_values_lens: torch.Tensor | None,
        packed_key_value_indexes: torch.Tensor | None,
        next_position_id: torch.Tensor,
        empty_cache_factory: Callable[[], Any],
        device: torch.device,
    ) -> None:
        self._cache = empty_cache_factory() if cache is None else deepcopy(cache)
        if key_values_lens is None:
            self._key_values_lens = torch.zeros(1, device=device, dtype=torch.int32)
        else:
            self._key_values_lens = key_values_lens.detach().clone().to(device=device, dtype=torch.int32)
        if packed_key_value_indexes is None:
            self._packed_key_value_indexes = torch.empty(0, device=device, dtype=torch.long)
        else:
            self._packed_key_value_indexes = (
                packed_key_value_indexes.detach().clone().to(device=device, dtype=torch.long)
            )
        self._next_position_ids = next_position_id.detach().reshape(1).to(device=device, dtype=torch.long)

    def ensure_empty(self, *, empty_cache_factory: Callable[[], Any], device: torch.device) -> None:
        if self._cache is None:
            self._cache = empty_cache_factory()
        if self._key_values_lens is None:
            self._key_values_lens = torch.zeros(1, device=device, dtype=torch.int32)
        if self._packed_key_value_indexes is None:
            self._packed_key_value_indexes = torch.empty(0, device=device, dtype=torch.long)
        if self._next_position_ids is None:
            self._next_position_ids = torch.zeros(1, device=device, dtype=torch.long)

    def append_packed_query(
        self,
        *,
        cache: Any,
        query_lens: torch.Tensor,
        device: torch.device,
        next_position_ids: torch.Tensor | None = None,
    ) -> None:
        self.require_ready()
        next_key_values_lens = self._key_values_lens + query_lens

        if next_position_ids is None:
            next_position_ids = self._next_position_ids + query_lens
        else:
            next_position_ids = next_position_ids.detach().reshape(1).to(device=device, dtype=torch.long)

        self._set_cache_state(
            cache=cache,
            key_values_lens=next_key_values_lens,
            packed_key_value_indexes=torch.arange(
                int(next_key_values_lens.sum().item()),
                device=device,
                dtype=torch.long,
            ),
            next_position_ids=next_position_ids,
        )

    # ── Cache accessors ──────────────────────────────────

    def require_ready(self) -> None:
        if (
            self._cache is None
            or self._key_values_lens is None
            or self._packed_key_value_indexes is None
            or self._next_position_ids is None
        ):
            raise RuntimeError(f"BAGEL {self.name} branch context is not initialized.")

    def cache_len(self) -> int:
        if self._key_values_lens is None:
            raise RuntimeError(f"BAGEL {self.name} branch cache lengths are not initialized.")
        return int(self._key_values_lens.sum().item())

    def repeated_position_ids(self, query_len: int, *, device: torch.device) -> torch.Tensor:
        if self._next_position_ids is None:
            raise RuntimeError(f"BAGEL {self.name} branch position ids are not initialized.")
        return self._next_position_ids.reshape(1).expand(query_len).to(device=device)

    def packed_query_args(
        self,
        query_len: int,
        *,
        device: torch.device,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.require_ready()
        cache_len = self.cache_len()
        query_lens = torch.tensor([query_len], device=device, dtype=torch.int32)
        packed_query_indexes = torch.arange(cache_len, cache_len + query_len, device=device, dtype=torch.long)

        if position_ids is None:
            packed_position_ids = self._next_position_ids.reshape(1) + torch.arange(
                query_len,
                device=device,
                dtype=torch.long,
            )
        else:
            packed_position_ids = position_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
            if int(packed_position_ids.numel()) != query_len:
                raise ValueError("BAGEL branch query position_ids length must match query_len.")

        return query_lens, packed_query_indexes, packed_position_ids


@dataclass
class MotGenerationState:
    _infer_mode: str = "und"
    _active_denoise_branch: str = "main"
    _velocity_buffer: dict[str, torch.Tensor] = field(default_factory=dict)
    _main: MotCacheContext = field(default_factory=lambda: MotCacheContext("main"))
    _cfg_text: MotCacheContext = field(default_factory=lambda: MotCacheContext("text CFG"))
    _cfg_img: MotCacheContext = field(default_factory=lambda: MotCacheContext("image CFG"))

    def reset(self) -> None:
        self._infer_mode = "und"
        self._active_denoise_branch = "main"
        self._velocity_buffer.clear()
        self._main.reset()
        self._cfg_text.reset()
        self._cfg_img.reset()

    def update_infer_mode(self, generation_kwargs: dict[str, Any]) -> str:
        self._infer_mode = str(generation_kwargs.get("infer_mode", self._infer_mode or "und"))
        return self._infer_mode

    @property
    def infer_mode(self) -> str:
        return self._infer_mode

    # ── CFG cache contexts ──────────────────────────────────

    @property
    def main(self) -> MotCacheContext:
        return self._main

    @property
    def cfg_text(self) -> MotCacheContext:
        return self._cfg_text

    @property
    def cfg_img(self) -> MotCacheContext:
        return self._cfg_img

    @property
    def active_denoise_cache(self) -> MotCacheContext:
        return self._cache_for_branch(self._active_denoise_branch)

    def _cache_for_branch(self, branch: str) -> MotCacheContext:
        if branch == "main":
            return self._main
        if branch == "cfg_text":
            return self._cfg_text
        if branch == "cfg_img":
            return self._cfg_img
        raise RuntimeError(f"Unsupported BAGEL denoise branch {branch!r}.")

    # ── CFG request policy ──────────────────────────────────

    def validate_cfg_request(self, generation_kwargs: dict[str, object]) -> None:
        cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
        cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
        if cfg_img_scale > 1.0 and cfg_text_scale <= 1.0:
            raise ValueError("cfg_img_scale > 1.0 requires cfg_text_scale > 1.0")

    def cfg_text_requested(self, generation_kwargs: dict[str, object]) -> bool:
        return float(generation_kwargs.get("cfg_text_scale", 1.0)) > 1.0

    def cfg_img_requested(self, generation_kwargs: dict[str, object]) -> bool:
        return float(generation_kwargs.get("cfg_img_scale", 1.0)) > 1.0

    # ── Denoise velocity round ──────────────────────────────────

    @property
    def active_denoise_branch(self) -> str:
        return self._active_denoise_branch

    def denoise_branches_for_timestep(self, generation_kwargs: dict[str, object], timestep: object) -> tuple[str, ...]:
        branches = ["main"]
        if self._cfg_text_active(generation_kwargs, timestep):
            branches.append("cfg_text")
        if self._cfg_img_active(generation_kwargs, timestep):
            branches.append("cfg_img")
        return tuple(branches)

    def collect_velocity(
        self,
        velocity: torch.Tensor,
        branches: tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> str:
        if len(branches) == 1:
            self._velocity_buffer.clear()
            self._active_denoise_branch = "main"
            return "ready"
        if self._active_denoise_branch not in branches:
            raise RuntimeError(f"Unsupported BAGEL denoise branch {self._active_denoise_branch!r}.")

        self._velocity_buffer[self._active_denoise_branch] = velocity.to(device=device, dtype=dtype)
        branch_index = branches.index(self._active_denoise_branch)
        if branch_index + 1 < len(branches):
            self._active_denoise_branch = branches[branch_index + 1]
            return "need_branch"
        return "merge"

    def merge_collected_velocity(
        self,
        generation_kwargs: dict[str, object],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        merged_velocity = (
            self._merged_velocity_without_cfg(device=device)
            if not self.cfg_text_requested(generation_kwargs)
            else self._merged_velocity_with_cfg(generation_kwargs, device=device)
        )
        self._velocity_buffer.clear()
        self._active_denoise_branch = "main"
        return merged_velocity

    def _merged_velocity_without_cfg(self, *, device: torch.device) -> torch.Tensor:
        main_velocity = self._velocity_buffer.get("main")
        if main_velocity is None:
            raise RuntimeError("BAGEL CFG merge requires a collected main branch velocity.")
        return main_velocity.to(device=device)

    def _merged_velocity_with_cfg(self, generation_kwargs: dict[str, object], *, device: torch.device) -> torch.Tensor:
        main_velocity = self._merged_velocity_without_cfg(device=device)

        cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
        cfg_text_velocity = self._velocity_buffer.get("cfg_text")
        if cfg_text_velocity is None:
            raise RuntimeError("BAGEL CFG merge requires a collected cfg_text branch velocity.")
        cfg_text_velocity = cfg_text_velocity.to(device=device, dtype=main_velocity.dtype)

        cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
        cfg_img_velocity = self._velocity_buffer.get("cfg_img")
        if cfg_img_scale > 1.0 and cfg_img_velocity is None:
            raise ValueError("BAGEL image CFG merge requires a collected cfg_img velocity.")
        if cfg_img_velocity is not None:
            cfg_img_velocity = cfg_img_velocity.to(device=device, dtype=main_velocity.dtype)

        cfg_renorm_min = float(generation_kwargs.get("cfg_renorm_min", 0.0))
        cfg_renorm_type = str(generation_kwargs.get("cfg_renorm_type", "global"))

        guided = cfg_text_velocity + cfg_text_scale * (main_velocity - cfg_text_velocity)
        if cfg_renorm_type == "text_channel":
            norm_main = torch.norm(main_velocity, dim=-1, keepdim=True)
            norm_text = torch.norm(guided, dim=-1, keepdim=True)
            scale = (norm_main / (norm_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            merged = guided * scale
            if cfg_img_scale > 1.0:
                assert cfg_img_velocity is not None
                merged = cfg_img_velocity + cfg_img_scale * (merged - cfg_img_velocity)
            return merged.to(device=device)

        if cfg_img_scale > 1.0:
            assert cfg_img_velocity is not None
            guided = cfg_img_velocity + cfg_img_scale * (guided - cfg_img_velocity)

        if cfg_renorm_type == "global":
            norm_main = torch.norm(main_velocity)
            norm_guided = torch.norm(guided)
        elif cfg_renorm_type == "channel":
            norm_main = torch.norm(main_velocity, dim=-1, keepdim=True)
            norm_guided = torch.norm(guided, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"BAGEL infer_gen CFG renorm type {cfg_renorm_type!r} is not implemented.")
        scale = (norm_main / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)

        return (guided * scale).to(device=device)

    def _cfg_text_active(self, generation_kwargs: dict[str, object], timestep: object) -> bool:
        cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
        if cfg_text_scale <= 1.0:
            return False

        interval = generation_kwargs.get("cfg_interval", (0.0, 1.0))
        if interval is None:
            lower, upper = 0.0, 1.0
        elif isinstance(interval, (list, tuple)) and len(interval) == 2:
            lower, upper = float(interval[0]), float(interval[1])
        else:
            lower, upper = float(interval), 1.0

        if torch.is_tensor(timestep):
            if timestep.numel() == 0:
                raise ValueError("BAGEL CFG timestep tensor must not be empty.")
            t = float(timestep.detach().reshape(-1)[0].item())
        elif timestep is None:
            raise ValueError("BAGEL CFG requires current-round meta['timestep'].")
        else:
            t = float(timestep)
        return t > lower and t <= upper

    def _cfg_img_active(self, generation_kwargs: dict[str, object], timestep: object) -> bool:
        self.validate_cfg_request(generation_kwargs)
        return self.cfg_img_requested(generation_kwargs) and self._cfg_text_active(generation_kwargs, timestep)


__all__ = ["MotCacheContext", "MotGenerationState"]
