"""Internal generation state for BAGEL Qwen2-MoT inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class BranchContext:
    name: str
    cache: Any | None = None
    key_values_lens: torch.Tensor | None = None
    packed_key_value_indexes: torch.Tensor | None = None
    next_position_ids: torch.Tensor | None = None

    def reset(self) -> None:
        self.cache = None
        self.key_values_lens = None
        self.packed_key_value_indexes = None
        self.next_position_ids = None

    def set(
        self,
        *,
        cache: Any,
        key_values_lens: torch.Tensor | None,
        packed_key_value_indexes: torch.Tensor | None,
        next_position_ids: torch.Tensor | None,
    ) -> None:
        self.cache = cache
        self.key_values_lens = key_values_lens
        self.packed_key_value_indexes = packed_key_value_indexes
        self.next_position_ids = next_position_ids

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
        self.cache = empty_cache_factory() if cache is None else deepcopy(cache)
        if key_values_lens is None:
            self.key_values_lens = torch.zeros(1, device=device, dtype=torch.int32)
        else:
            self.key_values_lens = key_values_lens.detach().clone().to(device=device, dtype=torch.int32)
        if packed_key_value_indexes is None:
            self.packed_key_value_indexes = torch.empty(0, device=device, dtype=torch.long)
        else:
            self.packed_key_value_indexes = (
                packed_key_value_indexes.detach().clone().to(device=device, dtype=torch.long)
            )
        self.next_position_ids = next_position_id.detach().reshape(1).to(device=device, dtype=torch.long)

    def ensure_empty(self, *, empty_cache_factory: Callable[[], Any], device: torch.device) -> None:
        if self.cache is None:
            self.cache = empty_cache_factory()
        if self.key_values_lens is None:
            self.key_values_lens = torch.zeros(1, device=device, dtype=torch.int32)
        if self.packed_key_value_indexes is None:
            self.packed_key_value_indexes = torch.empty(0, device=device, dtype=torch.long)
        if self.next_position_ids is None:
            self.next_position_ids = torch.zeros(1, device=device, dtype=torch.long)

    def require_ready(self) -> None:
        if self.cache is None or self.key_values_lens is None or self.next_position_ids is None:
            raise RuntimeError(f"BAGEL {self.name} branch context is not initialized.")

    def cache_len(self) -> int:
        if self.key_values_lens is None:
            raise RuntimeError(f"BAGEL {self.name} branch cache lengths are not initialized.")
        return int(self.key_values_lens.sum().item())

    def position_ids(self, query_len: int, *, device: torch.device) -> torch.Tensor:
        if self.next_position_ids is None:
            raise RuntimeError(f"BAGEL {self.name} branch position ids are not initialized.")
        return self.next_position_ids.reshape(1).expand(query_len).to(device=device)


@dataclass
class MotGenerationState:
    infer_mode: str = "und"
    denoise_branch: str = "main"
    velocity_buffer: dict[str, torch.Tensor] = field(default_factory=dict)
    main: BranchContext = field(default_factory=lambda: BranchContext("main"))
    cfg_text: BranchContext = field(default_factory=lambda: BranchContext("text CFG"))
    cfg_img: BranchContext = field(default_factory=lambda: BranchContext("image CFG"))

    def reset(self) -> None:
        self.infer_mode = "und"
        self.denoise_branch = "main"
        self.velocity_buffer.clear()
        self.main.reset()
        self.cfg_text.reset()
        self.cfg_img.reset()

    def branch_context(self, branch: str | None = None) -> BranchContext:
        branch = branch or self.denoise_branch
        if branch == "main":
            return self.main
        if branch == "cfg_text":
            return self.cfg_text
        if branch == "cfg_img":
            return self.cfg_img
        raise RuntimeError(f"Unsupported BAGEL denoise branch {branch!r}.")

    def collect_velocity(
        self,
        velocity: torch.Tensor,
        branches: tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> str:
        if len(branches) == 1:
            self.velocity_buffer.clear()
            self.denoise_branch = "main"
            return "ready"
        if self.denoise_branch not in branches:
            raise RuntimeError(f"Unsupported BAGEL denoise branch {self.denoise_branch!r}.")

        self.velocity_buffer[self.denoise_branch] = velocity.to(device=device, dtype=dtype)
        branch_index = branches.index(self.denoise_branch)
        if branch_index + 1 < len(branches):
            self.denoise_branch = branches[branch_index + 1]
            return "need_branch"
        return "merge"

    def velocities_for_merge(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        main_velocity = self.velocity_buffer.get("main")
        if main_velocity is None:
            raise RuntimeError("BAGEL CFG merge requires a collected main branch velocity.")
        cfg_text_velocity = self.velocity_buffer.get("cfg_text")
        if cfg_text_velocity is None:
            raise RuntimeError("BAGEL CFG merge requires a collected cfg_text branch velocity.")
        return main_velocity, cfg_text_velocity, self.velocity_buffer.get("cfg_img")

    def finish_velocity_round(self) -> None:
        self.velocity_buffer.clear()
        self.denoise_branch = "main"


__all__ = ["BranchContext", "MotGenerationState"]
