# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense LoRA layers for the VeOmni-native stack.

:class:`LoraLinear` wraps an ``nn.Linear`` target with the exact PEFT
``LoraLayer`` sub-module layout — ``base_layer`` + ``lora_A``/``lora_B``
``nn.ModuleDict`` s keyed by adapter name — so the resulting state-dict keys
(``base_model.model.<...>.{base_layer,lora_A,lora_B}.default.weight``) are
byte-identical to a stock PEFT LoRA checkpoint. VeOmni ships a single
``"default"`` adapter; the ModuleDict layout is kept for PEFT compatibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


DEFAULT_ADAPTER = "default"


class LoraLayer:
    """Mixin holding the LoRA sub-modules shared by all dense LoRA layers.

    Not an ``nn.Module`` itself — concrete layers (e.g. :class:`LoraLinear`)
    multiply-inherit from ``nn.Module`` and this mixin, and call
    :meth:`_init_lora_layer` from their ``__init__`` after ``super().__init__``.
    """

    def _init_lora_layer(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.scaling: dict[str, float] = {}
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, int] = {}
        self.use_rslora: dict[str, bool] = {}
        self._merged_adapters: list[str] = []

    @property
    def merged(self) -> bool:
        return bool(self._merged_adapters)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        init_lora_weights: bool,
    ) -> None:
        """Create the ``A``/``B``/dropout sub-modules for ``adapter_name``."""
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}.")

        ref = self.base_layer.weight
        factory_kwargs = {"dtype": ref.dtype, "device": ref.device}

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.use_rslora[adapter_name] = use_rslora
        self.scaling[adapter_name] = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False, **factory_kwargs)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False, **factory_kwargs)

        # Defer init while on meta device — weight loading / post-load init
        # (``reset_lora_parameters``) runs once real storage exists.
        if init_lora_weights and not ref.is_meta:
            self.reset_lora_parameters(adapter_name)

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name: str | None = None, init_lora_weights: bool = True) -> None:
        """Kaiming-uniform ``A`` + zero ``B`` (PEFT default). Idempotent.

        ``adapter_name=None`` resets every adapter, matching PEFT's signature so
        the shared init dispatcher can call it without special-casing.
        """
        if not init_lora_weights:
            return
        for name in self.lora_A:
            if adapter_name is not None and name != adapter_name:
                continue
            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name].weight)


class LoraLinear(nn.Module, LoraLayer):
    """LoRA-augmented ``nn.Linear`` with PEFT-identical FQNs.

    Forward: ``base_layer(x) + lora_B(lora_A(dropout(x))) * scaling``. The base
    ``nn.Linear`` (weight + optional bias) lives at ``base_layer`` and stays
    frozen; only ``lora_A``/``lora_B`` are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ) -> None:
        super().__init__()
        self._init_lora_layer(base_layer)
        self.active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        if self.merged:
            return result
        adapter = self.active_adapter
        if adapter not in self.lora_A:
            return result
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        dropout = self.lora_dropout[adapter]
        scaling = self.scaling[adapter]
        x = x.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(dropout(x))) * scaling
        return result + delta.to(result.dtype)

    @torch.no_grad()
    def merge(self, adapter_name: str | None = None) -> None:
        """Fold ``B @ A * scaling`` into ``base_layer.weight`` for ``adapter_name``."""
        adapter = adapter_name or self.active_adapter
        if adapter in self._merged_adapters or adapter not in self.lora_A:
            return
        delta = self._delta_weight(adapter)
        self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)
        self._merged_adapters.append(adapter)

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo any merged adapters, restoring the original base weight."""
        while self._merged_adapters:
            adapter = self._merged_adapters.pop()
            if adapter not in self.lora_A:
                continue
            delta = self._delta_weight(adapter)
            self.base_layer.weight.data -= delta.to(self.base_layer.weight.dtype)

    def _delta_weight(self, adapter: str) -> torch.Tensor:
        """``[out, in]`` delta ``B @ A * scaling`` for ``adapter``."""
        a = self.lora_A[adapter].weight  # [r, in]
        b = self.lora_B[adapter].weight  # [out, r]
        return (b @ a) * self.scaling[adapter]

    def extra_repr(self) -> str:
        r = self.r.get(self.active_adapter)
        alpha = self.lora_alpha.get(self.active_adapter)
        return f"r={r}, alpha={alpha}, in={self.in_features}, out={self.out_features}"


def is_lora_linear(module: nn.Module) -> bool:
    """Type-stable check for the dense LoRA layer."""
    return isinstance(module, LoraLinear)
