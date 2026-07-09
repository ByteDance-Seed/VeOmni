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

"""``VeOmniLoraModel`` — the PEFT-free LoRA model wrapper.

Mirrors ``peft.PeftModel`` structurally: ``self.base_model.model`` is the
original model, so every LoRA parameter FQN and every saved adapter key carries
the ``base_model.model.`` prefix, byte-identical to a PEFT checkpoint. Attribute
access and ``forward`` are forwarded to the wrapped model so trainers, loss
computation, and ``generate`` keep working unchanged.

The wrapper owns adapter injection (dense ``nn.Linear`` via
:mod:`veomni.lora.mapping`; MoE experts via :mod:`veomni.lora.moe_layers`),
the ``requires_grad`` policy, and PEFT-format save/load.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..utils import logging
from .config import VeOmniLoraConfig
from .layers import DEFAULT_ADAPTER
from .mapping import inject_dense_lora, mark_lora_trainable


logger = logging.get_logger(__name__)


class LoraModel(nn.Module):
    """Inner container holding the base model under ``.model`` (PEFT-aligned).

    Injection happens here so that ``self.model.<fqn>`` gains the wrappers while
    the outer :class:`VeOmniLoraModel` exposes them at ``base_model.model.<fqn>``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: VeOmniLoraConfig,
        adapter_name: str = DEFAULT_ADAPTER,
        inject: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.active_adapter = adapter_name
        # Keyed by adapter name for PEFT-shaped introspection; single adapter today.
        self.veomni_lora_config: dict[str, VeOmniLoraConfig] = {adapter_name: config}
        self.wrapped_dense: list[str] = []
        self.wrapped_moe: list[str] = []
        if inject:
            self._inject(config, adapter_name)

    def _inject(self, config: VeOmniLoraConfig, adapter_name: str) -> None:
        if config.modules_to_save:
            raise NotImplementedError(
                "VeOmniLoraModel does not support `modules_to_save` yet. Correctly keeping "
                "extra modules fully trainable under FSDP2 meta-device init requires seeding "
                "the trainable copy from the freshly-loaded base weights in the custom loader, "
                "which is not implemented. Remove `modules_to_save` from lora_config."
            )
        self.wrapped_dense = inject_dense_lora(self.model, config, adapter_name)
        if config.has_moe:
            from .moe_layers import inject_moe_lora

            self.wrapped_moe = inject_moe_lora(self.model, config, adapter_name)
        mark_lora_trainable(self.model, bias=config.bias)

    def get_lora_config(self, adapter_name: str | None = None) -> VeOmniLoraConfig:
        """Return the LoRA config for ``adapter_name`` (default: active).

        Note: intentionally *not* named ``config`` so that ``model.config``
        keeps resolving (via ``__getattr__``) to the base model's HF config,
        exactly as ``peft.PeftModel`` does.
        """
        return self.veomni_lora_config[adapter_name or self.active_adapter]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class VeOmniLoraModel(nn.Module):
    """PEFT-free drop-in for ``peft.PeftModel`` (single ``"default"`` adapter).

    Args:
        model: the base ``nn.Module`` (typically a ``PreTrainedModel``).
        config: the LoRA configuration.
        adapter_name: adapter key (``"default"``).
        inject: when ``False`` no wrappers are installed (internal use for
            :meth:`from_pretrained` when wrappers are re-installed separately).
    """

    def __init__(
        self,
        model: nn.Module,
        config: VeOmniLoraConfig,
        adapter_name: str = DEFAULT_ADAPTER,
        inject: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = LoraModel(model, config, adapter_name=adapter_name, inject=inject)
        self.active_adapter = adapter_name

    # ------------------------------------------------------------------
    # PeftModel-compatible surface
    # ------------------------------------------------------------------

    def get_lora_config(self, adapter_name: str | None = None) -> VeOmniLoraConfig:
        """Active LoRA config. ``model.config`` still resolves to the HF config."""
        return self.base_model.get_lora_config(adapter_name)

    def get_base_model(self) -> nn.Module:
        return self.base_model.model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    # ------------------------------------------------------------------
    # Save / load (PEFT-format)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True) -> None:
        """Write ``adapter_config.json`` + adapter weights (single-process).

        For distributed FSDP2 training the trainer uses
        :func:`veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp`
        instead (DCP parallel write + rank-0 consolidation). This method is the
        simple path for merged / single-process export and tests.
        """
        from .state_dict import get_lora_state_dict, save_adapter_file

        cfg = self.get_lora_config()
        cfg.save_pretrained(save_directory)
        state = get_lora_state_dict(self, adapter_name=self.active_adapter, config=cfg)
        save_adapter_file(state, save_directory, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        adapter_path: str,
        adapter_name: str = DEFAULT_ADAPTER,
        is_trainable: bool = True,
        config: VeOmniLoraConfig | None = None,
    ) -> VeOmniLoraModel:
        """Install LoRA wrappers described by ``adapter_path`` onto ``model``.

        Reads ``adapter_config.json`` (PEFT- or VeOmni-written) and rebuilds the
        matching dense + MoE wrappers. Adapter *weights* are NOT loaded here —
        under FSDP2 meta-device init they are streamed in later during
        parallelization (``build_parallelize_model`` with ``adapter_path``),
        mirroring the historical ``PeftModel.from_pretrained`` flow.

        When the config lacks a ``veomni_lora`` MoE block (stock PEFT adapter),
        MoE mode is inferred from the on-disk adapter tensor shapes.
        """
        if config is None:
            config = VeOmniLoraConfig.from_pretrained(adapter_path)
            if config.has_moe and config.moe_mode is None:
                from .state_dict import infer_moe_mode_from_adapter

                config = config.replace(moe_mode=infer_moe_mode_from_adapter(adapter_path))

        obj = cls(model, config, adapter_name=adapter_name, inject=True)
        if not is_trainable:
            for p in obj.parameters():
                p.requires_grad = False
        return obj

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_and_unload(self) -> nn.Module:
        """Fold dense adapter deltas into the base weights and return the base model.

        Dense ``LoraLinear`` layers are merged and replaced by their (updated)
        base ``nn.Linear``.

        MoE expert LoRA (``LoraIndependentExperts`` / ``LoraSharedExperts``) is
        **not** supported by ``merge_and_unload``: the wrappers drain the original
        experts module (its params are lifted, the object is not retained), so
        there is no clean base module to restore, and folding the per-half deltas
        into the fused ``gate_up_proj`` correctly under EP/FSDP sharding is a
        separate, unverified concern. A clear error is raised rather than
        returning a model that still silently applies LoRA in its forward.
        """
        from .layers import LoraLinear
        from .mapping import _replace_module

        base = self.base_model.model

        try:
            from .moe_layers import is_lora_moe_experts

            if any(is_lora_moe_experts(m) for m in base.modules()):
                raise NotImplementedError(
                    "merge_and_unload() does not support MoE expert LoRA "
                    "(LoraIndependentExperts / LoraSharedExperts). Keep the adapter "
                    "un-merged and load it alongside the base model at inference time."
                )
        except ImportError:
            pass

        dense_targets = [fqn for fqn, m in base.named_modules() if isinstance(m, LoraLinear)]
        for fqn in dense_targets:
            module: LoraLinear = base.get_submodule(fqn)
            module.merge()
            _replace_module(base, fqn, module.base_layer)

        return base
