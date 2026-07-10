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

"""Configuration for the VeOmni-native LoRA stack.

:class:`VeOmniLoraConfig` is the single source of truth for a LoRA run. It is
serialized to ``adapter_config.json`` using PEFT's field names so a stock
``peft`` install can read the file, plus a namespaced ``veomni_lora`` block for
VeOmni-only settings (MoE mode, gate/up split). PEFT ignores unknown top-level
keys, so the file stays loadable by ``peft.PeftModel.from_pretrained`` while
carrying everything VeOmni needs to rebuild the same adapter on resume.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from ..utils import logging


logger = logging.get_logger(__name__)

ADAPTER_CONFIG_NAME = "adapter_config.json"

# Namespaced block inside adapter_config.json for VeOmni-only settings. PEFT's
# ``LoraConfig.from_pretrained`` drops unknown top-level keys, so anything the
# native stack needs beyond PEFT's schema lives here.
VEOMNI_LORA_KEY = "veomni_lora"
_VEOMNI_SCHEMA_VERSION = 1

BiasType = Literal["none", "all", "lora_only"]
MoEMode = Literal["independent", "shared"]


@dataclass
class VeOmniLoraConfig:
    """LoRA configuration, PEFT-format compatible.

    The field set mirrors ``peft.LoraConfig`` for the features VeOmni supports
    (see the design spec). Fields not listed here (``use_dora``, ``loftq_config``,
    ``layer_replication`` etc.) are neither produced nor consumed; if present in
    a loaded PEFT ``adapter_config.json`` they are ignored with a warning when
    they carry a non-default value.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scaling numerator. Effective scale is
            ``lora_alpha / r`` (or ``lora_alpha / sqrt(r)`` when ``use_rslora``).
        target_modules: ``nn.Linear`` targets. ``list[str]`` (substring/suffix
            match on module FQN, PEFT semantics) or a single regex ``str``.
        exclude_modules: modules to skip even if matched by ``target_modules``.
        target_parameters: glob patterns matching 3-D MoE expert
            ``nn.Parameter`` s (e.g. ``model.layers.*.mlp.experts.gate_up_proj``).
        lora_dropout: dropout probability applied to the LoRA input.
        bias: which biases stay trainable — ``none`` / ``all`` / ``lora_only``.
        use_rslora: rank-stabilised scaling (``alpha / sqrt(r)``).
        init_lora_weights: kaiming-uniform ``A`` + zero ``B`` (no-op at init).
        modules_to_save: extra non-LoRA modules kept fully trainable and saved
            (e.g. a resized embedding / classifier head).
        rank_pattern: ``{regex: r}`` per-module rank overrides.
        alpha_pattern: ``{regex: lora_alpha}`` per-module alpha overrides.
        task_type: PEFT task-type tag written to ``adapter_config.json``
            (informational for VeOmni).
        base_model_name_or_path: informational, written to the config.
        moe_mode: ``independent`` (per-expert LoRA) or ``shared`` (one pair per
            layer). ``None`` when there are no ``target_parameters``.
        moe_two_lora_gate_up: split the fused ``gate_up_proj`` into two
            independent rank-``r`` adapters (gate / up). Always ``True`` today.
    """

    r: int = 8
    lora_alpha: int = 8
    target_modules: list[str] | str | None = None
    exclude_modules: list[str] | str | None = None
    target_parameters: list[str] | None = None
    lora_dropout: float = 0.0
    bias: BiasType = "none"
    use_rslora: bool = False
    init_lora_weights: bool = True
    modules_to_save: list[str] | None = None
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, int] = field(default_factory=dict)
    task_type: str | None = None
    base_model_name_or_path: str | None = None

    # VeOmni-only (serialized under the ``veomni_lora`` block).
    moe_mode: MoEMode | None = None
    moe_two_lora_gate_up: bool = True

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {self.r}.")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"`bias` must be one of none/all/lora_only, got {self.bias!r}.")
        if self.moe_mode is not None and self.moe_mode not in ("independent", "shared"):
            raise ValueError(f"`moe_mode` must be independent/shared/None, got {self.moe_mode!r}.")
        if not self.target_modules and not self.target_parameters:
            raise ValueError(
                "VeOmniLoraConfig needs at least one of `target_modules` (nn.Linear LoRA) or "
                "`target_parameters` (MoE expert LoRA)."
            )

    # ------------------------------------------------------------------
    # MoE helpers
    # ------------------------------------------------------------------

    @property
    def has_moe(self) -> bool:
        return bool(self.target_parameters)

    @property
    def scaling(self) -> float:
        """Effective LoRA scale for the base ``r`` / ``lora_alpha``."""
        import math

        denom = math.sqrt(self.r) if self.use_rslora else self.r
        return self.lora_alpha / denom

    def rank_for(self, module_fqn: str) -> int:
        """Rank for ``module_fqn`` honouring ``rank_pattern`` (first regex match)."""
        return _pattern_lookup(self.rank_pattern, module_fqn, self.r)

    def alpha_for(self, module_fqn: str) -> int:
        """Alpha for ``module_fqn`` honouring ``alpha_pattern`` (first regex match)."""
        return _pattern_lookup(self.alpha_pattern, module_fqn, self.lora_alpha)

    # ------------------------------------------------------------------
    # YAML (model.lora_config) <-> config
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, lora_config: dict[str, Any]) -> VeOmniLoraConfig:
        """Build from the trainer's ``model.lora_config`` YAML dict.

        Preserves the historical VeOmni key names (``rank``/``alpha``/
        ``lora_modules``/``share_expert_lora``) so existing configs keep working,
        while also accepting the PEFT-style names (``r``/``lora_alpha``/
        ``target_modules``) if a user prefers them.
        """
        cfg = dict(lora_config)

        def pick(*names, default=None):
            for n in names:
                if n in cfg and cfg[n] is not None:
                    return cfg[n]
            return default

        target_parameters = pick("target_parameters")
        share_expert_lora = pick("share_expert_lora", default=False)
        moe_mode: MoEMode | None = None
        if target_parameters:
            moe_mode = "shared" if share_expert_lora else "independent"

        return cls(
            r=pick("rank", "r", default=8),
            lora_alpha=pick("alpha", "lora_alpha", default=8),
            target_modules=pick("lora_modules", "target_modules"),
            exclude_modules=pick("exclude_modules"),
            target_parameters=target_parameters,
            lora_dropout=pick("lora_dropout", default=0.0),
            bias=pick("bias", default="none"),
            use_rslora=pick("use_rslora", default=False),
            init_lora_weights=pick("init_lora_weights", default=True),
            modules_to_save=pick("modules_to_save"),
            rank_pattern=pick("rank_pattern", default={}) or {},
            alpha_pattern=pick("alpha_pattern", default={}) or {},
            task_type=pick("task_type"),
            base_model_name_or_path=pick("base_model_name_or_path"),
            moe_mode=moe_mode,
        )

    # ------------------------------------------------------------------
    # adapter_config.json (PEFT-compatible) <-> config
    # ------------------------------------------------------------------

    def to_peft_dict(self) -> dict[str, Any]:
        """Serialize to a PEFT-loadable ``adapter_config.json`` dict.

        Emits PEFT field names at the top level (so ``peft`` can read the file)
        and stashes VeOmni-only settings under :data:`VEOMNI_LORA_KEY`.
        """
        d: dict[str, Any] = {
            "peft_type": "LORA",
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "init_lora_weights": self.init_lora_weights,
            "target_modules": _normalize_list(self.target_modules),
            "exclude_modules": _normalize_list(self.exclude_modules),
            "target_parameters": list(self.target_parameters) if self.target_parameters else None,
            "modules_to_save": list(self.modules_to_save) if self.modules_to_save else None,
            "rank_pattern": dict(self.rank_pattern),
            "alpha_pattern": dict(self.alpha_pattern),
            "task_type": self.task_type,
            "base_model_name_or_path": self.base_model_name_or_path,
            # PEFT fields we always default so the file is a clean LoRA config.
            "fan_in_fan_out": False,
            "inference_mode": False,
            "use_dora": False,
        }
        d[VEOMNI_LORA_KEY] = {
            "schema_version": _VEOMNI_SCHEMA_VERSION,
            "moe_mode": self.moe_mode,
            "moe_two_lora_gate_up": self.moe_two_lora_gate_up,
        }
        return d

    @classmethod
    def from_peft_dict(cls, d: dict[str, Any]) -> VeOmniLoraConfig:
        """Build from an ``adapter_config.json`` dict.

        Accepts both VeOmni-written configs (with a ``veomni_lora`` block) and
        stock PEFT configs (without one — MoE mode is then left ``None`` and
        inferred from tensor shapes at load time). Unsupported PEFT features
        carrying non-default values are warned about, not silently dropped.
        """
        peft_type = d.get("peft_type", "LORA")
        if peft_type not in ("LORA", None):
            raise ValueError(f"VeOmniLoraConfig only supports peft_type=LORA, got {peft_type!r}.")

        _warn_unsupported(d)

        veomni_block = d.get(VEOMNI_LORA_KEY) or {}
        moe_mode = veomni_block.get("moe_mode")

        return cls(
            r=d.get("r", 8),
            lora_alpha=d.get("lora_alpha", 8),
            target_modules=d.get("target_modules"),
            exclude_modules=d.get("exclude_modules"),
            target_parameters=d.get("target_parameters"),
            lora_dropout=d.get("lora_dropout", 0.0),
            bias=d.get("bias", "none"),
            use_rslora=d.get("use_rslora", False),
            init_lora_weights=bool(d.get("init_lora_weights", True)),
            modules_to_save=d.get("modules_to_save"),
            rank_pattern=d.get("rank_pattern") or {},
            alpha_pattern=d.get("alpha_pattern") or {},
            task_type=d.get("task_type"),
            base_model_name_or_path=d.get("base_model_name_or_path"),
            moe_mode=moe_mode,
            moe_two_lora_gate_up=veomni_block.get("moe_two_lora_gate_up", True),
        )

    def save_pretrained(self, save_directory: str) -> str:
        """Write ``adapter_config.json`` into ``save_directory``. Returns the path."""
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, ADAPTER_CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(self.to_peft_dict(), f, indent=2, sort_keys=True)
        return path

    @classmethod
    def from_pretrained(cls, adapter_path: str) -> VeOmniLoraConfig:
        """Read ``adapter_config.json`` from ``adapter_path``."""
        path = os.path.join(adapter_path, ADAPTER_CONFIG_NAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No {ADAPTER_CONFIG_NAME} found in {adapter_path!r}.")
        with open(path) as f:
            return cls.from_peft_dict(json.load(f))

    def replace(self, **changes: Any) -> VeOmniLoraConfig:
        """Return a copy with ``changes`` applied (dataclasses.replace wrapper)."""
        return dataclasses.replace(self, **changes)


def _normalize_list(value: list[str] | str | None) -> list[str] | str | None:
    """Pass through a regex ``str`` target spec, else materialize a plain list."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return list(value)


def _pattern_lookup(pattern: dict[str, int], fqn: str, default: int) -> int:
    """First-match lookup of a ``{regex: value}`` PEFT pattern against ``fqn``."""
    import re

    for regex, value in pattern.items():
        if re.search(regex, fqn):
            return value
    return default


# PEFT features VeOmni does not implement. If a loaded adapter_config.json sets
# any of these to a non-default value the produced model would silently differ,
# so we surface a warning rather than ignore it quietly.
_UNSUPPORTED_NONDEFAULT = {
    "use_dora": False,
    "fan_in_fan_out": False,
    "layer_replication": None,
    "loftq_config": {},
    "megatron_config": None,
    "trainable_token_indices": None,
}


def _warn_unsupported(d: dict[str, Any]) -> None:
    for key, default in _UNSUPPORTED_NONDEFAULT.items():
        if key in d and d[key] not in (default, None):
            logger.warning_rank0(
                f"VeOmniLoraConfig: adapter_config.json sets unsupported field {key}={d[key]!r}; "
                "it will be ignored. Behaviour may differ from the original PEFT adapter."
            )
