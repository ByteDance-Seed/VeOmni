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

"""Argument schema for OmniModel V2 (SeedOmni) training + inference.

Why a dedicated omni schema?
----------------------------
The base :class:`~veomni.arguments.arguments_types.VeOmniArguments` nests the
parallelism / FSDP topology under ``train.accelerator``.  That made sense when
``accelerator`` was a *training-only* concern, but OmniModel inference also
needs FSDP (large vision encoders sharded across ranks), so :class:`OmniArguments`
**pulls ``accelerator`` out to the top level** — a peer of ``model`` / ``data`` /
``train`` — so a single ``base.yaml`` drives both the trainer and the
inferencer.

Two further omni-specific changes:

* ``model.model_path`` is a **split-checkpoint root folder** (one subfolder per
  OmniModule), not a single HF checkpoint.
* The omni graphs live in their own files and are referenced from the args:
  ``model.modules`` / ``model.train_graph`` for training and
  ``infer.modules`` / ``infer.infer_graph`` / ``infer.infer_type`` for
  inference.

Compatibility with :class:`~veomni.trainer.base.BaseTrainer`
------------------------------------------------------------
The omni trainers reuse ``BaseTrainer`` build helpers, which read
``args.train.accelerator``.  To keep that working without editing every
trainer, :meth:`OmniArguments.__post_init__` injects the top-level
``accelerator`` back onto ``train`` and then runs the accelerator-dependent
validation/derivation that the (deferred) :class:`OmniTrainingArguments` skipped.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from ..utils import logging
from .arguments_types import (
    AcceleratorConfig,
    DataArguments,
    ModelArguments,
    TrainingArguments,
    VeOmniArguments,
)


logger = logging.get_logger(__name__)


@dataclass
class OmniTrainingArguments(TrainingArguments):
    """``train.*`` for OmniModel — identical to :class:`TrainingArguments`
    except that ``accelerator`` is owned by the top-level :class:`OmniArguments`.

    The inherited ``accelerator`` field is retained (so downstream
    ``BaseTrainer`` code that reads ``args.train.accelerator`` keeps working);
    it is injected from the top level by :meth:`OmniArguments.__post_init__`.
    Accelerator-dependent validation (``_validate_accelerator``) and batch
    derivation (``_derive_batch_config``) are **deferred** to the root, because
    they need the real (top-level) accelerator rather than this field's default.
    """

    def __post_init__(self):
        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.global_rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        # NOTE: _validate_accelerator() + _derive_batch_config() are deferred to
        # OmniArguments.__post_init__ (run after the top-level accelerator is
        # injected). Only the accelerator-independent setup runs here.
        self._resolve_checkpoint_paths()
        self._resolve_profile()


@dataclass
class OmniModelArguments(ModelArguments):
    """``model.*`` for OmniModel V2 training.

    ``model_path`` points at a **split-checkpoint root folder** (one subfolder
    per OmniModule).  ``modules`` / ``train_graph`` reference the per-module
    override file and the training-graph file respectively.
    """

    modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the per-module training override YAML (e.g. "
                "configs/seed_omni/Janus/janus_1.3b/modules_train.yaml). "
                "Each top-level key is a module name; its block may carry "
                "``model`` / ``train`` / ``accelerator`` overrides deep-merged "
                "onto the global args. May also be provided inline as a dict "
                "(e.g. via --model.modules.<name>.* CLI overrides)."
            )
        },
    )
    train_graph: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the training-graph YAML (flat list of `module[.method]` "
                "edges under a `training_graph:` key, or a bare list)."
            )
        },
    )


@dataclass
class OmniGraphProfileArguments:
    """``graph_profile.*`` — SeedOmni graph profiler settings.

    This is separate from ``train.profile.*``. The latter owns PyTorch profiler
    traces; this block controls graph-node execution records and optional
    wall/CUDA/memory suffixes.
    """

    train_start_step: int = field(
        default=1,
        metadata={
            "help": (
                "First training global step to save graph profiler records for. "
                "Only used when at least one graph profiler detail switch is enabled."
            )
        },
    )
    train_end_step: int = field(
        default=2,
        metadata={"help": "Last training global step to save graph profiler records for."},
    )
    enable_wall_time: bool = field(
        default=False,
        metadata={"help": "Append per graph-node wall-clock timing to graph profiler records."},
    )
    enable_cuda_events: bool = field(
        default=False,
        metadata={"help": "Append per graph-node CUDA event timing to graph profiler records."},
    )
    enable_memory: bool = field(
        default=False,
        metadata={"help": "Append request-local peak device memory to graph profiler records."},
    )

    def enable_graph_profiling(self) -> bool:
        return self.enable_wall_time or self.enable_cuda_events or self.enable_memory


@dataclass
class OmniInferArguments:
    """``infer.*`` — OmniModel V2 inference configuration + per-call knobs.

    ``modules`` overrides the **training** modules for inference (defaults to
    eager ``from_pretrained`` per module; opt-in FSDP via a module's
    ``accelerator.fsdp_config`` block).  ``infer_graph`` maps a scenario name to
    its generation-graph YAML; ``infer_type`` selects the active scenario.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Split-checkpoint root folder for inference. Defaults to "
                "`model.model_path` when unset; override to infer from a "
                "trained checkpoint while keeping the training base elsewhere."
            )
        },
    )
    modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the per-module inference override YAML (e.g. "
                "modules_infer.yaml). Deep-merged onto the training modules; "
                "modules default to eager load unless this opts into FSDP."
            )
        },
    )
    infer_graph: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Mapping of scenario name -> generation-graph YAML path."},
    )
    infer_type: Optional[str] = field(
        default=None,
        metadata={"help": "Active scenario key into `infer_graph`. Defaults to the first entry."},
    )
    generation_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Free-form generation kwargs passed to the generation graph. "
                "Arbitrary keys may be added via --infer.generation_kwargs.<key> <value>."
            )
        },
    )
    # ---- per-invocation runtime knobs ------------------------------------
    prompt: str = field(
        default="",
        metadata={"help": "User text prompt (required at generate time)."},
    )
    image: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path or http(s) URL to an image. Omit for text-to-image generation."},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "Root output directory; artefacts are nested under <output_dir>/<infer_type>/."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )


@dataclass
class OmniDataArguments(DataArguments):
    """``data.*`` for OmniModel — adds the multimodal-IO config block.

    ``mm_configs`` is forwarded to the ``seedomni`` data transform (and thence to
    ``fetch_images`` / ``fetch_videos``), e.g.
    ``mm_configs: {use_audio_in_video: false, fps: 2.0, max_frames: 16}``.
    """

    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input (forwarded to the seedomni data transform)."},
    )


@dataclass
class OmniArguments(VeOmniArguments):
    """Root config for OmniModel V2 — assembles model / data / train /
    accelerator / infer.  Consumed by :func:`veomni.arguments.parse_omni_args`.

    A single ``base.yaml`` carries both the training and inference blocks; the
    trainer ignores ``infer`` and the inferencer ignores the training-only
    fields, so the same config drives both.
    """

    model: OmniModelArguments = field(default_factory=OmniModelArguments)
    data: OmniDataArguments = field(default_factory=OmniDataArguments)
    train: OmniTrainingArguments = field(default_factory=OmniTrainingArguments)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    graph_profile: OmniGraphProfileArguments = field(default_factory=OmniGraphProfileArguments)
    infer: OmniInferArguments = field(default_factory=OmniInferArguments)

    def __post_init__(self):
        # accelerator is pulled to the top level; inject it onto `train` so the
        # reused BaseTrainer build helpers (which read args.train.accelerator)
        # keep working, then run the deferred accelerator-dependent validation.
        self.train.accelerator = self.accelerator
        self.train._validate_accelerator()
        self.train._derive_batch_config()
        super().__post_init__()

    def _to_base_args(self) -> VeOmniArguments:
        """Project this omni root onto a plain :class:`VeOmniArguments`.

        Strips omni-only model fields (``modules`` / ``train_graph``) and the
        top-level ``infer`` block, and keeps the accelerator under
        ``train.accelerator`` (already injected in :meth:`__post_init__`).  Used
        as the base onto which per-module overrides are deep-merged by
        :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`.
        """
        model_kwargs = {f.name: getattr(self.model, f.name) for f in fields(ModelArguments)}
        return VeOmniArguments(
            model=ModelArguments(**model_kwargs),
            data=self.data,
            train=self.train,
        )

    def load_omni_config(self):
        """Build the :class:`OmniConfig` for **training** (modules + training graph)."""
        from ..models.seed_omni.configuration_omni import OmniConfig

        if not self.model.model_path:
            raise ValueError("`model.model_path` (split-checkpoint root) is required for OmniModel V2.")
        if not self.model.modules:
            raise ValueError("`model.modules` (per-module override YAML) is required for OmniModel V2 training.")

        return OmniConfig.from_omni_args(
            global_args=self._to_base_args(),
            model_path=self.model.model_path,
            modules=self.model.modules,
            train_graph=self.model.train_graph,
        )

    def load_omni_infer_config(self):
        """Build the :class:`OmniConfig` for **inference** (modules + generation graph)."""
        from ..models.seed_omni.configuration_omni import OmniConfig

        infer = self.infer
        # `infer.model_path` overrides the training base for inference (e.g. a
        # trained checkpoint root); falls back to `model.model_path`.
        model_path = infer.model_path or self.model.model_path
        if not model_path:
            raise ValueError(
                "A split-checkpoint root is required for OmniModel V2 inference: "
                "set `infer.model_path` or `model.model_path`."
            )

        graph_map = infer.infer_graph or {}
        if not graph_map:
            raise ValueError("`infer.infer_graph` (scenario -> generation-graph YAML) is required for inference.")
        selected = infer.infer_type
        if selected is None:
            selected = next(iter(graph_map))
            infer.infer_type = selected
        if selected not in graph_map:
            known = ", ".join(sorted(graph_map)) or "(none)"
            raise KeyError(f"Unknown infer.infer_type {selected!r}; expected one of: {known}.")

        return OmniConfig.from_omni_args(
            global_args=self._to_base_args(),
            model_path=model_path,
            modules=self.model.modules,
            infer_modules=infer.modules,
            infer_graph=graph_map[selected],
            generation_kwargs=infer.generation_kwargs,
        )


__all__ = [
    "OmniTrainingArguments",
    "OmniModelArguments",
    "OmniInferArguments",
    "OmniGraphProfileArguments",
    "OmniArguments",
]
