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

"""OmniInferencer — SeedOmni V2 inference driver.

Standalone from :class:`OmniTrainer`.  Each sub-module is built by its own
:class:`OmniModuleInferencer` (default: eager ``from_pretrained`` +
``device_map='auto'``).  Per-module FSDP2 is opt-in via the module's YAML
``train.accelerator.fsdp_config`` block — see ``infer_*.yaml`` ``modules:``
overrides deep-merged into ``train.yaml``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import torch

from ..arguments import DataArguments, TrainingArguments, VeOmniArguments
from ..arguments.arguments_types import ModelArguments
from ..data.multimodal.image_utils import load_image
from ..models.seed_omni import build_conversation
from ..models.seed_omni.modeling_omni import OmniModel
from ..utils import helper
from .base import BaseTrainer
from .omni_trainer import OmniModelArguments, OmniModuleTrainer, OmniTrainer


if TYPE_CHECKING:
    from ..models.seed_omni.configuration_omni import OmniConfig


logger = helper.create_logger(__name__)


def _module_uses_fsdp(mod_cfg: dict[str, Any]) -> bool:
    """True when a module YAML block explicitly requests FSDP (``train.accelerator.fsdp_config``)."""
    train = mod_cfg.get("train") or {}
    accelerator = train.get("accelerator") or mod_cfg.get("accelerator") or {}
    fsdp = accelerator.get("fsdp_config") or {}
    if "fsdp_mode" not in fsdp:
        return False
    mode = fsdp.get("fsdp_mode")
    return bool(mode and str(mode).lower() not in ("none", "disabled", "off", "ddp"))


class OmniModuleInferencer(OmniModuleTrainer):
    """Per-module inference builder — extends :class:`OmniModuleTrainer`.

    * **Default (eager)** — ``from_pretrained(..., device_map='auto')`` on one
      device; no distributed init required.
    * **Optional FSDP** — when the module's YAML carries
      ``train.accelerator.fsdp_config.fsdp_mode: fsdp2``, reuses the training
      meta-init → FSDP2 → weight-load path from :class:`OmniModuleTrainer`
      (without checkpoint callbacks).
    """

    def __init__(
        self,
        args: VeOmniArguments,
        subfolder_name: str,
    ):
        self.subfolder_name = subfolder_name
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self._build_model()
        self._build_model_assets()

    @property
    def model(self) -> torch.nn.Module:
        return self.base.model

    def _build_model(self) -> None:
        """Eager ``from_pretrained`` — default inference load path."""
        args: VeOmniArguments = self.base.args
        assert args.train.accelerator.fsdp_config.fsdp_mode == "eager", "Inferencer now only support fsdp_mode"
        from ..models.seed_omni import OMNI_MODEL_REGISTRY, read_model_type

        model_path = args.model.model_path
        overrides = dict(args.model.model_config or {})
        model_type = read_model_type(model_path)
        cls = OMNI_MODEL_REGISTRY[model_type]()
        logger.info_rank0(
            f"OmniModuleInferencer '{self.subfolder_name}': eager load "
            f"(model_type={model_type}, cls={cls.__name__}) from {model_path}"
        )
        self.base.model = cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **overrides,
        ).eval()


@dataclass
class OmniInferModelArguments(OmniModelArguments):
    """``model.*`` for SeedOmni V2 inference — extends :class:`OmniModelArguments`."""

    omni_infer_yaml_path: dict[str, str] | None = field(
        default_factory=dict,
        metadata={
            "help": (
                "Mapping of inference scenario name → inference YAML path.  "
                "The selected scenario's YAML overlays ``omni_train_yaml_path`` "
                "at runtime (``generation_graph`` via flat dict.update; "
                "``modules`` deep-merged per module name for per-module "
                "inference load overrides).  Example keys: infer_gen / infer_und / "
                "infer_interleave."
            )
        },
    )
    omni_infer_type: str | None = field(
        default=None,
        metadata={"help": "Active inference scenario key into omni_infer_yaml_path (inference only)."},
    )

    def load_omni_config(self, global_args: VeOmniArguments) -> OmniConfig:
        from ..models.seed_omni.configuration_omni import OmniConfig

        infer_map = self.omni_infer_yaml_path
        selected = self.omni_infer_type
        if selected is not None:
            if selected not in infer_map:
                known = ", ".join(sorted(infer_map)) or "(none)"
                raise KeyError(f"Unknown omni_infer_type {selected!r}; expected one of: {known}.")
            infer_yaml_path = infer_map[selected]
        else:
            selected = next(iter(infer_map))
            self.omni_infer_type = selected
            infer_yaml_path = infer_map[selected]

        return OmniConfig._init(
            global_args=global_args,
            model_path=self.model_path,
            train_yaml_path=self.omni_train_yaml_path,
            infer_yaml_path=infer_yaml_path,
        )


@dataclass
class GenerationKwargsArguments:
    """``generation_kwargs.*`` — per-invocation generation knobs for ``infer_omni``."""

    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for sampling."},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to sample."},
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={"help": "Guidance scale for sampling."},
    )


@dataclass
class OmniInferArguments:
    """``infer.*`` — per-invocation inference knobs for ``infer_omni``."""

    model_path: str = field(
        metadata={"help": "Local path/HDFS path to the pre-trained model."},
    )
    prompt: str = field(
        default="",
        metadata={"help": "User text prompt (required; non-empty)."},
    )
    image: str | None = field(
        default=None,
        metadata={"help": "Optional path or http(s) URL to an image.  Omit for text-to-image generation."},
    )
    output_dir: str = field(
        default="output",
        metadata={
            "help": (
                "Root output directory.  ``OmniInferenceArguments.__post_init__`` nests artefacts "
                "under ``<output_dir>/<omni_infer_type>/``."
            )
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    generation_kwargs: GenerationKwargsArguments | dict[str, Any] | None = field(
        default_factory=GenerationKwargsArguments,
        metadata={"help": "Generation kwargs."},
    )

    def __post_init__(self):
        assert self.prompt, "--infer.prompt is required (use a non-empty string)."


@dataclass
class OmniInferenceArguments:
    """Root config for SeedOmni V2 inference — consumed by :func:`parse_args`."""

    model: OmniInferModelArguments = field(default_factory=OmniInferModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)
    # train offload config / fsdp config / parallel config could be used for inference
    infer: OmniInferArguments = field(default_factory=OmniInferArguments)

    def __post_init__(self):
        self.model.model_path = self.infer.model_path
        self.infer.output_dir = os.path.join(self.infer.output_dir, self.model.omni_infer_type)
        logger.info_rank0(f"OmniInferencer: model_path = {self.model.model_path}")
        logger.info_rank0(f"OmniInferencer: scenario = {self.model.omni_infer_type}")
        logger.info_rank0(f"OmniInferencer: output_dir = {self.infer.output_dir}")

    def _to_base_args(self) -> VeOmniArguments:
        omni_model = self.model
        model_kwargs = {f.name: getattr(omni_model, f.name) for f in fields(ModelArguments)}
        return VeOmniArguments(
            model=ModelArguments(**model_kwargs),
            data=self.data,
            train=self.train,
        )


@dataclass
class InferenceRequest:
    """A single inference call."""

    prompt: str
    images: list[Any] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


class OmniInferencer(OmniTrainer):
    """SeedOmni V2 inference driver — one :class:`OmniModuleInferencer` per module."""

    module_inferencers: dict[str, OmniModuleInferencer]

    def __init__(self, args: OmniInferenceArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        helper.set_seed(args.infer.seed)
        self._build_model()

    @property
    def model(self) -> OmniModel:
        return self.base.model

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return self.base.model.modules_dict

    @property
    def args(self) -> OmniInferenceArguments:
        return self.base.args

    def _build_model(self) -> None:
        base = self.base
        args: OmniInferenceArguments = base.args
        self.omni_config = args.model.load_omni_config(args._to_base_args())

        if not self.omni_config.has_generation_graph():
            raise ValueError("OmniConfig has no generation_graph — inference requires an infer YAML scenario.")

        default_generation_kwargs = deepcopy(self.omni_config.generation_kwargs)
        default_generation_kwargs.update(asdict(args.infer.generation_kwargs))
        args.infer.generation_kwargs = default_generation_kwargs

        self.module_names = self.omni_config.module_names
        self.module_inferencers: dict[str, OmniModuleInferencer] = {}
        modules: dict[str, torch.nn.Module] = {}
        for name in self.module_names:
            module_config = self.omni_config.module_config(name)
            inferencer = OmniModuleInferencer(
                module_config,
                subfolder_name=name,
            )
            self.module_inferencers[name] = inferencer
            modules[name] = inferencer.model
            logger.info_rank0(
                f"OmniInferencer: built module-inferencer '{name}' from {module_config.model.model_path}"
            )
        self.base.model = OmniModel(self.omni_config, modules)
        self.base.model_config = self.omni_config
        logger.info_rank0(
            f"OmniInferencer: composed OmniModel with {len(self.module_names)} modules ({self.module_names})."
        )

    # ── Inference entry point ─────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs)."""
        infer_args = self.args.infer
        request = InferenceRequest(
            prompt=infer_args.prompt,
            images=[load_image(infer_args.image)] if infer_args.image else [],
            generation_kwargs=infer_args.generation_kwargs,
        )
        ctx = self._run(request)
        self.finalize(ctx, output_dir=infer_args.output_dir)
        return ctx

    def finalize(
        self,
        ctx: dict[str, Any],
        *,
        output_dir: str,
    ) -> None:
        """Persist reply / images / trace from one ``generate`` call."""
        os.makedirs(output_dir, exist_ok=True)

        reply = _extract_generated_text(self.model.generated)
        reply_path = os.path.join(output_dir, "reply.txt")
        with open(reply_path, "w", encoding="utf-8") as f:
            f.write(reply + ("\n" if reply and not reply.endswith("\n") else ""))
        logger.info_rank0(f"finalize: reply ({len(reply)} chars) → {reply_path}")
        if reply:
            logger.info_rank0(f"--- reply ---\n{reply}\n-------------")

        images_out = [
            item["value"]
            for item in self.model.generated
            if isinstance(item, dict) and item.get("type") == "image" and item.get("value") is not None
        ]
        for idx, image in enumerate(images_out):
            out_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            image.save(out_path)
            logger.info_rank0(f"finalize: image #{idx} → {out_path}")

        trace = ctx.get("trace") or []
        trace_path = os.path.join(output_dir, "trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace) + "\n")
        logger.info_rank0(f"finalize: FSM trace ({len(trace)} lines) → {trace_path}")

        if not reply and not images_out:
            logger.warning_rank0("finalize: FSM produced no reply and no images.")

    @torch.inference_mode()
    def _run(self, req: InferenceRequest) -> dict[str, Any]:
        for module in self.modules.values():
            if hasattr(module, "reset_global_inference_state"):
                module.reset_global_inference_state()
            elif hasattr(module, "reset_local_inference_state"):
                module.reset_local_inference_state()

        conversation = build_conversation(prompt=req.prompt, images=req.images)
        request_dict: dict[str, Any] = {
            "conversation_list": conversation,
        }
        self.model.reset()
        trace_buf: list[str] = []
        ctx = self.model.generate(
            request=request_dict,
            trace=trace_buf,
            generation_kwargs=req.generation_kwargs,
        )
        ctx["trace"] = trace_buf
        return ctx


def _extract_generated_text(generated: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in generated:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        value = item.get("value")
        if value is None:
            continue
        text = str(value)
        if text:
            parts.append(text)
    return "\n".join(parts)


__all__ = [
    "OmniInferencer",
    "OmniModuleInferencer",
    "InferenceRequest",
    "GenerationKwargsArguments",
    "OmniInferArguments",
    "OmniInferModelArguments",
    "OmniInferenceArguments",
]
