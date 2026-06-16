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
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from ..arguments import OmniArguments, VeOmniArguments
from ..data.multimodal.image_utils import load_image
from ..distributed.parallel_state import use_parallel_state
from ..models.seed_omni import build_conversation
from ..models.seed_omni.modeling_omni import OmniModel
from ..utils import helper
from ..utils.device import get_device_type
from .base import BaseTrainer
from .omni_trainer import OmniModuleTrainer, OmniTrainer


logger = helper.create_logger(__name__)


def _module_needs_distributed(mod_cfg: dict[str, Any]) -> bool:
    """True when a module's inference YAML opts into a distributed build (FSDP2 / ExtraParallel / DDP).

    A module needs an initialised process group + its own :class:`ParallelState`
    whenever it is **not** a single-process ``eager`` load — i.e. ``fsdp2``
    (incl. expert-parallel ``ep`` / vocab-parallel ``emb``) or ``ddp`` (a replicated backbone alongside
    the sharded modules). ``eager`` is the inference default
    (``OmniConfig._resolve_default_accelerator``) and loads via
    ``device_map`` without any collectives.
    """
    fsdp_config = mod_cfg.get("train", {}).get("accelerator", {}).get("fsdp_config", {})
    if "fsdp_mode" not in fsdp_config:
        return False
    fsdp_mode = fsdp_config.get("fsdp_mode")
    return bool(fsdp_mode and str(fsdp_mode).lower() not in ("eager"))


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

        if args.train.accelerator.fsdp_config.fsdp_mode == "eager":
            self._build_model_eager()
            self._build_model_assets()
        else:
            # FSDP / extra-parallel (e.g. ep / emb) inference: reuse the training build
            # (per-module ParallelState -> meta-init -> FSDP2 wrap + weight load),
            # in eval mode and without optimizer / checkpoint callbacks.

            # TODO(WJC)
            # Disable mixed precision so FSDP params stay bf16 (no float32 master)
            # — matching the eager (bf16) modules; otherwise the float32 embeds
            # collide with the eager backbone's bf16 weights (dtype mismatch).
            args.train.accelerator.fsdp_config.mixed_precision.enable = False

            # ``_setup`` builds this module's ParallelState without mutating the
            # global current state, so scope the meta-init + FSDP/DDP wrap (which
            # read ``get_parallel_state()``) to it.
            self._setup()
            with use_parallel_state(self.parallel_state):
                self.base._build_model()
                self._build_model_assets()
                self._freeze_model_module()
                self.base._build_parallelized_model()
                self.base.model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self.base.model

    def _build_model_eager(self) -> None:
        """Eager ``from_pretrained`` load with a launch-aware ``device_map``.

        * **Single-process** (no torchrun) — ``device_map='auto'`` lets accelerate
          spread the one model across all visible GPUs (and offload if needed).
        * **Distributed** (torchrun, mixed with FSDP/emb modules) — every rank
          sees all GPUs. Pin a **full replica to this rank's own device**
          (``{"": "<device>:<local_rank>"}``; the ``""`` key targets the whole
          model), co-located with this rank's sharded modules.
        """
        args: VeOmniArguments = self.base.args
        assert args.train.accelerator.fsdp_config.fsdp_mode == "eager"
        from ..models.seed_omni import OMNI_MODEL_REGISTRY, read_model_type

        model_path = args.model.model_path
        overrides = dict(args.model.model_config or {})
        model_type = read_model_type(model_path)
        cls = OMNI_MODEL_REGISTRY[model_type]()
        if dist.is_initialized():
            # Distributed launch: one full replica pinned to this rank's device.
            device_map = {"": f"{get_device_type()}:{int(os.getenv('LOCAL_RANK', 0))}"}
        else:
            device_map = "auto"
        logger.info_rank0(
            f"OmniModuleInferencer '{self.subfolder_name}': eager load "
            f"(model_type={model_type}, cls={cls.__name__}, device_map={device_map}) from {model_path}"
        )
        self.base.model = cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            **overrides,
        ).eval()


@dataclass
class InferenceRequest:
    """A single inference call."""

    prompt: str
    images: list[Any] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


class OmniInferencer(OmniTrainer):
    """SeedOmni V2 inference driver — one :class:`OmniModuleInferencer` per module."""

    module_inferencers: dict[str, OmniModuleInferencer]

    def __init__(self, args: OmniArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        # Resolve the inference OmniConfig up front (modules + generation graph):
        # the per-module merged accelerators tell us whether any module opts into
        # FSDP + extra-parallel — which requires a distributed (torchrun)
        # launch with an initialised process group + default ParallelState.
        self.omni_config = args.load_omni_infer_config()
        if not self.omni_config.has_generation_graph():
            raise ValueError("OmniConfig has no generation_graph — inference requires an infer graph scenario.")

        self._distributed = any(
            _module_needs_distributed(self.omni_config.modules[name]) for name in self.omni_config.module_names
        )
        if self._distributed:
            # Initialize the process group and default ParallelState.
            self.base._setup()
        helper.set_seed(args.infer.seed)
        self._build_model()

        # Nest artefacts under <output_dir>/<infer_type>/ (infer_type is resolved
        # during load_omni_infer_config when left unset).
        args.infer.output_dir = os.path.join(args.infer.output_dir, args.infer.infer_type)
        logger.info_rank0(f"OmniInferencer: model_path = {args.infer.model_path or args.model.model_path}")
        logger.info_rank0(f"OmniInferencer: scenario = {args.infer.infer_type}")
        logger.info_rank0(f"OmniInferencer: output_dir = {args.infer.output_dir}")

    @property
    def model(self) -> OmniModel:
        return self.base.model

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return self.base.model.modules_dict

    @property
    def args(self) -> OmniArguments:
        return self.base.args

    def _build_model(self) -> None:
        omni_config = self.omni_config  # resolved in __init__

        self.module_names = omni_config.module_names
        self.module_inferencers: dict[str, OmniModuleInferencer] = {}
        modules: dict[str, torch.nn.Module] = {}

        for name in self.module_names:
            module_config = omni_config.module_config(name)
            module_inferencer = OmniModuleInferencer(
                module_config,
                subfolder_name=name,
            )
            self.module_inferencers[name] = module_inferencer
            modules[name] = module_inferencer.model
            logger.info_rank0(
                f"OmniInferencer: built module-inferencer '{name}' from {module_config.model.model_path}"
            )

        self.base.model = OmniModel(omni_config, modules)

        if self._distributed:
            module_parallel_states = {
                name: mi.parallel_state
                for name, mi in self.module_inferencers.items()
                if hasattr(mi, "parallel_state")
            }
            if module_parallel_states:
                self.base.model.set_module_parallel_states(module_parallel_states)
        self.base.model_config = omni_config
        logger.info_rank0(
            f"OmniInferencer: composed OmniModel with {len(self.module_names)} modules ({self.module_names})."
        )

    # ── Inference entry point ─────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs)."""
        infer_args = self.args.infer
        assert infer_args.prompt, "--infer.prompt is required (use a non-empty string)."
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
        """Persist reply / images / trace from one ``generate`` call.

        Under a distributed launch every rank runs the FSM (the collectives need
        all ranks) and — with replicated/greedy decoding — produces the same
        output, so only rank 0 writes the outputs to disk.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
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

    @torch.no_grad()
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
]
