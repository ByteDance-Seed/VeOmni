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

Standalone from :class:`OmniTrainer`, and like it holds exactly one model
handle: ``self.model``.  Which of the two SeedOmni build paths produces it
depends on whether any module opts into a distributed build via its YAML
``accelerator.fsdp_config`` block (see ``infer_*.yaml`` ``modules:``
overrides deep-merged into the launcher base):

* all-``eager`` (the inference default) — no VeOmni infrastructure is needed, so
  the handle is a plain composed ``PreTrainedModel`` from
  :meth:`OmniModel.from_pretrained`, with standard ``PreTrainedModel``
  sub-modules placed by ``device_map``.
* any FSDP2 / DDP / ExtraParallel module — the handle is an
  :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`
  composing one :class:`ModuleRuntime` per module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from ...arguments import OmniArguments
from ...arguments.omni_arguments_types import OmniModuleRuntimeArguments
from ...models.seed_omni.accelerator import OmniModelRuntime
from ...models.seed_omni.modeling_omni import OmniModel
from ...models.seed_omni.processing_omni import OmniProcessor
from ...models.seed_omni.utils.graph_profiler import GraphProfiler
from ...utils import helper
from .omni_trainer import OmniTrainer


logger = helper.create_logger(__name__)


def _module_needs_distributed(module_args: OmniModuleRuntimeArguments) -> bool:
    """True when a module opts into a distributed build (FSDP2 / ExtraParallel / DDP).

    A module needs an initialised process group + its own :class:`ParallelState`
    whenever it is **not** a single-process ``eager`` load — i.e. ``fsdp2``
    (incl. expert-parallel ``ep`` / vocab-parallel ``emb``) or ``ddp`` (a replicated backbone alongside
    the sharded modules). ``eager`` is the inference default
    (``build_module_runtime_args``) and loads via ``device_map`` without collectives.
    """
    fsdp_mode = module_args.accelerator.fsdp_config.fsdp_mode
    return bool(fsdp_mode and str(fsdp_mode).lower() not in ("eager",))


@dataclass
class InferenceRequest:
    """A single inference call."""

    prompt: str
    images: list[Any] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


class OmniInferencer:
    """SeedOmni V2 inference driver over a single model handle."""

    model: OmniModel | OmniModelRuntime
    processor: OmniProcessor

    def __init__(self, args: OmniArguments):
        self.args = args

        self.checkpoint_root = args.model.model_path
        self.omni_model_runtime = args.resolve_model(for_inference=True)

        self._distributed = any(
            _module_needs_distributed(self.omni_model_runtime.modules[name])
            for name in self.omni_model_runtime.module_names
        )
        if self._distributed:
            self.device = OmniTrainer.setup_distributed(args)
        helper.set_seed(args.infer.seed)
        self._build_model()

        # Nest artefacts under <output_dir>/<infer_type>/ (infer_type is resolved
        # during resolve_model(for_inference=True) when left unset).
        infer_type = args.model.launcher_config("infer_type")
        args.infer.output_dir = os.path.join(args.infer.output_dir, infer_type)
        logger.info_rank0(f"OmniInferencer: model_path = {self.checkpoint_root}")
        logger.info_rank0(f"OmniInferencer: scenario = {infer_type}")
        logger.info_rank0(f"OmniInferencer: output_dir = {args.infer.output_dir}")

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return self.model.modules_dict

    def _build_model(self) -> None:
        """Build ``self.model`` — bare HF :class:`OmniModel` or :class:`OmniModelRuntime`.

        With every module on ``fsdp_mode: eager`` there are no collectives and no
        per-module meta-init to orchestrate, so the graph runs on a plain
        composed ``PreTrainedModel`` loaded straight from the split checkpoint —
        the same object a non-VeOmni user gets from
        :meth:`OmniModel.from_pretrained`.  The launcher's
        :class:`OmniModelRuntimeArguments` is projected onto an :class:`OmniConfig` here
        so ``model.model_config.modules`` overrides (graphs, per-module
        ``model_config``) still apply.

        As soon as one module needs FSDP2 / DDP / ExtraParallel, every module is
        built by its own :class:`ModuleRuntime` under its own
        :class:`ParallelState` and the handle becomes the VeOmni runtime.
        """
        self.module_names = self.omni_model_runtime.module_names
        if self._distributed:
            self.model = OmniModelRuntime.from_model_runtime(
                self.omni_model_runtime,
                for_inference=True,
            )
        else:
            self.model = OmniModel.from_pretrained(
                self.checkpoint_root,
                config=self.omni_model_runtime.to_hf_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
            logger.info_rank0(
                f"OmniInferencer: eager load of {len(self.module_names)} module(s) ({self.module_names}) "
                f"from {self.checkpoint_root}."
            )
        self.processor = OmniProcessor.from_config(self.model.config, checkpoint_root=self.checkpoint_root)
        self.model_config = self.model.config

    # ── Inference entry point ─────────────────────────────────────────────────

    def _runtime_generation_kwargs(self) -> dict[str, Any]:
        """Return per-request generation kwargs with the resolved V2 scenario attached."""
        infer_args = self.args.infer
        generation_kwargs = dict(infer_args.generation_kwargs)
        requested_infer_type = generation_kwargs.get("infer_type")
        active_infer_type = self.args.model.launcher_config("infer_type")
        if requested_infer_type is not None and requested_infer_type != active_infer_type:
            raise ValueError(
                "`infer.generation_kwargs.infer_type` conflicts with `model.model_config.infer_type`: "
                f"{requested_infer_type!r} != {active_infer_type!r}."
            )
        generation_kwargs["infer_type"] = active_infer_type
        return generation_kwargs

    def generate(self) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs)."""
        infer_args = self.args.infer
        assert infer_args.prompt, "--infer.prompt is required (use a non-empty string)."
        request = InferenceRequest(
            prompt=infer_args.prompt,
            images=list(infer_args.images),
            generation_kwargs=self._runtime_generation_kwargs(),
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

        reply = _extract_generated_text(ctx["generated"])
        reply_path = os.path.join(output_dir, "reply.txt")
        with open(reply_path, "w", encoding="utf-8") as f:
            f.write(reply + ("\n" if reply and not reply.endswith("\n") else ""))
        logger.info_rank0(f"finalize: reply ({len(reply)} chars) → {reply_path}")
        if reply:
            logger.info_rank0(f"--- reply ---\n{reply}\n-------------")

        images_out = [
            item["value"]
            for item in ctx["generated"]
            if isinstance(item, dict) and item.get("type") == "image" and item.get("value") is not None
        ]
        for idx, image in enumerate(images_out):
            out_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            image.save(out_path)
            logger.info_rank0(f"finalize: image #{idx} → {out_path}")

        profiler = ctx.get("profiler")
        trace = profiler.save_records() if isinstance(profiler, GraphProfiler) else []
        trace_path = os.path.join(output_dir, "trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace) + "\n")
        logger.info_rank0(f"finalize: FSM trace ({len(trace)} lines) → {trace_path}")

        if not reply and not images_out:
            logger.warning_rank0("finalize: FSM produced no reply and no images.")

    def _begin_graph_trace(self) -> GraphProfiler | None:
        """Open a per-request FSM trace on the VeOmni runtime handle.

        Inference always traces (there is no train-step window to gate on). The
        bare-HF handle has no profiler hook — :meth:`finalize` then writes an
        empty trace.
        """
        if isinstance(self.model, OmniModelRuntime):
            return self.model.begin_request_trace(self.args.train.graph_profile)
        return None

    def _run(self, req: InferenceRequest) -> dict[str, Any]:
        request_dict = self.processor(
            text=req.prompt,
            images=req.images or None,
            inference=True,
        )
        self.model.reset()
        profiler = self._begin_graph_trace()
        with torch.no_grad():
            generated = self.model.generate(request_dict, generation_kwargs=req.generation_kwargs)
        return {"generated": generated, "profiler": profiler}


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
    "InferenceRequest",
]
