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

"""OmniInferencer — SeedOmni V2 inference driver (subclass of :class:`OmniTrainer`).

Reuses :class:`OmniTrainer`'s module build path.  By default modules load
eagerly via ``from_pretrained`` on a single device; set ``model.infer_use_fsdp``
to reuse the training FSDP2 build (requires distributed init).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from ..arguments import DataArguments, InferArguments, TrainingArguments
from ..data.multimodal.image_utils import load_image
from ..models.seed_omni import build_conversation
from ..utils import helper
from .omni_trainer import OmniModelArguments, OmniTrainer, VeOmniOmniArguments


logger = helper.create_logger(__name__)


@dataclass
class OmniInferModelArguments(OmniModelArguments):
    """``model.*`` for SeedOmni V2 inference — extends :class:`OmniModelArguments`."""

    infer_use_fsdp: bool = field(
        default=False,
        metadata={
            "help": (
                "Build modules with FSDP2 (same path as training).  Requires "
                "distributed init; default is eager single-device ``from_pretrained``."
            )
        },
    )


@dataclass
class OmniInferRunArguments(InferArguments):
    """``infer.*`` — per-invocation inference knobs for ``infer_omni``."""

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
    guidance_scale: float = field(
        default=5.0,
        metadata={"help": "Classifier-free guidance weight for T2I (Janus default = 5.0)."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.prompt, "--infer.prompt is required (use a non-empty string)."


@dataclass
class OmniInferenceArguments:
    """Root config for SeedOmni V2 inference — consumed by :func:`parse_args`."""

    model: OmniInferModelArguments = field(default_factory=OmniInferModelArguments)
    infer: OmniInferRunArguments = field(default_factory=OmniInferRunArguments)

    def __post_init__(self):
        self.model.model_path = self.infer.model_path
        self.model.tokenizer_path = self.infer.tokenizer_path
        self.infer.output_dir = os.path.join(self.infer.output_dir, self.model.omni_infer_type or "")
        logger.info_rank0(f"OmniInferencer: model_path = {self.model.model_path}")
        logger.info_rank0(f"OmniInferencer: scenario = {self.model.omni_infer_type}")
        logger.info_rank0(f"OmniInferencer: output_dir = {self.infer.output_dir}")


@dataclass
class InferenceRequest:
    """A single inference call."""

    prompt: str
    images: list[Any] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 2048


def _to_ve_omni_arguments(args: OmniInferenceArguments) -> VeOmniOmniArguments:
    """Map inference CLI args onto :class:`VeOmniOmniArguments` for shared build."""
    model = OmniInferModelArguments(
        model_path=args.model.model_path,
        tokenizer_path=args.model.tokenizer_path,
        omni_train_yaml_path=args.model.omni_train_yaml_path,
        omni_infer_yaml_path=dict(args.model.omni_infer_yaml_path or {}),
        omni_infer_type=args.model.omni_infer_type,
        infer_use_fsdp=args.model.infer_use_fsdp,
    )
    return VeOmniOmniArguments(
        model=model,
        data=DataArguments(train_path="/tmp/unused_infer"),
        train=TrainingArguments(),
    )


class OmniInferencer(OmniTrainer):
    """SeedOmni V2 inference driver — inherits module build from :class:`OmniTrainer`."""

    def __init__(self, args: OmniInferenceArguments):
        self.inference_args = args
        helper.set_seed(args.infer.seed)
        super().__init__(_to_ve_omni_arguments(args), runtime="infer")
        if not self.omni_config.has_generation_graph():
            raise ValueError("Generation graph is required for inference")

    @property
    def model(self) -> torch.nn.Module:
        return self.base.model

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return self.base.model.modules_dict

    @property
    def args(self) -> OmniInferenceArguments:
        return self.inference_args

    # ── Inference entry point ─────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs)."""
        infer_args = self.inference_args.infer
        has_image = bool(infer_args.image)
        request = InferenceRequest(
            prompt=infer_args.prompt,
            images=[load_image(infer_args.image)] if has_image else [],
            generation_kwargs={
                "temperature": infer_args.temperature,
                "top_p": infer_args.top_p,
                "do_sample": infer_args.do_sample,
                "guidance_scale": float(infer_args.guidance_scale),
            },
            max_new_tokens=infer_args.max_tokens,
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
            if hasattr(module, "reset_inference_state"):
                module.reset_inference_state()

        conversation = build_conversation(prompt=req.prompt, images=req.images)
        request_dict: dict[str, Any] = {
            "conversation_list": conversation,
            "generation_kwargs": req.generation_kwargs,
        }
        self.model.reset()
        trace_buf: list[str] = []
        ctx = self.model.generate(
            request=request_dict,
            context=request_dict,
            max_new_tokens=req.max_new_tokens,
            trace=trace_buf,
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
    "InferenceRequest",
    "OmniInferModelArguments",
    "OmniInferRunArguments",
    "OmniInferenceArguments",
]
