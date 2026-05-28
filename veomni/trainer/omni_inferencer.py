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

"""OmniInferencer — SeedOmni V2 single-process inference driver.

Usage::

    args = parse_args(OmniInferenceArguments)
    OmniInferencer(args).generate()

Lifecycle
---------
1. :class:`OmniInferenceArguments` is parsed by
   :func:`veomni.arguments.parse_args`.  Its ``__post_init__`` copies
   ``--infer.model_path`` / ``--infer.tokenizer_path`` onto :attr:`model`
   and nests ``infer.output_dir`` under the active ``omni_infer_type``.

2. ``OmniInferencer(args)`` builds the merged :class:`OmniConfig` from
   ``model.omni_train_yaml_path`` + ``omni_infer_yaml_path[omni_infer_type]``
   (see :meth:`OmniConfig.from_paths`), then constructs each declared
   module by reading its ``weights_path/config.json`` for ``model_type``
   and dispatching through :data:`OMNI_MODEL_REGISTRY`.  Modules load
   eagerly on a single device via ``cls.from_pretrained`` — no FSDP, no
   meta init, ``device_map`` deferred to a follow-up PR.

3. The global tokenizer is loaded from ``OmniConfig.tokenizer_path`` and
   wired into every module that exposes ``set_tokenizer`` (notably
   :class:`JanusTextEncoder`, which resolves bos / boi / eoi / eos ids at
   this point).  Per-module processors (vision / audio / …) are auto-
   loaded inside :meth:`OmniModule.from_pretrained` from the same
   weights folder when the subclass declares ``processor_class`` —
   each module owns its own raw-input tensorisation in
   :meth:`generate`.

4. Inference is a three-layer call stack:

   * :meth:`generate` — high-level: reads :attr:`self.args.infer`, loads
     any image, applies the CFG / sampling knobs, calls
     :meth:`run_request`, then writes every artifact to disk via
     :meth:`finalize`.
   * :meth:`run_request` — mid-level: accepts a built
     :class:`InferenceRequest`, returns the raw ``ctx`` dict without
     touching the filesystem.  Use for programmatic / batched flows.
   * :meth:`_run` — internal: resets per-module state, builds the
     conversation-part list via :func:`build_conversation`, packs it
     into a ``request`` / ``context`` dict and hands off to
     :meth:`OmniModel.generate`.  Raw-PIL → tensor conversion is the
     receiving module's responsibility (see :class:`JanusSiglip`).

   The returned ``ctx`` carries the FSM trace, the accumulated
   ``generated_images_collected`` PIL images (already postprocessed
   by each emitting module's processor — e.g.
   :meth:`JanusVqvaeProcessor.postprocess`), and the per-module
   ``finalize`` outputs (e.g. decoded text under
   ``finalize['janus_text_encoder']['text']``).

Generation kwargs distribution
------------------------------
``generation_kwargs`` is a plain dict that :meth:`run_request` (and the
public :meth:`generate` wrapper) forwards into ``request`` / ``ctx``.
Every module's ``generate`` reads the slot from kwargs and filters by
the keys it recognises (:meth:`JanusTextEncoder._extract_sampling_kwargs`
/ :meth:`JanusVqvae._extract_sampling_kwargs`) — same shape as
HuggingFace ``GenerationMixin._prepare_generation_config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer

from ..arguments import InferArguments
from ..data.multimodal.image_utils import load_image
from ..models.seed_omni import (
    OMNI_MODEL_REGISTRY,
    OmniConfig,
    OmniModel,
    build_conversation,
    read_model_type,
)
from ..utils import helper
from ..utils.device import get_device_type


logger = helper.create_logger(__name__)


@dataclass
class OmniInferModelArguments:
    """``model.*`` — slim model arguments for SeedOmni V2 inference / visualization.

    Carries ONLY the fields actually consumed by the FSM-driven inference
    path: the three ``omni_*`` graph pointers + ``model_path`` /
    ``tokenizer_path`` (which :class:`OmniInferenceArguments.__post_init__`
    auto-fills from ``args.infer.*`` so users only set them once on the CLI).
    Intentionally does NOT inherit from :class:`ModelArguments` /
    :class:`OmniModelArguments` — dragging their training-only knobs
    (``ops_implementation``, ``encoders`` / ``decoders``, ``lora_config``,
    ``safetensor_idx_path``, ``basic_modules``, …) into an inference CLI
    would bloat ``--help`` with dozens of irrelevant flags.

    See :class:`veomni.trainer.omni_trainer.OmniModelArguments` for the
    training-side analogue (which DOES inherit ``ModelArguments`` because
    the trainer needs the full surface).
    """

    model_path: str | None = field(
        default=None,
        metadata={"help": "Auto-filled from --infer.model_path by OmniInferenceArguments.__post_init__."},
    )
    tokenizer_path: str | None = field(
        default=None,
        metadata={"help": "Auto-filled from --infer.tokenizer_path by OmniInferenceArguments.__post_init__."},
    )
    omni_train_yaml_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the master training YAML (carries ``modules / nodes / "
                "edges / training_graph``).  Read at inference time strictly for "
                "the graph topology — the rest of the train YAML's training-only "
                "fields (optimizer / wandb / dataloader / ...) are ignored."
            )
        },
    )
    omni_infer_yaml_path: dict[str, str] | None = field(
        default_factory=dict,
        metadata={
            "help": (
                "Mapping of inference scenario name → inference YAML path.  The "
                "selected scenario's YAML overlays ``omni_train_yaml_path`` to "
                "inject a ``generation_graph``.  Example keys: ``infer_gen`` "
                "(T2I), ``infer_und`` (I2T), ``infer_interleave``."
            )
        },
    )
    omni_infer_type: str | None = field(
        default=None,
        metadata={"help": "Active scenario key into ``omni_infer_yaml_path`` (required for inference)."},
    )

    def __post_init__(self):
        assert self.omni_train_yaml_path is not None, "model.omni_train_yaml_path is required"


@dataclass
class OmniInferRunArguments(InferArguments):
    """``infer.*`` — per-invocation inference knobs for ``infer_omni``.

    Extends framework :class:`~veomni.arguments.InferArguments` with the
    SeedOmni V2 driver's prompt / image / output / CFG fields.  Inherits
    ``model_path`` / ``tokenizer_path`` (the canonical CLI flags) plus
    standard sampling knobs (``seed``, ``temperature``, ``top_p``,
    ``max_tokens``, ``do_sample``).

    Lives in this module (not the script) so other drivers (RL eval,
    batched-prompt benchmarks, future REST shims) can reuse the same CLI
    surface without copy-pasting the dataclass.
    """

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
                "under ``<output_dir>/<omni_infer_type>/`` so different scenarios don't clobber "
                "each other on re-runs."
            )
        },
    )
    guidance_scale: float = field(
        default=5.0,
        metadata={
            "help": (
                "Classifier-free guidance weight for text-to-image generation (Janus default = 5.0). "
                "Ignored for I2T scenarios and when ≤ 1.0.  Same shape as the HF baseline "
                "`scripts/multimodal/infer/janus_hf_infer_gen.py`."
            )
        },
    )
    trace: bool = field(
        default=False,
        metadata={"help": "Dump FSM step / transition log to <output_dir>/trace.txt (debugging aid)."},
    )
    progress: bool = field(
        default=True,
        metadata={
            "help": (
                "Print one ``[FSM] step <N>: <state>`` line per FSM state entry to stdout "
                "(coarse progress bar — see :meth:`OmniModel.generate`).  Default true for "
                "CLI runs; set false for CI / notebook embeds that don't want the stdout spam."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.prompt, "--infer.prompt is required (use a non-empty string)."
        # TODO: diffusion-only modules have no tokenizer — relax this when we add such a scenario.


@dataclass
class OmniInferenceArguments:
    """Root config for SeedOmni V2 inference — consumed by :func:`parse_args`.

    The launcher YAML's top-level ``model:`` section populates :attr:`model`
    (graph YAML pointers + the active scenario key).  ``data:`` / ``train:``
    and every other training-only ``model.*`` field declared in the launcher
    YAML are silently ignored because they're not declared on
    :class:`OmniInferModelArguments`.

    ``__post_init__`` does the minimum bookkeeping to give every downstream
    consumer a single source of truth: copies ``--infer.{model_path,
    tokenizer_path}`` onto :attr:`model` (so the user only sets them once
    on the CLI) and nests :attr:`infer.output_dir` under the active
    ``omni_infer_type`` (so different scenarios don't clobber each other on
    re-runs against the same ``--infer.output_dir``).

    Use::

        args = parse_args(OmniInferenceArguments)
        OmniInferencer(args).generate()
    """

    model: OmniInferModelArguments = field(default_factory=OmniInferModelArguments)
    infer: OmniInferRunArguments = field(default_factory=OmniInferRunArguments)

    def __post_init__(self):
        self.model.model_path = self.infer.model_path
        self.model.tokenizer_path = self.infer.tokenizer_path
        self.infer.output_dir = os.path.join(self.infer.output_dir, self.model.omni_infer_type)
        logger.info_rank0(f"OmniInferencer: model_path = {self.model.model_path}")
        logger.info_rank0(f"OmniInferencer: scenario = {self.model.omni_infer_type}")
        logger.info_rank0(f"OmniInferencer: output_dir = {self.infer.output_dir}")


@dataclass
class InferenceRequest:
    """A single inference call.

    Attributes
    ----------
    prompt:
        Free-form text from the user.
    images:
        Optional PIL images (or already-tensor ``pixel_values`` per
        image).  Placed at the head of the conversation list by
        :func:`build_conversation` per the V2 layout contract (images
        first, then user text, then assistant marker).
    force_image_gen:
        When ``True`` the FSM is steered into the image-VQ branch on
        the first sampling step regardless of the LM head output.  See
        :meth:`JanusTextEncoder.decode`.
    generation_kwargs:
        Free-form keyword arguments forwarded to every module's
        ``generate`` (each module filters by signature).  Common keys:
        ``temperature``, ``top_p``, ``do_sample``.
    max_new_tokens:
        Hard upper bound on total FSM iterations across all states.
    """

    prompt: str
    images: list[Any] = field(default_factory=list)
    force_image_gen: bool = False
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 2048


class OmniInferencer:
    """SeedOmni V2 inference driver — per-module ``from_pretrained`` + FSM.

    Single-process, single-device, no FSDP.  Designed to be the inference twin
    of :class:`OmniTrainer`; both share the registry-based module construction
    contract documented in :mod:`veomni.models.seed_omni.modules`.

    Usage::

        args = parse_args(OmniInferenceArguments)
        OmniInferencer(args).generate()

    The tokenizer is best-effort: if loading fails (e.g. a diffusion-only
    split-checkpoint with no ``tokenizer.json`` at the root) the inferencer
    warns and leaves ``self.tokenizer = None`` — only modules that need it
    via ``set_tokenizer`` are affected.
    """

    def __init__(self, args: OmniInferenceArguments):
        self.args: OmniInferenceArguments = args
        self.device = torch.device(get_device_type())
        helper.set_seed(args.infer.seed)

        self.omni_config = OmniConfig.from_paths(
            model_path=args.model.model_path,
            tokenizer_path=args.model.tokenizer_path,
            train_yaml_path=args.model.omni_train_yaml_path,
            infer_yaml_path=args.model.omni_infer_yaml_path[args.model.omni_infer_type],
        )
        logger.info_rank0(
            f"OmniInferencer: loaded OmniConfig with {len(self.omni_config.module_names)} "
            f"modules ({self.omni_config.module_names})."
        )

        self.modules: dict[str, torch.nn.Module] = self._build_modules()

        # Tokenizer is best-effort — diffusion-only checkpoints have none.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.omni_config.tokenizer_path)
            for module in self.modules.values():
                module.set_tokenizer(self.tokenizer)
        except Exception as e:
            self.tokenizer = None
            logger.warning_rank0(
                f"OmniInferencer: tokenizer load failed ({e}); modules that need set_tokenizer will be skipped."
            )

        self.model = OmniModel(self.omni_config, self.modules).eval()

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_modules(self) -> dict[str, torch.nn.Module]:
        """Build every declared module via ``<cls>.from_pretrained(weights_path)``.

        ``model_type`` is read from each ``<weights_path>/config.json`` via
        :func:`transformers.AutoConfig.from_pretrained` (so the YAML
        ``modules:`` block never has to repeat it) and dispatched through
        :data:`OMNI_MODEL_REGISTRY`.  Module-level overrides (everything in
        the YAML other than ``weights_path``) are forwarded as kwargs to
        the underlying ``__init__`` via ``from_pretrained``'s
        ``**model_kwargs`` mechanism — same shape ``OmniConfig.modules``
        already uses (e.g. ``freeze_vqvae: true``).
        """
        modules: dict[str, torch.nn.Module] = {}
        for name in self.omni_config.module_names:
            mod_cfg = self.omni_config.module_config(name)
            weights_path = mod_cfg.pop("weights_path", None)
            if not weights_path:
                raise ValueError(f"Module '{name}' is missing `weights_path` in OmniConfig.modules.")
            if not os.path.isdir(weights_path):
                raise FileNotFoundError(
                    f"Module '{name}' weights_path does not exist or is not a directory: {weights_path}"
                )
            model_type = read_model_type(weights_path)
            cls = OMNI_MODEL_REGISTRY[model_type]()
            logger.info_rank0(
                f"  building module '{name}' (model_type={model_type}, cls={cls.__name__}) from {weights_path}"
            )
            module = cls.from_pretrained(
                weights_path,
                torch_dtype=torch.bfloat16,
                **{k: v for k, v in mod_cfg.items() if not k.startswith("_")},
            )
            # TODO(omni-inferencer): weights materialise on CPU then copy to
            # the accelerator, doubling peak host RAM per module.  Switch to
            # ``device_map={"": self.device}`` once multi-device dispatch lands.
            module = module.to(self.device).eval()
            modules[name] = module
        return modules

    # ── Inference entry point ─────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs).

        Reads ``self.args.infer``, loads any image, applies CFG / sampling
        knobs, runs the FSM, and finally writes ``reply.txt`` /
        ``generated_image_*.png`` / ``trace.txt`` under
        ``self.args.infer.output_dir`` via :meth:`finalize`.

        Returned ``ctx`` carries (at minimum):

        * ``conversation_list`` — the final list of parts (prompt +
          assistant-side sampled tokens).
        * ``generated_images_collected`` — every image decoded by
          :class:`JanusVqvae` during the run (each ``(1, H, W, 3)`` float
          tensor in ``[-1, 1]``).
        * ``finalize`` — per-module dict from each :meth:`OmniModule.finalize`
          hook (notably ``finalize['janus_text_encoder']['text']`` for the
          decoded reply).

        For programmatic use that doesn't need on-disk artifacts (e.g. batched
        eval, RL rollouts), call :meth:`run_request` directly with a
        :class:`InferenceRequest`.
        """
        infer_args: OmniInferRunArguments = self.args.infer

        has_image = bool(infer_args.image)
        # T2I scenarios steer to image-VQ immediately; I2T stays in text_ar.
        force_image_gen = not has_image
        # CFG only kicks in for T2I — squelch to 1.0 for I2T because the
        # janus_text_encoder uncond branch isn't well-defined for prompts
        # containing image_und parts (see `_maybe_build_cfg_uncond_embeds`).
        guidance_scale = float(infer_args.guidance_scale) if force_image_gen else 1.0

        request = InferenceRequest(
            prompt=infer_args.prompt,
            images=[load_image(infer_args.image)] if has_image else [],
            force_image_gen=force_image_gen,
            generation_kwargs={
                "temperature": infer_args.temperature,
                "top_p": infer_args.top_p,
                "do_sample": infer_args.do_sample,
                "guidance_scale": guidance_scale,
            },
            max_new_tokens=infer_args.max_tokens,
        )
        trace_buf: list[str] | None = [] if infer_args.trace else None
        ctx = self.run_request(request, trace=trace_buf, progress=infer_args.progress)
        self.finalize(ctx, output_dir=infer_args.output_dir, trace=trace_buf)
        return ctx

    def run_request(
        self,
        request: InferenceRequest,
        *,
        trace: list[str] | None = None,
        progress: bool = False,
    ) -> dict[str, Any]:
        """Low-level entry: accept a fully-built :class:`InferenceRequest`.

        Returns the raw ``ctx`` dict without touching the filesystem — use
        this for programmatic / batched flows where the caller manages
        output persistence.  For one-shot script use, prefer :meth:`generate`
        (which also calls :meth:`finalize`).

        ``progress`` opts into :class:`OmniModel.generate`'s per-state stdout
        trail (e.g. ``[FSM] step    0: prompt_encode``).  Defaults to
        ``False`` here so programmatic batched callers stay quiet.
        """
        return self._run(request, trace=trace, progress=progress)

    # ── Output persistence ────────────────────────────────────────────────────

    def finalize(
        self,
        ctx: dict[str, Any],
        *,
        output_dir: str,
        trace: list[str] | None = None,
    ) -> None:
        """Persist every multimodal artifact produced by one ``generate`` call.

        Writes (under ``output_dir``, created if missing):

        * ``reply.txt`` — decoded assistant text from
          ``ctx['finalize'][...]['text']`` (UTF-8; Janus is multilingual).
          Always written, even when the reply is empty, so the file's
          existence signals "the FSM ran to completion".  When non-empty,
          also echoed to stdout via the logger (rank-0 gated) so CLI users
          see it without opening the file.
        * ``generated_image_{i}.png`` — every PIL image in
          ``ctx['generated_images_collected']``, one PNG per image.  The
          per-module ``generate`` (e.g. :class:`JanusVqvae`) already
          delegated the model-specific denormalisation to its own
          processor (:meth:`JanusVqvaeProcessor.postprocess`) so the
          inferencer just calls ``img.save(path)`` — no model-specific
          tensor math here.
        * ``trace.txt`` — FSM step / transition log (when ``trace`` was
          collected during this run).

        Stale-files behaviour: ``finalize`` does NOT clear existing
        ``generated_image_*.png`` / ``reply.txt`` / ``trace.txt`` from
        ``output_dir`` before writing.  A re-run that produces fewer images
        than the previous run will leave the extras on disk.  Callers that
        need a clean slate should ``rm -rf`` the directory themselves
        beforehand — keeping that policy in the caller's hands lets eval
        harnesses choose between "overwrite" and "append with separate dir".

        Logs a warning when the FSM produced no reply AND no images — that's
        usually a sign that the launcher YAML / scenario don't match the
        prompt (e.g. I2T scenario with no ``--infer.image``).
        """
        os.makedirs(output_dir, exist_ok=True)

        reply = _extract_reply(ctx)
        reply_path = os.path.join(output_dir, "reply.txt")
        # encoding="utf-8" is load-bearing — Janus is multilingual so reply
        # text may carry CJK / emoji.  Default-locale opens crash on `LANG=C`.
        with open(reply_path, "w", encoding="utf-8") as f:
            f.write(reply + ("\n" if reply and not reply.endswith("\n") else ""))
        logger.info_rank0(f"finalize: reply ({len(reply)} chars) → {reply_path}")
        if reply:
            # Long-form echo for CLI users — rank-0 gated via logger so it
            # doesn't multiplex on a future multi-rank dispatch.
            logger.info_rank0(f"--- reply ---\n{reply}\n-------------")

        images_out = list(ctx.get("generated_images_collected") or [])
        for idx, image in enumerate(images_out):
            out_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            image.save(out_path)
            logger.info_rank0(f"finalize: image #{idx} → {out_path}")

        if trace is not None:
            trace_path = os.path.join(output_dir, "trace.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("\n".join(trace) + "\n")
            logger.info_rank0(f"finalize: FSM trace ({len(trace)} lines) → {trace_path}")

        if not reply and not images_out:
            logger.warning_rank0("finalize: FSM produced no reply and no images.")

    # ── Internal ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _run(
        self,
        req: InferenceRequest,
        *,
        trace: list[str] | None = None,
        progress: bool = False,
    ) -> dict[str, Any]:
        # Reset per-call buffers held inside modules (VQ token grid, etc.).
        for module in self.modules.values():
            if hasattr(module, "reset_inference_state"):
                module.reset_inference_state()

        conversation = build_conversation(prompt=req.prompt, images=req.images)

        request_dict: dict[str, Any] = {
            "conversation_list": conversation,
            "force_image_gen": req.force_image_gen,
            "generation_kwargs": req.generation_kwargs,
        }
        # ``OmniModel.generate`` initialises ``ctx`` from ``context`` (or
        # from ``request`` when ``context`` is None) — we pass the same dict
        # for both so every key is visible to module kwargs.
        ctx = self.model.generate(
            request=request_dict,
            context=request_dict,
            max_new_tokens=req.max_new_tokens,
            trace=trace,
            progress=progress,
        )
        return ctx


# ── Module helpers ───────────────────────────────────────────────────────────


def _extract_reply(ctx: dict[str, Any]) -> str:
    """Pluck the decoded text from any module's ``finalize`` payload.

    The FSM exposes per-module finalize outputs at ``ctx['finalize'][name]``.
    Janus' text encoder writes the decoded reply under
    ``ctx['finalize']['janus_text_encoder']['text']`` but other models may
    name their module differently — so we scan every payload that's a dict
    carrying a ``"text"`` key.  Returns ``""`` when none are found (e.g. for
    pure T2I where the FSM never enters ``text_ar`` after the prompt).

    Contract for multi-module futures: when MORE than one finalize payload
    carries a ``"text"`` key, the FIRST one (by FSM finalize-call order,
    which is dict-insertion order) wins and a warning is logged.  Callers
    needing deterministic disambiguation should read ``ctx['finalize']``
    directly with an explicit module name.
    """
    finalize = ctx.get("finalize") or {}
    text_payloads = [(name, p) for name, p in finalize.items() if isinstance(p, dict) and "text" in p]
    if len(text_payloads) > 1:
        names = ", ".join(name for name, _ in text_payloads)
        logger.warning_rank0(
            f"_extract_reply: multiple modules produced text ({names}); returning "
            f"the first ('{text_payloads[0][0]}').  Read ctx['finalize'] directly to disambiguate."
        )
    if text_payloads:
        return text_payloads[0][1]["text"]
    return ""


__all__ = [
    "OmniInferencer",
    "InferenceRequest",
    "OmniInferModelArguments",
    "OmniInferRunArguments",
    "OmniInferenceArguments",
]
