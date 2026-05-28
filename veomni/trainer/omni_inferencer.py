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

Lifecycle
---------
1. ``OmniInferencer(cfg)`` takes a fully-resolved :class:`OmniConfig`
   (already deep-merged with the desired ``generation_graph`` from
   training + inference YAMLs, with module ``weights_path`` values
   joined under the split-checkpoint root).  Two convenience factories
   build a ready inferencer in one call:

   * :meth:`OmniInferencer.from_args` — preferred for CLI / script
     entry points.  Accepts a :class:`OmniInferenceArguments` produced
     by :func:`veomni.arguments.parse_args`; resolves the scenario,
     loads the merged :class:`OmniConfig`, and stashes ``args.infer``
     so :meth:`generate` can be called with no arguments.
   * :meth:`OmniInferencer.from_launcher` — programmatic alternative
     that takes a launcher YAML path + optional ``infer_type`` /
     ``model_path`` overrides.  Does NOT stash run-time args, so the
     caller must pass an :class:`OmniInferRunArguments` to
     :meth:`generate` (or use :meth:`run_request` instead).

2. The constructor builds each declared module by reading its
   ``weights_path/config.json`` to look up ``model_type`` and
   dispatching through :data:`OMNI_MODEL_REGISTRY` to the right
   :class:`PreTrainedModel`-derived class.  Modules are loaded eagerly
   on a single device via ``cls.from_pretrained(weights_path,
   torch_dtype, **module_overrides)`` — no FSDP, no meta init,
   ``device_map`` deferred to a follow-up PR.

3. The global tokenizer is loaded from ``OmniConfig.tokenizer_path``
   and wired into every module that exposes ``set_tokenizer``
   (notably :class:`JanusTextEncoder`, which resolves bos / boi / eoi /
   eos ids at this point).

4. Per-module processors (vision / vqvae) are loaded the same way via
   :data:`OMNI_PROCESSOR_REGISTRY` and stashed for use by
   :meth:`_run` to turn raw PIL images into ``pixel_values``
   before the FSM runs.

5. Inference is a three-layer call stack:

   * :meth:`generate` — high-level: reads :class:`OmniInferRunArguments`
     (passed explicitly or stashed via :meth:`from_args`), loads any
     image, applies the CFG / sampling knobs, calls :meth:`run_request`,
     then writes every artifact to disk via :meth:`finalize`.
   * :meth:`run_request` — mid-level: accepts a built
     :class:`InferenceRequest`, returns the raw ``ctx`` dict without
     touching the filesystem.  Use for programmatic / batched flows.
   * :meth:`_run` — internal: resets per-module state, builds the
     :class:`ConversationPart` list via :func:`build_conversation`,
     attaches per-image ``pixel_values`` via the matching processor,
     packs the inputs as a ``request`` / ``context`` dict and hands
     off to :meth:`OmniModel.generate`.

   The returned ``ctx`` carries the FSM trace, the accumulated
   ``generated_images_collected`` tensors, and the per-module
   ``finalize`` outputs (e.g. decoded text under
   ``finalize['janus_text_encoder']['text']``).

Generation kwargs distribution
------------------------------
``generation_kwargs`` is a plain dict that :meth:`run_request` (and
the public :meth:`generate` wrapper) forwards into ``request`` /
``ctx``.  Every module's ``generate`` reads the slot from kwargs and
filters by the keys it recognises
(:meth:`JanusTextEncoder._extract_sampling_kwargs` /
:meth:`JanusVqvae._extract_sampling_kwargs`) — same shape as
HuggingFace ``GenerationMixin._prepare_generation_config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer, PretrainedConfig

from ..arguments import InferArguments
from ..data.multimodal.image_utils import load_image, save_image_tensors_to_file
from ..models.seed_omni import (
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    ConversationPart,
    OmniConfig,
    OmniModel,
    build_conversation,
)
from ..utils import helper
from ..utils.device import get_device_type


logger = helper.create_logger(__name__)


@dataclass
class OmniInferModelArguments:
    """``model.*`` — slim model arguments for SeedOmni V2 inference / visualization.

    Carries ONLY the four fields actually consumed by the FSM-driven inference
    path: ``model_path`` + the three ``omni_*`` graph pointers.  Intentionally
    does NOT inherit from :class:`ModelArguments` / :class:`OmniModelArguments`
    (which carry training-only knobs like ``ops_implementation``, ``encoders``
    /``decoders``, ``lora_config``, ``safetensor_idx_path``, ``basic_modules``,
    etc.) — dragging those into an inference CLI would bloat ``--help`` with
    dozens of irrelevant flags and silently honour training defaults that have
    no effect at inference time.

    Use with :func:`veomni.arguments.parse_args` like::

        from veomni.arguments import parse_args
        from veomni.trainer.omni_inferencer import OmniInferenceArguments, OmniInferencer

        args = parse_args(OmniInferenceArguments)
        OmniInferencer.from_args(args).generate()  # writes outputs to args.infer.output_dir

    See :class:`veomni.trainer.omni_trainer.OmniModelArguments` for the
    training-side analogue (which DOES inherit ``ModelArguments`` because the
    trainer needs the full surface).
    """

    model_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Local/HDFS path to the split-checkpoint root (e.g. "
                "``/tmp/janus_1.3b_split``).  Required for inference (the global "
                "tokenizer and per-module weights are loaded from here); "
                "optional for graph visualization (which doesn't load weights)."
            )
        },
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
                "Mapping of inference scenario name → inference YAML path.  Each "
                "scenario YAML deep-merges over ``omni_train_yaml_path`` to inject "
                "a ``generation_graph``.  Example keys: ``infer_gen`` (T2I), "
                "``infer_und`` (I2T), ``infer_interleave``."
            )
        },
    )
    omni_infer_type: str | None = field(
        default=None,
        metadata={
            "help": (
                "Active scenario key into ``omni_infer_yaml_path``.  If unset, the "
                "calling script may auto-pick from context (e.g. ``infer_omni.py`` "
                "picks ``infer_und`` when ``--infer.image`` is given, ``infer_gen`` "
                "otherwise)."
            )
        },
    )

    def load_omni_config(self, *, infer_type: str | None = None) -> OmniConfig:
        """Build an :class:`OmniConfig` with per-module ``weights_path`` resolved.

        Mirrors :meth:`veomni.trainer.omni_trainer.OmniModelArguments.load_omni_config`
        so the inference path doesn't drift from training-side path resolution.
        ``infer_type`` (if provided) overrides :attr:`omni_infer_type` for this
        call only — the dataclass field is not mutated.
        """
        from ..models.seed_omni.configuration_seed_omni import apply_model_path

        if not self.omni_train_yaml_path:
            raise ValueError("`model.omni_train_yaml_path` is required for OmniModel V2 inference.")
        if not self.model_path:
            raise ValueError("`model.model_path` is required for OmniModel V2 inference.")

        paths = [self.omni_train_yaml_path]
        selected = infer_type or self.omni_infer_type
        if selected is not None:
            infer_map = self.omni_infer_yaml_path or {}
            if selected not in infer_map:
                known = ", ".join(sorted(infer_map)) or "(none)"
                raise KeyError(f"Unknown omni_infer_type {selected!r}; expected one of: {known}.")
            paths.append(infer_map[selected])

        cfg = OmniConfig.from_yamls(*paths)
        return apply_model_path(cfg, self.model_path)


@dataclass
class OmniInferRunArguments(InferArguments):
    """``infer.*`` — per-invocation inference knobs for ``infer_omni``.

    Extends framework :class:`~veomni.arguments.InferArguments` with the
    SeedOmni V2 driver's prompt / image / output / CFG fields.

    Inherited and used: ``seed``, ``temperature``, ``top_p``, ``max_tokens``,
    ``do_sample``.

    Inherited and silently ignored: ``model_path`` / ``tokenizer_path``.
    The V2 entry-point sources the model from the launcher YAML's ``model:``
    section via :class:`OmniInferModelArguments`, so ``--model.model_path`` is
    the canonical override.  These two fields are overridden here to default
    to ``None`` (the base class declares ``model_path`` as mandatory) and are
    documented as "ignored" so users don't reach for the wrong flag.

    Lives in this module (not the script) so other drivers (RL eval,
    batched-prompt benchmarks, future REST shims) can reuse the same CLI
    surface without copy-pasting the dataclass.
    """

    model_path: str | None = field(
        default=None,
        metadata={"help": "Ignored — use --model.model_path to override the launcher YAML."},
    )
    tokenizer_path: str | None = field(
        default=None,
        metadata={"help": "Ignored — tokenizer is loaded from --model.model_path (the split-ckpt root)."},
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
        metadata={"help": "Directory for reply.txt + generated_image_*.png + trace.txt (created if missing)."},
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
        # Base-class __post_init__ tries `self.tokenizer_path = self.model_path`
        # when tokenizer_path is None.  That's harmless when both are None
        # (which is the V2 case), but we skip the super call to make the
        # "both ignored" semantics explicit and avoid mutating an unused field.
        pass


@dataclass
class OmniInferenceArguments:
    """Root config for SeedOmni V2 inference — consumed by :func:`parse_args`.

    The launcher YAML's top-level ``model:`` section populates :attr:`model`,
    keeping only the four fields that actually drive inference (model path +
    graph YAML pointers).  ``data:`` / ``train:`` and every other training-only
    ``model.*`` field declared in the launcher YAML (``ops_implementation``,
    ``encoders``, ``lora_config``, ...) are silently ignored because they're
    not declared on :class:`OmniInferModelArguments`.

    Use::

        from veomni.arguments import parse_args
        from veomni.trainer.omni_inferencer import OmniInferenceArguments, OmniInferencer

        args = parse_args(OmniInferenceArguments)
        OmniInferencer.from_args(args).generate()
    """

    model: OmniInferModelArguments = field(default_factory=OmniInferModelArguments)
    infer: OmniInferRunArguments = field(default_factory=OmniInferRunArguments)


def _select_scenario(model_args: OmniInferModelArguments, *, has_image: bool) -> tuple[str, str]:
    """Pick the inference scenario (an entry in ``model.omni_infer_yaml_path``).

    Priority:

    1. ``model.omni_infer_type`` if set (via CLI ``--model.omni_infer_type`` or YAML).
       An unknown value raises ``KeyError`` — silently falling through to auto-pick
       would mask typos (e.g. ``--model.omni_infer_type infrr_gen``) that the user
       would otherwise want to catch immediately.
    2. Auto: image present → ``infer_und``, absent → ``infer_gen``.
    3. Any scenario declared in ``omni_infer_yaml_path`` (first key).

    Returns
    -------
    ``(scenario, source)`` where ``source`` is one of ``"yaml/cli"`` / ``"auto"``
    / ``"fallback"`` — the caller logs this so users can tell which priority
    path fired (especially important since the launcher YAML may declare
    ``omni_infer_type`` for training-time generation samples, which now also
    applies to inference).
    """
    infer_map = dict(model_args.omni_infer_yaml_path or {})
    if not infer_map:
        raise ValueError(
            "Launcher YAML declares no `model.omni_infer_yaml_path` entries — cannot pick an inference scenario."
        )
    if model_args.omni_infer_type:
        if model_args.omni_infer_type not in infer_map:
            known = ", ".join(sorted(infer_map)) or "(none)"
            raise KeyError(
                f"Unknown model.omni_infer_type {model_args.omni_infer_type!r}; expected one of: {known}.  "
                f"Set via CLI: --model.omni_infer_type <key>."
            )
        return model_args.omni_infer_type, "yaml/cli (model.omni_infer_type)"
    preferred = "infer_und" if has_image else "infer_gen"
    if preferred in infer_map:
        return preferred, f"auto (has_image={has_image})"
    return next(iter(infer_map)), "fallback (first available)"


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

    Single-process, single-device, no FSDP.  Designed to be the
    inference twin of :class:`OmniTrainer`; both share the
    registry-based module construction contract documented in
    :mod:`veomni.models.seed_omni.modules`, but the inferencer takes a
    pre-built :class:`OmniConfig` directly (the trainer-side dataclass
    plumbing is irrelevant for one-shot inference scripts).

    Parameters
    ----------
    cfg:
        Fully resolved :class:`OmniConfig` — must already carry
        ``modules`` with absolute / resolved ``weights_path`` values,
        a ``tokenizer_path``, and a ``generation_graph``.  Use
        :meth:`from_launcher` to build one from a launcher YAML.
    dtype:
        ``"float16"`` / ``"bfloat16"`` / ``"float32"``.  Defaults to
        ``"bfloat16"`` on GPU / NPU and ``"float32"`` on CPU.
    seed:
        Random seed for the sampler.  Forwarded to
        :func:`veomni.utils.helper.set_seed`.
    """

    def __init__(
        self,
        cfg: OmniConfig,
        *,
        dtype: str | None = None,
        seed: int = 42,
        infer_args: OmniInferRunArguments | None = None,
    ):
        self.device = torch.device(get_device_type())
        self.dtype = _resolve_dtype(dtype or "bfloat16", self.device)
        helper.set_seed(seed)

        if not isinstance(cfg, OmniConfig):
            raise TypeError(f"OmniInferencer expects an OmniConfig, got {type(cfg).__name__}.")
        if not cfg.tokenizer_path:
            raise ValueError(
                "OmniConfig.tokenizer_path is unset — use OmniInferencer.from_launcher() or "
                "apply_model_path(cfg, model_path) before constructing OmniInferencer."
            )
        if not cfg.has_generation_graph():
            raise ValueError(
                "OmniInferencer requires an OmniConfig with a `generation_graph` (deep-merge an "
                "infer_*.yaml on top of the training YAML, or pass `infer_type=...` to "
                "OmniInferencer.from_launcher)."
            )
        self.cfg = cfg

        # Stashed per-run knobs (set by :meth:`from_args`).  ``generate()``
        # consults this when no explicit overrides are passed — the common
        # "script just calls ``inferencer.generate()``" flow.  None for
        # programmatic constructions; explicit args on ``generate()`` are
        # required in that case.
        self.infer_args: OmniInferRunArguments | None = infer_args

        logger.info_rank0(
            f"OmniInferencer: loaded OmniConfig with {len(self.cfg.module_names)} modules ({self.cfg.module_names})."
        )

        # Per-module HF PreTrainedModel instances (eager init, single device).
        self.modules: dict[str, torch.nn.Module] = self._build_modules()
        # Per-module ProcessorMixin (only modules with a registered processor).
        self.processors: dict[str, Any] = self._build_processors()

        # Global tokenizer — wired into every text-aware module so they can
        # resolve special-token ids (bos / boi / eoi / eos) at runtime
        # instead of baking them into config.json.
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_path)
        for module in self.modules.values():
            if hasattr(module, "set_tokenizer"):
                module.set_tokenizer(self.tokenizer)

        # Compose the runtime OmniModel.  The runtime is the same one the
        # trainer uses — graph traversal + finalize hook + per-step
        # image collection.
        self.model = OmniModel(self.cfg, self.modules).eval()
        # `OmniModel.modules_dict` is a back-compat view; module instances
        # are already on-device from `_build_modules`, no `.to()` cascade.

    # ── Factories ────────────────────────────────────────────────────────────

    @classmethod
    def from_args(
        cls,
        args: OmniInferenceArguments,
        *,
        dtype: str | None = None,
    ) -> OmniInferencer:
        """Build an :class:`OmniInferencer` from a fully-parsed root config.

        One-stop constructor for CLI / script entry points: resolves the
        inference scenario, loads the merged :class:`OmniConfig`, builds the
        per-module weights, and stashes :attr:`args.infer` so subsequent
        ``inferencer.generate()`` calls without arguments use that payload.

        Validates the two CLI mandatories (``--infer.prompt`` non-empty,
        ``--model.model_path`` set) up front so the user sees a clear error
        before module construction starts (which is the slow / expensive part).
        """
        if not args.infer.prompt:
            raise ValueError(
                "`--infer.prompt` is required (use a non-empty string).  Example: --infer.prompt 'a cat'."
            )
        if not args.model.model_path:
            raise ValueError("`--model.model_path` is required (launcher YAML must declare `model.model_path`).")

        scenario, source = _select_scenario(args.model, has_image=bool(args.infer.image))
        logger.info_rank0(f"OmniInferencer.from_args: model_path = {args.model.model_path}")
        logger.info_rank0(f"OmniInferencer.from_args: scenario = {scenario}  (source: {source})")

        cfg = args.model.load_omni_config(infer_type=scenario)
        return cls(cfg, dtype=dtype, seed=args.infer.seed, infer_args=args.infer)

    @classmethod
    def from_launcher(
        cls,
        launcher_yaml: str | os.PathLike,
        *,
        infer_type: str | None = None,
        model_path: str | None = None,
        dtype: str | None = None,
        seed: int = 42,
    ) -> OmniInferencer:
        """Build an :class:`OmniInferencer` from a launcher YAML path.

        Reads the launcher's ``model:`` section to discover the split
        checkpoint root, the training YAML, and the inference YAML
        catalogue.  ``infer_type`` overrides the launcher's
        ``omni_infer_type`` so callers can swap scenarios per request
        without touching the YAML on disk.  ``model_path`` overrides
        the launcher's ``model.model_path`` for the same reason
        (typical use: the in-repo YAML points at a placeholder under
        ``seed_omni/janus_1.3b`` and the real split lives somewhere
        else on the host).
        """
        from ..models.seed_omni.configuration_seed_omni import apply_model_path, load_launcher_model_section

        if model_path is None:
            cfg = OmniConfig.from_launcher(launcher_yaml, infer_type=infer_type)
        else:
            # Mirror OmniConfig.from_launcher but apply the override before
            # path resolution so relative module paths join under the new root.
            model = load_launcher_model_section(launcher_yaml)
            train_yaml = model.get("omni_train_yaml_path")
            if not train_yaml:
                raise ValueError(f"`model.omni_train_yaml_path` is required in {launcher_yaml!s}.")
            infer_map = model.get("omni_infer_yaml_path") or {}
            selected = infer_type or model.get("omni_infer_type")
            paths = [train_yaml]
            if selected is not None:
                if selected not in infer_map:
                    known = ", ".join(sorted(infer_map)) or "(none)"
                    raise KeyError(
                        f"Unknown omni_infer_type {selected!r} in {launcher_yaml!s}; expected one of: {known}."
                    )
                paths.append(infer_map[selected])
            cfg = apply_model_path(OmniConfig.from_yamls(*paths), str(model_path))

        return cls(cfg, dtype=dtype, seed=seed)

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
        for name in self.cfg.module_names:
            mod_cfg = self.cfg.module_config(name)
            weights_path = mod_cfg.pop("weights_path", None)
            if not weights_path:
                raise ValueError(f"Module '{name}' is missing `weights_path` in OmniConfig.modules.")
            if not os.path.isdir(weights_path):
                raise FileNotFoundError(
                    f"Module '{name}' weights_path does not exist or is not a directory: {weights_path}"
                )
            model_type = _read_model_type(weights_path)
            cls = OMNI_MODEL_REGISTRY[model_type]()
            logger.info_rank0(
                f"  building module '{name}' (model_type={model_type}, cls={cls.__name__}) from {weights_path}"
            )
            module = cls.from_pretrained(
                weights_path,
                torch_dtype=self.dtype,
                **{k: v for k, v in mod_cfg.items() if not k.startswith("_")},
            )
            # TODO(omni-inferencer): weights materialise on CPU first and
            # then get copied to the accelerator, doubling peak host RAM
            # while a module is in flight.  Once we add multi-device
            # dispatch we should switch to `device_map={"": self.device}`
            # on `from_pretrained` so weights land directly on-device.
            module = module.to(self.device).eval()
            modules[name] = module
        return modules

    def _build_processors(self) -> dict[str, Any]:
        """Best-effort load of per-module processors.

        Modules without a registered processor (``janus_llama``,
        ``janus_text_encoder``) are silently skipped — they consume
        already-tokenised / already-tensorised inputs.
        """
        processors: dict[str, Any] = {}
        proc_keys = set(OMNI_PROCESSOR_REGISTRY.valid_keys())
        for name in self.cfg.module_names:
            mod_cfg = self.cfg.module_config(name)
            weights_path = mod_cfg.get("weights_path")
            if not weights_path:
                continue
            model_type = _read_model_type(weights_path)
            if model_type not in proc_keys:
                continue
            proc_cls = OMNI_PROCESSOR_REGISTRY[model_type]()
            try:
                processors[name] = proc_cls.from_pretrained(weights_path)
                logger.info_rank0(f"  loaded processor for '{name}' ({proc_cls.__name__})")
            except Exception as exc:  # noqa: BLE001 — processors are best-effort
                logger.warning_rank0(f"  skipping processor for '{name}' ({proc_cls.__name__}): {exc}")
        return processors

    # ── Inference entry point ─────────────────────────────────────────────────

    def generate(self, infer_args: OmniInferRunArguments | None = None) -> dict[str, Any]:
        """Run one inference request end-to-end (FSM + save outputs).

        High-level convenience method matching the "build inferencer, call
        ``generate()``" usage pattern.  Reads :class:`OmniInferRunArguments`
        from ``infer_args`` (when provided) or from :attr:`self.infer_args`
        (set by :meth:`from_args`).  After the FSM completes, :meth:`finalize`
        writes ``reply.txt`` / ``generated_image_*.png`` / ``trace.txt`` under
        ``infer_args.output_dir``.

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
        args = infer_args or self.infer_args
        if args is None:
            raise ValueError(
                "OmniInferencer.generate() requires either an explicit `infer_args` argument "
                "or a prior `OmniInferencer.from_args(...)` to stash one.  For programmatic "
                "use without on-disk artifacts, call `OmniInferencer.run_request(InferenceRequest(...))` instead."
            )
        if not args.prompt:
            raise ValueError("OmniInferRunArguments.prompt is empty — set a non-empty prompt.")

        has_image = bool(args.image)
        # T2I scenarios steer to image-VQ immediately; I2T stays in text_ar.
        force_image_gen = not has_image
        # CFG only kicks in for T2I — for I2T we squelch back to 1.0 so the
        # janus_text_encoder skips the uncond branch (its construction rule
        # isn't well-defined for prompts containing image_und parts; see
        # `_maybe_build_cfg_uncond_embeds`).
        guidance_scale = float(args.guidance_scale) if force_image_gen else 1.0

        request = InferenceRequest(
            prompt=args.prompt,
            images=[load_image(args.image)] if has_image else [],
            force_image_gen=force_image_gen,
            generation_kwargs={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
                "guidance_scale": guidance_scale,
            },
            max_new_tokens=args.max_tokens,
        )
        trace_buf: list[str] | None = [] if args.trace else None
        ctx = self.run_request(request, trace=trace_buf, progress=args.progress)
        self.finalize(ctx, output_dir=args.output_dir, trace=trace_buf)
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

        ``progress`` opts into :class:`OmniModel.generate`'s per-state
        stdout trail (e.g. ``[FSM] step    0: prompt_encode``).  Defaults
        to ``False`` here so programmatic batched callers stay quiet; the
        :meth:`generate` wrapper passes ``args.progress`` (default True)
        for CLI use.
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
        * ``generated_image_{i}.png`` — every tensor in
          ``ctx['generated_images_collected']``, one PNG per image.  Each
          tensor is squeezed to ``(H, W, 3)``, mapped from VQVAE's ``[-1, 1]``
          range to ``[0, 1]``, then handed to
          :func:`veomni.data.multimodal.image_utils.save_image_tensors_to_file`
          (which does ``×255 → round → uint8 → PIL``).
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

        images_out: list[torch.Tensor] = list(ctx.get("generated_images_collected") or [])
        for idx, tensor in enumerate(images_out):
            out_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            _save_generated_image(tensor, out_path)
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
        self._attach_pixel_values(conversation)

        request_dict: dict[str, Any] = {
            "conversation_list": conversation,
            "force_image_gen": req.force_image_gen,
            "generation_kwargs": req.generation_kwargs,
        }
        # ``OmniModel.generate`` initialises ``ctx`` from ``context``
        # (or from ``request`` when ``context`` is None) — we pass the
        # same dict for both so every key is visible to module kwargs.
        # ``progress`` flows from the caller (False by default for
        # batched / programmatic flows; the CLI :meth:`generate` wrapper
        # forwards ``args.progress`` which defaults to True).
        ctx = self.model.generate(
            request=request_dict,
            context=request_dict,
            max_new_tokens=req.max_new_tokens,
            trace=trace,
            progress=progress,
        )
        return ctx

    def _attach_pixel_values(self, conversation: list[ConversationPart]) -> None:
        """Tensorise every ``image_und`` part's raw PIL via the siglip processor.

        Modules consume already-tensor ``pixel_values`` so we run the
        processor here (single-process, CPU is fine) and let the
        :meth:`JanusSiglip.generate` per-part loop move the tensors to
        the model's device.  Skipped silently for parts that already
        carry a ``pixel_values`` tensor (callers may pre-process).
        """
        proc = self.processors.get("janus_siglip")
        for part in conversation:
            if part.kind != "image_und" or part.pixel_values is not None:
                continue
            if part.image is None:
                continue
            if proc is None:
                raise RuntimeError(
                    "OmniInferencer: an `image_und` part has no `pixel_values` and no "
                    "siglip processor was registered. Either pre-tensorise the part or "
                    "ensure `janus_siglip/preprocessor_config.json` exists in the split "
                    "checkpoint."
                )
            out = proc(images=[part.image], return_tensors="pt")
            pv = out["pixel_values"]
            if pv.dim() == 4 and pv.size(0) == 1:
                pv = pv.squeeze(0)
            part.pixel_values = pv


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


def _save_generated_image(tensor: torch.Tensor, out_path: str) -> None:
    """Save a Janus-VQVAE-decoded image tensor as PNG.

    Janus emits ``(1, H, W, 3)`` (or ``(H, W, 3)``) float tensors in ``[-1, 1]``.
    We squeeze the batch dim, map ``[-1, 1] → [0, 1]``, then delegate to
    :func:`veomni.data.multimodal.image_utils.save_image_tensors_to_file`
    (which expects ``[0, 1]`` H×W×3 and does the
    ``×255 → clamp → round → uint8 → PIL`` dance — note the rounding step
    is what keeps the saved PNG bit-exact against the HF baseline
    ``scripts/multimodal/infer/janus_hf_infer_gen.py`` reference encoder).

    Lives here rather than in ``image_utils`` because the ``[-1, 1]`` range
    convention is Janus-specific — other decoders (SD VAE, MAGVIT, ...) emit
    different ranges and would do their own normalisation before calling the
    shared sink.
    """
    img = tensor.detach().to(dtype=torch.float32, device="cpu")
    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
    if img.dim() != 3 or img.size(-1) != 3:
        raise ValueError(f"Cannot save image with shape {tuple(img.shape)}; expected (H, W, 3).")
    # `clamp` guards against out-of-range entries from a misbehaving decoder;
    # the +1/2 mapping is the exact inverse of Janus' ``2x - 1`` preprocess.
    img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image_tensors_to_file(img, out_path)


def _read_model_type(weights_path: str) -> str:
    """Read ``model_type`` from a module's ``config.json``.

    Uses :meth:`PretrainedConfig.get_config_dict` (not
    :class:`AutoConfig.from_pretrained`) because Janus split-checkpoint
    configs declare custom ``model_type`` values
    (``janus_siglip`` / ``janus_text_encoder`` / ``janus_llama`` /
    ``janus_vqvae``) that are not in HF's :data:`CONFIG_MAPPING`.
    ``AutoConfig`` would raise on those families before we even get a
    chance to consult :data:`OMNI_MODEL_REGISTRY`; reading the raw dict
    sidesteps that.  See :mod:`veomni.models.loader` for the same
    pattern in the foundation-model loader.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(weights_path)
    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Module config at {weights_path} has no `model_type` — cannot resolve OmniModule class.")
    # Note: :class:`Registry.__getitem__` raises ``ValueError`` (not
    # ``KeyError``) on miss, so the default ``in`` test on a MutableMapping
    # subclass would mis-route the exception.  Use ``valid_keys()`` to
    # decide registration explicitly.
    config_keys = set(OMNI_CONFIG_REGISTRY.valid_keys())
    model_keys = set(OMNI_MODEL_REGISTRY.valid_keys())
    if model_type in config_keys:
        # Validate the config can be re-read by the registered subclass so
        # downstream `from_pretrained` doesn't hit a surprise schema gap.
        cfg_cls = OMNI_CONFIG_REGISTRY[model_type]()
        cfg_cls.from_pretrained(weights_path)
    if model_type not in model_keys:
        raise KeyError(
            f"Module model_type {model_type!r} (from {weights_path}) is not registered in "
            f"OMNI_MODEL_REGISTRY. Known: {sorted(model_keys)}."
        )
    return model_type


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}.")
    return mapping[name]


__all__ = [
    "OmniInferencer",
    "InferenceRequest",
    "OmniInferModelArguments",
    "OmniInferRunArguments",
    "OmniInferenceArguments",
]
