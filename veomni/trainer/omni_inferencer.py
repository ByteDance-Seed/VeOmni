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
   joined under the split-checkpoint root).  Use
   :meth:`OmniInferencer.from_launcher` to build one from a launcher
   YAML path in one call.

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
   :meth:`generate` to turn raw PIL images into ``pixel_values``
   before the FSM runs.

5. :meth:`generate` builds a :class:`ConversationPart` list from the
   user payload, attaches per-image ``pixel_values`` via the matching
   processor, packs the inputs as a ``request`` / ``context`` dict
   (including ``force_image_gen`` and a free-form
   ``generation_kwargs`` payload), and hands off to
   :meth:`OmniModel.generate`.  The returned ``ctx`` carries the FSM
   trace, the accumulated ``generated_images_collected`` tensors, and
   the per-module ``finalize`` outputs (e.g. decoded text under
   ``finalize['janus_text_encoder']['text']``).

Generation kwargs distribution
------------------------------
``generation_kwargs`` is a plain dict that the inferencer simply
forwards into ``request`` / ``ctx``.  Every module's ``generate``
reads the slot from kwargs and filters by the keys it recognises
(:meth:`JanusTextEncoder._extract_sampling_kwargs` /
:meth:`JanusVqvae._extract_sampling_kwargs`) — same shape as
HuggingFace ``GenerationMixin._prepare_generation_config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer, PretrainedConfig

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
    def from_launcher(
        cls,
        launcher_yaml: str | Path,
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

    def generate(
        self,
        *,
        prompt: str,
        images: list[Any] | None = None,
        force_image_gen: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
        max_new_tokens: int = 2048,
        trace: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one inference request through the FSM, return ``ctx``.

        The returned dict carries (at minimum):

        * ``conversation_list`` — the final list of parts (prompt +
          assistant-side sampled tokens).
        * ``generated_images_collected`` — every image decoded by
          :class:`JanusVqvae` during the run (each ``(1, H, W, 3)``
          float tensor in ``[-1, 1]``).
        * ``finalize`` — per-module dict from each
          :meth:`OmniModule.finalize` hook (notably
          ``finalize['janus_text_encoder']['text']`` for the decoded
          reply).
        """
        request = InferenceRequest(
            prompt=prompt,
            images=list(images or []),
            force_image_gen=force_image_gen,
            generation_kwargs=dict(generation_kwargs or {}),
            max_new_tokens=max_new_tokens,
        )
        return self._run(request, trace=trace)

    def generate_from_request(
        self,
        request: InferenceRequest,
        *,
        trace: list[str] | None = None,
    ) -> dict[str, Any]:
        """Lower-level entry: accept a fully-built :class:`InferenceRequest`."""
        return self._run(request, trace=trace)

    # ── Internal ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _run(self, req: InferenceRequest, *, trace: list[str] | None = None) -> dict[str, Any]:
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
        ctx = self.model.generate(
            request=request_dict,
            context=request_dict,
            max_new_tokens=req.max_new_tokens,
            trace=trace,
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


__all__ = ["OmniInferencer", "InferenceRequest"]
