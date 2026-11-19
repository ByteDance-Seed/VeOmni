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

"""HF-native base for every SeedOmni sub-module ``modeling.py``."""

from __future__ import annotations

import os
from typing import Any

from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel


class PretrainedOmniModule(PreTrainedModel):
    """Base for every ``modules/<family>/<sub>/modeling.py`` class.

    Subclasses hold weights, ``forward``, and FSM ``generate`` endpoints only.
    """

    #: Whether this module can hand its sequence-parallel shard straight to the
    #: next module in the graph instead of all-gathering it back to the full
    #: sequence (``ParallelState.skip_sp_gather_between_modules``).
    #:
    #: ``False`` by default. A module that implements the handoff sets
    #: ``supports_sp_sharded_handoff = True`` on its own class, beside its other
    #: class attributes; a family whose reason is not obvious says why in a
    #: comment next to it.
    #:
    #: What ``True`` claims: once ``pre_forward`` has sliced this rank's
    #: ``1 / sp_size`` chunk and ``forward`` has run, ``post_forward`` writes, item
    #: by item and in carrier order, exactly this rank's slice of the packed
    #: backbone sequence onto ``item.value`` -- no gather, no reordering, and no
    #: padding left on the last item beyond the SP pad tail, which the consumer
    #: recomputes from the same lengths. A module that gathers unconditionally, or
    #: whose outputs are not a plain concatenation of the carrier items (a packed
    #: MoT scattering through a dense two-stream mask, an image-batch-dim VAE),
    #: leaves this ``False``. A module that never moves sequence activations
    #: through the carrier (an n-gram table lookup) satisfies it trivially and may
    #: say ``True``.
    #:
    #: :func:`~veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.build_omni_model_runtime`
    #: rejects a run that turns the switch on while any participating module
    #: leaves this ``False``, so the two sides of a handoff can never disagree.
    supports_sp_sharded_handoff: bool = False

    #: Whether this module's blocks may be wrapped by ``torch.compile``.
    #:
    #: ``False`` by default. A module sets ``supports_torch_compile = True`` on
    #: its own class once its compile path has been measured to pay off; the
    #: flag is the per-module gate, and ``accelerator.torch_compile.enable``
    #: remains the run-wide switch — a module that opts in is compiled only when
    #: the run turns compile on, and a module that does not is never compiled.
    #:
    #: Measured on A100-80GB, one decoder block per module, forward+backward
    #: (``dynamic=False``, ``backend="inductor"``; larger is better):
    #:
    #: * ``qwen3_llm`` / ``qwen3vl_llm`` / ``janus_llama`` — ~1.32-1.58x. The
    #:   text backbones are where Inductor's fusion beats eager by the most.
    #: * ``janus_vqvae`` — ~1.74x over its 37 conv/attention blocks.
    #: * ``janus_siglip`` — ~1.12x, close to the noise floor for a 4-block
    #:   tower; kept on because the cost is a one-off compile.
    #: * ``bagel_siglip_navit`` — ~1.02x, i.e. nothing. Its patch embedding and
    #:   positional add dominate, so it stays ``False``.
    #:
    #: What ``True`` claims: this module's compile path was measured to pay off.
    #: It does **not** claim the shapes are stable — SeedOmni's collator does no
    #: padding (``SeedOmniCollator``: "no stacking, no padding"), so a text
    #: backbone sees a different packed length every step and needs
    #: :attr:`supports_torch_compile_dynamic` to avoid a fresh Inductor compile
    #: per new length (measured: 40 distinct lengths cost 362 s with
    #: ``dynamic=False`` against 13.5 s with ``dynamic=True``). A module whose
    #: shapes are already fixed by the processor (a tower, a codec) sets that
    #: flag ``False`` and pays nothing.
    supports_torch_compile: bool = False

    #: Whether this module's compile path tolerates ``dynamic=True``.
    #:
    #: ``dynamic`` lets one Inductor graph cover a range of shapes, which is what
    #: a multimodal run wants when the packed token count moves between steps.
    #: It is not free: Inductor emits shape-polymorphic kernels that cost more
    #: per call, and on a kernel-heavy module the tax can exceed the compile win
    #: outright. Measured per module (eager / static / dynamic, A100-80GB):
    #:
    #: * ``qwen3_llm`` / ``qwen3vl_llm`` / ``janus_llama`` — static and dynamic are
    #:   indistinguishable (1.00x). Text blocks are a handful of large matmuls;
    #:   shape polymorphism changes nothing.
    #: * ``janus_vqvae`` — dynamic is **30% slower** than static (30.7 -> 44.2 ms
    #:   per step, reproduced three times), i.e. it gives back most of the 1.73x
    #:   compile win. Its 37 small conv/attention blocks are exactly the shape
    #:   that pays the polymorphic-kernel tax.
    #:
    #: So a module whose input shape is fixed by the image processor sets this
    #: ``False`` — the run-level ``dynamic`` flag is then overridden off for this
    #: module only, and the other modules keep it. A module whose shape moves
    #: sets it ``True``; for the text backbones that is not optional, since the
    #: collator pads nothing (see :attr:`supports_torch_compile`).
    supports_torch_compile_dynamic: bool = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then bind module-owned processor / tokenizer sidecars."""
        from .module_processing_base import bind_module_assets

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # ``kwargs`` may carry an HF ``config=<PretrainedConfig>`` (the loaded
        # module config, forwarded by ``OmniModel._load_modules``) — that's a
        # different "config" than the launcher/runtime overrides dict
        # ``config_overrides`` represents. Strip the HF ``config`` key before
        # binding so it cannot collide with a preprocessor helper that also
        # takes a positional ``config``.
        config_overrides = {k: v for k, v in kwargs.items() if k != "config"}
        bind_module_assets(
            model,
            checkpoint_path=str(pretrained_model_name_or_path),
            config_overrides=config_overrides,
        )
        return model

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *args: Any,
        save_module_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write this module's config and its processor / tokenizer sidecars.

        ``save_module_weights=False`` skips the weight files. The sidecars are
        written either way, so a weights-free export can still preprocess.
        Weights go last so this module's ``config.json`` is the one that remains.
        """
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
            asset = getattr(self, attr, None)
            if asset is not None and hasattr(asset, "save_pretrained"):
                asset.save_pretrained(save_directory)
        if not save_module_weights:
            self.config.save_pretrained(save_directory)
            return
        super().save_pretrained(save_directory, *args, **kwargs)

    def get_assets(self) -> list[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    def generate(self, generation_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Default FSM endpoint: same as training ``forward``. Override when inference differs.

        :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` calls
        ``generate(**ctx, generation_kwargs=...)`` for every generation-graph
        node. Bare endpoints resolve to this method.
        """
        del generation_kwargs
        return self.forward(**kwargs)

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing generation request."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full request-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Abort-only flush of leftover buffers into a one-shot ``generated`` payload.

        Invoked when generation hits ``max_new_tokens`` before ``done``, not
        after a normal FSM completion.
        """
        del ctx
        return {}


__all__ = ["PretrainedOmniModule"]
