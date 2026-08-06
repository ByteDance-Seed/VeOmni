"""
BaseMixin — minimal SeedOmni V2 lifecycle + shared graph-hook registry.

Every module composes capability mixins (training / inference / meter / …) into a
local ``VeOmniMixin``; ``modeling.py`` inherits only ``VeOmniMixin`` +
``PreTrainedModel``.

Layout
------
* ``base_mixin.py`` — :class:`BaseMixin` (``from_pretrained``, ``get_assets``, hook registry)
* ``training_module_mixin.py`` — :class:`TrainingModuleMixin` (``pre_forward`` / ``post_forward``)
* ``inference_module_mixin.py`` — :class:`InferenceModuleMixin` (``pre_generate`` / ``post_generate``)
* ``modules/<family>/<sub>/modulemixin.py``::

    class TrainingMixin(TrainingModuleMixin): ...
    class InferenceMixin(InferenceModuleMixin): ...
    class MeterMixin(MetricMeterMixin): ...
    class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin, MeterMixin): ...

* ``modules/<family>/<sub>/modeling.py``::

    class Xxx(VeOmniMixin, PreTrainedModel): ...
"""

from __future__ import annotations

from typing import Any


class BaseMixin:
    """SeedOmni V2 base — checkpoint load, asset export, shared hook-name registry."""

    @classmethod
    def _omni_hook_name(cls, marker: str, context: str) -> str | None:
        """Resolve the method name tagged ``marker`` for call-site ``context``.

        Shared by training (``@pre_forward`` / ``@post_forward``) and inference
        (``@pre_generate`` / ``@post_generate``) dispatchers.
        """
        cache_attr = f"__omni_hooks_{marker}__"
        registry: dict[str, str] | None = cls.__dict__.get(cache_attr)
        if registry is None:
            registry = {}
            for klass in reversed(cls.__mro__):
                for name, attr in vars(klass).items():
                    contexts = getattr(attr, marker, None)
                    if contexts is not None:
                        for ctx in contexts:
                            registry[ctx] = name
            setattr(cls, cache_attr, registry)
        return registry.get(context)

    def get_assets(self) -> list[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then copy HF assets from the module's CPU worker if declared."""
        from ..processing.binding import bind_module_assets

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        bind_module_assets(
            model,
            checkpoint_path=str(pretrained_model_name_or_path),
            config_overrides=kwargs,
        )
        return model


__all__ = ["BaseMixin"]
