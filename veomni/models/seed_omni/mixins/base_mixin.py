"""
BaseMixin — minimal SeedOmni lifecycle + shared graph-hook registry.

Every module's ``modeling.py`` inherits :class:`OmniPreTrainedModel`.
Training / inference graph hooks live on :class:`TrainingModuleMixin` /
:class:`InferenceModuleMixin` and are composed onto a module when a later
PR needs them.

Layout
------
* ``modules/module_modeling_base.py`` — :class:`OmniPreTrainedModel` (native HF sub-models)
* ``base_mixin.py`` — :class:`BaseMixin` (runtime hook registry)
* ``training_module_mixin.py`` — :class:`TrainingModuleMixin` (``pre_forward`` / ``post_forward``)
* ``inference_module_mixin.py`` — :class:`InferenceModuleMixin` (runtime ``pre_generate`` / ``post_generate``)
* ``modules/<family>/<sub>/modeling.py``::

    class Xxx(OmniPreTrainedModel): ...
"""

from __future__ import annotations


class BaseMixin:
    """Shared graph-hook registry for training / inference mixins."""

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


__all__ = ["BaseMixin"]
