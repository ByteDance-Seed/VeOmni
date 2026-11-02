"""
BaseMixin — minimal SeedOmni lifecycle + shared graph-hook registry.

Every module's ``modeling.py`` inherits :class:`PretrainedOmniModule`.
Training / inference graph hooks live on :class:`TrainingModuleMixin` /
:class:`InferenceModuleMixin` and are composed onto a module when a later
PR needs them.

Layout
------
* ``modules/module_modeling_base.py`` — :class:`PretrainedOmniModule` (native HF sub-models)
* ``base_mixin.py`` — :class:`BaseMixin` (runtime hook registry)
* ``training_module_mixin.py`` — :class:`TrainingModuleMixin` (``pre_forward`` / ``post_forward``)
* ``inference_module_mixin.py`` — :class:`InferenceModuleMixin` (runtime ``pre_generate`` / ``post_generate``)
* ``modules/<family>/<sub>/modeling.py``::

    class Xxx(PretrainedOmniModule): ...
"""

from __future__ import annotations

from typing import Callable


def mark_hook(fn: Callable, marker: str, contexts: tuple[str, ...]) -> Callable:
    """Tag ``fn`` with the call-sites it hooks, for :func:`hook_name` to find.

    The tag rides on the function object because the registry is built by
    walking ``vars(klass)`` before any instance exists. ``setattr`` rather than
    a plain attribute write: a function has no such slot to assign to.
    """
    setattr(fn, marker, contexts)
    return fn


def hook_name(cls: type, marker: str, context: str) -> str | None:
    """Resolve the method name on ``cls`` tagged ``marker`` for call-site ``context``.

    The read side of :func:`mark_hook`, shared by the training
    (``@pre_forward`` / ``@post_forward``) and inference
    (``@pre_generate`` / ``@post_generate``) dispatchers.

    A plain function rather than a :class:`BaseMixin` method because the mixins
    that dispatch do **not** inherit ``BaseMixin`` — the host class lists both,
    and in the established order (``VeOmniMixin(BaseMixin, TrainingMixin, ...)``)
    making them inherit it would make the MRO unsolvable.
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


class BaseMixin:
    """Shared graph-hook registry for training / inference mixins."""

    @classmethod
    def _omni_hook_name(cls, marker: str, context: str) -> str | None:
        """Resolve a hook name on this class — see :func:`hook_name`."""
        return hook_name(cls, marker, context)


__all__ = ["BaseMixin", "hook_name", "mark_hook"]
