"""SeedOmni module mixin registry.

Each entry maps a HuggingFace ``model_type`` string (the ``model_type``
field of each module's :class:`PretrainedConfig` subclass) to the
:class:`~veomni.models.seed_omni.mixins.base_mixin.BaseMixin` subclass that backs it.  At trainer-build time the
flow is::

    model_type = read_model_type(<weights_path>)        # reads config.json
    cfg_cls = OMNI_CONFIG_REGISTRY[model_type]()
    cls     = OMNI_MODEL_REGISTRY[model_type]()
    cfg     = cfg_cls.from_pretrained(<weights_path>)
    module  = cls.from_pretrained(<weights_path>)       # HF PreTrainedModel API
    # → FSDP-wrapped by build_parallelize_model inside OmniTrainer

These modules are **not** registered with HuggingFace ``AutoConfig`` /
``AutoModel`` — always resolve the class via ``OMNI_*_REGISTRY`` first,
then call ``from_pretrained`` on that class.

``OMNI_ACCELERATED_MODEL_REGISTRY`` maps the same ``model_type`` to the
VeOmni-side subclass (``modules/<family>/<sub>/accelerated/``) that mixes the
graph hooks (:class:`BaseMixin`, :class:`TrainingModuleMixin`,
:class:`InferenceModuleMixin`) onto the HF-native class.
:func:`veomni.models.loader.get_model_class` builds that subclass when one is
registered, so a :class:`ModuleRuntime` gets a graph participant while bare
``OmniModel.from_pretrained`` keeps the plain HF class.

Factory functions registered on ``OMNI_*_REGISTRY`` lazy-import the
concrete config / model / processor classes on first call — importing this
package only wires up the registry table, it does not load modeling code.

File layout
-----------
Shared bases live next to the families, not at the ``seed_omni/`` package root:

* ``module_modeling_base.py`` — :class:`PretrainedOmniModule`
* ``module_processing_base.py`` — :class:`ModulePreprocessorBase` + :func:`bind_module_assets`
* ``module_configuration_base.py`` — :class:`OmniModuleConfig` (base config for a module, and for its ``OmniConfig._module_entries`` slot)

Concrete modules: ``modules/<family>/<sub_module>/(configuration.py,
modeling.py[, processing.py])``.  Each sub-module gets its own folder; the
folder name carries the namespace so the inner files use short names rather
than re-spelling ``<family>_<sub_module>`` per file.
"""

from transformers import PretrainedConfig

from ....utils.registry import Registry  # VeOmni shared name→factory registry; not seed_omni-local.
from .module_configuration_base import OmniModuleConfig
from .module_modeling_base import PretrainedOmniModule
from .module_processing_base import MODULE_ASSET_ATTRS, ModulePreprocessorBase, bind_module_assets


OMNI_CONFIG_REGISTRY = Registry("OmniConfig")
OMNI_MODEL_REGISTRY = Registry("OmniModel")
OMNI_PROCESSOR_REGISTRY = Registry("OmniProcessor")
OMNI_ACCELERATED_MODEL_REGISTRY = Registry("OmniAcceleratedModel")


def read_hf_model_type(model_path: str) -> str:
    """Read the upstream ``model_type`` from a HuggingFace ``config.json``.

    Generic reader (no registry validation): returns the raw ``model_type``
    string. Shared by any caller that needs to dispatch on a checkpoint's
    declared family — the SeedOmni convert pipeline
    (:func:`~veomni.models.seed_omni.utils.convert_registry.convert_checkpoint`,
    which dispatches on the *upstream* HF type) and as the primitive behind
    :func:`read_model_type`.

    Uses :meth:`PretrainedConfig.get_config_dict` rather than
    :class:`AutoConfig.from_pretrained` because split-checkpoint modules
    declare custom ``model_type`` values (``module_A`` / ``module_B`` / …)
    that are NOT in HF's :data:`CONFIG_MAPPING`.  ``AutoConfig`` would raise
    on those families before we even get a chance to consult the registries;
    reading the raw dict sidesteps that.  See :mod:`veomni.models.loader`
    for the same pattern in the foundation-model loader.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(model_path)
    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Checkpoint at {model_path} has no `model_type` in config.json.")
    return model_type


def read_model_type(model_path: str) -> str:
    """Read ``model_type`` from a module's ``config.json`` and validate registration.

    Shared helper for any caller that needs to dispatch from a
    split-checkpoint subfolder to the matching module class —
    today that's :class:`OmniInferencer` (eager ``from_pretrained``) and
    :meth:`OmniTrainer._build_model` (meta-init via
    :func:`build_foundation_model`).  Centralised here so both paths use
    the same registration gate and emit identical error messages.

    Builds on :func:`read_hf_model_type` (the raw ``config.json`` read) and
    then gates the result on the SeedOmni registries.
    """
    model_type = read_hf_model_type(model_path)
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
        cfg_cls.from_pretrained(model_path)
    if model_type not in model_keys:
        raise KeyError(
            f"Module model_type {model_type!r} (from {model_path}) is not registered in "
            f"OMNI_MODEL_REGISTRY. Known: {sorted(model_keys)}."
        )
    return model_type


from . import fake_model  # noqa: F401  E402


__all__ = [
    "MODULE_ASSET_ATTRS",
    "OMNI_ACCELERATED_MODEL_REGISTRY",
    "OMNI_CONFIG_REGISTRY",
    "OMNI_MODEL_REGISTRY",
    "OMNI_PROCESSOR_REGISTRY",
    "ModulePreprocessorBase",
    "OmniModuleConfig",
    "PretrainedOmniModule",
    "bind_module_assets",
    "read_hf_model_type",
    "read_model_type",
]
