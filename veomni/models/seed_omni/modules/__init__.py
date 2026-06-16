"""SeedOmni V2 module mixin registry.

Each entry maps a HuggingFace ``model_type`` string (the ``model_type``
field of each module's :class:`PretrainedConfig` subclass) to the
:class:`~veomni.models.seed_omni.module.ModuleMixin` subclass that backs it.  At trainer-build time the
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

Factory functions registered on ``OMNI_*_REGISTRY`` lazy-import the
concrete config / model / processor classes on first call — importing this
package only wires up the registry table, it does not load modeling code.

File layout
-----------
``modules/<family>/<sub_module>/(configuration.py, modeling.py[,
processing.py])``.  Each sub-module gets its own folder; the folder name
carries the namespace so the inner files use short names rather than
re-spelling ``<family>_<sub_module>`` per file.  Cross-family
lightweight modules live under ``modules/base/<sub_module>/``.
"""

from transformers import PretrainedConfig

from ....utils.registry import Registry


OMNI_CONFIG_REGISTRY = Registry("OmniConfig")
OMNI_MODEL_REGISTRY = Registry("OmniModel")
OMNI_PROCESSOR_REGISTRY = Registry("OmniProcessor")

# Side-effect only: attach @register factories under base/, janus/, qwen3/, qwen3_moe/.
from . import base, janus, qwen3, qwen3_moe  # noqa: F401  E402


def read_model_type(model_path: str) -> str:
    """Read ``model_type`` from a module's ``config.json`` and validate registration.

    Shared helper for any caller that needs to dispatch from a
    split-checkpoint subfolder to the matching module class —
    today that's :class:`OmniInferencer` (eager ``from_pretrained``) and
    :meth:`OmniTrainer._build_model` (meta-init via
    :func:`build_foundation_model`).  Centralised here so both paths use
    the same registration gate and emit identical error messages.

    Uses :meth:`PretrainedConfig.get_config_dict` rather than
    :class:`AutoConfig.from_pretrained` because Janus / future split
    checkpoints declare custom ``model_type`` values
    (``janus_siglip`` / ``janus_text_encoder`` / ``janus_llama`` /
    ``janus_vqvae``) that are NOT in HF's :data:`CONFIG_MAPPING`.
    ``AutoConfig`` would raise on those families before we even get a
    chance to consult :data:`OMNI_MODEL_REGISTRY`; reading the raw dict
    sidesteps that.  See :mod:`veomni.models.loader` for the same
    pattern in the foundation-model loader.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(model_path)
    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Module config at {model_path} has no `model_type` — cannot resolve module class.")
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


__all__ = [
    "OMNI_CONFIG_REGISTRY",
    "OMNI_MODEL_REGISTRY",
    "OMNI_PROCESSOR_REGISTRY",
    "read_model_type",
]
