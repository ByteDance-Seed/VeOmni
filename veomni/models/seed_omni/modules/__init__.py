"""SeedOmni V2 module mixin registry.

Each entry maps a HuggingFace ``model_type`` string (the ``model_type``
field of each module's :class:`PretrainedConfig` subclass) to the
:class:`OmniModule` mixin class that backs it.  At trainer-build time the
flow is::

    cfg_cls = OMNI_CONFIG_REGISTRY[model_type]()
    cls     = OMNI_MODEL_REGISTRY[model_type]()
    cfg     = cfg_cls.from_pretrained(<weights_path>)   # reads config.json
    module  = cls.from_pretrained(<weights_path>)       # HF PreTrainedModel API
    # → wrapped by build_parallelize_model in OmniTrainer (Step 2)

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

from ....utils.registry import Registry

OMNI_CONFIG_REGISTRY = Registry("OmniConfig")
OMNI_MODEL_REGISTRY = Registry("OmniModel")
OMNI_PROCESSOR_REGISTRY = Registry("OmniProcessor")

# Side-effect only: attach @register factories under base/ and janus/.
from . import base, janus  # noqa: F401


__all__ = [
    "OMNI_CONFIG_REGISTRY",
    "OMNI_MODEL_REGISTRY",
    "OMNI_PROCESSOR_REGISTRY",
]
