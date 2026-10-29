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

"""Construct registries for models classes.

This is a separate ``Registry`` instance from any other modeling registry.
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTextToWaveform,
    AutoModelForTokenClassification,
    AutoProcessor,
    PretrainedConfig,
)


try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = AutoModelForImageTextToText

from veomni.utils import logging
from veomni.utils.env import get_env
from veomni.utils.registry import Registry


MODELING_REGISTRY = Registry("Modeling")
MODEL_CONFIG_REGISTRY = Registry("ModelConfig")
MODEL_PROCESSOR_REGISTRY = Registry("ModelProcessor")

logger = logging.get_logger(__name__)


def _omni_registries():
    """Import SeedOmni registries lazily.

    SeedOmni modeling imports back into ``veomni.models``, so this stays
    behind a function instead of a module-level import.

    Returns ``(OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY)``.
    """
    from .seed_omni.modules import (
        OMNI_CONFIG_REGISTRY,
        OMNI_MODEL_REGISTRY,
        OMNI_PROCESSOR_REGISTRY,
    )

    return OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


def raise_unsupported_veomni_modeling(model_name: str) -> None:
    """Raise a model-build error that names the available VeOmni model types."""
    supported = ", ".join(sorted(MODELING_REGISTRY.valid_keys())) or "none"
    raise RuntimeError(
        f"{model_name} is not registered in veomni.models; registered model types: {supported}. "
        "Set MODELING_BACKEND=hf to load the upstream HuggingFace class without VeOmni model kernels."
    )


def get_model_config(config_path: str, **kwargs):
    """Load a config, replacing it with a registered class when one exists."""
    from .configuration_base import get_veomni_config_class

    config_dict, _ = PretrainedConfig.get_config_dict(config_path, **kwargs)
    model_type = config_dict.get("model_type", config_dict.get("_class_name"))
    config_class = None
    if get_env("MODELING_BACKEND") != "hf":
        if model_type == "omni":
            from .seed_omni.configuration_omni import OmniConfig

            config_class = OmniConfig
        elif model_type in MODEL_CONFIG_REGISTRY.valid_keys():
            config_class = MODEL_CONFIG_REGISTRY[model_type]()
        else:
            omni_config_registry, _, _ = _omni_registries()
            if model_type in omni_config_registry.valid_keys():
                config_class = omni_config_registry[model_type]()
    if config_class is None:
        # Resolve the schema before initialization so native parent attention
        # broadcasts cannot erase explicit child choices.
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if model_type in CONFIG_MAPPING and not config_dict.get("auto_map"):
            config_class = CONFIG_MAPPING[model_type]
        else:
            config_class = type(AutoConfig.from_pretrained(config_path, **kwargs))
    kwargs.pop("trust_remote_code", None)
    # Pass overrides through the selected class's own loader. Composed configs
    # must apply root choices before they load and validate module sidecars.
    return get_veomni_config_class(config_class).from_pretrained(config_path, **kwargs)


def get_model_processor(processor_path: str, **kwargs):
    """Load a processor, replacing it with a registered class when one exists."""
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        logger.info_rank0("[PROCESSOR] Force loading model processor from Huggingface.")
        return AutoProcessor.from_pretrained(processor_path, **kwargs)

    try:
        processor = AutoProcessor.from_pretrained(processor_path, **kwargs)
        processor_class_name = getattr(type(processor), "__name__", None)
        if processor_class_name in MODEL_PROCESSOR_REGISTRY.valid_keys():
            kwargs.pop("trust_remote_code", None)
            processor = MODEL_PROCESSOR_REGISTRY[processor_class_name]().from_pretrained(processor_path, **kwargs)
            logger.info_rank0(
                f"[PROCESSOR] Loading {processor_class_name} from Huggingface and replaced with customized processor."
            )
            return processor
        logger.info_rank0(
            f"[PROCESSOR] Loading {processor_class_name} from Huggingface as no customized processor registered."
        )
        return processor
    except Exception:
        # SeedOmni sub-module processors are keyed by ``model_type``
        # (read from the module's ``config.json``), not by processor class
        # name — consult the OMNI registry first.
        try:
            hub_kwargs = {
                key: kwargs[key]
                for key in ("token", "revision", "cache_dir", "subfolder", "local_files_only")
                if key in kwargs
            }
            cfg_dict, _ = PretrainedConfig.get_config_dict(processor_path, **hub_kwargs)
            omni_model_type = cfg_dict.get("model_type")
        except Exception:
            omni_model_type = None
        _, _, omni_processor_registry = _omni_registries()
        if omni_model_type is not None and omni_model_type in set(omni_processor_registry.valid_keys()):
            kwargs.pop("trust_remote_code", None)
            logger.info_rank0(f"[PROCESSOR] Loading {omni_model_type} from OMNI processor registry.")
            return omni_processor_registry[omni_model_type]().from_pretrained(processor_path, **kwargs)

        from transformers.processing_utils import ProcessorMixin
        from transformers.utils import PROCESSOR_NAME, cached_file

        processor_config_file = cached_file(processor_path, PROCESSOR_NAME)
        config_dict, _ = ProcessorMixin.get_processor_dict(processor_config_file, **kwargs)
        processor_class_name = config_dict["processor_class"]
        logger.info_rank0(f"[PROCESSOR] Loading {processor_class_name} from custom processor.")
        kwargs.pop("trust_remote_code", None)
        return MODEL_PROCESSOR_REGISTRY[processor_class_name]().from_pretrained(processor_path, **kwargs)


def get_model_class(model_config: PretrainedConfig):
    """Return the registered modeling class, or the HuggingFace Auto class."""

    def get_model_arch_from_config(model_config):
        """Return the first architecture name declared by a model config."""
        arch_name = model_config.architectures
        if isinstance(arch_name, list):
            arch_name = arch_name[0]
        return arch_name

    arch_name = get_model_arch_from_config(model_config)
    model_type = model_config.model_type
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend != "hf":
        # A module built here is a graph participant, so prefer the subclass
        # that carries the graph hooks; the HF-native class is the fallback.
        _, omni_model_registry, _ = _omni_registries()
        if model_type in set(omni_model_registry.valid_keys()):
            from .seed_omni.modules import OMNI_ACCELERATED_MODEL_REGISTRY

            if model_type in set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys()):
                return OMNI_ACCELERATED_MODEL_REGISTRY[model_type]()
            return omni_model_registry[model_type]()
        if model_type not in MODELING_REGISTRY.valid_keys():
            raise_unsupported_veomni_modeling(model_type)
        return MODELING_REGISTRY[model_type](arch_name)
    config_class = type(model_config).__dict__.get("_hf_config_class", type(model_config))
    if config_class in AutoModelForImageTextToText._model_mapping.keys():
        load_class = AutoModelForImageTextToText
    elif config_class in AutoModelForVision2Seq._model_mapping.keys():
        load_class = AutoModelForVision2Seq
    elif config_class in AutoModelForTextToWaveform._model_mapping.keys():
        load_class = AutoModelForTextToWaveform
    elif (
        arch_name is not None
        and "ForCausalLM" in arch_name
        and config_class in AutoModelForCausalLM._model_mapping.keys()
    ):
        load_class = AutoModelForCausalLM
    elif (
        arch_name is not None
        and "ForTokenClassification" in arch_name
        and config_class in AutoModelForTokenClassification._model_mapping.keys()
    ):
        load_class = AutoModelForTokenClassification
    elif (
        arch_name is not None
        and "ForSequenceClassification" in arch_name
        and config_class in AutoModelForSequenceClassification._model_mapping.keys()
    ):
        load_class = AutoModelForSequenceClassification
    else:
        load_class = AutoModel
    return load_class
