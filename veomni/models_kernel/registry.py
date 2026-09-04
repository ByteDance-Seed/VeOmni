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

"""Construct registries for models_kernel classes.

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


def raise_unsupported_veomni_modeling(model_name: str) -> None:
    raise RuntimeError(
        f"{model_name} does not have a VeOmni modeling path. Set MODELING_BACKEND=hf "
        f"to bypass VeOmni patches and load upstream HuggingFace classes directly."
    )


def get_model_config(config_path: str, **kwargs):
    """Load a config, replacing it with a registered class when one exists."""
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        logger.info_rank0("[CONFIG] Force loading model config from Huggingface.")
        return AutoConfig.from_pretrained(config_path, **kwargs)

    try:
        config = AutoConfig.from_pretrained(config_path, **kwargs)
        model_type = config.model_type
        if model_type in MODEL_CONFIG_REGISTRY.valid_keys():
            kwargs.pop("trust_remote_code", None)
            config = MODEL_CONFIG_REGISTRY[model_type]().from_pretrained(config_path, **kwargs)
            logger.info_rank0(f"[CONFIG] Loading {model_type} from Huggingface and replaced with customized config.")
            return config
        logger.info_rank0(f"[CONFIG] Loading {model_type} from Huggingface as no customized config registered.")
        return config
    except Exception:
        config_dict, _ = PretrainedConfig.get_config_dict(config_path, **kwargs)
        model_type = config_dict["model_type"] if "model_type" in config_dict else config_dict["_class_name"]
        logger.info_rank0(f"[CONFIG] Loading {model_type} from custom config.")
        kwargs.pop("trust_remote_code", None)
        return MODEL_CONFIG_REGISTRY[model_type]().from_pretrained(config_path, **kwargs)


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
        arch_name = model_config.architectures
        if isinstance(arch_name, list):
            arch_name = arch_name[0]
        return arch_name

    arch_name = get_model_arch_from_config(model_config)
    model_type = model_config.model_type
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend != "hf":
        return MODELING_REGISTRY[model_type](arch_name)
    if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
        load_class = AutoModelForImageTextToText
    elif type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        load_class = AutoModelForVision2Seq
    elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():
        load_class = AutoModelForTextToWaveform
    elif (
        arch_name is not None
        and "ForCausalLM" in arch_name
        and type(model_config) in AutoModelForCausalLM._model_mapping.keys()
    ):
        load_class = AutoModelForCausalLM
    elif (
        arch_name is not None
        and "ForTokenClassification" in arch_name
        and type(model_config) in AutoModelForTokenClassification._model_mapping.keys()
    ):
        load_class = AutoModelForTokenClassification
    elif (
        arch_name is not None
        and "ForSequenceClassification" in arch_name
        and type(model_config) in AutoModelForSequenceClassification._model_mapping.keys()
    ):
        load_class = AutoModelForSequenceClassification
    else:
        load_class = AutoModel
    return load_class
