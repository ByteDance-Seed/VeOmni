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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py

from abc import ABC

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..utils import logging
from ..utils.registry import Registry
from .module_utils import init_empty_weights, load_model_weights


MODELING_REGISTRY = Registry("Modeling")
MODEL_CONFIG_REGISTRY = Registry("ModelConfig")
MODEL_PROCESSOR_REGISTRY = Registry("ModelProcessor")

logger = logging.get_logger(__name__)


def get_model_config(config_path: str, force_use_huggingface: bool = False, **kwargs):
    if force_use_huggingface:
        logger.info_rank0("[CONFIG] Force loading model config from Huggingface.")
        return AutoConfig.from_pretrained(config_path, **kwargs)
    else:
        try:  # first load from hf, then replace with veomni
            config = AutoConfig.from_pretrained(config_path, **kwargs)
            model_type = config.model_type
            if model_type in MODEL_CONFIG_REGISTRY.valid_keys():
                kwargs.pop("trust_remote_code", None)
                config = MODEL_CONFIG_REGISTRY[model_type]().from_pretrained(config_path, **kwargs)
                logger.info_rank0(
                    f"[CONFIG] Loading {model_type} from Huggingface and replaced with customized config."
                )
                return config
            else:
                logger.info_rank0(
                    f"[CONFIG] Loading {model_type} from Huggingface as no customized config registered."
                )
                return config
        except Exception:  # load from veomni
            config_dict, _ = PretrainedConfig.get_config_dict(config_path, **kwargs)
            model_type = config_dict["model_type"]
            logger.info_rank0(f"[CONFIG] Loading {model_type} from custom config.")
            kwargs.pop("trust_remote_code", None)
            return MODEL_CONFIG_REGISTRY[model_type]().from_pretrained(config_path, **kwargs)


def get_model_processor(processor_path: str, force_use_huggingface: bool = False, **kwargs):
    if force_use_huggingface:
        logger.info_rank0("[PROCESSOR] Force loading model processor from Huggingface.")
        return AutoProcessor.from_pretrained(processor_path, **kwargs)
    else:
        try:  # first load from hf, then replace with veomni
            processor = AutoProcessor.from_pretrained(processor_path, **kwargs)
            processor_class_name = getattr(type(processor), "__name__", None)
            if processor_class_name in MODEL_PROCESSOR_REGISTRY.valid_keys():
                kwargs.pop("trust_remote_code", None)
                processor = MODEL_PROCESSOR_REGISTRY[processor_class_name]().from_pretrained(processor_path, **kwargs)
                logger.info_rank0(
                    f"[PROCESSOR] Loading {processor_class_name} from Huggingface and replaced with customized processor."
                )
                return processor
            else:
                logger.info_rank0(
                    f"[PROCESSOR] Loading {processor_class_name} from Huggingface as no customized processor registered."
                )
                return processor
        except Exception:  # load from veomni
            from transformers.processing_utils import ProcessorMixin
            from transformers.utils import PROCESSOR_NAME, cached_file

            processor_config_file = cached_file(processor_path, PROCESSOR_NAME)
            config_dict, _ = ProcessorMixin.get_processor_dict(processor_config_file, **kwargs)
            processor_class_name = config_dict["processor_class"]
            logger.info_rank0(f"[PROCESSOR] Loading {processor_class_name} from custom processor.")
            kwargs.pop("trust_remote_code", None)
            return MODEL_PROCESSOR_REGISTRY[processor_class_name]().from_pretrained(processor_path, **kwargs)


def get_model_class(model_config: PretrainedConfig, force_use_huggingface: bool = False):
    def get_model_arch_from_config(model_config):
        arch_name = model_config.architectures
        if isinstance(arch_name, list):
            arch_name = arch_name[0]
        return arch_name

    arch_name = get_model_arch_from_config(model_config)
    model_type = model_config.model_type
    if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # assume built-in models
        load_class = AutoModelForImageTextToText
    elif type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
        load_class = AutoModelForVision2Seq
    elif (
        arch_name is not None
        and "ForCausalLM" in arch_name
        and type(model_config) in AutoModelForCausalLM._model_mapping.keys()
    ):
        load_class = AutoModelForCausalLM
    else:
        load_class = AutoModel
    if force_use_huggingface:
        return load_class
    return MODELING_REGISTRY[model_type](arch_name)


class BaseModelLoader(ABC):
    def __init__(self):
        pass

    def load_model(self, model_config, **kwargs):
        raise NotImplementedError


class HuggingfaceLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls

    def load_model(self, init_kwargs: dict, **kwargs):
        init_kwargs.pop("trust_remote_code", True)

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from Huggingface modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if init_device == "meta":
                with init_empty_weights():
                    logger.info_rank0("Init empty model on meta device from config without init_weights.")
                    model = self.model_cls.from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = self.model_cls.from_config(**init_kwargs)
        else:
            with init_empty_weights():
                model = self.model_cls.from_config(**init_kwargs)
            if not empty_init:
                load_model_weights(model, weights_path, init_device)

        return model


class CustomizedModelingLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls

    def load_model(self, init_kwargs: dict, **kwargs):
        init_kwargs.pop("trust_remote_code", True)

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from customized modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if init_device == "meta":
                with init_empty_weights():
                    logger.info_rank0("Init empty model on meta device from config without init_weights.")
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = self.model_cls._from_config(**init_kwargs)
        else:
            with init_empty_weights(), no_init_weights():
                model = self.model_cls._from_config(**init_kwargs)

            if not empty_init:
                load_model_weights(model, weights_path, init_device)

            # we should tie embeddings after loading weights because init_empty_weights() leads to untied weights,
            if getattr(model.config, "tie_word_embeddings", True):
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
                except Exception as e:
                    logger.info_rank0(f"Failed to tie embeddings: {e}")

        return model


def get_loader(model_config, force_use_huggingface):
    model_cls = get_model_class(model_config, force_use_huggingface)
    if force_use_huggingface:
        loader = HuggingfaceLoader(model_cls=model_cls)
    else:
        loader = CustomizedModelingLoader(model_cls=model_cls)

    # TODO: there's no difference between HuggingfaceLoader and CustomizedModelingLoader except tie_word_embedding. Check if able to merge
    return loader
