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

from abc import ABC, abstractmethod

import torch
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
    PreTrainedModel,
)


# transformers v5 deleted the AutoModelForVision2Seq class, so we use AutoModelForImageTextToText as a fallback
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = AutoModelForImageTextToText

from transformers.initialization import no_init_weights

from ..utils import logging
from ..utils.env import get_env
from ..utils.registry import Registry
from .module_utils import init_empty_weights, load_model_weights


MODELING_REGISTRY = Registry("Modeling")
MODEL_CONFIG_REGISTRY = Registry("ModelConfig")
MODEL_PROCESSOR_REGISTRY = Registry("ModelProcessor")

logger = logging.get_logger(__name__)


def _omni_registries():
    """Lazily fetch the SeedOmni V2 sub-module registries.

    SeedOmni V2 splits a composite model (e.g. Janus) into self-contained
    sub-modules — ``janus_siglip`` / ``janus_vqvae`` / ``janus_llama`` /
    ``janus_text_encoder`` — whose ``model_type`` values are **not** in HF's
    ``CONFIG_MAPPING`` and live in their own registries rather than
    ``MODEL_CONFIG_REGISTRY`` / ``MODELING_REGISTRY`` / ``MODEL_PROCESSOR_REGISTRY``.

    The ``get_model_*`` functions below consult these so the shared
    :func:`veomni.models.build_foundation_model` / :func:`build_processor`
    loader path works for an OmniModule sub-module exactly like a normal
    single model (``OmniTrainer._build_model`` builds each sub-module this
    way).  Imported lazily (inside the loader functions) to avoid a circular
    import: ``veomni.models.seed_omni`` modeling imports back into
    ``veomni.models``.

    Returns ``(OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY)``.
    """
    from .seed_omni.modules import (
        OMNI_CONFIG_REGISTRY,
        OMNI_MODEL_REGISTRY,
        OMNI_PROCESSOR_REGISTRY,
    )

    return OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


def raise_unsupported_veomni_modeling(model_name: str) -> None:
    # Gate for models whose VeOmni modeling path has NOT been ported to the
    # patchgen/generated flow. ``get_model_class`` in this module short-circuits
    # when MODELING_BACKEND=hf, so this function is only reached when the caller
    # wants VeOmni's patched classes — fail loudly instead of returning a stub
    # that would silently produce broken graphs.
    raise RuntimeError(
        f"{model_name} does not have a VeOmni modeling path. Set MODELING_BACKEND=hf "
        f"to bypass VeOmni patches and load upstream HuggingFace classes directly."
    )


def get_model_config(config_path: str, **kwargs):
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
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
        except Exception:  # load from veomni (custom / OmniModule model_type)
            config_dict, _ = PretrainedConfig.get_config_dict(config_path, **kwargs)
            model_type = (
                config_dict["model_type"] if "model_type" in config_dict else config_dict["_class_name"]
            )  # diffusers use _class_name
            kwargs.pop("trust_remote_code", None)
            # SeedOmni V2 sub-module configs (janus_siglip / ...) live in the
            # OMNI registry, not MODEL_CONFIG_REGISTRY.
            omni_config_registry, _, _ = _omni_registries()
            if model_type in set(omni_config_registry.valid_keys()):
                logger.info_rank0(f"[CONFIG] Loading {model_type} from OMNI config registry.")
                return omni_config_registry[model_type]().from_pretrained(config_path, **kwargs)
            logger.info_rank0(f"[CONFIG] Loading {model_type} from custom config.")
            return MODEL_CONFIG_REGISTRY[model_type]().from_pretrained(config_path, **kwargs)


def get_model_processor(processor_path: str, **kwargs):
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
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
        except Exception:  # load from veomni (custom / OmniModule processor)
            # SeedOmni V2 sub-module processors are keyed by ``model_type``
            # (read from the module's ``config.json``), not by processor class
            # name — consult the OMNI registry first.
            try:
                cfg_dict, _ = PretrainedConfig.get_config_dict(processor_path)
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
    def get_model_arch_from_config(model_config):
        arch_name = model_config.architectures
        if isinstance(arch_name, list):
            arch_name = arch_name[0]
        return arch_name

    arch_name = get_model_arch_from_config(model_config)
    model_type = model_config.model_type
    modeling_backend = get_env("MODELING_BACKEND")
    if not modeling_backend == "hf":
        # SeedOmni V2 sub-modules resolve to their OmniModule class via the
        # OMNI registry (keyed by model_type; the factory takes no arch_name).
        _, omni_model_registry, _ = _omni_registries()
        if model_type in set(omni_model_registry.valid_keys()):
            return omni_model_registry[model_type]()
        return MODELING_REGISTRY[model_type](arch_name)
    if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # assume built-in models
        load_class = AutoModelForImageTextToText
    elif type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
        load_class = AutoModelForVision2Seq
    elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():  # assume built-in models
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


class BaseModelLoader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
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

            # init_empty_weights() leaves embeddings untied; re-tie only when
            # the config asks for it. Nested multimodal layouts can disable tying
            # on either side (InternVL on inner, Qwen3VLMoe on outer with inner
            # silent), so AND both. Treat unset as True so a silent side does not
            # override an explicit True, but require at least one side to set the
            # flag -- if neither does, default to False (matches HF v5).
            text_config = (
                model.config.get_text_config(decoder=True)
                if hasattr(model.config, "get_text_config")
                else model.config
            )
            if (
                (hasattr(model.config, "tie_word_embeddings") or hasattr(text_config, "tie_word_embeddings"))
                and getattr(model.config, "tie_word_embeddings", True)
                and getattr(text_config, "tie_word_embeddings", True)
            ):
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
                except Exception as e:
                    logger.info_rank0(f"Failed to tie embeddings: {e}")

        return model


def get_loader(model_config):
    model_cls = get_model_class(model_config)
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        loader = HuggingfaceLoader(model_cls=model_cls)
    else:
        loader = CustomizedModelingLoader(model_cls=model_cls)

    # TODO: there's no difference between HuggingfaceLoader and CustomizedModelingLoader except tie_word_embedding. Check if able to merge
    return loader
