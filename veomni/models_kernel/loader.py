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

"""Construct a model class, then optionally load weights."""

from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel
from transformers.initialization import no_init_weights

from veomni.models_kernel.checkpoint.weights import init_empty_weights, load_model_weights
from veomni.models_kernel.registry import get_model_class
from veomni.utils import logging
from veomni.utils.env import get_env


logger = logging.get_logger(__name__)


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

        if weights_path is None:
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

        if weights_path is None:
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
    if get_env("MODELING_BACKEND") == "hf":
        return HuggingfaceLoader(model_cls=model_cls)
    return CustomizedModelingLoader(model_cls=model_cls)
