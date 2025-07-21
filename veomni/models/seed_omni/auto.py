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


from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import no_init_weights

from ..auto import build_foundation_model, build_processor
from ..module_utils import init_empty_weights, load_model_weights
from .configuration_seed_omni import SeedOmniConfig
from .modeling_seed_omni import SeedOmniModel
from .processing_seed_omni import SeedOmniProcessor


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, ProcessorMixin


def build_omni_processor(
    config_path: str,
    tokenizer_path: str,
    encoders: Dict[str, Dict[str, str]] = {},
    decoders: Dict[str, Dict[str, str]] = {},
    input_encoder: str = "encoder",
    output_encoder: str = "decoder",
    encode_target: bool = False,
    max_pixels: int = 28 * 28 * 768,
) -> "ProcessorMixin":
    """
    Builds omni modality processor using foundation tokenizer, encoders and decoders.
    """
    foundation_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    if isinstance(foundation_config, SeedOmniConfig):
        return build_processor(tokenizer_path)

    processor_dict = {"tokenizer": AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)}

    input_encoders = encoders if input_encoder == "encoder" else decoders
    output_encoders = decoders if output_encoder == "decoder" else encoders

    for encoder_type, encoder_args in input_encoders.items():
        processor = AutoProcessor.from_pretrained(
            encoder_args["config_path"], max_pixels=max_pixels, trust_remote_code=True
        )
        processor_dict[f"input_{encoder_type}_processor"] = processor

    for encoder_type, encoder_args in output_encoders.items():
        processor = AutoProcessor.from_pretrained(
            encoder_args["config_path"], max_pixels=max_pixels, trust_remote_code=True
        )
        processor_dict[f"output_{encoder_type}_processor"] = processor

    if encode_target:
        for decoder_type, decoder_args in decoders.items():
            processor = AutoProcessor.from_pretrained(decoder_args["config_path"], trust_remote_code=True)
            processor_dict[f"target_{decoder_type}_processor"] = processor

    omni_processor = SeedOmniProcessor(**processor_dict)
    return omni_processor


def build_omni_model(
    config_path: str,
    weights_path: Optional[str] = None,
    encoders: Dict[str, Dict[str, str]] = {},
    decoders: Dict[str, Dict[str, str]] = {},
    input_encoder: str = "encoder",
    output_encoder: str = "decoder",
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "flash_attention_2",
    empty_init: bool = False,
    init_device: Literal["cpu", "cuda"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: bool = False,
) -> "PreTrainedModel":
    """
    Builds omni modality model using foundation model, encoders, and decoders.
    """
    if config_kwargs is None:
        config_kwargs = {}

    foundation_config: "PretrainedConfig" = AutoConfig.from_pretrained(
        config_path, trust_remote_code=True, **config_kwargs
    )
    if isinstance(foundation_config, SeedOmniConfig):
        return build_foundation_model(
            config_path=config_path,
            weights_path=weights_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            empty_init=empty_init,
            init_device=init_device,
            config_kwargs=config_kwargs,
            force_use_huggingface=force_use_huggingface,
        )

    foundation_config = foundation_config.to_dict()
    encoder_config = {"text_config": foundation_config}  # TODO: only keep nessesary keys
    for encoder_type, encoder_args in encoders.items():
        extra_args = {key: value for key, value in encoder_args.items() if key not in ["config_path", "model_path"]}
        encoder_config[f"{encoder_type}_config"] = AutoConfig.from_pretrained(
            encoder_args["config_path"],
            trust_remote_code=True,
            output_size=foundation_config["hidden_size"],
            **extra_args,
            **config_kwargs,
        ).to_dict()

    encoder_config["encode_input"] = input_encoder == "encoder"
    encoder_config["encode_output"] = output_encoder == "encoder"
    decoder_config = {}
    for decoder_type, decoder_args in decoders.items():
        extra_args = {key: value for key, value in decoder_args.items() if key not in ["config_path", "model_path"]}
        decoder_config[f"{decoder_type}_config"] = AutoConfig.from_pretrained(
            decoder_args["config_path"],
            trust_remote_code=True,
            output_size=foundation_config["hidden_size"],
            **extra_args,
            **config_kwargs,
        ).to_dict()

    decoder_config["encode_input"] = input_encoder == "decoder"
    decoder_config["encode_output"] = output_encoder == "decoder"

    config = SeedOmniConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        foundation_config=foundation_config,
        tie_word_embeddings=foundation_config["tie_word_embeddings"],
        **config_kwargs,
    )
    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
    }
    if weights_path is None:  # init empty model
        with torch.device(init_device):
            model = SeedOmniModel._from_config(**init_kwargs)
    else:
        with init_empty_weights(), no_init_weights():
            model = SeedOmniModel._from_config(**init_kwargs)

    if weights_path is not None and not empty_init:
        load_model_weights(model.foundation, weights_path, init_device)

    for encoder_type, encoder_args in encoders.items():
        if encoder_args.get("model_path"):
            load_model_weights(
                getattr(model.encoder, f"{encoder_type}_encoder"), encoder_args["model_path"], init_device
            )

    for decoder_type, decoder_args in decoders.items():
        if decoder_args.get("model_path"):
            load_model_weights(
                getattr(model.decoder, f"{decoder_type}_decoder"), decoder_args["model_path"], init_device
            )

    # tie embeddings
    model.get_input_embeddings()._parameters["weight"] = model.foundation.get_input_embeddings()._parameters["weight"]
    if getattr(model.foundation.config, "tie_word_embeddings", True):
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]

    return model
