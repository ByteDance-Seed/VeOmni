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
"""Convert an upstream Bagel checkpoint into VeOmni's per-submodule seed_omni layout.

The upstream monolithic Bagel checkpoint stores everything in one state dict
(``ema.safetensors`` for the model + ``ae.safetensors`` for the VAE). This script splits
it into the three VeOmni submodules consumed by ``build_omni_model``:

    <output_dir>/bagel_foundation/        # MoT language model (+ tokenizer)
    <output_dir>/bagel_vision_encoder/    # SigLIP NaViT understanding encoder
    <output_dir>/bagel_vision_decoder/    # FLUX VAE + rectified-flow generation head

Usage::

    python scripts/multimodal/convert_bagel.py --model_path /path/to/BAGEL-7B --output_dir ./bagel_veomni
"""

import argparse
import os

import torch
from safetensors.torch import load_file

from veomni.models import build_tokenizer
from veomni.models.seed_omni.decoder.bagel_vision_decoder.configuring_bagel_vision_decoder import (
    BagelVisionDecoderConfig,
)
from veomni.models.seed_omni.decoder.bagel_vision_decoder.modeling_bagel_vision_decoder import BagelVisionDecoder
from veomni.models.seed_omni.decoder.bagel_vision_decoder.processing_bagel_vision_decoder import (
    BagelVisionDecoderProcessor,
)
from veomni.models.seed_omni.encoder.bagel_vision_encoder.configuration_bagel_vision_encoder import (
    BagelVisionEncoderConfig,
)
from veomni.models.seed_omni.encoder.bagel_vision_encoder.modeling_bagel_vision_encoder import BagelVisionEncoder
from veomni.models.seed_omni.encoder.bagel_vision_encoder.processing_bagel_vision_encoder import (
    BagelVisionEncoderProcessor,
)
from veomni.models.seed_omni.foundation.bagel_foundation.configuration_bagel_foundation import BagelFoundationConfig
from veomni.models.seed_omni.foundation.bagel_foundation.modeling_bagel_foundation import BagelFoundationModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the upstream Bagel checkpoint.")
    parser.add_argument("--output_dir", type=str, default="bagel_veomni", help="Where to write the submodules.")
    parser.add_argument("--vit_select_layer", type=int, default=-2)
    parser.add_argument("--vit_rope", action="store_true", help="Keep RoPE in the ViT encoder.")
    args = parser.parse_args()

    model_path = args.model_path
    os.makedirs(args.output_dir, exist_ok=True)

    model_state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
    # Non-persistent sinusoidal position buffers are regenerated on load.
    model_state_dict.pop("latent_pos_embed.pos_embed", None)
    model_state_dict.pop("vit_pos_embed.pos_embed", None)

    # ------------------------------------------------------------------ foundation
    llm_config = BagelFoundationConfig.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    with torch.device("meta"):
        foundation_model = BagelFoundationModel._from_config(llm_config)

    foundation_state_dict = {}
    for k in list(model_state_dict.keys()):
        if k.startswith("language_model."):
            foundation_state_dict[k.replace("language_model.", "")] = model_state_dict.pop(k)
    foundation_model.load_state_dict(foundation_state_dict, strict=True, assign=True)
    foundation_dir = os.path.join(args.output_dir, "bagel_foundation")
    foundation_model.save_pretrained(foundation_dir)
    build_tokenizer(model_path).save_pretrained(foundation_dir)
    print(f"[bagel] foundation -> {foundation_dir}")

    # -------------------------------------------------------------- vision encoder
    vit_config = BagelVisionEncoderConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + args.vit_select_layer
    vit_config.rope = args.vit_rope
    vit_config.output_size = llm_config.hidden_size
    with torch.device("meta"):
        vision_encoder = BagelVisionEncoder._from_config(vit_config)

    encoder_state_dict = {}
    for k in list(model_state_dict.keys()):
        if k.startswith("vit_model."):
            encoder_state_dict[k.replace("vit_model.", "")] = model_state_dict.pop(k)
        elif k.startswith("connector."):
            encoder_state_dict[k] = model_state_dict.pop(k)
    vision_encoder.load_state_dict(encoder_state_dict, strict=True, assign=True)
    encoder_dir = os.path.join(args.output_dir, "bagel_vision_encoder")
    vision_encoder.save_pretrained(encoder_dir)
    BagelVisionEncoderProcessor().save_pretrained(encoder_dir)
    print(f"[bagel] vision encoder -> {encoder_dir}")

    # -------------------------------------------------------------- vision decoder
    ae_state_dict = load_file(os.path.join(model_path, "ae.safetensors"), device="cpu")
    model_state_dict.update(ae_state_dict)
    decoder_config = BagelVisionDecoderConfig()
    decoder_config.output_size = llm_config.hidden_size
    with torch.device("meta"):
        vision_decoder = BagelVisionDecoder._from_config(decoder_config)

    decoder_state_dict = {}
    for k in list(model_state_dict.keys()):
        if k.startswith("encoder.") or k.startswith("decoder."):
            decoder_state_dict["auto_encoder." + k] = model_state_dict.pop(k)
        else:
            decoder_state_dict[k] = model_state_dict.pop(k)
    vision_decoder.load_state_dict(decoder_state_dict, strict=True, assign=True)
    decoder_dir = os.path.join(args.output_dir, "bagel_vision_decoder")
    vision_decoder.save_pretrained(decoder_dir)
    BagelVisionDecoderProcessor().save_pretrained(decoder_dir)
    print(f"[bagel] vision decoder -> {decoder_dir}")


if __name__ == "__main__":
    main()
