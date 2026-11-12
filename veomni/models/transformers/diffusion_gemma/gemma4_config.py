# Copyright 2025 ByteDance Ltd. and/or its affiliates
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

"""Derive a DiffusionGemma config from an unmodified Gemma 4 config."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


RAW_GEMMA4_ARCHITECTURE = "Gemma4ForConditionalGeneration"
DIFFUSION_GEMMA_ARCHITECTURE = "DiffusionGemmaForBlockDiffusion"

_RAW_ONLY_TOP_LEVEL_CONFIG_KEYS = frozenset(
    {
        "audio_config",
        "audio_token_id",
        "boa_token_id",
        "eoa_token_id",
        "eoa_token_index",
        "video_token_id",
    }
)
_RAW_ONLY_TEXT_CONFIG_KEYS = frozenset(
    {
        "attention_k_eq_v",
        "enable_moe_block",
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "use_cache",
        "use_double_wide_mlp",
        "vocab_size_per_layer_input",
    }
)


def convert_gemma4_config_dict(source: dict) -> dict:
    """Create a DiffusionGemma architecture config from an official Gemma 4 config."""
    if source.get("architectures") != [RAW_GEMMA4_ARCHITECTURE]:
        raise ValueError(f"Expected architecture {RAW_GEMMA4_ARCHITECTURE}, got {source.get('architectures')}")
    if source.get("model_type") != "gemma4":
        raise ValueError(f"Expected model_type='gemma4', got {source.get('model_type')!r}")
    if source.get("text_config", {}).get("model_type") != "gemma4_text":
        raise ValueError("Expected text_config.model_type='gemma4_text'.")

    converted = copy.deepcopy(source)
    for key in _RAW_ONLY_TOP_LEVEL_CONFIG_KEYS:
        converted.pop(key, None)
    converted["architectures"] = [DIFFUSION_GEMMA_ARCHITECTURE]
    converted["model_type"] = "diffusion_gemma"
    converted.update(
        {
            "canvas_length": 256,
            "max_canvases": 8,
            "canvas_start": 0,
            "self_conditioning_prob": 0.5,
            "encoder_ar_loss": True,
            "mask_prompt_ar_loss": False,
            "mask_other_tail_eos": True,
            "compute_all_canvas_df_loss": False,
            "canvas_batching": "serial",
            "print_ar_df_norm": False,
        }
    )
    text_config = converted["text_config"]
    for key in _RAW_ONLY_TEXT_CONFIG_KEYS:
        text_config.pop(key, None)
    text_config["model_type"] = "diffusion_gemma_text"
    return converted


def write_diffusion_gemma_config(source_path: Path, output_dir: Path) -> dict:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    converted = convert_gemma4_config_dict(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config.json"
    rendered = (json.dumps(converted, indent=2, sort_keys=True) + "\n").encode()
    output_path.write_bytes(rendered)
    return {
        "source_config": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_config": str(output_path),
        "output_sha256": hashlib.sha256(rendered).hexdigest(),
        "source_architecture": RAW_GEMMA4_ARCHITECTURE,
        "target_architecture": DIFFUSION_GEMMA_ARCHITECTURE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_diffusion_gemma_config(args.source_config, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
