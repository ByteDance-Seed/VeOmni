"""Precompute Gemma text features for LTX-2 offline training.

This script encodes text prompts through Gemma3 + FeatureExtractor and saves
the intermediate features to disk. During training, only the lightweight
EmbeddingsProcessor connectors run on these features.

Usage:
    python veomni/models/diffusers/ltx2_3/precompute_ltx2_gemma_features.py \
        --data_dir /path/to/training/data \
        --gemma_model_path /path/to/gemma3 \
        --checkpoint_path /path/to/ltx2-checkpoint.safetensors \
        --max_sequence_length 256

Expected input directory structure:
    data_dir/
    ├── videos/          # Raw video files (mp4, etc.)
    ├── prompts.txt      # One text prompt per line (aligned with videos)
    └── (optional) audio/ # Raw audio files

Output directory structure:
    data_dir/
    ├── latents/         # VAE-encoded video latents (.pt)
    ├── conditions/      # Gemma features (.pt)
    └── audio_latents/   # VAE-encoded audio latents (.pt, optional)
"""

import argparse
import json
from pathlib import Path

import torch
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    EMBEDDINGS_PROCESSOR_KEY_REMAP,
    _create_feature_extractor,
)
from tqdm import tqdm

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401
from veomni.utils.device import get_device_type


def load_feature_extractor(checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load feature extractor weights from LTX-2 checkpoint.

    Uses safetensors safe_open to selectively read only feature extractor keys,
    avoiding loading the entire (potentially 20+ GB) checkpoint into memory.
    """
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_path)

    transformer_config_path = (
        ckpt_path / "transformer" / "config.json"
        if ckpt_path.is_dir()
        else ckpt_path.parent / "transformer" / "config.json"
    )
    if not transformer_config_path.exists():
        transformer_config_path = ckpt_path / "config.json" if ckpt_path.is_dir() else ckpt_path.parent / "config.json"

    with open(transformer_config_path) as f:
        config = json.load(f)

    transformer_config = config.get("transformer", config)
    feature_extractor = _create_feature_extractor(transformer_config)

    fe_prefixes = {
        old_prefix: new_prefix
        for old_prefix, new_prefix in EMBEDDINGS_PROCESSOR_KEY_REMAP.items()
        if new_prefix.startswith("feature_extractor.")
    }
    target_prefixes = tuple(fe_prefixes.keys())

    fe_state_dict = {}

    if ckpt_path.is_dir():
        safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
    elif ckpt_path.suffix == ".safetensors":
        safetensor_files = [ckpt_path]
    else:
        safetensor_files = []

    if safetensor_files:
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt", device=str(device)) as f:
                for key in f.keys():
                    if key.startswith(target_prefixes):
                        for old_prefix in fe_prefixes:
                            if key.startswith(old_prefix):
                                fe_key = "feature_extractor." + key[len(old_prefix) :]
                                fe_state_dict[fe_key] = f.get_tensor(key).to(dtype)
                                break
    else:
        if ckpt_path.is_dir():
            bin_files = sorted(ckpt_path.glob("*.bin")) + sorted(ckpt_path.glob("*.pt"))
        elif ckpt_path.suffix in (".bin", ".pt"):
            bin_files = [ckpt_path]
        else:
            bin_files = []

        for bin_path in bin_files:
            all_state_dict = torch.load(str(bin_path), map_location=device, weights_only=True)
            for key, value in all_state_dict.items():
                for old_prefix in fe_prefixes:
                    if key.startswith(old_prefix):
                        fe_key = "feature_extractor." + key[len(old_prefix) :]
                        fe_state_dict[fe_key] = value.to(dtype)
                        break
            del all_state_dict

    if fe_state_dict:
        missing, unexpected = feature_extractor.load_state_dict(fe_state_dict, strict=False)
        if missing:
            print(f"Feature extractor missing keys: {missing}")
        if unexpected:
            print(f"Feature extractor unexpected keys: {unexpected}")
    else:
        print("WARNING: No feature extractor weights found in checkpoint!")

    return feature_extractor.to(device=device, dtype=dtype)


def encode_prompt(text_encoder, feature_extractor, prompt: str, device, dtype):
    """Encode a single text prompt through Gemma + feature extractor."""
    hidden_states, attention_mask = text_encoder.encode(prompt)

    video_feats, audio_feats = feature_extractor(hidden_states, attention_mask, padding_side="left")

    return {
        "video_prompt_embeds": video_feats[0].cpu(),
        "audio_prompt_embeds": audio_feats[0].cpu() if audio_feats is not None else None,
        "prompt_attention_mask": attention_mask[0].cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute Gemma features for LTX-2 training")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--gemma_model_path", type=str, required=True, help="Path to Gemma3 model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to LTX-2 checkpoint")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max token sequence length")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to prompts file (one per line)")
    parser.add_argument("--device", type=str, default=get_device_type(), help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("Loading Gemma text encoder...")
    text_encoder = GemmaTextEncoder.from_pretrained(args.gemma_model_path, max_length=args.max_sequence_length)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    print("Loading feature extractor from checkpoint...")
    feature_extractor = load_feature_extractor(args.checkpoint_path, device, dtype)
    feature_extractor.eval()

    data_dir = Path(args.data_dir)
    conditions_dir = data_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = args.prompts_file or (data_dir / "prompts.txt")
    if not Path(prompts_file).exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(prompts)} prompts...")

    for idx, prompt in enumerate(tqdm(prompts)):
        output_path = conditions_dir / f"{idx:06d}.pt"
        if output_path.exists():
            continue

        with torch.no_grad():
            features = encode_prompt(text_encoder, feature_extractor, prompt, device, dtype)

        torch.save(features, output_path)

    print(f"Done! Features saved to {conditions_dir}")


if __name__ == "__main__":
    main()
