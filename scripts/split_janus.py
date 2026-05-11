"""
Split a Janus-1.3B checkpoint into three OmniModule sub-module checkpoints.

Usage
-----
  python scripts/split_janus.py \
      --model_path /mnt/hdfs/veomni/models/transformers/Janus-1.3B \
      --output_dir /mnt/hdfs/veomni/models/seed_omni/janus_1.3b

Output structure
----------------
  <output_dir>/
    vision_encoder/
      config.json          (JanusVisionEncoderConfig)
      model.safetensors    (vision_model + aligner weights)
    vq_decoder/
      config.json          (JanusVQDecoderConfig)
      model.safetensors    (vqmodel + gen_embeddings + gen_aligner + gen_head)
    ar_llm/
      config.json          (JanusLLMConfig)
      model.safetensors    (language_model + lm_head weights)
    tokenizer/             (copy of original tokenizer files)

After splitting, set the following paths in janus_1.3b.yaml:
  modules:
    vision_encoder:
      weights_path: <output_dir>/vision_encoder
    vq_decoder:
      weights_path: <output_dir>/vq_decoder
    ar_llm:
      weights_path: <output_dir>/ar_llm
"""

import argparse
import json
import os
import shutil

import torch


def _save_state_dict(state_dict: dict, output_dir: str) -> None:
    """Save a state dict as safetensors (falling back to .pt if safetensors unavailable)."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print(f"  [warn] safetensors not available; saved as pytorch_model.bin")


def _save_config(cfg_dict: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)


def split_janus(model_path: str, output_dir: str, dtype: str = "bfloat16") -> None:
    print(f"Loading Janus from: {model_path}")
    from transformers import JanusForConditionalGeneration

    torch_dtype = getattr(torch, dtype)
    model = JanusForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="cpu"
    )
    model.eval()
    cfg = model.config
    inner = model.model

    # ── 1. vision_encoder (vision_model + aligner) ────────────────────────────
    print("Extracting vision_encoder ...")
    ve_state = {}
    for k, v in inner.vision_model.state_dict().items():
        ve_state[f"vision_model.{k}"] = v
    for k, v in inner.aligner.state_dict().items():
        ve_state[f"aligner.{k}"] = v

    vision_cfg_dict = cfg.vision_config.to_dict() if hasattr(cfg.vision_config, "to_dict") else vars(cfg.vision_config)
    ve_config = {
        "model_type": "janus_vision_encoder",
        "vision_config": vision_cfg_dict,
    }
    ve_dir = os.path.join(output_dir, "vision_encoder")
    _save_state_dict(ve_state, ve_dir)
    _save_config(ve_config, ve_dir)
    print(f"  saved {len(ve_state)} tensors → {ve_dir}")

    # ── 2. vq_decoder (vqmodel + generation_* layers) ────────────────────────
    print("Extracting vq_decoder ...")
    vq_state = {}
    for k, v in inner.vqmodel.state_dict().items():
        vq_state[f"vqmodel.{k}"] = v
    for k, v in inner.generation_embeddings.state_dict().items():
        vq_state[f"generation_embeddings.{k}"] = v
    for k, v in inner.generation_aligner.state_dict().items():
        vq_state[f"generation_aligner.{k}"] = v
    for k, v in inner.generation_head.state_dict().items():
        vq_state[f"generation_head.{k}"] = v

    vq_cfg_dict = cfg.vq_config.to_dict() if hasattr(cfg.vq_config, "to_dict") else vars(cfg.vq_config)
    vq_config = {
        "model_type": "janus_vq_decoder",
        "vq_config": vq_cfg_dict,
        "freeze_vqvae": True,
    }
    vq_dir = os.path.join(output_dir, "vq_decoder")
    _save_state_dict(vq_state, vq_dir)
    _save_config(vq_config, vq_dir)
    print(f"  saved {len(vq_state)} tensors → {vq_dir}")

    # ── 3. ar_llm (language_model + lm_head) ────────────────────────────────
    print("Extracting ar_llm ...")
    llm_state = {}
    for k, v in inner.language_model.state_dict().items():
        llm_state[f"language_model.{k}"] = v
    for k, v in model.lm_head.state_dict().items():
        llm_state[f"lm_head.{k}"] = v

    text_cfg_dict = cfg.text_config.to_dict() if hasattr(cfg.text_config, "to_dict") else vars(cfg.text_config)
    llm_config = {
        "model_type": "janus_llm",
        "text_config": text_cfg_dict,
        "image_token_id": getattr(cfg, "image_token_id", 100577),
        "gen_image_token_id": getattr(cfg, "gen_image_token_id", 100578),
    }
    llm_dir = os.path.join(output_dir, "ar_llm")
    _save_state_dict(llm_state, llm_dir)
    _save_config(llm_config, llm_dir)
    print(f"  saved {len(llm_state)} tensors → {llm_dir}")

    # ── 4. copy tokenizer ─────────────────────────────────────────────────────
    print("Copying tokenizer ...")
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    for fname in os.listdir(model_path):
        if any(fname.startswith(p) for p in ("tokenizer", "special_tokens", "vocab")):
            shutil.copy2(os.path.join(model_path, fname), os.path.join(tok_dir, fname))
    print(f"  tokenizer → {tok_dir}")

    print(f"\nDone!  Split checkpoint saved to: {output_dir}")
    print("Use the following in your janus_1.3b.yaml:")
    print(f"  vision_encoder weights_path: {ve_dir}")
    print(f"  vq_decoder    weights_path: {vq_dir}")
    print(f"  ar_llm        weights_path: {llm_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Janus-1.3B into OmniModule sub-checkpoints")
    parser.add_argument(
        "--model_path",
        default="/mnt/hdfs/veomni/models/transformers/Janus-1.3B",
        help="Path to the original Janus checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/hdfs/veomni/models/seed_omni/janus_1.3b",
        help="Directory to write the split sub-checkpoints",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="Torch dtype for loading the checkpoint",
    )
    args = parser.parse_args()
    split_janus(args.model_path, args.output_dir, args.dtype)
