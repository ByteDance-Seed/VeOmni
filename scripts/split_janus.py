"""
Split a Janus-1.3B checkpoint into four SeedOmni V2 sub-module checkpoints.

Usage
-----
  python scripts/split_janus.py \
      --model_path /mnt/hdfs/veomni/models/transformers/Janus-1.3B \
      --output_dir /mnt/hdfs/veomni/models/seed_omni/janus_1.3b

Output structure (matches ``design.md`` §11)
--------------------------------------------
  <output_dir>/
    janus_siglip/
      config.json                     # JanusSiglipConfig (model_type=janus_siglip)
      model.safetensors               # vision_model.* + aligner.*
      preprocessor_config.json        # JanusSiglipProcessor (per-module asset)
    janus_vqvae/
      config.json                     # JanusVqvaeConfig (model_type=janus_vqvae)
      model.safetensors               # vqmodel.* + generation_*.*
      preprocessor_config.json        # JanusVqvaeProcessor (per-module asset)
    janus_llama/
      config.json                     # JanusLlamaConfig (model_type=janus_llama)
      model.safetensors               # language_model.* (no embed_tokens / no lm_head)
    text_embed/
      config.json                     # JanusTextEmbedConfig (model_type=janus_text_embed)
      model.safetensors               # embed_tokens.* (+ lm_head.* if untied)
    tokenizer/                        # global asset (one tokenizer per Omni model)

The OmniConfig YAML side then reads each module from
``<output_dir>/<module_name>``; ``AutoConfig.from_pretrained`` picks up
``model_type`` from each ``config.json`` and the V2
``MODULE_MIXIN_REGISTRY`` resolves the corresponding mixin class.

Why a separate ``text_embed``?
------------------------------
``embed_tokens`` and ``lm_head`` are vocabulary-dependent and conceptually
belong with the tokenizer side of the model — not the LLM backbone.
Pulling them out as a generic :class:`TextEmbed` OmniModule lets the same
graph treat text-vocab encode/decode and image-VQ encode/decode
symmetrically (both have ``encode`` / ``decode`` call-site methods).

Janus subclass: :class:`JanusTextEmbed`
---------------------------------------
Janus also needs to *emit* :code:`<begin_of_image>` / :code:`<end_of_image>`
boundary tokens around a VQ image span (the framework has no notion of
these; emitting them is a model concern).  We use
:class:`JanusTextEmbed` (model_type ``janus_text_embed``) — a thin
subclass of :class:`TextEmbed` that adds ``emit_image_start`` /
``emit_image_end`` call-site methods plus the two boundary token ids.
The split script reads the actual boi/eoi ids from the tokenizer so the
checkpoint stays self-describing.
"""

import argparse
import os
import shutil

import torch


def _save_state_dict(state_dict: dict, output_dir: str) -> None:
    """Save a state dict as safetensors (falling back to .pt if unavailable)."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print("  [warn] safetensors not available; saved as pytorch_model.bin")


def _resolve_token_id(tokenizer, candidates, default: int) -> int:
    """Look up a special-token id by trying each candidate name.

    ``candidates`` is an iterable of candidate token *strings* (or
    ``None``); the first one that resolves to a non-``unk`` id wins.
    Falls back to ``default`` if none match — keeping the script
    robust against tokenizer changes upstream.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in candidates:
        if not cand:
            continue
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != unk:
            return int(tid)
    return default


def split_janus(model_path: str, output_dir: str, dtype: str = "bfloat16") -> None:
    """Split the Janus checkpoint into the four V2 sub-module folders."""
    print(f"Loading Janus from: {model_path}")
    from transformers import AutoTokenizer, JanusForConditionalGeneration

    # Force registration of V2 module configs so save_pretrained works.
    from veomni.models.seed_omni.modules import (
        JanusLlama,
        JanusLlamaConfig,
        JanusSiglip,
        JanusSiglipConfig,
        JanusTextEmbed,
        JanusTextEmbedConfig,
        JanusVqvae,
        JanusVqvaeConfig,
    )
    from veomni.models.seed_omni.modules.janus import JanusSiglipProcessor, JanusVqvaeProcessor

    torch_dtype = getattr(torch, dtype)
    model = JanusForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="cpu")
    model.eval()
    cfg = model.config
    inner = model.model

    # Pull the boundary-token ids straight from the tokenizer's special
    # tokens — Janus uses ``<begin_of_image>`` (boi) and ``<end_of_image>``
    # (eoi) to delimit a VQ image span.  Storing them in the
    # JanusTextEmbedConfig keeps the checkpoint self-describing so the
    # FSM yaml doesn't need to hard-code vocabulary specifics.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    boi_id = _resolve_token_id(tokenizer, ("<begin_of_image>", getattr(tokenizer, "boi_token", None)), default=100016)
    eoi_id = _resolve_token_id(tokenizer, ("<end_of_image>", getattr(tokenizer, "eoi_token", None)), default=100593)
    image_token_id = getattr(cfg, "image_token_id", 100581)
    print(f"  tokenizer boi={boi_id}  eoi={eoi_id}  image_token={image_token_id}")

    # ── 1. janus_siglip (vision_model + aligner) ────────────────────────────
    print("Extracting janus_siglip ...")
    vision_cfg_dict = cfg.vision_config.to_dict() if hasattr(cfg.vision_config, "to_dict") else vars(cfg.vision_config)
    siglip_cfg = JanusSiglipConfig(vision_config=vision_cfg_dict)
    siglip = JanusSiglip(siglip_cfg)
    siglip.vision_model.load_state_dict(inner.vision_model.state_dict())
    siglip.aligner.load_state_dict(inner.aligner.state_dict())

    siglip_dir = os.path.join(output_dir, "janus_siglip")
    siglip.to(torch_dtype).save_pretrained(siglip_dir, safe_serialization=True)
    # Per-module asset: vision processor.  The original Janus image processor
    # already encapsulates the right resize + normalise constants.
    try:
        from transformers.models.janus.processing_janus import JanusProcessor

        proc = JanusProcessor.from_pretrained(model_path)
        # Keep just the image processor side; tokenizer is global.
        ip = proc.image_processor
        siglip_proc = JanusSiglipProcessor(**ip.to_dict())
        siglip_proc.save_pretrained(siglip_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not extract image processor for siglip: {e}")
    print(f"  saved → {siglip_dir}")

    # ── 2. janus_vqvae (vqmodel + generation_*) ───────────────────────────
    print("Extracting janus_vqvae ...")
    vq_cfg_dict = cfg.vq_config.to_dict() if hasattr(cfg.vq_config, "to_dict") else vars(cfg.vq_config)
    vqvae_cfg = JanusVqvaeConfig(vq_config=vq_cfg_dict, freeze_vqvae=True)
    vqvae = JanusVqvae(vqvae_cfg)
    vqvae.vqmodel.load_state_dict(inner.vqmodel.state_dict())
    vqvae.generation_embeddings.load_state_dict(inner.generation_embeddings.state_dict())
    vqvae.generation_aligner.load_state_dict(inner.generation_aligner.state_dict())
    vqvae.generation_head.load_state_dict(inner.generation_head.state_dict())

    vqvae_dir = os.path.join(output_dir, "janus_vqvae")
    vqvae.to(torch_dtype).save_pretrained(vqvae_dir, safe_serialization=True)
    try:
        # VQVAE shares the same image processor as SigLIP in the original Janus
        # checkpoint; copy the JSON next to the VQVAE module so the per-module
        # checkpoint is self-contained.
        from transformers.models.janus.processing_janus import JanusProcessor

        proc = JanusProcessor.from_pretrained(model_path)
        ip = proc.image_processor
        vqvae_proc = JanusVqvaeProcessor(**ip.to_dict())
        vqvae_proc.save_pretrained(vqvae_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not extract image processor for vqvae: {e}")
    print(f"  saved → {vqvae_dir}")

    # ── 3. text_embed (language_model.embed_tokens + lm_head) ──────────────
    # Saved as JanusTextEmbed (model_type=janus_text_embed) so the model
    # owns the boi/eoi emitter methods used by the inference FSM bridge
    # states.  The config is a JanusTextEmbedConfig — vocab_size /
    # hidden_size / tie_word_embeddings come from the original LLaMA
    # config; boi/eoi come from the tokenizer.
    print("Extracting text_embed (as janus_text_embed) ...")
    text_cfg = cfg.text_config
    text_cfg_dict = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else vars(text_cfg)
    tie = bool(text_cfg_dict.get("tie_word_embeddings", False))
    te_cfg = JanusTextEmbedConfig(
        vocab_size=text_cfg_dict.get("vocab_size"),
        hidden_size=text_cfg_dict.get("hidden_size"),
        tie_word_embeddings=tie,
        # LLaMA / Janus convention: lm_head has no bias.
        lm_head_bias=False,
        begin_of_image_token_id=boi_id,
        end_of_image_token_id=eoi_id,
    )
    te = JanusTextEmbed(te_cfg)
    te.embed_tokens.load_state_dict(inner.language_model.embed_tokens.state_dict())
    if not tie:
        # Untied: load the original lm_head linear.
        te.lm_head.load_state_dict(model.lm_head.state_dict())

    te_dir = os.path.join(output_dir, "text_embed")
    te.to(torch_dtype).save_pretrained(te_dir, safe_serialization=True)
    print(f"  saved → {te_dir} (tie_word_embeddings={tie}, boi={boi_id}, eoi={eoi_id})")

    # ── 4. janus_llama (LlamaModel backbone, no embed_tokens / no lm_head) ─
    print("Extracting janus_llama ...")
    llama_cfg = JanusLlamaConfig(
        text_config=text_cfg_dict,
        image_token_id=image_token_id,
        gen_image_token_id=image_token_id,
    )
    llama = JanusLlama(llama_cfg)
    src = inner.language_model.state_dict()
    src = {k: v for k, v in src.items() if not k.startswith("embed_tokens.")}
    llama.language_model.load_state_dict(src, strict=False)

    llama_dir = os.path.join(output_dir, "janus_llama")
    llama.to(torch_dtype).save_pretrained(llama_dir, safe_serialization=True)
    print(f"  saved → {llama_dir} (no embed_tokens / no lm_head)")

    # ── 5. global tokenizer ─────────────────────────────────────────────────
    # One tokenizer per Omni model; lives at the root rather than per-module.
    print("Copying global tokenizer ...")
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    try:
        tokenizer.save_pretrained(tok_dir)
    except Exception:  # noqa: BLE001
        # Fall back to a raw file copy (covers tokenizers that don't round-trip
        # cleanly through ``save_pretrained``).
        for fname in os.listdir(model_path):
            if any(fname.startswith(p) for p in ("tokenizer", "special_tokens", "vocab")):
                shutil.copy2(os.path.join(model_path, fname), os.path.join(tok_dir, fname))
    print(f"  tokenizer → {tok_dir}")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")
    print("janus_1.3b.yaml (V2) should reference:")
    print(f"  tokenizer_path : {tok_dir}")
    print(f"  modules.janus_siglip.weights_path : {siglip_dir}")
    print(f"  modules.janus_vqvae.weights_path  : {vqvae_dir}")
    print(f"  modules.text_embed.weights_path   : {te_dir}")
    print(f"  modules.janus_llama.weights_path  : {llama_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Janus-1.3B into SeedOmni V2 sub-checkpoints")
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
