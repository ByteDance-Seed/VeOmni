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
    text_encoder/
      config.json                     # JanusTextEncoderConfig (model_type=janus_text_encoder)
      model.safetensors               # embed_tokens.* (+ lm_head.* if untied)
    tokenizer/                        # global asset (one tokenizer per Omni model)

The OmniConfig YAML side then reads each module from
``<output_dir>/<module_name>``; ``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY``
resolve ``model_type`` from each ``config.json`` to the mixin class.

Why a separate ``text_encoder``?
------------------------------
``embed_tokens`` and ``lm_head`` are vocabulary-dependent and conceptually
belong with the tokenizer side of the model — not the LLM backbone.
Pulling them out as a generic :class:`TextEncoder` OmniModule lets the same
graph treat text-vocab encode/decode and image-VQ encode/decode
symmetrically (both have ``encode`` / ``decode`` call-site methods).

Janus subclass: :class:`JanusTextEncoder`
---------------------------------------
Janus also needs to *emit* :code:`<begin_of_image>` / :code:`<end_of_image>`
boundary tokens around a VQ image span (the framework has no notion of
these; emitting them is a model concern).  We use
:class:`JanusTextEncoder` (model_type ``janus_text_encoder``) — a thin
subclass of :class:`TextEncoder` that adds ``emit_image_start`` /
``emit_image_end`` call-site methods plus the two boundary token ids.
Boundary / placeholder token ids are resolved at runtime from the module
tokenizer (not stored in ``config.json``).
"""

import argparse
import os

import torch
from transformers import LlamaConfig
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.models.seed_omni.modules.janus.convert_janus_weight_to_hf import convert_model


def _save_state_dict(state_dict: dict, output_dir: str) -> None:
    """Save a state dict as safetensors (falling back to .pt if unavailable)."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print("  [warn] safetensors not available; saved as pytorch_model.bin")


def split_janus(model_path: str, output_dir: str) -> None:
    """Split the Janus checkpoint into the four sub-module folders."""
    print(f"Loading Janus from: {model_path}")
    from transformers import AutoTokenizer, JanusForConditionalGeneration, JanusProcessor

    # Resolve V2 module classes lazily via registry factories.
    import veomni.models.seed_omni.modules  # noqa: F401 — register factories
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY

    JanusLlama = OMNI_MODEL_REGISTRY["janus_llama"]()
    JanusLlamaConfig = OMNI_CONFIG_REGISTRY["janus_llama"]()
    JanusSiglip = OMNI_MODEL_REGISTRY["janus_siglip"]()
    JanusSiglipConfig = OMNI_CONFIG_REGISTRY["janus_siglip"]()
    JanusTextEncoder = OMNI_MODEL_REGISTRY["janus_text_encoder"]()
    JanusTextEncoderConfig = OMNI_CONFIG_REGISTRY["janus_text_encoder"]()
    JanusVqvae = OMNI_MODEL_REGISTRY["janus_vqvae"]()
    JanusVqvaeConfig = OMNI_CONFIG_REGISTRY["janus_vqvae"]()
    JanusSiglipProcessor = OMNI_PROCESSOR_REGISTRY["janus_siglip"]()
    JanusVqvaeProcessor = OMNI_PROCESSOR_REGISTRY["janus_vqvae"]()

    model = JanusForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    processor = JanusProcessor.from_pretrained(model_path)
    model.eval()
    cfg = model.config
    inner = model.model

    # ── 1. janus_siglip (vision_model + aligner) ────────────────────────────
    print("Extracting janus_siglip ...")
    vision_cfg_dict = cfg.vision_config.to_dict()
    siglip_cfg = JanusSiglipConfig(vision_config=vision_cfg_dict)
    with no_init_weights(), init_empty_weights():
        siglip = JanusSiglip._from_config(siglip_cfg)
    siglip.vision_model.load_state_dict(inner.vision_model.state_dict(), assign=True)
    siglip.aligner.load_state_dict(inner.aligner.state_dict(), assign=True)

    siglip_dir = os.path.join(output_dir, "janus_siglip")
    siglip.save_pretrained(siglip_dir, safe_serialization=True)
    image_processor = processor.image_processor
    siglip_processor = JanusSiglipProcessor(**image_processor.to_dict())
    siglip_processor.save_pretrained(siglip_dir)
    print(f"  saved → {siglip_dir}")

    # ── 2. janus_vqvae (vqmodel + generation_*) ───────────────────────────
    print("Extracting janus_vqvae ...")
    vq_cfg_dict = cfg.vq_config.to_dict()
    vqvae_cfg = JanusVqvaeConfig(vq_config=vq_cfg_dict, freeze_vqvae=True)
    with no_init_weights(), init_empty_weights():
        vqvae = JanusVqvae._from_config(vqvae_cfg)
    vqvae.vqmodel.load_state_dict(inner.vqmodel.state_dict(), assign=True)
    vqvae.generation_embeddings.load_state_dict(inner.generation_embeddings.state_dict(), assign=True)
    vqvae.generation_aligner.load_state_dict(inner.generation_aligner.state_dict(), assign=True)
    vqvae.generation_head.load_state_dict(inner.generation_head.state_dict(), assign=True)

    vqvae_dir = os.path.join(output_dir, "janus_vqvae")
    vqvae.save_pretrained(vqvae_dir, safe_serialization=True)
    vqvae_processor = JanusVqvaeProcessor(**image_processor.to_dict())
    vqvae_processor.save_pretrained(vqvae_dir)
    print(f"  saved → {vqvae_dir}")

    # ── 3. text_encoder (language_model.embed_tokens + lm_head) ──────────────
    # Saved as JanusTextEncoder (model_type=janus_text_encoder) so the model
    # owns the boi/eoi emitter methods used by the inference FSM bridge
    # states.  The config is a JanusTextEncoderConfig — vocab_size /
    # hidden_size / tie_word_embeddings come from the original LLaMA
    # config; boi/eoi ids are resolved at runtime from the module tokenizer.
    print("Extracting text_encoder (as janus_text_encoder) ...")
    text_cfg: LlamaConfig = cfg.text_config
    text_cfg_dict = text_cfg.to_dict()
    te_cfg = JanusTextEncoderConfig(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
        # LLaMA / Janus convention: lm_head has no bias.
        lm_head_bias=False,
    )
    with no_init_weights(), init_empty_weights():
        te = JanusTextEncoder._from_config(te_cfg)
    te.embed_tokens.load_state_dict(inner.language_model.embed_tokens.state_dict(), assign=True)
    if not text_cfg.tie_word_embeddings:
        # Untied: load the original lm_head linear.
        te.lm_head.load_state_dict(model.lm_head.state_dict(), assign=True)

    te_dir = os.path.join(output_dir, "text_encoder")
    te.save_pretrained(te_dir, safe_serialization=True)
    print(f"  saved → {te_dir} (tie_word_embeddings={text_cfg.tie_word_embeddings})")

    # ── 4. janus_llama (LlamaModel backbone, no embed_tokens / no lm_head) ─
    print("Extracting janus_llama ...")
    llama_cfg = JanusLlamaConfig(text_config=text_cfg_dict)
    with no_init_weights(), init_empty_weights():
        llama = JanusLlama._from_config(llama_cfg)
    src = inner.language_model.state_dict()
    src = {k: v for k, v in src.items() if not k.startswith("embed_tokens.")}
    llama.language_model.load_state_dict(src, assign=True)

    llama_dir = os.path.join(output_dir, "janus_llama")
    llama.save_pretrained(llama_dir, safe_serialization=True)
    print(f"  saved → {llama_dir} (no embed_tokens / no lm_head)")

    # ── 5. global tokenizer ─────────────────────────────────────────────────
    # One tokenizer per Omni model; lives at the root rather than per-module.
    print(f"Copying global tokenizer to {output_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)
    print(f"  tokenizer → {output_dir}")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")
    print("janus_1.3b.yaml (V2) should reference:")
    print(f"  tokenizer_path : {output_dir}")
    print(f"  modules.janus_siglip : {siglip_dir}")
    print(f"  modules.janus_vqvae : {vqvae_dir}")
    print(f"  modules.text_encoder : {te_dir}")
    print(f"  modules.janus_llama : {llama_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Janus-1.3B into SeedOmni V2 sub-checkpoints")
    parser.add_argument(
        "--model_path",
        default="transformers/Janus-1.3B",
        help="Path to the original Janus checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        default="seed_omni/Janus-1.3B",
        help="Directory to write the split sub-checkpoints",
    )
    args = parser.parse_args()
    origin_model_path = args.model_path
    hf_model_path = args.model_path + "-hf"
    convert_model(
        local_dir=origin_model_path,
        output_dir=hf_model_path,
    )
    split_janus(hf_model_path, args.output_dir)
