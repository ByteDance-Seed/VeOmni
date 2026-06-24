"""Split a Janus checkpoint into SeedOmni V2 module subfolders."""

from __future__ import annotations

import os

from transformers import AutoTokenizer, JanusForConditionalGeneration, JanusProcessor, LlamaConfig
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY

from .convert_janus_weight_to_hf import convert_model as convert_janus_to_hf


def convert_janus_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Split an upstream Janus checkpoint into four V2 module subfolders."""
    del kwargs
    origin_model_path = model_path
    if not origin_model_path.endswith("-hf"):
        hf_model_path = model_path + "-hf"
        convert_janus_to_hf(local_dir=origin_model_path, output_dir=hf_model_path)
    else:
        hf_model_path = origin_model_path
    _split_janus_hf(hf_model_path, output_dir)


def _split_janus_hf(model_path: str, output_dir: str) -> None:
    print(f"Loading Janus from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
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

    model = JanusForConditionalGeneration.from_pretrained(model_path, device_map="cpu")
    processor = JanusProcessor.from_pretrained(model_path)
    model.eval()
    cfg = model.config
    inner = model.model

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

    print("Extracting janus_vqvae ...")
    vq_cfg_dict = cfg.vq_config.to_dict()
    vqvae_cfg = JanusVqvaeConfig(vq_config=vq_cfg_dict)
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

    print("Extracting janus_text_encoder ...")
    text_cfg: LlamaConfig = cfg.text_config
    text_cfg_dict = text_cfg.to_dict()
    te_cfg = JanusTextEncoderConfig(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
        lm_head_bias=False,
    )
    with no_init_weights(), init_empty_weights():
        te = JanusTextEncoder._from_config(te_cfg)
    te.embed_tokens.load_state_dict(inner.language_model.embed_tokens.state_dict(), assign=True)
    if not text_cfg.tie_word_embeddings:
        src_sd = {k: v.detach().clone() for k, v in model.lm_head.state_dict().items()}
        te.lm_head.load_state_dict(src_sd, assign=True)

    te_dir = os.path.join(output_dir, "janus_text_encoder")
    te.save_pretrained(te_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(te_dir)
    print(f"  saved → {te_dir} (tie_word_embeddings={text_cfg.tie_word_embeddings})")

    print("Extracting janus_llama ...")
    gen_image_token_id = int(tokenizer.convert_tokens_to_ids("<image_0>"))
    llama_cfg = JanusLlamaConfig(
        text_config=text_cfg_dict,
        image_token_id=int(tokenizer.image_token_id),
        gen_image_token_id=gen_image_token_id,
    )
    with no_init_weights(), init_empty_weights():
        llama = JanusLlama._from_config(llama_cfg)
    src = inner.language_model.state_dict()
    src = {k: v for k, v in src.items() if not k.startswith("embed_tokens.")}
    llama.language_model.load_state_dict(src, assign=True)

    llama_dir = os.path.join(output_dir, "janus_llama")
    llama.save_pretrained(llama_dir, safe_serialization=True)
    print(f"  saved → {llama_dir} (no embed_tokens / no lm_head)")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")


@OMNI_CONVERT_REGISTRY.register("janus")  # hf ckpt
@OMNI_CONVERT_REGISTRY.register("multi_modality")  # deepseek ckpt
def _register_janus_convert():
    return convert_janus_checkpoint


__all__ = ["convert_janus_checkpoint"]
