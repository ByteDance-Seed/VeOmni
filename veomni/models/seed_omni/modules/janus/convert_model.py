"""Split a Janus checkpoint into SeedOmni V2 modules.

Registered under ``OMNI_CONVERT_REGISTRY["janus"]`` (HF) and
``["multi_modality"]`` (DeepSeek); :func:`convert_checkpoint` writes the omni
checkpoint. Graphs come from ``configs/seed_omni/Janus/janus_1.3b/``.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

from transformers import AutoTokenizer, JanusForConditionalGeneration, JanusProcessor, LlamaConfig
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.models.seed_omni.utils.convert_registry import (
    OMNI_CONVERT_REGISTRY,
    attach_module_assets,
    load_family_graphs,
)

from .convert_janus_weight_to_hf import convert_model as convert_janus_to_hf


def convert_janus_checkpoint(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Split an upstream Janus checkpoint into four V2 modules.

    A DeepSeek-format source (path not ending in ``-hf``) is first converted to
    the HF layout in a temporary directory.
    """
    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/Janus/janus_1.3b",
        training="train/graph_train.yaml",
        generation={
            "infer_gen": "infer/graph_infer_gen.yaml",
            "infer_und": "infer/graph_infer_und.yaml",
            "infer_interleave": "infer/graph_infer_interleave.yaml",
        },
        training_graph=kwargs.pop("training_graph", None),
        generation_graph=kwargs.pop("generation_graph", None),
    )
    del kwargs
    if model_path.endswith("-hf"):
        modules = _split_janus_hf(model_path)
    else:
        with tempfile.TemporaryDirectory(prefix="janus-hf-") as hf_model_path:
            convert_janus_to_hf(local_dir=model_path, output_dir=hf_model_path)
            modules = _split_janus_hf(hf_model_path)
    return {
        "modules": modules,
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "infer_interleave",
    }


def _split_janus_hf(model_path: str) -> dict[str, Any]:
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
    image_processor = JanusProcessor.from_pretrained(model_path).image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Janus's vocab is byte-level BPE. An HF export written before
    # convert_janus_weight_to_hf bypassed the mislabelled ``LlamaTokenizer``
    # class carries a rebuilt Metaspace pipeline that tokenizes differently.
    pre_tokenizer = json.loads(tokenizer.backend_tokenizer.to_str()).get("pre_tokenizer") or {}
    if pre_tokenizer.get("type") == "Metaspace":
        raise ValueError(
            f"{model_path} holds a Metaspace tokenizer; Janus needs the byte-level one. "
            "Convert from the DeepSeek checkpoint (the directory without the -hf suffix) instead."
        )
    model.eval()
    cfg = model.config
    inner = model.model

    print("Extracting janus_siglip ...")
    siglip_cfg = JanusSiglipConfig(vision_config=cfg.vision_config.to_dict())
    with no_init_weights(), init_empty_weights():
        siglip = JanusSiglip._from_config(siglip_cfg)
    siglip.vision_model.load_state_dict(inner.vision_model.state_dict(), assign=True)
    siglip.aligner.load_state_dict(inner.aligner.state_dict(), assign=True)
    attach_module_assets(siglip, image_processor=JanusSiglipProcessor(**image_processor.to_dict()))

    print("Extracting janus_vqvae ...")
    vqvae_cfg = JanusVqvaeConfig(vq_config=cfg.vq_config.to_dict())
    with no_init_weights(), init_empty_weights():
        vqvae = JanusVqvae._from_config(vqvae_cfg)
    vqvae.vqmodel.load_state_dict(inner.vqmodel.state_dict(), assign=True)
    vqvae.generation_embeddings.load_state_dict(inner.generation_embeddings.state_dict(), assign=True)
    vqvae.generation_aligner.load_state_dict(inner.generation_aligner.state_dict(), assign=True)
    vqvae.generation_head.load_state_dict(inner.generation_head.state_dict(), assign=True)
    attach_module_assets(vqvae, image_processor=JanusVqvaeProcessor(**image_processor.to_dict()))

    print("Extracting janus_text_encoder ...")
    text_cfg: LlamaConfig = cfg.text_config
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
    attach_module_assets(te, tokenizer=tokenizer)
    print(f"  tie_word_embeddings={text_cfg.tie_word_embeddings}")

    print("Extracting janus_llama ...")
    llama_cfg = JanusLlamaConfig(
        text_config=text_cfg.to_dict(),
        image_token_id=int(tokenizer.image_token_id),
        gen_image_token_id=int(tokenizer.convert_tokens_to_ids("<image_0>")),
    )
    with no_init_weights(), init_empty_weights():
        llama = JanusLlama._from_config(llama_cfg)
    src = {k: v for k, v in inner.language_model.state_dict().items() if not k.startswith("embed_tokens.")}
    llama.language_model.load_state_dict(src, assign=True)

    return {
        "janus_siglip": siglip,
        "janus_vqvae": vqvae,
        "janus_text_encoder": te,
        "janus_llama": llama,
    }


@OMNI_CONVERT_REGISTRY.register("janus")  # hf ckpt
@OMNI_CONVERT_REGISTRY.register("multi_modality")  # deepseek ckpt
def _register_janus_convert():
    return convert_janus_checkpoint


__all__ = ["convert_janus_checkpoint"]
