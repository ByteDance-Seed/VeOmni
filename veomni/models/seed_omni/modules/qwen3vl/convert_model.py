"""Split a Qwen3-VL checkpoint into SeedOmni V2 module subfolders.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3_vl"]`` and dispatched by
``scripts/convert_model.py`` (via :func:`convert_checkpoint`).

Output layout::

    <output_dir>/
      qwen3vl_vision/         # ViT + patch merger + deepstack mergers + image processor
      qwen3vl_text_encoder/   # embed_tokens (+ lm_head if untied) + tokenizer
      qwen3vl_llm/            # text backbone (no embed_tokens / no lm_head)
"""

from __future__ import annotations

import os

from transformers import AutoProcessor, AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.utils.device import IS_NPU_AVAILABLE

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import (
        Qwen3VLForConditionalGeneration,
    )
else:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLForConditionalGeneration,
    )


def convert_qwen3vl_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Split an upstream Qwen3-VL checkpoint into three V2 module subfolders."""
    del kwargs
    print(f"Loading Qwen3-VL from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    Qwen3VLVisionEncoder = OMNI_MODEL_REGISTRY["qwen3vl_vision"]()
    Qwen3VLVisionEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()
    Qwen3VLTextEncoder = OMNI_MODEL_REGISTRY["qwen3vl_text_encoder"]()
    Qwen3VLTextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()
    Qwen3VLLlm = OMNI_MODEL_REGISTRY["qwen3vl_llm"]()
    Qwen3VLLlmConfig = OMNI_CONFIG_REGISTRY["qwen3vl_llm"]()

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map="cpu")
    model.eval()
    cfg = model.config
    text_cfg = cfg.text_config
    inner = model.model  # Qwen3VLModel: .visual + .language_model

    print("Extracting qwen3vl_vision ...")
    vision_cfg = Qwen3VLVisionEncoderConfig(vision_config=cfg.vision_config.to_dict())
    with no_init_weights(), init_empty_weights():
        vision = Qwen3VLVisionEncoder._from_config(vision_cfg)
    vision.visual.load_state_dict(inner.visual.state_dict(), assign=True)
    vision_dir = os.path.join(output_dir, "qwen3vl_vision")
    vision.save_pretrained(vision_dir, safe_serialization=True)
    # Two distinct processors / configs, both loaded by the vision module:
    #   image -> preprocessor_config.json       (Qwen2VLImageProcessor)
    #   video -> video_preprocessor_config.json (Qwen3VLVideoProcessor)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.save_pretrained(vision_dir)
    processor.video_processor.save_pretrained(vision_dir)
    print(f"  saved → {vision_dir}")

    print("Extracting qwen3vl_text_encoder ...")
    te_cfg = Qwen3VLTextEncoderConfig(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
        lm_head_bias=False,
    )
    with no_init_weights(), init_empty_weights():
        te = Qwen3VLTextEncoder._from_config(te_cfg)
    te.embed_tokens.load_state_dict(inner.language_model.embed_tokens.state_dict(), assign=True)
    if not text_cfg.tie_word_embeddings and model.lm_head is not None:
        src_sd = {k: v.detach().clone() for k, v in model.lm_head.state_dict().items()}
        te.lm_head.load_state_dict(src_sd, assign=True)
    te_dir = os.path.join(output_dir, "qwen3vl_text_encoder")
    te.save_pretrained(te_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(te_dir)
    print(f"  saved → {te_dir} (tie_word_embeddings={text_cfg.tie_word_embeddings})")

    print("Extracting qwen3vl_llm ...")
    llm_cfg = Qwen3VLLlmConfig(
        text_config=text_cfg.to_dict(),
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
        image_token_id=cfg.image_token_id,
    )
    with no_init_weights(), init_empty_weights():
        llm = Qwen3VLLlm._from_config(llm_cfg)
    src = {k: v for k, v in inner.language_model.state_dict().items() if not k.startswith("embed_tokens.")}
    llm.language_model.load_state_dict(src, assign=True, strict=False)
    llm_dir = os.path.join(output_dir, "qwen3vl_llm")
    llm.save_pretrained(llm_dir, safe_serialization=True)
    print(f"  saved → {llm_dir} (no embed_tokens / no lm_head)")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")


@OMNI_CONVERT_REGISTRY.register("qwen3_vl")
def _register_qwen3vl_convert():
    return convert_qwen3vl_checkpoint


__all__ = ["convert_qwen3vl_checkpoint"]
