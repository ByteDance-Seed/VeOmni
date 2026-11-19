"""Split a Qwen3-VL checkpoint into SeedOmni modules.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3_vl"]`` and dispatched by
``scripts/seed_omni/convert_model.py`` (via :func:`convert_checkpoint`).
Graphs come from ``configs/seed_omni/Qwen/qwen3vl/``.

Output layout::

    <output_dir>/
      qwen3vl_vision/         # ViT + patch merger + deepstack mergers + image processor
      qwen3vl_text_encoder/   # embed_tokens (+ lm_head if untied) + tokenizer
      qwen3vl_llm/            # text backbone (no embed_tokens / no lm_head)
"""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.checkpoint.weights import init_empty_weights
from veomni.utils import logging
from veomni.utils.device import IS_NPU_AVAILABLE

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, attach_module_assets, load_family_graphs


def _qwen3vl_for_causal_lm():
    """The patched Qwen3-VL class for this accelerator.

    Imported inside the call, not at module import time: the family package is
    imported by every ``veomni.models.seed_omni.modules`` load, and building the
    patched modeling eagerly would make every SeedOmni run pay for it.
    """
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import (
            Qwen3VLForConditionalGeneration,
        )
    else:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLForConditionalGeneration,
        )
    return Qwen3VLForConditionalGeneration


def _build_kwargs(attn_implementation: str | None) -> dict[str, Any]:
    """``_from_config`` kwargs recording an explicit attention backend, if one was asked for.

    A module built without one keeps its config default, which is what a plain
    convert wants; on a host that cannot run that default (CPU, no flash-attn)
    the caller passes ``eager`` and the split records it.
    """
    return {"attn_implementation": attn_implementation} if attn_implementation is not None else {}


logger = logging.get_logger(__name__)


def convert_qwen3vl_checkpoint(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Split an upstream Qwen3-VL checkpoint into three SeedOmni modules."""
    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/Qwen/qwen3vl",
        training="train/graph_train.yaml",
        generation={"vision_understanding": "infer/graph_infer.yaml"},
        training_graph=kwargs.pop("training_graph", None),
        generation_graph=kwargs.pop("generation_graph", None),
    )
    # The upstream model is instantiated to read its weights, so it needs an
    # attention backend this host can actually build. `convert_with_hf_layout`
    # honours this argument for layout-driven families; the converter path has to
    # do the same or a CPU host dies in `_check_and_adjust_attn_implementation`.
    attn_implementation = kwargs.pop("attn_implementation", None)
    del kwargs
    logger.info_rank0(f"Loading Qwen3-VL from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    Qwen3VLVisionEncoder = OMNI_MODEL_REGISTRY["qwen3vl_vision"]()
    Qwen3VLVisionEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()
    Qwen3VLTextEncoder = OMNI_MODEL_REGISTRY["qwen3vl_text_encoder"]()
    Qwen3VLTextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()
    Qwen3VLLlm = OMNI_MODEL_REGISTRY["qwen3vl_llm"]()
    Qwen3VLLlmConfig = OMNI_CONFIG_REGISTRY["qwen3vl_llm"]()

    model = _qwen3vl_for_causal_lm().from_pretrained(
        model_path,
        device_map="cpu",
        **({"attn_implementation": attn_implementation} if attn_implementation is not None else {}),
    )
    model.eval()
    cfg = model.config
    text_cfg = cfg.text_config
    inner = model.model  # Qwen3VLModel: .visual + .language_model

    logger.info_rank0("Extracting qwen3vl_vision ...")
    vision_cfg = Qwen3VLVisionEncoderConfig(vision_config=cfg.vision_config.to_dict())
    with no_init_weights(), init_empty_weights():
        vision = Qwen3VLVisionEncoder._from_config(vision_cfg, **_build_kwargs(attn_implementation))
    vision.visual.load_state_dict(inner.visual.state_dict(), assign=True)
    # Two distinct processors / configs, both loaded by the vision module:
    #   image -> preprocessor_config.json       (Qwen2VLImageProcessor)
    #   video -> video_preprocessor_config.json (Qwen3VLVideoProcessor)
    processor = AutoProcessor.from_pretrained(model_path)
    attach_module_assets(vision, image_processor=processor.image_processor, video_processor=processor.video_processor)

    logger.info_rank0("Extracting qwen3vl_text_encoder ...")
    # Official Qwen3-VL puts `tie_word_embeddings` on the top-level config
    # (2B=True, 8B=False). Nested `text_config` often omits the field.
    tie_word_embeddings = bool(getattr(cfg, "tie_word_embeddings", getattr(text_cfg, "tie_word_embeddings", True)))
    te_cfg = Qwen3VLTextEncoderConfig(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        tie_word_embeddings=tie_word_embeddings,
        lm_head_bias=False,
    )
    with no_init_weights(), init_empty_weights():
        te = Qwen3VLTextEncoder._from_config(te_cfg, **_build_kwargs(attn_implementation))
    te.embed_tokens.load_state_dict(inner.language_model.embed_tokens.state_dict(), assign=True)
    if not tie_word_embeddings and model.lm_head is not None:
        src_sd = {k: v.detach().clone() for k, v in model.lm_head.state_dict().items()}
        te.lm_head.load_state_dict(src_sd, assign=True)
    attach_module_assets(te, tokenizer=AutoTokenizer.from_pretrained(model_path))
    logger.info_rank0(f"  tie_word_embeddings={tie_word_embeddings}")

    logger.info_rank0("Extracting qwen3vl_llm ...")
    llm_cfg = Qwen3VLLlmConfig(
        text_config=text_cfg.to_dict(),
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    with no_init_weights(), init_empty_weights():
        llm = Qwen3VLLlm._from_config(llm_cfg, **_build_kwargs(attn_implementation))
    # The module owns no `embed_tokens` / `lm_head` (the text encoder does), so
    # those two keys are the *only* legitimate difference. `strict=True` would
    # reject them outright, and `strict=False` alone would also accept a
    # half-populated module — assert the exact set instead, so a renamed or
    # dropped layer fails here rather than as a meta tensor at the first step.
    dropped = ("embed_tokens.", "lm_head.")
    src = {k: v for k, v in inner.language_model.state_dict().items() if not k.startswith(dropped)}
    src_keys = set(src)
    expected_keys = set(llm.language_model.state_dict())
    if src_keys != expected_keys:
        raise RuntimeError(
            "convert qwen3vl_llm: unexpected parameter mismatch. "
            f"missing={sorted(expected_keys - src_keys)} unused={sorted(src_keys - expected_keys)}"
        )
    llm.language_model.load_state_dict(src, assign=True, strict=True)

    return {
        "modules": {"qwen3vl_vision": vision, "qwen3vl_text_encoder": te, "qwen3vl_llm": llm},
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "vision_understanding",
    }


@OMNI_CONVERT_REGISTRY.register("qwen3_vl")
def _register_qwen3vl_convert():
    return convert_qwen3vl_checkpoint


__all__ = ["convert_qwen3vl_checkpoint"]
