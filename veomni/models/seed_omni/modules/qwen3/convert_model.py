"""Split a Qwen3 checkpoint into SeedOmni V2 module subfolders.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3"]`` and dispatched by
``scripts/convert_model.py`` (via :func:`convert_checkpoint`).

Output layout::

    <output_dir>/
      qwen3_text_encoder/   # embed_tokens (+ lm_head if untied) + tokenizer
      qwen3_llm/            # decoder backbone (no embed_tokens / no lm_head)
"""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.utils.device import IS_NPU_AVAILABLE

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_npu import Qwen3ForCausalLM
else:
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM


def convert_qwen3_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Split an upstream Qwen3 checkpoint into two V2 module subfolders."""
    del kwargs
    print(f"Loading Qwen3 from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    Qwen3TextEncoder = OMNI_MODEL_REGISTRY["qwen3_text_encoder"]()
    Qwen3TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()

    model = Qwen3ForCausalLM.from_pretrained(model_path, device_map="cpu")
    model.eval()
    cfg = model.config

    print("Extracting qwen3_text_encoder ...")
    te_cfg = Qwen3TextEncoderConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        tie_word_embeddings=cfg.tie_word_embeddings,
        lm_head_bias=False,
    )
    with no_init_weights(), init_empty_weights():
        te = Qwen3TextEncoder._from_config(te_cfg)
    te.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict(), assign=True)
    if not cfg.tie_word_embeddings and model.lm_head is not None:
        src_sd = {k: v.detach().clone() for k, v in model.lm_head.state_dict().items()}
        te.lm_head.load_state_dict(src_sd, assign=True)

    te_dir = os.path.join(output_dir, "qwen3_text_encoder")
    te.save_pretrained(te_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(te_dir)
    print(f"  saved → {te_dir} (tie_word_embeddings={cfg.tie_word_embeddings})")

    print("Extracting qwen3_llm ...")
    llm_cfg = Qwen3LlmConfig(text_config=cfg.to_dict())
    with no_init_weights(), init_empty_weights():
        llm = Qwen3Llm._from_config(llm_cfg)
    src = {k: v for k, v in model.model.state_dict().items() if not k.startswith("embed_tokens.")}
    llm.language_model.load_state_dict(src, assign=True)

    llm_dir = os.path.join(output_dir, "qwen3_llm")
    llm.save_pretrained(llm_dir, safe_serialization=True)
    print(f"  saved → {llm_dir} (no embed_tokens / no lm_head)")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")


@OMNI_CONVERT_REGISTRY.register("qwen3")
def _register_qwen3_convert():
    return convert_qwen3_checkpoint


__all__ = ["convert_qwen3_checkpoint"]
