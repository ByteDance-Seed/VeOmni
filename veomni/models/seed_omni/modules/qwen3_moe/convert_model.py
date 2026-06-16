"""Split a Qwen3-MoE checkpoint into SeedOmni V2 module subfolders.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3_moe"]`` and dispatched by
``scripts/convert_model.py`` (via :func:`convert_checkpoint`).

The upstream is a standard HF checkpoint (``Qwen3MoeForCausalLM``); no DeepSeek
-> HF pre-step is needed (unlike Janus).  ``Qwen3MoeForCausalLM.from_pretrained``
merges the per-expert HF weights into the v5 **fused** layout at load time via
the registered ``Qwen3MoeCheckpointTensorConverter``, so the saved backbone
subfolder is already fused.

The text encoder (``embed_tokens`` + ``lm_head``) is vocabulary-only and
MoE-agnostic, so it reuses the dense ``qwen3_text_encoder`` module verbatim.

Output layout::

    <output_dir>/
      qwen3_text_encoder/   # embed_tokens (+ lm_head if untied) + tokenizer
      qwen3_moe_llm/        # MoE decoder backbone (fused experts; no embed_tokens / no lm_head)
"""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.utils.device import IS_NPU_AVAILABLE

from ...convert_registry import OMNI_CONVERT_REGISTRY


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import Qwen3MoeForCausalLM
else:
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeForCausalLM


def convert_qwen3_moe_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Split an upstream Qwen3-MoE checkpoint into two V2 module subfolders."""
    del kwargs
    print(f"Loading Qwen3-MoE from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    Qwen3MoeLlm = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    Qwen3MoeLlmConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    # The text encoder is MoE-agnostic — reuse the dense qwen3 module.
    Qwen3TextEncoder = OMNI_MODEL_REGISTRY["qwen3_text_encoder"]()
    Qwen3TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()

    # from_pretrained merges per-expert HF weights into the fused v5 layout.
    model = Qwen3MoeForCausalLM.from_pretrained(model_path, device_map="cpu")
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

    print("Extracting qwen3_moe_llm ...")
    llm_cfg = Qwen3MoeLlmConfig(text_config=cfg.to_dict())
    with no_init_weights(), init_empty_weights():
        llm = Qwen3MoeLlm._from_config(llm_cfg)
    # Experts are already fused at this point; drop the embedding (lives in text encoder).
    src = {k: v for k, v in model.model.state_dict().items() if not k.startswith("embed_tokens.")}
    llm.language_model.load_state_dict(src, assign=True)

    llm_dir = os.path.join(output_dir, "qwen3_moe_llm")
    llm.save_pretrained(llm_dir, safe_serialization=True)
    print(f"  saved → {llm_dir} (fused experts; no embed_tokens / no lm_head)")

    print(f"\nDone.  Split checkpoint saved to: {output_dir}")


@OMNI_CONVERT_REGISTRY.register("qwen3_moe")
def _register_qwen3_moe_convert():
    return convert_qwen3_moe_checkpoint


__all__ = ["convert_qwen3_moe_checkpoint"]
