"""Split a Qwen3-MoE checkpoint into SeedOmni modules.

Registered under ``OMNI_CONVERT_REGISTRY["qwen3_moe"]`` and dispatched by
``scripts/seed_omni/convert_model.py`` (via :func:`convert_checkpoint`).
Graphs come from ``configs/seed_omni/Qwen/qwen3_30b_a3b/train/``.

The upstream is a standard HF checkpoint (``Qwen3MoeForCausalLM``); no DeepSeek
-> HF pre-step is needed (unlike Janus).  The source is loaded with the stock
transformers class, whose weight-conversion mapping merges the per-expert HF
weights into the v5 **fused** layout; the VeOmni patched class skips that
mapping (VeOmni's own loader fuses instead), so it would leave the experts at
their init.  The saved backbone subfolder is therefore already fused.

The text encoder (``embed_tokens`` + ``lm_head``) is vocabulary-only and
MoE-agnostic, so it reuses the dense ``qwen3_text_encoder`` module verbatim.

Output layout::

    <output_dir>/
      qwen3_text_encoder/   # embed_tokens (+ lm_head if untied) + tokenizer
      qwen3_moe_llm/        # MoE decoder backbone (fused experts; no embed_tokens / no lm_head)
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, Qwen3MoeForCausalLM
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, attach_module_assets, load_family_graphs


def convert_qwen3_moe_checkpoint(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Split an upstream Qwen3-MoE checkpoint into two SeedOmni modules."""
    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/Qwen/qwen3_30b_a3b/train",
        training="graph_train.yaml",
        generation={"infer_text": "graph_infer.yaml"},
        training_graph=kwargs.pop("training_graph", None),
        generation_graph=kwargs.pop("generation_graph", None),
    )
    del kwargs
    print(f"Loading Qwen3-MoE from: {model_path}")
    import veomni.models.seed_omni.modules  # noqa: F401
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    Qwen3MoeLlm = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    Qwen3MoeLlmConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    # The text encoder is MoE-agnostic — reuse the dense qwen3 module.
    Qwen3TextEncoder = OMNI_MODEL_REGISTRY["qwen3_text_encoder"]()
    Qwen3TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3_text_encoder"]()

    model, loading_info = Qwen3MoeForCausalLM.from_pretrained(model_path, device_map="cpu", output_loading_info=True)
    if loading_info["missing_keys"]:
        missing = sorted(loading_info["missing_keys"])
        raise RuntimeError(f"Qwen3-MoE weights missing from {model_path}: {missing[:8]} ({len(missing)} total)")
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
    attach_module_assets(te, tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
    print(f"  tie_word_embeddings={cfg.tie_word_embeddings}")

    print("Extracting qwen3_moe_llm ...")
    llm_cfg = Qwen3MoeLlmConfig(text_config=cfg.to_dict())
    with no_init_weights(), init_empty_weights():
        llm = Qwen3MoeLlm._from_config(llm_cfg)
    # Experts are already fused at this point; drop the embedding (lives in text encoder).
    src = {k: v for k, v in model.model.state_dict().items() if not k.startswith("embed_tokens.")}
    llm.language_model.load_state_dict(src, assign=True)

    return {
        "modules": {"qwen3_text_encoder": te, "qwen3_moe_llm": llm},
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "infer_text",
    }


@OMNI_CONVERT_REGISTRY.register("qwen3_moe")
def _register_qwen3_moe_convert():
    return convert_qwen3_moe_checkpoint


__all__ = ["convert_qwen3_moe_checkpoint"]
