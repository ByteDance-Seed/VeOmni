"""Thin wrapper around the vendored official BAGEL implementation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.initialization import no_init_weights
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tests.seed_omni.bagel.transformers.vendor.data.data_utils import add_special_tokens
from tests.seed_omni.bagel.transformers.vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from tests.seed_omni.bagel.transformers.vendor.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from veomni.models.module_utils import init_empty_weights


class BagelOfficialReferenceWrapper(nn.Module):
    """Own the official BAGEL assembly so captures do not import vendor internals."""

    def __init__(
        self,
        model: Bagel,
        *,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids or {}
        self.config = model.config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        *,
        llm_config: Qwen2Config,
        vit_config: SiglipVisionConfig | None = None,
        vae_config: Any | None = None,
        visual_gen: bool,
        visual_und: bool,
        tokenizer: Qwen2Tokenizer | None = None,
        new_token_ids: dict[str, int] | None = None,
        init_on_meta: bool = True,
    ) -> BagelOfficialReferenceWrapper:
        _ensure_default_rope_init()
        _normalize_llm_config(llm_config)
        config = BagelConfig(
            visual_gen=visual_gen,
            visual_und=visual_und,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
        )
        context = _empty_init_context() if init_on_meta else nullcontext()
        with context:
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config) if visual_und and vit_config is not None else None
            model = Bagel(language_model, vit_model, config)
            if visual_und and vit_model is not None:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=init_on_meta)
        return cls(model, tokenizer=tokenizer, new_token_ids=new_token_ids)

    @classmethod
    def from_model_root(
        cls,
        model_root: str | Path,
        *,
        visual_gen: bool,
        visual_und: bool,
        init_on_meta: bool = True,
    ) -> BagelOfficialReferenceWrapper:
        root = Path(model_root)
        llm_config = Qwen2Config.from_json_file(str(root / "llm_config.json"))
        _normalize_llm_config(llm_config)

        tokenizer = Qwen2Tokenizer.from_pretrained(str(root), local_files_only=True)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens > 0:
            llm_config.vocab_size = len(tokenizer)

        vit_config = SiglipVisionConfig.from_json_file(str(root / "vit_config.json")) if visual_und else None
        return cls.from_configs(
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=None,
            visual_gen=visual_gen,
            visual_und=visual_und,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            init_on_meta=init_on_meta,
        )


@contextmanager
def _empty_init_context():
    with no_init_weights(), init_empty_weights():
        yield


def _normalize_llm_config(llm_config: Qwen2Config) -> None:
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "bos_token_id", 0)


def _ensure_default_rope_init() -> None:
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _compute_default_rope_parameters(config: Any, device: torch.device | None = None, **kwargs: Any):
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0
