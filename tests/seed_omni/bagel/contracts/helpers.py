from __future__ import annotations

from pathlib import Path

from veomni.arguments import OmniArguments, OmniModelArguments
from veomni.arguments.arguments_types import DataArguments
from veomni.models.seed_omni import OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY


def bagel_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Bagel" / "bagel_7b_mot"


def config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def model_cls(model_type: str):
    return OMNI_MODEL_REGISTRY[model_type]()


def load_omni_config(
    *,
    model_path: str = "",
    modules_path: Path,
    train_graph_path: Path | None = None,
    infer_modules: dict | Path | None = None,
    infer_graph_path: Path | None = None,
    generation_kwargs: dict | None = None,
) -> OmniConfig:
    model_args = OmniModelArguments(
        model_path=model_path or ".",
        config_path=model_path or ".",
        modules=str(modules_path),
    )
    base = OmniArguments(
        model=model_args,
        data=DataArguments(train_path=""),
    )._to_base_args()
    return OmniConfig.from_omni_args(
        global_args=base,
        model_path=model_path,
        modules=str(modules_path),
        train_graph=str(train_graph_path) if train_graph_path else None,
        infer_modules=infer_modules,
        infer_graph=str(infer_graph_path) if infer_graph_path else None,
        generation_kwargs=generation_kwargs,
    )


def tiny_bagel_qwen2_cfg() -> dict:
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
