"""Shared helpers for BAGEL tests."""

from __future__ import annotations

from pathlib import Path

from veomni.arguments import OmniArguments, OmniDataArguments
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.omni_arguments import OmniModelRuntimeArguments, build_module_runtime_args, build_omni_model_runtime


def bagel_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "Bagel" / "bagel_7b_mot"


def config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def model_cls(model_type: str):
    """Return the VeOmni-accelerated module class (training / graph hooks)."""
    return OMNI_ACCELERATED_MODEL_REGISTRY[model_type]()


def native_model_cls(model_type: str):
    """Return the HF-native module class (eager inference weights)."""
    return OMNI_MODEL_REGISTRY[model_type]()


def load_module_runtime_args(
    *,
    model_path: str = "",
    modules_path: Path,
    for_inference: bool = False,
) -> dict:
    base = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path or ".",
            model_config={"modules": str(modules_path)},
        ),
        data=OmniDataArguments(train_path=""),
    )._to_module_global_args()
    return build_module_runtime_args(
        global_args=base,
        model_path=model_path,
        modules=str(modules_path),
        for_inference=for_inference,
    )


def load_omni_config(
    *,
    model_path: str = "",
    modules_path: Path,
    train_graph_path: Path | None = None,
    infer_graph_path: Path | None = None,
    generation_kwargs: dict | None = None,
) -> OmniConfig:
    model_path = model_path or "."
    model_config = {"modules": str(modules_path)}
    if train_graph_path is not None:
        model_config["train_graph"] = str(train_graph_path)
    base = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config=model_config,
        ),
        data=OmniDataArguments(train_path="."),
    )._to_module_global_args()
    return build_omni_model_runtime(
        global_args=base,
        model_path=model_path,
        train_modules=str(modules_path),
        train_graph=str(train_graph_path) if train_graph_path else None,
        infer_graph=str(infer_graph_path) if infer_graph_path else None,
        generation_kwargs=generation_kwargs,
    ).to_hf_config()


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
