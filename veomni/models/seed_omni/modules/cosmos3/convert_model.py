"""Split a released Cosmos3-Nano checkpoint into the four SeedOmni modules.

The release is a diffusers-style tree with no root ``config.json``::

    Cosmos3-Nano/
    ├── transformer/      both towers + flow connector (``model_type: qwen3_vl_text``)
    ├── vae/              Wan2.2 ``AutoencoderKLWan``
    └── text_tokenizer/

so the converter is registered on the transformer's ``model_type`` and is run on
that subfolder::

    python scripts/seed_omni/convert_model.py \\
        --model_path pretrained_models/Cosmos3-Nano/transformer \\
        --output_dir pretrained_models/Cosmos3-Nano-omni

Every transformer tensor is routed by an explicit prefix table; the names need
no remapping. Two families of keys are dropped on purpose: the Qwen3-VL vision
encoder (this recipe never runs the ViT) and the audio / action pathways (Nano
trains neither). Any other key is an error rather than a silent skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, attach_module_assets, load_family_graphs


GRAPH_DIR = "configs/seed_omni/Cosmos3/cosmos3_nano"
TRAINING_GRAPHS = {
    "online_training": "train/graph_online.yaml",
    "offline_embedding": "train/graph_offline_embedding.yaml",
    "offline_training": "train/graph_offline_training.yaml",
}
# Training-only: there is no flow sampler yet, so the checkpoint carries no generation graph.
GENERATION_GRAPHS: dict[str, str] = {}

TEXT_ENCODER = "cosmos3_text_encoder"
VAE = "cosmos3_vae"
MOT = "cosmos3_mot"
FLOW_CONNECTOR = "cosmos3_flow_connector"

WEIGHT_ROUTES = (
    (("embed_tokens.", "lm_head."), TEXT_ENCODER),
    (("layers.", "norm.", "norm_moe_gen."), MOT),
    (("proj_in.", "proj_out.", "time_embedder."), FLOW_CONNECTOR),
)
DROPPED_PREFIXES = (
    # Qwen3-VL vision encoder.
    "blocks.",
    "merger.",
    "deepstack_merger_list.",
    "patch_embed.",
    "pos_embed.",
    # Audio and action pathways.
    "audio_proj_in.",
    "audio_proj_out.",
    "audio_modality_embed",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
)
# Keys only a Cosmos3 transformer config carries; a plain Qwen3-VL text tower has none of them.
COSMOS3_CONFIG_KEYS = ("patch_latent_dim", "qk_norm_for_diffusion", "latent_patch_size")


def route_weight(key: str) -> str | None:
    """Module a checkpoint tensor belongs to, or ``None`` when it is dropped on purpose."""
    if key.startswith(DROPPED_PREFIXES):
        return None
    for prefixes, module in WEIGHT_ROUTES:
        if key.startswith(prefixes):
            return module
    raise ValueError(f"Cosmos3 converter has no route for checkpoint key {key!r}.")


def _iter_safetensors(directory: Path) -> Iterator[tuple[str, Any]]:
    from safetensors import safe_open

    # HF names the index ``model.safetensors.index.json``, diffusers ``diffusion_pytorch_model...``.
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        files = sorted({file for index in indexes for file in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(path.name for path in directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights under {directory}.")
    for name in files:
        with safe_open(str(directory / name), framework="pt") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _load_tokenizer(root: Path) -> Any:
    """The checkpoint processor's tokenizer when there is one, else ``text_tokenizer/``."""
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_dir = root / "text_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.is_dir() else root))
    try:
        processor = AutoProcessor.from_pretrained(str(root))
    except Exception:
        processor = None
    return getattr(processor, "tokenizer", None) or tokenizer


def _module_configs(transformer: dict[str, Any], vae: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from .flow_connector.configuration import Cosmos3FlowConnectorConfig
    from .mot.configuration import Cosmos3MotConfig
    from .text_encoder.configuration import Cosmos3TextEncoderConfig
    from .vae.configuration import Cosmos3VaeConfig

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    # The release keeps the mRoPE knobs under ``rope_scaling``.
    rope = transformer.get("rope_scaling") or {}
    mot_keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "hidden_act",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "qk_norm_for_text",
        "qk_norm_for_diffusion",
        "use_und_k_norm_for_gen",
        "rope_theta",
        "base_fps",
        "enable_fps_modulation",
        "temporal_compression_factor",
        "unified_3d_mrope_temporal_modality_margin",
        "latent_patch_size",
    )
    return {
        TEXT_ENCODER: Cosmos3TextEncoderConfig(
            vocab_size=transformer["vocab_size"],
            hidden_size=transformer["hidden_size"],
            tie_word_embeddings=False,
            generation_marker_ids=[int(eos), int(vision_start)],
        ),
        VAE: Cosmos3VaeConfig(vae_config=vae, latent_patch_size=transformer["latent_patch_size"]),
        MOT: Cosmos3MotConfig(
            **{key: transformer[key] for key in mot_keys if key in transformer},
            mrope_section=rope.get("mrope_section"),
            mrope_interleaved=rope.get("mrope_interleaved", True),
        ),
        FLOW_CONNECTOR: Cosmos3FlowConnectorConfig(
            **{
                key: transformer[key]
                for key in ("hidden_size", "patch_latent_dim", "latent_patch_size", "timestep_scale")
                if key in transformer
            }
        ),
    }


def convert_cosmos3(
    model_path: str,
    *,
    training_graph: str | None = None,
    generation_graph: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the four modules on meta, then assign every released tensor to its owner."""
    import torch
    from diffusers import AutoencoderKLWan

    from .flow_connector.modeling import Cosmos3FlowConnector
    from .mot.modeling import Cosmos3Mot
    from .text_encoder.modeling import Cosmos3TextEncoder
    from .vae.modeling import Cosmos3Vae

    del kwargs
    transformer_dir = Path(model_path)
    transformer = json.loads((transformer_dir / "config.json").read_text())
    missing = [key for key in COSMOS3_CONFIG_KEYS if key not in transformer]
    if missing:
        raise ValueError(
            f"{transformer_dir} is a {transformer.get('model_type')!r} checkpoint without {missing}, so it is not a "
            "Cosmos3 transformer; no other qwen3_vl_text checkpoint has a SeedOmni converter."
        )
    root = transformer_dir.parent
    vae_dir = root / "vae"
    vae_config = json.loads((vae_dir / "config.json").read_text())
    tokenizer = _load_tokenizer(root)

    configs = _module_configs(transformer, vae_config, tokenizer)
    classes = {
        TEXT_ENCODER: Cosmos3TextEncoder,
        VAE: Cosmos3Vae,
        MOT: Cosmos3Mot,
        FLOW_CONNECTOR: Cosmos3FlowConnector,
    }
    with torch.device("meta"):
        modules = {name: cls(configs[name]) for name, cls in classes.items()}

    states: dict[str, dict[str, Any]] = {name: {} for name in modules}
    for key, tensor in _iter_safetensors(transformer_dir):
        target = route_weight(key)
        if target is not None:
            states[target][key] = tensor
    vae = AutoencoderKLWan.from_pretrained(str(root), subfolder="vae")
    states[VAE] = {f"vae.{key}": tensor for key, tensor in vae.state_dict().items()}
    for name, module in modules.items():
        # ``strict``: a module whose checkpoint keys do not match its parameters exactly is an error.
        module.load_state_dict(states.pop(name), strict=True, assign=True)
    attach_module_assets(modules[TEXT_ENCODER], tokenizer=tokenizer)

    training_graphs, generation_graphs = load_family_graphs(
        GRAPH_DIR,
        training=TRAINING_GRAPHS,
        generation=GENERATION_GRAPHS,
        training_graph=training_graph,
        generation_graph=generation_graph,
    )
    return {
        "modules": modules,
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "train_type": "online_training",
        "infer_type": next(iter(generation_graphs), None),
    }


@OMNI_CONVERT_REGISTRY.register("qwen3_vl_text")
def register_cosmos3_convert():
    return convert_cosmos3


__all__ = ["convert_cosmos3", "route_weight"]
