"""Load Cosmos3-Nano's transformer, Wan VAE and tokenizer from the release tree.

Direct loading and the generic offline splitter both use this layout. A temporary
HF index links the original component shards; no weights are copied to load them.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from safetensors import safe_open
from transformers import PretrainedConfig

from ...utils.hf_layout import (
    OMNI_HF_LAYOUT_DETECTORS,
    OMNI_HF_LAYOUT_REGISTRY,
    OmniHFLayout,
    OmniHFModuleLayout,
    copy_source_non_weight_files,
)
from .flow_connector.configuration import Cosmos3FlowConnectorConfig
from .mot.configuration import Cosmos3MotConfig
from .text_encoder.configuration import Cosmos3TextEncoderConfig
from .vae.configuration import Cosmos3VaeConfig


TEXT = "cosmos3_text_encoder"
VAE = "cosmos3_vae"
MOT = "cosmos3_mot"
FLOW = "cosmos3_flow_connector"
_SIGNATURE = {"patch_latent_dim", "qk_norm_for_diffusion", "latent_patch_size"}
_UNUSED = (
    "blocks.",
    "merger.",
    "deepstack_merger_list.",
    "patch_embed.",
    "pos_embed.",
    "audio_proj_in.",
    "audio_proj_out.",
    "audio_modality_embed",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _root(path: str) -> Path:
    path = Path(path)
    return path.parent if path.name == "transformer" and not (path / "transformer").is_dir() else path


def is_cosmos3_component_tree(path: str) -> bool:
    root = _root(path)
    config = root / "transformer" / "config.json"
    if not config.is_file() or not (root / "vae" / "config.json").is_file():
        return False
    transformer = _json(config)
    return transformer.get("model_type") == "qwen3_vl_text" and _SIGNATURE <= transformer.keys()


def _tokenizer(root: str):
    from transformers import AutoProcessor, AutoTokenizer

    # Preserve the offline converter's preference for the release processor's tokenizer.
    try:
        processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    except (OSError, ValueError, KeyError, ImportError, AttributeError):
        processor = None
    if getattr(processor, "tokenizer", None) is not None:
        return processor.tokenizer
    directory = Path(root) / "text_tokenizer"
    return AutoTokenizer.from_pretrained(directory if directory.is_dir() else root, local_files_only=True)


def _config(root: str):
    if not is_cosmos3_component_tree(root):
        raise ValueError(f"Not a Cosmos3-Nano component tree: {root}")
    transformer = _json(Path(root) / "transformer/config.json")
    tokenizer = _tokenizer(root)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return PretrainedConfig(
        transformer=transformer,
        vae=_json(Path(root) / "vae/config.json"),
        generation_marker_ids=[eos, tokenizer.convert_tokens_to_ids("<|vision_start|>")],
        module_dtypes=_manifest(root)[1],
    )


def _text_config(c):
    return Cosmos3TextEncoderConfig(
        vocab_size=c.transformer["vocab_size"],
        hidden_size=c.transformer["hidden_size"],
        tie_word_embeddings=False,
        generation_marker_ids=c.generation_marker_ids,
        dtype=c.module_dtypes[TEXT],
    )


def _mot_config(c):
    keys = (
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
    rope = c.transformer.get("rope_scaling") or {}
    return Cosmos3MotConfig(
        **{k: c.transformer[k] for k in keys if k in c.transformer},
        mrope_section=rope.get("mrope_section"),
        mrope_interleaved=rope.get("mrope_interleaved", True),
        dtype=c.module_dtypes[MOT],
    )


def _flow_config(c):
    keys = ("hidden_size", "patch_latent_dim", "latent_patch_size", "timestep_scale")
    return Cosmos3FlowConnectorConfig(
        **{k: c.transformer[k] for k in keys if k in c.transformer}, dtype=c.module_dtypes[FLOW]
    )


def _vae_config(c):
    return Cosmos3VaeConfig(
        vae_config=c.vae, latent_patch_size=c.transformer["latent_patch_size"], dtype=c.module_dtypes[VAE]
    )


def _component_shards(directory: Path):
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"Ambiguous component weight indexes under {directory}")
    if not indexes:
        return sorted(directory.glob("*.safetensors")), None
    index = _json(indexes[0])["weight_map"]
    names = sorted(set(index.values()))
    if any(Path(n).name != n or "/" in n or "\\" in n or ":" in n or not n.endswith(".safetensors") for n in names):
        raise ValueError(f"Expected component-local safetensors shard names in {indexes[0]}")
    return [directory / n for n in names], index


@lru_cache(maxsize=8)
def _manifest(root: str):
    """Read only headers. Keep the old strict load contract even though the common loader can initialize missing keys."""
    files, dtypes, shapes = {}, {}, {name: {} for name in COSMOS3_HF_LAYOUT.modules}
    for component in ("transformer", "vae"):
        directory = Path(root) / component
        shards, index = _component_shards(directory)
        if not shards:
            raise FileNotFoundError(f"No safetensors weights under {directory}")
        component_map = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in files:
                        raise ValueError(f"Duplicate Cosmos3 weight {key!r} in {shard} and {files[key]}")
                    files[key] = f"{component}/{shard.name}"
                    component_map[key] = shard.name
                    route = COSMOS3_HF_LAYOUT.route(key)
                    if route is None and component == "transformer":
                        continue
                    if route is None or (route[0] == VAE) != (component == "vae"):
                        raise ValueError(f"Weight {key!r} is in the wrong Cosmos3 component: {component}")
                    name, target = route
                    tensor = f.get_slice(key)
                    shapes[name][target] = tuple(tensor.get_shape())
                    dtype = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}.get(tensor.get_dtype())
                    if dtype is None or dtypes.setdefault(name, dtype) != dtype:
                        raise ValueError(f"Unsupported or mixed weight dtype in {name}: {key} ({tensor.get_dtype()})")
        if index is not None and index != component_map:
            raise ValueError(f"Weight index does not match shard contents in {directory}")
    if missing := shapes.keys() - dtypes.keys():
        raise ValueError(f"Missing Cosmos3 module weights: {sorted(missing)}")
    return files, dtypes, shapes


class Cosmos3HFLayout(OmniHFLayout):
    def resolve_source_path(self, source_path):
        return str(_root(source_path))

    def route(self, source_key):
        routed = super().route(source_key)
        if routed is None and not source_key.startswith(_UNUSED):
            raise ValueError(f"Cosmos3 has no route for checkpoint key {source_key!r}.")
        return routed

    def validate_module_state(self, root, name, model):
        expected = {key: tuple(t.shape) for key, t in model.state_dict().items()}
        actual = _manifest(root)[2][name]
        if expected != actual:
            mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
            raise ValueError(
                f"{name} weight mismatch: missing={sorted(expected.keys() - actual.keys())}, "
                f"unexpected={sorted(actual.keys() - expected.keys())}, shape mismatch={mismatched}"
            )

    def source_weight_files(self, source_path):
        return _manifest(source_path)[0]

    def resolve_weights_path(self, source_path):
        return _staged_weights(source_path)

    def build_fqn_to_index_mapping(self, weight_files):
        indices = {path: i for i, path in enumerate(sorted(set(weight_files.values())), 1)}
        return {key: indices[path] for key, path in weight_files.items()}

    def finalize_exported_shards(self, source_path, save_path, weight_files: Mapping[str, str]):
        root, out = Path(source_path), Path(save_path)
        copy_source_non_weight_files(str(root), str(out))
        files = sorted(set(weight_files.values()))
        for i, name in enumerate(files, 1):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (out / f"model-{i:05d}-of-{len(files):05d}.safetensors").replace(target)
        (out / "model.safetensors.index.json").unlink(missing_ok=True)
        for component in ("transformer", "vae"):
            copy_source_non_weight_files(str(root / component), str(out / component))
            for index in (root / component).glob("*.safetensors.index.json"):
                shutil.copyfile(index, out / component / index.name)
        # Other pipeline components are opaque to the four training modules.
        # Keep their original assets (and a root-level upstream index, if present).
        for name in ("text_tokenizer", "sound_tokenizer", "vision_encoder", "scheduler", "assets", "images"):
            if (root / name).is_dir():
                shutil.copytree(root / name, out / name, dirs_exist_ok=True)
        root_index = root / "model.safetensors.index.json"
        if root_index.is_file():
            shutil.copyfile(root_index, out / root_index.name)


@lru_cache(maxsize=8)
def _staged_weights(root: str) -> str:
    files = _manifest(root)[0]
    staged = Path(tempfile.mkdtemp(prefix="veomni_cosmos3_weights_"))
    atexit.register(shutil.rmtree, staged, True)
    names = {}
    for i, path in enumerate(sorted(set(files.values()))):
        alias = f"component-{i}.safetensors"
        names[path] = alias
        try:
            (staged / alias).symlink_to(Path(root) / path)
        except OSError:
            os.link(Path(root) / path, staged / alias)
    index = {"metadata": {}, "weight_map": {k: names[v] for k, v in files.items()}}
    (staged / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return str(staged)


def _prefixes(*names, target=""):
    return tuple((name + ".", target + name + ".") for name in names)


COSMOS3_HF_LAYOUT = Cosmos3HFLayout(
    modules={
        TEXT: OmniHFModuleLayout(
            _prefixes("embed_tokens", "lm_head"), _text_config, lambda root: {"tokenizer": _tokenizer(root)}
        ),
        VAE: OmniHFModuleLayout(
            _prefixes("encoder", "decoder", "quant_conv", "post_quant_conv", target="vae."), _vae_config
        ),
        MOT: OmniHFModuleLayout(_prefixes("layers", "norm", "norm_moe_gen"), _mot_config),
        FLOW: OmniHFModuleLayout(_prefixes("proj_in", "proj_out", "time_embedder"), _flow_config),
    },
    graph_dir="configs/seed_omni/Cosmos3/cosmos3_nano",
    training_graphs={
        "online_training": "train/graph_online.yaml",
        "offline_embedding": "train/graph_offline_embedding.yaml",
        "offline_training": "train/graph_offline_training.yaml",
    },
    load_hf_config=_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("cosmos3_omni")
def _register_cosmos3_hf_layout():
    return COSMOS3_HF_LAYOUT


@OMNI_HF_LAYOUT_DETECTORS.register("cosmos3_omni")
def _detect_cosmos3_hf_layout(path: str) -> str | None:
    return "cosmos3_omni" if is_cosmos3_component_tree(path) else None
