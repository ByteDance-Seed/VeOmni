"""LTX-2.5 BF16 split-pack loading for the shared LTX training path.

The transformer keeps official LTX parameter names. Frozen text projections
and connectors use the same checkpoint prefix mapping as LTX-2.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


DIT_PREFIX = "model.diffusion_model."
CONNECTOR_PREFIXES = (DIT_PREFIX + "video_embeddings_connector.", DIT_PREFIX + "audio_embeddings_connector.")


def read_metadata(path: str | Path, key: str = "config") -> dict:
    """Decode a required JSON metadata entry from a safetensors checkpoint."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if key not in metadata:
        raise ValueError(f"{path} is missing safetensors metadata {key!r}.")
    return json.loads(metadata[key])


def prepare_training_directory(transformer_path: str, output_dir: str) -> None:
    """Validate the BF16 dev transformer and expose it to VeOmni's weight loader.

    Only sidecars and a symlink are written; the 22B checkpoint is never copied.
    Use a new directory so existing runs cannot silently change their base model.
    """
    from ..ltx_transformer.configuration_ltx2_3_transformer import LTXVideoTransformerModelConfig
    from ..ltx_transformer.modeling_ltx2_3_transformer import LTXVideoTransformerModel

    source = Path(transformer_path).resolve(strict=True)
    metadata = read_metadata(source)
    config = dict(metadata["transformer"])
    if config.get("ff_bias", True) or not config.get("use_keyframes_abs_pos_embedding", False):
        raise ValueError("Expected an LTX-2.5 checkpoint with ff_bias=false and keyframe embeddings enabled.")
    config.update(with_audio=True, tie_word_embeddings=False)
    model_config = LTXVideoTransformerModelConfig(**config)
    with torch.device("meta"):
        model = LTXVideoTransformerModel(model_config)
    expected = model.state_dict()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        actual = {
            key.removeprefix(DIT_PREFIX): key
            for key in handle.keys()
            if key.startswith(DIT_PREFIX) and not key.startswith(CONNECTOR_PREFIXES)
        }
        if expected.keys() != actual.keys():
            raise ValueError(
                f"Transformer keys differ: missing={sorted(expected.keys() - actual.keys())}, "
                f"unexpected={sorted(actual.keys() - expected.keys())}"
            )
        for name, tensor in expected.items():
            stored = handle.get_slice(actual[name])
            if tuple(stored.get_shape()) != tuple(tensor.shape) or stored.get_dtype() not in ("BF16", "F32"):
                raise ValueError(f"Unsupported shape or dtype for {actual[name]}; use the BF16 dev checkpoint.")
        if not any(key.startswith(CONNECTOR_PREFIXES) for key in handle.keys()):
            raise ValueError("The transformer checkpoint must include its text connectors.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    model_config.save_pretrained(destination)
    # Condition loading reads the full original metadata from this same file.
    (destination / "model.safetensors").symlink_to(source)


def _load_packed_tokenizer(handle, max_length: int):
    """Build the tokenizer in memory using LTX-2's GemmaAssets conventions."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    def read_asset(filename: str, key: str) -> bytes:
        """Read a packed tokenizer asset from tensor bytes or checkpoint metadata."""
        if key in handle.keys():
            # Published packs may store byte assets as signed integers.
            return handle.get_tensor(key).to(torch.uint8).numpy().tobytes()
        metadata = handle.metadata() or {}
        if filename in metadata:
            return metadata[filename].encode()
        raise ValueError(f"Packed Gemma checkpoint is missing {filename!r}.")

    config = json.loads(read_asset("tokenizer_config.json", "hf_asset__tokenizer_config.json"))
    # These fields configure from_pretrained reconstruction, not the in-memory constructor.
    skip = {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(read_asset("tokenizer.json", "tokenizer_json")),
        model_max_length=max_length,
        **{key: value for key, value in config.items() if key not in skip},
    )


class PackedGemma4Encoder(torch.nn.Module):
    """Frozen text-only adaptation of the official LTX-2 Gemma loader.

    Config dispatch, packed assets and flat/nested text keys follow Lightricks/LTX-2:
    https://github.com/Lightricks/LTX-2/tree/9ec55f9f22798a3198d9c923856824821bc3317e/packages/ltx-core/src/ltx_core/text_encoders/gemma
    Only the language model is materialized; prompt enhancement and modality towers
    are not needed for offline caption encoding. Tokenizer assets stay in memory.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype, max_length: int = 1024):
        """Load the declared Gemma-4 text model and tokenizer as frozen encoding components."""
        super().__init__()
        from accelerate import init_empty_weights
        from transformers import AutoModel
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        config = read_metadata(checkpoint_path, "gemma_config")["text_config"]
        model_type = config.get("model_type")
        if model_type not in ("gemma4_text", "gemma4_unified_text"):
            raise ValueError(f"Expected a Gemma-4 text config, got {model_type!r}.")
        text_config = CONFIG_MAPPING[model_type].from_dict(config)
        with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            self.tokenizer = _load_packed_tokenizer(handle, max_length)
            # Leave nonpersistent rotary/embed-scale buffers on CPU, not meta.
            with init_empty_weights():
                self.model = AutoModel.from_config(text_config, dtype=torch.float32)
            keys = handle.keys()
            prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in keys) else "model."
            tower_prefixes = ("model.vision_embedder.", "model.embed_vision.", "model.embed_audio.")
            text_keys = [key for key in keys if key.startswith(prefix) and not key.startswith(tower_prefixes)]
            for key in text_keys:
                if handle.get_slice(key).get_dtype() not in ("BF16", "F32"):
                    raise ValueError(f"Unsupported dtype for {key}; use the BF16 dev checkpoint.")
            state = {key.removeprefix(prefix): handle.get_tensor(key) for key in text_keys}
            self.model.load_state_dict(state, strict=True, assign=True)
            del state
        self.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def encode(self, prompt: str, padding_side: str = "left"):
        """Return all text hidden states and the attention mask for a left-padded prompt."""
        if padding_side != "left":
            raise ValueError("LTX-2.5 prompt features require left padding.")
        tokens = self.tokenizer(prompt.strip(), truncation=True, max_length=self.max_length)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("LTX Gemma-4 tokenizer must declare its BOS token.")
        if not tokens or tokens[0] != bos:
            tokens = [bos, *tokens][: self.max_length]
        inputs = self.tokenizer.pad(
            {"input_ids": [tokens]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.model.device)
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return output.hidden_states, inputs.attention_mask


def load_packed_feature_extractor(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load frozen audio and video text projections from the packed Gemma checkpoint."""
    from ..ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2

    config = read_metadata(checkpoint_path, "gemma_config")["text_config"]
    prefix = "text_embedding_projection."
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        state = {key.removeprefix(prefix): handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    flat_dim = config["hidden_size"] * (config["num_hidden_layers"] + 1)
    with torch.device("meta"):
        extractor = FeatureExtractorV2(
            video_aggregate_embed=torch.nn.Linear(flat_dim, state["video_aggregate_embed.weight"].shape[0]),
            audio_aggregate_embed=torch.nn.Linear(flat_dim, state["audio_aggregate_embed.weight"].shape[0]),
            embedding_dim=config["hidden_size"],
        )
    extractor.load_state_dict(state, strict=True, assign=True)
    return extractor.to(device=device, dtype=dtype).eval().requires_grad_(False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare an LTX-2.5 BF16 dev transformer for VeOmni training.")
    parser.add_argument("--transformer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    prepare_training_directory(args.transformer_path, args.output_dir)
