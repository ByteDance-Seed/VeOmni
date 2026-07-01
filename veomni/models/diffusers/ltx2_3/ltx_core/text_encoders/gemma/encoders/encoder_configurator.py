import torch
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config."""
    gemma_text_config = GEMMA3_CONFIG_FOR_LTX.text_config
    embedding_dim = gemma_text_config.hidden_size
    num_layers = gemma_text_config.num_hidden_layers + 1
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config — missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


def build_embeddings_processor(config: dict, with_feature_extractor: bool = False) -> EmbeddingsProcessor:
    """Build EmbeddingsProcessor from model config.

    Args:
        config: Model config dict (must contain 'transformer' key with connector params).
        with_feature_extractor: If True, also creates the feature extractor (for precompute).
    """
    transformer_config = config.get("transformer", {})

    video_connector = Embeddings1DConnectorConfigurator.from_config(config)
    audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)
    feature_extractor = _create_feature_extractor(transformer_config) if with_feature_extractor else None

    return EmbeddingsProcessor(
        video_connector=video_connector,
        audio_connector=audio_connector,
        feature_extractor=feature_extractor,
    )


EMBEDDINGS_PROCESSOR_KEY_REMAP = {
    "text_embedding_projection.aggregate_embed.": "feature_extractor.aggregate_embed.",
    "text_embedding_projection.video_aggregate_embed.": "feature_extractor.video_aggregate_embed.",
    "text_embedding_projection.audio_aggregate_embed.": "feature_extractor.audio_aggregate_embed.",
    "model.diffusion_model.video_embeddings_connector.": "video_connector.",
    "model.diffusion_model.audio_embeddings_connector.": "audio_connector.",
}
