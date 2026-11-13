"""Janus VQ codebook lookup must match by cosine, not raw euclidean, distance.

Janus trains its codebook with ``codebook_l2_norm=True``, so the reference
tokenizer l2-normalises both the encoder latent and the codebook rows before the
nearest-neighbour search. HF's ``JanusVQVAEVectorQuantizer`` searches the raw
space (it only normalises in ``get_codebook_entry``, on the decode side), which
disagrees with the reference on the majority of real images and drives the
frozen-model image cross-entropy to near chance.
"""

import torch
import torch.nn.functional as F
from transformers.models.janus.configuration_janus import JanusVQVAEConfig

from veomni.models.seed_omni.modules.janus.vqvae.modeling import JanusVqvaeVectorQuantizer


def _quantizer(embedding_weight: torch.Tensor) -> JanusVqvaeVectorQuantizer:
    num_embeddings, embed_dim = embedding_weight.shape
    config = JanusVQVAEConfig(num_embeddings=num_embeddings, embed_dim=embed_dim, num_patches=1)
    quantizer = JanusVqvaeVectorQuantizer(config)
    with torch.no_grad():
        quantizer.embedding.weight.copy_(embedding_weight)
    return quantizer


def test_lookup_picks_cosine_nearest_not_euclidean_nearest():
    # Row 0 points the same way as the query but is far from it in raw space;
    # row 1 sits close in raw space but points elsewhere. Cosine picks row 0.
    embedding = torch.tensor([[10.0, 0.0], [0.9, 0.9]])
    latent = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)

    _, _, indices = _quantizer(embedding)(latent)

    assert indices.tolist() == [0]
    raw_nearest = (embedding - latent.view(1, 2)).pow(2).sum(dim=1).argmin().item()
    assert raw_nearest == 1, "fixture no longer distinguishes the two metrics"


def test_quantized_output_is_the_normalized_codebook_row():
    embedding = torch.tensor([[3.0, 4.0], [-1.0, 0.0]])
    latent = torch.tensor([0.6, 0.8]).view(1, 2, 1, 1)

    quantized, _, indices = _quantizer(embedding)(latent)

    expected = F.normalize(embedding[indices], p=2, dim=-1).view(1, 2, 1, 1)
    torch.testing.assert_close(quantized, expected)


def test_gradient_flows_straight_through_the_lookup():
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    latent = torch.tensor([0.8, 0.6]).view(1, 2, 1, 1).requires_grad_()

    quantized, _, _ = _quantizer(embedding)(latent)
    quantized.sum().backward()

    assert latent.grad is not None and latent.grad.abs().sum() > 0
