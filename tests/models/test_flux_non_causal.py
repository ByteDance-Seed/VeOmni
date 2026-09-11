import torch

from veomni.models.transformers.flux.modeling_flux import (
    FluxJointAttention,
    FluxSingleTransformerBlock,
)


def _rotary_embedding(seq_len, head_dim):
    # Per-position 2x2 rotation blocks with shape (1, 1, seq_len, head_dim // 2, 2, 2).
    # RoPE is a fixed rotation applied identically to q and k, so its exact values do not
    # affect the causal/non-causal distinction this test checks.
    pos = torch.arange(seq_len, dtype=torch.float32)
    half = torch.arange(head_dim // 2, dtype=torch.float32)
    ang = torch.outer(pos, half)
    cos, sin = torch.cos(ang), torch.sin(ang)
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, 1, seq_len, head_dim // 2, 2, 2)


def test_flux_joint_attention_is_non_causal():
    torch.manual_seed(0)
    dim, num_heads, head_dim = 32, 4, 8
    block = FluxJointAttention(dim, dim, num_heads, head_dim).eval()

    seq_text, seq_img = 1, 2
    text = torch.randn(1, seq_text, dim)  # text tokens come first in the sequence
    image = torch.randn(1, seq_img, dim)  # image tokens come after
    rotary = _rotary_embedding(seq_text + seq_img, head_dim)

    def run(image):
        with torch.no_grad():
            _, out_b = block(image, text, rotary)
        return out_b

    out_b0 = run(image)
    image_perturbed = image.clone()
    image_perturbed[0, -1] += 1.0  # perturb a later (image) token
    out_b1 = run(image_perturbed)

    # The leading text token must be influenced by a later image token under
    # bidirectional attention; under causal attention it would be unchanged.
    assert not torch.allclose(out_b0[0, 0], out_b1[0, 0], atol=1e-5)


def test_flux_single_transformer_block_is_non_causal():
    torch.manual_seed(0)
    dim, num_heads = 32, 4
    head_dim = dim // num_heads
    block = FluxSingleTransformerBlock(dim, num_heads).eval()

    seq = 3
    hidden_states = torch.randn(1, seq, 3 * num_heads * head_dim)
    rotary = _rotary_embedding(seq, head_dim)

    with torch.no_grad():
        out0 = block.process_attention(hidden_states, rotary)

    hidden_states_perturbed = hidden_states.clone()
    hidden_states_perturbed[0, -1] += 1.0
    with torch.no_grad():
        out1 = block.process_attention(hidden_states_perturbed, rotary)

    assert not torch.allclose(out0[0, 0], out1[0, 0], atol=1e-5)
