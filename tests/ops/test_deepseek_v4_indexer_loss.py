# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def reference_compressed_target(q, kv, attn_sink, topk_idxs, compressed_start, sm_scale):
    """Paper-correct teacher for DeepSeek-V3.2 eq. (4), in fp32.

    Deliberately written without reference to any LSE: the per-head denominator
    comes from one softmax over ``[selected slots ‖ sink]``, so the window and
    sink contributions are structurally present. This is the property Megatron's
    teacher violates (NVIDIA/Megatron-LM#5776) and the reason this reference must
    never be replaced by a call into the implementation.

    Args:
        q:               [B, S, H, D] any float dtype
        kv:              [B, S_kv, D] any float dtype
        attn_sink:       [H]
        topk_idxs:       [B, S, W + C] int, -1 for misses
        compressed_start: W, the index where the compressed slice begins
        sm_scale:        float

    Returns:
        [B, S, C] fp32, L1-normalised over the compressed slice.
    """
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    batch_index = torch.arange(b, device=kv.device).view(b, 1, 1)
    gathered = kv[batch_index, topk_idxs.clamp_min(0)]  # [B, S, W + C, D]
    logits = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
    compressed = probs[..., compressed_start:].sum(dim=2)  # head sum -> [B, S, C]
    return compressed / compressed.sum(-1, keepdim=True).clamp_min(1e-20)


def test_reference_target_responds_to_sink_and_window():
    """The two perturbations Megatron's compressed-only teacher is blind to.

    A per-head softmax over the compressed entries alone would make both of
    these no-ops, so this test is what distinguishes a correct teacher from a
    plausible one. Two heads with opposing preferences are required: with one
    head the outer normalisation cancels the denominator entirely.
    """
    torch.manual_seed(0)
    b, s, h, d, w, c = 1, 3, 2, 16, 2, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, w + c, d)
    topk = torch.arange(w + c).view(1, 1, -1).expand(b, s, -1).contiguous().to(torch.int32)
    sink = torch.zeros(h)
    scale = d**-0.5

    base = reference_compressed_target(q, kv, sink, topk, w, scale)

    bumped_sink = sink.clone()
    bumped_sink[0] = 5.0
    assert not torch.allclose(base, reference_compressed_target(q, kv, bumped_sink, topk, w, scale), atol=1e-4)

    bumped_kv = kv.clone()
    bumped_kv[:, 0] *= 4.0  # a window row, not a compressed row
    assert not torch.allclose(base, reference_compressed_target(q, bumped_kv, sink, topk, w, scale), atol=1e-4)


def test_reference_target_is_normalised():
    torch.manual_seed(1)
    q = torch.randn(2, 5, 4, 16)
    kv = torch.randn(2, 12, 16)
    topk = torch.randint(0, 12, (2, 5, 8), dtype=torch.int32)
    topk[0, 0, 0] = -1
    target = reference_compressed_target(q, kv, torch.zeros(4), topk, 4, 16**-0.5)
    assert torch.allclose(target.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert (target >= 0).all()
