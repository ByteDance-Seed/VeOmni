"""Local source comparison; native attention/norm, bounded equivalent RoPE grid."""

import argparse
import json
import sys
import types

import torch
import torch.nn.functional as F

from veomni.models.diffusers.seedvr2.configuration_seedvr2 import SeedVR2Config
from veomni.models.diffusers.seedvr2.core import na
from veomni.models.diffusers.seedvr2.modeling_seedvr2 import SeedVR2Model


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
args = parser.parse_args()
torch.set_num_threads(4)
sys.path.insert(0, args.source)

# The Ascend host cannot execute FlashAttention's CUDA extension. Use SDPA
# inside the unmodified upstream FlashAttentionVarlen adapter; this comparison
# proves source model/layout parity, not CUDA kernel parity.
flash = types.ModuleType("flash_attn")


def flash_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    out = []
    qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(qs) - 1):
        query = q[qs[i] : qs[i + 1]].transpose(0, 1).unsqueeze(0)
        key = k[ks[i] : ks[i + 1]].transpose(0, 1).unsqueeze(0)
        value = v[ks[i] : ks[i + 1]].transpose(0, 1).unsqueeze(0)
        out.append(F.scaled_dot_product_attention(query, key, value).squeeze(0).transpose(0, 1))
    return torch.cat(out)


flash.flash_attn_varlen_func = flash_attention
sys.modules["flash_attn"] = flash
import diffusers.models.normalization as source_normalization  # noqa: E402
from models.dit_v2.nadit import NaDiT  # noqa: E402 -- requires the explicit reference-only shim above
from models.dit_v2.rope import NaMMRotaryEmbedding3d  # noqa: E402


source_normalization.is_torch_npu_available = lambda: False


# Language frequencies are independent of the allocated grid extent.
# All tested positions lie inside this grid; only unused storage is removed.
def bounded_grid(self, *dims):
    return self.rope.get_axial_freqs(*(min(d, 64) if i == 0 else min(d, 16) for i, d in enumerate(dims)))


NaMMRotaryEmbedding3d.get_axial_freqs = bounded_grid

torch.manual_seed(1030)
config = SeedVR2Config.from_pretrained("tests/toy_config/seedvr2_toy")
config.norm = config.qk_norm = config.vid_out_norm = "rms"
config.txt_in_norm = "layer"
source = NaDiT(**config.backbone_kwargs()).train()
target = SeedVR2Model(config).train()
target.dit.load_state_dict(source.state_dict(), strict=True)
vid, vs = na.flatten([torch.randn(3, 8, 8, 9), torch.randn(1, 4, 4, 9)])
txt, ts = na.flatten([torch.randn(3, 24), torch.randn(5, 24)])
kwargs = dict(vid=vid, txt=txt, vid_shape=vs, txt_shape=ts, timestep=torch.tensor([1000.0, 800.0]))
a = source(**kwargs).vid_sample
b = target(**kwargs).vid_sample
torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
a.float().square().mean().backward()
b.float().square().mean().backward()
max_grad = 0.0
for name, p in source.named_parameters():
    q = dict(target.dit.named_parameters())[name]
    assert p.grad is not None and q.grad is not None, name
    torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
    max_grad = max(max_grad, (p.grad - q.grad).abs().max().item())
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    mixed_source = source(**kwargs).vid_sample
    mixed_target = target(**kwargs).vid_sample
torch.testing.assert_close(mixed_source, mixed_target, atol=2e-2, rtol=2e-2)
print(
    json.dumps(
        {
            "source_revision": "e4de8c24441a67e1b7df56abea10645059bb1185",
            "shape": list(a.shape),
            "max_output_error": (a - b).abs().max().item(),
            "max_gradient_error": max_grad,
            "atol": 1e-5,
            "rtol": 1e-4,
            "bf16_max_output_error": (mixed_source - mixed_target).abs().max().item(),
            "bf16_atol": 2e-2,
            "bf16_rtol": 2e-2,
            "substitutions": [
                "CUDA FlashAttention -> torch SDPA",
                "Apex fused RMSNorm -> upstream Diffusers RMSNorm",
                "RoPE grid bounded to used-position superset",
            ],
            "original_cuda_parity": False,
        },
        indent=2,
    )
)
