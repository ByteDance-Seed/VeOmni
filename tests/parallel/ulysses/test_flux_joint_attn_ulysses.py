"""Multi-process regression test for Flux joint attention under Ulysses SP.

`FluxJointAttention` all-to-alls Q/K/V of the image and text streams, attends over the
concatenated `[text_full, image_full]` sequence, then used one reverse all-to-all on that
concatenation. That hands each rank a contiguous slice of `[text_full, image_full]`, which the
code then split at the local text length -- so every rank got a mix of text and image tokens
from the wrong positions, even with divisible lengths and no mask.

This drives `FluxJointAttention.forward` directly on local shards with `ulysses_size=world_size`
and checks the gathered image and text outputs against a full-sequence run of the same module.
"""

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d NCCL not available, skipping tests", allow_module_level=True)

from torch.testing._internal.common_utils import run_tests

from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.models.transformers.flux.modeling_flux import FluxJointAttention

from .utils import SequenceParallelTest


def _rotary_embedding(seq_len, head_dim, device):
    pos = torch.arange(seq_len, dtype=torch.float32)
    ang = torch.outer(pos, torch.arange(head_dim // 2, dtype=torch.float32) * 0.1)
    cos, sin = torch.cos(ang), torch.sin(ang)
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, 1, seq_len, head_dim // 2, 2, 2).to(device)


def _gather(x, group, world_size):
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    c10d.all_gather(chunks, x.contiguous(), group=group)
    return torch.cat(chunks, dim=1)


class FluxJointAttentionUlyssesTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_matches_full_sequence_reference(self):
        group = self._get_process_group()
        device = torch.device(get_device_type(), self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        dim, num_heads, head_dim, txt_len, img_len = 64, 4, 16, 4, 16
        assert txt_len % self.world_size == 0 and img_len % self.world_size == 0  # isolate from padding

        torch.manual_seed(0)
        attn = FluxJointAttention(dim, dim, num_heads, head_dim).to(device)
        text = torch.randn(1, txt_len, dim, device=device)
        image = torch.randn(1, img_len, dim, device=device)
        rotary = _rotary_embedding(txt_len + img_len, head_dim, device)
        # Entity-style mask: text token 0 may not attend to the second half of the image.
        mask = torch.zeros(1, 1, txt_len + img_len, txt_len + img_len, device=device)
        mask[..., 0, txt_len + img_len // 2 :] = float("-inf")
        mask[..., txt_len + img_len // 2 :, 0] = float("-inf")

        try:
            _init_parallel_state(dp_size=self.world_size, ulysses_size=1, device_type=get_device_type(), name=None)
            with torch.no_grad():
                refs = {name: attn(image, text, rotary, m) for name, m in [("no_mask", None), ("mask", mask)]}
            clear_parallel_state()

            _init_parallel_state(dp_size=1, ulysses_size=self.world_size, device_type=get_device_type(), name=None)
            t_chunk, i_chunk = txt_len // self.world_size, img_len // self.world_size
            text_local = text[:, self.rank * t_chunk : (self.rank + 1) * t_chunk]
            image_local = image[:, self.rank * i_chunk : (self.rank + 1) * i_chunk]
            for name, m in [("no_mask", None), ("mask", mask)]:
                with torch.no_grad():
                    out_image, out_text = attn(image_local, text_local, rotary, m)
                ref_image, ref_text = refs[name]
                torch.testing.assert_close(
                    _gather(out_image, group, self.world_size), ref_image, atol=1e-5, rtol=1e-4, msg=f"{name}: image"
                )
                torch.testing.assert_close(
                    _gather(out_text, group, self.world_size), ref_text, atol=1e-5, rtol=1e-4, msg=f"{name}: text"
                )
        finally:
            clear_parallel_state()


if __name__ == "__main__":
    run_tests()
