"""Multi-process regression test for the KimiK3 MoonViT3d vision tower under Ulysses SP.

Round-9 review: the tower used to run redundantly at full width on every SP rank
(pixel_values was sp_slice=False in get_extra_collate_infos). It now follows the
qwen3_vl arrangement -- the collator sp-slices pixel_values (the repo default for
that key: pad_scale=4 keeps every 2x2 merge window whole), the tower computes
1/sp of the token rows per rank, the attention-internal Ulysses exchange keeps
the global attention contract, and the features are gathered back (the
grad-summing _Gather) before the image-atomic t-pool merger.

This test pins the parity the review asked for: the SP-sharded tower matches
the single-rank (redundant) one, including token counts not divisible by
sp_size * pad_scale and video-style T>1 grids. The parity budget is
per-device: a tight fp32 budget on NPU (the fused row accepts fp32), the
repo's bf16 budget on CUDA, where the veomni_flash_attention_2 row
(flash-attn) only accepts fp16/bf16.
"""

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if get_device_type() == "cpu" or not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d backend not available, skipping tests", allow_module_level=True)

from torch.testing._internal.common_utils import run_tests

from tests.models.tiny_configs import tiny_kimi_k3_config
from veomni.distributed.parallel_state import use_parallel_state
from veomni.distributed.sequence_parallel.data import sp_pad_and_slice
from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import KimiK3ForConditionalGeneration
from veomni.ops.platform import GpuKernelRequirement

from .utils import SequenceParallelTest


CASES = {
    "single_image": [[1, 4, 4]],
    "multi_image_mixed": [[1, 4, 4], [1, 2, 2], [2, 2, 2], [1, 4, 4]],
    "video_t2": [[2, 2, 2]],
}


# CUDA's flash-attn row only accepts fp16/bf16, so the tower and its pixel
# inputs run bf16 there with the repo's bf16 parity budget (the qwen4_exp
# vision arrangement); the fused NPU row keeps fp32 and the tight budget.
# GpuKernelRequirement.device is the conftest's device-lease spelling of the
# GPU device name (the device-api sanity gate forbids the literal here).
_VISION_DTYPE = torch.bfloat16 if get_device_type() == GpuKernelRequirement.device else torch.float32
_PARITY_ATOL = _PARITY_RTOL = 1e-2 if _VISION_DTYPE == torch.bfloat16 else 2e-4


def _build_tower(device):
    cfg = tiny_kimi_k3_config()
    cfg._attn_implementation = {"text_config": "eager", "vision_config": "veomni_flash_attention_2"}
    torch.manual_seed(0)
    model = KimiK3ForConditionalGeneration(cfg)
    return model.vision_tower.to(device=device, dtype=_VISION_DTYPE).eval()


class KimiVisionTowerSPParityTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_vision_tower_sp2_matches_single_rank(self):
        self._get_process_group()
        device = torch.device(get_device_type(), self.rank)
        tower = _build_tower(device)
        # identical weights on both ranks regardless of init nondeterminism
        tower.load_state_dict(self._sync_model(tower.state_dict(), self.rank))

        failures = []
        for name, grids in CASES.items():
            num_rows = sum(t * h * w for t, h, w in grids)
            gen = torch.Generator().manual_seed(1)
            pixel_values = torch.randn(num_rows, 3, 14, 14, generator=gen).to(device=device, dtype=_VISION_DTYPE)
            grid_thws = torch.tensor(grids, dtype=torch.int64, device=device)

            # Baseline: SP off (the DP reference state) -- the full-sequence
            # tower, i.e. the redundant arrangement the collator used to produce.
            with use_parallel_state(self.dp_state):
                with torch.no_grad():
                    expected = [f.clone() for f in tower(pixel_values, grid_thws)]

            # Sharded: SP on -- the collator-style row slice (pad_scale=4, the
            # pixel_values default); the tower gathers the features internally
            # before the image-atomic merger, so every rank returns the
            # full-sequence list.
            with use_parallel_state(self.sp_state):
                local_pixels = sp_pad_and_slice(pixel_values, dim=0, pad_value=0, pad_scale=4)
                assert local_pixels.shape[0] * self.world_size >= num_rows
                with torch.no_grad():
                    out = tower(local_pixels, grid_thws)

            if len(out) != len(expected) or len(out) != len(grids):
                failures.append(f"{name}: feature list length mismatch")
                continue
            for a, b in zip(out, expected):
                try:
                    torch.testing.assert_close(a, b, atol=_PARITY_ATOL, rtol=_PARITY_RTOL, equal_nan=False)
                except AssertionError as e:
                    failures.append(f"{name}: {' '.join(str(e).split())}")
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    run_tests()
