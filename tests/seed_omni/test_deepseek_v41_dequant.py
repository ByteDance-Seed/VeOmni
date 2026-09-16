# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Dequantization, against the released converter's own arithmetic.

A dequantizer that is merely self-consistent is worthless: it produces a
checkpoint that loads, runs and is wrong by a scale factor or a transposition.
So the fp8 path is checked against the expression in ``inference/convert.py``
(reproduced here, because the file is not importable from the test
environment), and the fp4 path against a per-element decode of the bit layout
that makes no reference to the implementation being tested.

The two failure modes worth naming:

* a scale applied along the wrong axis still yields finite weights of the right
  shape, and the model trains to a worse loss rather than crashing;
* the fp4 nibble order transposes adjacent pairs of input features, which
  nothing downstream can detect.
"""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.dequant import (
    FP4_VALUES,
    dequant_fp4_blockwise,
    dequant_fp8_blockwise,
    fuse_gate_up,
    split_block_shape,
)


# Verbatim from inference/convert.py, so a divergence in ours shows up as a
# mismatch rather than as both sides being changed together.
REFERENCE_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def reference_fp8_dequant(weight, scale):
    """The `wo_a` branch of inference/convert.py's second pass, verbatim."""
    out_block_size = weight.size(0) // scale.size(0)
    in_block_size = weight.size(1) // scale.size(1)
    dequantized = (
        weight.unflatten(0, (-1, out_block_size)).unflatten(-1, (-1, in_block_size)).float()
        * scale[:, None, :, None].float()
    )
    return dequantized.flatten(2, 3).flatten(0, 1).bfloat16()


class TestFp4Table:
    def test_matches_the_released_table(self):
        assert torch.equal(torch.tensor(FP4_VALUES, dtype=torch.float32).abs(), REFERENCE_FP4_TABLE.abs())

    def test_the_high_half_is_the_negated_low_half(self):
        table = torch.tensor(FP4_VALUES, dtype=torch.float32)
        assert torch.equal(table[8:], -table[:8])

    def test_six_is_the_largest_magnitude(self):
        """`6.0 * 2**6 = 384 < 448` is what bounds the reference's offset bits."""
        assert max(abs(v) for v in FP4_VALUES) == 6.0


class TestFp8Blockwise:
    @pytest.mark.parametrize("block", [32, 128])
    def test_matches_the_reference_expression(self, block):
        torch.manual_seed(0)
        weight = torch.randn(block * 3, block * 2)
        scale = torch.rand(3, 2) + 0.5
        torch.testing.assert_close(dequant_fp8_blockwise(weight, scale), reference_fp8_dequant(weight, scale))

    def test_each_block_is_scaled_by_its_own_factor(self):
        """The check a transposed scale survives: make every block constant and
        every scale distinct, so the output reads back the scale grid."""
        weight = torch.ones(64, 64)
        scale = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        out = dequant_fp8_blockwise(weight, scale, out_dtype=torch.float32)
        assert out[0, 0] == 2.0 and out[0, 32] == 3.0
        assert out[32, 0] == 5.0 and out[32, 32] == 7.0

    def test_a_transposed_scale_would_be_caught_by_that(self):
        weight = torch.ones(64, 64)
        scale = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        out = dequant_fp8_blockwise(weight, scale, out_dtype=torch.float32)
        transposed = dequant_fp8_blockwise(weight, scale.t().contiguous(), out_dtype=torch.float32)
        assert not torch.equal(out, transposed)

    def test_an_unsupported_block_is_refused(self):
        with pytest.raises(ValueError, match="unsupported fp8 block"):
            dequant_fp8_blockwise(torch.ones(64, 64), torch.ones(4, 4))

    def test_a_non_dividing_scale_is_refused(self):
        with pytest.raises(ValueError, match="not an integer number of blocks"):
            dequant_fp8_blockwise(torch.ones(64, 65), torch.ones(2, 2))

    def test_block_shape_is_read_from_the_tensors(self):
        assert split_block_shape(torch.ones(128, 256), torch.ones(4, 8)) == (32, 32)


class TestFp4Blockwise:
    def _decode_by_hand(self, packed, scale, in_block):
        """Per-element decode of the bit layout, independent of the implementation."""
        out_dim, half = packed.shape
        result = torch.zeros(out_dim, half * 2)
        raw = packed.view(torch.uint8)
        for row in range(out_dim):
            for column in range(half):
                byte = int(raw[row, column])
                result[row, 2 * column] = REFERENCE_FP4_TABLE[byte & 0x0F]
                result[row, 2 * column + 1] = REFERENCE_FP4_TABLE[(byte >> 4) & 0x0F]
        for row in range(out_dim):
            for block in range(scale.size(1)):
                lo, hi = block * in_block, (block + 1) * in_block
                result[row, lo:hi] *= float(scale[row, block])
        return result

    def test_matches_a_hand_decode_of_the_bits(self):
        torch.manual_seed(1)
        packed = torch.randint(0, 256, (8, 16), dtype=torch.uint8)
        scale = torch.rand(8, 2) + 0.5
        ours = dequant_fp4_blockwise(packed, scale, out_dtype=torch.float32)
        torch.testing.assert_close(ours, self._decode_by_hand(packed, scale, in_block=16))

    def test_low_nibble_comes_first_along_the_input_dimension(self):
        """0x71 is low=1 (0.5), high=7 (6.0). Reversed, the pair transposes and
        nothing downstream can tell."""
        packed = torch.tensor([[0x71]], dtype=torch.uint8)
        out = dequant_fp4_blockwise(packed, torch.ones(1, 1), out_dtype=torch.float32)
        assert out[0, 0] == 0.5 and out[0, 1] == 6.0

    def test_the_sign_bit_is_the_high_bit_of_the_nibble(self):
        packed = torch.tensor([[0x9F]], dtype=torch.uint8)  # low=0xF (-6.0), high=0x9 (-0.5)
        out = dequant_fp4_blockwise(packed, torch.ones(1, 1), out_dtype=torch.float32)
        assert out[0, 0] == -6.0 and out[0, 1] == -0.5

    def test_unpacking_doubles_the_input_dimension(self):
        out = dequant_fp4_blockwise(torch.zeros(4, 8, dtype=torch.uint8), torch.ones(4, 1))
        assert out.shape == (4, 16)

    def test_int8_storage_is_accepted(self):
        """The checkpoint stores the packed nibbles as int8; reading them as
        signed would make every byte above 0x7F a negative index."""
        packed = torch.tensor([[-1]], dtype=torch.int8)  # 0xFF -> low=-6.0, high=-6.0
        out = dequant_fp4_blockwise(packed, torch.ones(1, 1), out_dtype=torch.float32)
        assert out[0, 0] == -6.0 and out[0, 1] == -6.0

    def test_a_float_weight_is_refused(self):
        with pytest.raises(ValueError, match="packed fp4 must arrive as int8/uint8"):
            dequant_fp4_blockwise(torch.zeros(4, 8), torch.ones(4, 1))


class TestGateUpFusion:
    def test_gate_rows_come_first(self):
        """`_apply_gate` chunks the projection output in two and clamps the
        halves differently; reversing these puts the one-sided clamp on `up`."""
        gate, up = torch.zeros(3, 5), torch.ones(3, 5)
        fused = fuse_gate_up(gate, up)
        assert fused.shape == (6, 5)
        assert (fused[:3] == 0).all() and (fused[3:] == 1).all()

    def test_the_fusion_round_trips_through_chunk(self):
        torch.manual_seed(0)
        gate, up = torch.randn(4, 6), torch.randn(4, 6)
        back_gate, back_up = fuse_gate_up(gate, up).chunk(2, dim=0)
        torch.testing.assert_close(back_gate, gate)
        torch.testing.assert_close(back_up, up)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="must match"):
            fuse_gate_up(torch.zeros(3, 5), torch.zeros(4, 5))


class TestAgainstTheReferenceExpertPath:
    """One expert end to end: fp4 storage -> the slab the experts module reads."""

    def test_a_dequantized_expert_reproduces_the_reference_forward(self):
        torch.manual_seed(2)
        hidden, inter = 32, 16
        packed_w1 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
        packed_w3 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
        scale = torch.rand(inter, 2) * 0.1 + 0.05

        w1 = dequant_fp4_blockwise(packed_w1, scale, out_dtype=torch.float32)
        w3 = dequant_fp4_blockwise(packed_w3, scale, out_dtype=torch.float32)
        fused = fuse_gate_up(w1, w3)

        x = torch.randn(5, hidden)
        gate_ref, up_ref = torch.nn.functional.linear(x, w1), torch.nn.functional.linear(x, w3)
        gate_ours, up_ours = torch.nn.functional.linear(x, fused).chunk(2, dim=-1)
        torch.testing.assert_close(gate_ours, gate_ref)
        torch.testing.assert_close(up_ours, up_ref)
