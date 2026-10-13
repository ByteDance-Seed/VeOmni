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

"""The n-gram tables holding the released fp8 rows.

These two tables are 196.6B of the parameters, so the property that matters
most is the one about memory: the table must stay fp8 for the whole run and
only the gathered rows may ever become bf16. A lookup that dequantized in bulk
would be numerically perfect and would also need 393 GB.
"""

from types import SimpleNamespace

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp8_blockwise
from veomni.models.seed_omni.modules.deepseek_v41.ngram.checkpoint import (
    NgramTableDequant,
    NgramTablePadding,
    create_ngram_table_converter,
)
from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
from veomni.models.seed_omni.modules.deepseek_v41.ngram.quantized import DeepseekV41NgramTable
from veomni.models.seed_omni.modules.deepseek_v41.quant_scheme import (
    QUANT_BLOCK_SIZE,
    SCALE_DTYPE,
    quantized_storage_dtype,
)


RELEASED_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

ROWS = 64
DIM = 256


def quantize_rows(values: torch.Tensor):
    """``[rows, dim]`` bf16 -> the fp8 rows and per-32-column scales the release ships.

    Written out rather than taken from a helper because it is the layout under
    test: one scale per row per 32 columns, as a power of two.
    """
    blocks = values.float().unflatten(-1, (values.shape[-1] // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    packed = (blocks / scale).to(torch.float8_e4m3fn)
    return packed.flatten(-2), scale.squeeze(-1).to(SCALE_DTYPE)


class TestTheTableItself:
    def test_bf16_leaves_an_ordinary_embedding(self):
        table = DeepseekV41NgramTable(ROWS, DIM)
        assert not table.is_quantized
        assert table.weight.dtype is torch.float32
        assert "weight_scale" not in table.state_dict()

    def test_a_quantized_table_carries_one_scale_per_row_per_block(self):
        """Not a square tile: a table's rows are unrelated memories."""
        table = DeepseekV41NgramTable(ROWS, DIM, quant_dtype=torch.float8_e4m3fn)
        assert table.weight.dtype is torch.float8_e4m3fn
        assert tuple(table.weight_scale.shape) == (ROWS, DIM // QUANT_BLOCK_SIZE)
        assert table.weight_scale.dtype is SCALE_DTYPE

    def test_a_row_that_is_not_whole_blocks_is_refused(self):
        with pytest.raises(ValueError, match="scale blocks"):
            DeepseekV41NgramTable(ROWS, DIM + 1, quant_dtype=torch.float8_e4m3fn)

    def test_the_table_is_not_trained(self):
        """196.6B frozen parameters; an optimizer stepping them would need moments for each."""
        table = DeepseekV41NgramTable(ROWS, DIM, quant_dtype=torch.float8_e4m3fn)
        assert not table.weight.requires_grad
        assert not table.weight_scale.requires_grad


class TestLookup:
    def test_a_lookup_returns_dequantized_rows(self):
        torch.manual_seed(0)
        table = DeepseekV41NgramTable(ROWS, DIM, quant_dtype=torch.float8_e4m3fn)
        values = torch.randn(ROWS, DIM, dtype=torch.bfloat16)
        packed, scale = quantize_rows(values)
        with torch.no_grad():
            table.weight.copy_(packed)
            table.weight_scale.copy_(scale)

        ids = torch.tensor([[3, 17], [0, ROWS - 1]])
        gathered = table(ids)

        assert gathered.dtype is torch.bfloat16
        assert tuple(gathered.shape) == (2, 2, DIM)
        expected = dequant_fp8_blockwise(packed[ids.flatten()], scale[ids.flatten()])
        torch.testing.assert_close(gathered.flatten(0, 1), expected, rtol=0, atol=0)

    def test_the_table_stays_fp8_after_a_lookup(self):
        """The whole point: only the gathered rows are ever widened."""
        table = DeepseekV41NgramTable(ROWS, DIM, quant_dtype=torch.float8_e4m3fn)
        table(torch.tensor([1, 2, 3]))
        assert table.weight.dtype is torch.float8_e4m3fn

    def test_a_lookup_recovers_the_values_it_was_given(self):
        """End to end, so a scale applied to the wrong columns would show."""
        torch.manual_seed(0)
        table = DeepseekV41NgramTable(ROWS, DIM, quant_dtype=torch.float8_e4m3fn)
        # Each 32-column block on a different scale, which is what per-block
        # scales are for and what a mis-indexed scale would visibly break.
        values = torch.randn(ROWS, DIM, dtype=torch.bfloat16)
        for block in range(DIM // QUANT_BLOCK_SIZE):
            values[:, block * QUANT_BLOCK_SIZE : (block + 1) * QUANT_BLOCK_SIZE] *= 2.0**block
        packed, scale = quantize_rows(values)
        with torch.no_grad():
            table.weight.copy_(packed)
            table.weight_scale.copy_(scale)

        gathered = table(torch.arange(ROWS)).float()
        error = (gathered - values.float()).abs() / values.float().abs().clamp(min=1e-6)
        # fp8_e4m3 keeps 3 mantissa bits, so a value lands within about 2^-4 of
        # itself once its block scale is removed.
        assert error.max() < 0.07


class TestMixedPrecisionDoesNotUpcastTheTables:
    def test_float_is_a_no_op_when_the_tables_are_fp8(self):
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram

        config = DeepseekV41NgramConfig(weight_quantization=RELEASED_QUANTIZATION)
        with torch.device("meta"):
            model = DeepseekV41Ngram(config)
        assert model.tables[0].weight.dtype is torch.float8_e4m3fn
        model.float()
        assert model.tables[0].weight.dtype is torch.float8_e4m3fn


class TestTheConfigSwitch:
    def test_a_bf16_ngram_config_asks_for_nothing(self):
        assert quantized_storage_dtype(DeepseekV41NgramConfig()) is None

    def test_the_native_ngram_config_asks_for_fp8(self):
        config = DeepseekV41NgramConfig(weight_quantization=RELEASED_QUANTIZATION)
        assert quantized_storage_dtype(config) is torch.float8_e4m3fn


class TestRowPadding:
    """The tables are allocated a little larger than the checkpoint holds.

    A row count that is a sum of primes divides by essentially nothing, and a
    row-sharded table needs one its group size divides. The padding is applied
    as the tensors stream in, so the artifact is untouched and the 91.6 GiB
    tables are never held whole.
    """

    def test_the_released_counts_do_not_divide_by_eight(self):
        """The premise. Without this the padding would be dead code."""
        config = DeepseekV41NgramConfig()
        assert config.num_embeddings == [384006168, 384016682]
        assert config.num_embeddings[1] % 8 != 0

    def test_padding_reaches_every_plausible_group_size(self):
        config = DeepseekV41NgramConfig()
        for rows in config.padded_num_embeddings:
            for group_size in (2, 4, 8, 16, 32, 64, 128, 256):
                assert rows % group_size == 0

    def test_padding_is_cheap(self):
        """446 rows between the two tables; the point is that it is not a tradeoff."""
        config = DeepseekV41NgramConfig()
        added = sum(p - r for r, p in zip(config.num_embeddings, config.padded_num_embeddings))
        assert added * config.head_dim < 200 * 1024

    def test_the_semantic_size_is_left_alone(self):
        """`num_embeddings` is the prime sum, which the hashing and the layout check use."""
        config = DeepseekV41NgramConfig()
        assert config.num_embeddings == [384006168, 384016682]
        config.validate_bucket_layout()

    def test_a_table_is_padded_up_and_its_rows_kept(self):
        config = DeepseekV41NgramConfig(num_embeddings=[384006168, 384016682])
        converter = NgramTablePadding(config.padded_num_embeddings)
        # Stands in for the real tensor; padding is a dim-0 operation and does
        # not care how many rows precede it.
        converted = converter.convert("tables.0.weight", torch.ones(64, 4))

        assert converted.tensor.shape[0] == config.padded_num_embeddings[0]
        assert (converted.tensor[:64] == 1).all()
        assert (converted.tensor[64:] == 0).all()

    def test_only_the_tables_are_claimed(self):
        """`token_map` and the hash constants are replicated, not row-sharded."""
        converter = NgramTablePadding([100, 100])
        assert converter.can_handle("tables.0.weight")
        assert converter.can_handle("tables.1.weight_scale")
        assert not converter.can_handle("token_map")
        assert not converter.can_handle("primes")

    def test_the_padding_is_declared_streamable(self):
        """Lets the loader read only a rank's slice instead of the whole table."""
        converter = NgramTablePadding([100, 100])
        assert converter.is_dim0_zero_pad("tables.0.weight")
        assert converter.finalize() == []

    def test_a_table_larger_than_allocated_is_refused(self):
        """A config describing a different table would otherwise read wrong buckets."""
        converter = NgramTablePadding([64, 64])
        with pytest.raises(ValueError, match="more than"):
            converter.convert("tables.0.weight", torch.ones(128, 4))

    def test_no_converter_when_nothing_needs_padding(self):
        config = DeepseekV41NgramConfig(
            num_embeddings=[384006400, 384016896], weight_quantization=RELEASED_QUANTIZATION
        )
        model = SimpleNamespace(config=config)
        assert create_ngram_table_converter(model) is None


class TestTrainableTablesDequantizeAtLoad:
    """A trainable module allocates bf16 tables, so the released fp8 rows are widened at load.

    Streaming is not optional here: every rank reading the whole table and
    dequantizing it would need 393 GB, so a rank must read only its rows of the
    fp8 table and of the scales, and dequantize just those.
    """

    def _released_rows(self, rows: int = 10):
        torch.manual_seed(0)
        return quantize_rows(torch.randn(rows, DIM, dtype=torch.bfloat16))

    def test_bf16_tables_get_the_dequantizing_converter(self):
        model = SimpleNamespace(config=DeepseekV41NgramConfig())
        assert isinstance(create_ngram_table_converter(model), NgramTableDequant)

    def test_the_scale_is_read_with_its_rows_not_on_its_own(self):
        converter = NgramTableDequant([16, 16])
        assert converter.dim0_rowwise_companion("tables.1.weight") == "tables.1.weight_scale"
        assert converter.dim0_rowwise_companion("tables.1.weight_scale") is None
        assert converter.should_skip_without_loading("tables.1.weight_scale")
        assert not converter.should_skip_without_loading("tables.1.weight")

    def test_whole_tensors_pair_up_in_either_order(self):
        packed, scale = self._released_rows()
        expected = dequant_fp8_blockwise(packed, scale)
        for order in (("weight", "weight_scale"), ("weight_scale", "weight")):
            converter = NgramTableDequant([16, 16])
            tensors = {"weight": packed, "weight_scale": scale}
            assert converter.convert(f"tables.0.{order[0]}", tensors[order[0]]) is None
            converted = converter.convert(f"tables.0.{order[1]}", tensors[order[1]])
            assert converted.name == "tables.0.weight"
            assert converted.tensor.dtype is torch.bfloat16
            torch.testing.assert_close(converted.tensor[:10], expected, rtol=0, atol=0)
            assert (converted.tensor[10:] == 0).all()
            assert converter.finalize() == []

    def test_a_table_without_its_scale_is_refused(self):
        packed, _ = self._released_rows()
        converter = NgramTableDequant([16, 16])
        converter.convert("tables.0.weight", packed)
        with pytest.raises(ValueError, match="missing"):
            converter.finalize()

    def test_a_bf16_checkpoint_table_is_only_padded(self):
        converter = NgramTableDequant([16, 16])
        converted = converter.convert("tables.0.weight", torch.ones(10, DIM, dtype=torch.bfloat16))
        assert converted.tensor.shape[0] == 16
        assert converter.is_dim0_zero_pad("tables.0.weight")

    def test_each_rank_streams_and_dequantizes_only_its_rows(self, tmp_path):
        """10 real rows padded to 16 over 2 ranks: rank 1 straddles the padding."""
        from safetensors import safe_open
        from safetensors.torch import save_file

        from veomni.models.module_utils import _companion_row_transform, _read_ep_dim0_slice

        packed, scale = self._released_rows()
        path = str(tmp_path / "ngram.safetensors")
        save_file({"tables.0.weight": packed, "tables.0.weight_scale": scale}, path)
        converter = NgramTableDequant([16, 16])
        row_transform = _companion_row_transform(converter, "tables.0.weight", path, "tables.0.weight_scale")

        shards = []
        with safe_open(path, framework="pt", device="cpu") as f:
            for rank in range(2):
                shards.append(
                    _read_ep_dim0_slice(
                        f.get_slice("tables.0.weight"),
                        name="tables.0.weight",
                        target0=8,
                        para_rank=rank,
                        expected_full0=16,
                        zero_pad=True,
                        row_transform=row_transform,
                    )
                )

        full = torch.cat(shards)
        assert full.dtype is torch.bfloat16
        torch.testing.assert_close(full[:10], dequant_fp8_blockwise(packed, scale), rtol=0, atol=0)
        assert (full[10:] == 0).all()
