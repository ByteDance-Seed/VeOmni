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

"""Config for :class:`DeepseekV41Ngram` -- the n-gram hash lookup tables."""

from typing import List, Optional

from transformers import PretrainedConfig


class DeepseekV41NgramConfig(PretrainedConfig):
    """Bucket layout of DeepSeek-V4.1's two n-gram memory tables.

    Every field is copied verbatim from the monolithic checkpoint's
    ``text_config``; none is invented here. The row counts in
    ``num_embeddings`` are *derived* upstream rather than free parameters --
    each engram layer draws ``(max_ngram_size - 1) * n_heads`` distinct primes
    just above ``vocab_size`` and sums them -- so they are stored rather than
    recomputed, and :meth:`validate_bucket_layout` checks the two agree.

    ``wkv`` and the gate are deliberately NOT here: they read the backbone's
    mHC residual stream and therefore belong to ``deepseek_v41_llm`` (the
    lookup is the module boundary; see the G1 design doc).
    """

    model_type = "deepseek_v41_ngram"

    def __init__(
        self,
        layer_ids: Optional[List[int]] = None,
        num_embeddings: Optional[List[int]] = None,
        max_ngram_size: int = 4,
        n_heads: int = 8,
        head_dim: int = 256,
        vocab_size: int = 16000000,
        compressed_vocab_size: int = 99092,
        pad_token_id: int = 2,
        **kwargs,
    ):
        self.layer_ids = list(layer_ids) if layer_ids is not None else [1, 14]
        self.num_embeddings = list(num_embeddings) if num_embeddings is not None else [384006168, 384016682]
        self.max_ngram_size = max_ngram_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.compressed_vocab_size = compressed_vocab_size
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def n_hash_cols(self) -> int:
        """Hash ids emitted per position per engram layer: one per (n-gram size, head)."""
        return (self.max_ngram_size - 1) * self.n_heads

    def validate_bucket_layout(self) -> None:
        """Refuse a table whose row count cannot be a sum of the drawn primes.

        A mismatch here is the failure mode that stays silent: the hash ids are
        ``rolling % prime + offset``, so a table sized for a different prime set
        still indexes in range and still trains -- it just reads rows that
        belong to a different (n-gram size, head) bucket. The exact prime draw
        depends on ``sympy`` and is reproduced at build time; this bound check
        is what a config alone can assert.
        """
        if len(self.num_embeddings) != len(self.layer_ids):
            raise ValueError(
                f"num_embeddings has {len(self.num_embeddings)} entries but layer_ids has "
                f"{len(self.layer_ids)}: one table per engram layer is required."
            )
        floor = self.n_hash_cols * self.vocab_size
        for layer_id, rows in zip(self.layer_ids, self.num_embeddings):
            if rows < floor:
                raise ValueError(
                    f"engram layer {layer_id}: num_embeddings={rows} is below the "
                    f"{self.n_hash_cols} x vocab_size={self.vocab_size} = {floor} floor, so the "
                    "drawn primes cannot fit. A truncated table silently reads wrong rows."
                )
