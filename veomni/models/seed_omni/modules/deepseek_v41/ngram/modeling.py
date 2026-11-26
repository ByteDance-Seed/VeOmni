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

"""DeepSeek-V4.1 n-gram memory lookup (``deepseek_v41_ngram``).

This module is the *lookup* half of what the released checkpoint calls Engram:
compress the token ids, hash each position's 2-/3-/4-grams, and gather the
matching rows of two ~384M-row tables. It depends on nothing but ``input_ids``
and the image mask, so it runs ahead of the backbone as its own graph node.

The other half -- ``wkv`` and the gate -- reads the backbone's mHC residual
stream at layers 1 and 14 and therefore lives in ``deepseek_v41_llm``. The
boundary is also the cheap one to cut: the gathered rows are ~24 KB/token,
where the residual stream would be ~102 KB/token.

The two tables hold 196.6B of the model's parameters. They are expected to stay
frozen and in FP8, row-sharded over the ``emb`` extra-parallel group
(:class:`~veomni.distributed.emb_parallel.ShardedEmbedding`), which puts ~23 GiB on each of 8 ranks.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...module_modeling_base import PretrainedOmniModule
from ..quant_scheme import quantized_storage_dtype
from .checkpoint import create_ngram_table_converter
from .configuration import DeepseekV41NgramConfig
from .hashing import (
    DEAD,
    build_bucket_offsets,
    build_bucket_primes,
    build_hash_multipliers,
    gather_lookback,
    rolling_hash,
)
from .quantized import DeepseekV41NgramTable


class DeepseekV41Ngram(PretrainedOmniModule):
    """Two n-gram hash tables plus the hashing that addresses them.

    ``token_map`` is a **persistent** buffer: it collapses token ids that
    normalize alike (``" The"`` / ``"the"`` / ``"THE"`` share a compressed id)
    and is derived from the tokenizer, which this module does not own. The
    convert step bakes it in rather than re-deriving it here, because every hash
    multiplier is a function of the compressed vocab size -- a map rebuilt from
    a different tokenizer revision would silently rehash both whole tables.

    The bucket layout (``primes`` / ``offsets`` / ``multipliers``) is
    deterministic given the config and is rebuilt at construction.
    """

    config_class = DeepseekV41NgramConfig
    # A table lookup, not a sequence activation: it emits ``ngram_embeds`` rows that
    # the backbone indexes positionally and never writes ``item.value``, so there is
    # no full-sequence tensor to hand on. The rows it does emit are already CP-local
    # (``forward_pre`` passes ``cp_rank`` / ``cp_size``), so it takes part in the
    # handoff trivially rather than being excluded from it.
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_ngram"
    main_input_name = "input_ids"
    _no_split_modules = ["DeepseekV41NgramTable"]
    supports_gradient_checkpointing = False

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # This module hashes and gathers n-gram table rows; it does not execute attention.
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: "DeepseekV41Ngram"):
        """Zero-pad the checkpoint's table rows up to the allocated counts; dequantize for bf16 tables."""
        return create_ngram_table_converter(model)

    def __init__(self, config: DeepseekV41NgramConfig):
        super().__init__(config)
        config.validate_bucket_layout()
        self.config = config

        quant_dtype = quantized_storage_dtype(config)
        # Allocated on the padded counts, not the prime sums: a table has to be
        # divisible by the `emb` group size to be row-sharded, and a sum of
        # primes is not. The extra rows are unreachable by construction -- see
        # `padded_num_embeddings` -- and the checkpoint's real rows are
        # zero-padded into them by `_NgramTablePadding` at load.
        self.tables = nn.ModuleList(
            [
                DeepseekV41NgramTable(rows, config.head_dim, quant_dtype=quant_dtype)
                for rows in config.padded_num_embeddings
            ]
        )

        primes = build_bucket_primes(
            len(config.layer_ids), config.max_ngram_size, config.n_heads, config.hash_vocab_size
        )
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", build_bucket_offsets(primes), persistent=False)
        self.register_buffer(
            "multipliers",
            build_hash_multipliers(config.layer_ids, config.max_ngram_size, config.compressed_vocab_size),
            persistent=False,
        )
        # Persistent: produced by the convert step from the tokenizer (see class docstring).
        # Sized by the *tokenizer's* vocab, not the hash base -- it is indexed by raw token id.
        self.register_buffer("token_map", torch.zeros(config.vocab_size, dtype=torch.int64), persistent=True)
        self._ngram_cache = None
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)

    def freeze_model(self) -> None:
        """When ``config.freeze``, freeze the tables -- which is the whole module.

        The runtime calls this only if it exists (``model_runtime._freeze_model_module``
        looks the attribute up), so a config field without this method would be
        accepted and ignored.
        """
        if self.config.freeze:
            self.tables.requires_grad_(False)

    def float(self):
        """Keep released fp8 tables in fp8.

        ``build_parallelize_model`` upcasts to float32 when mixed precision is
        on, so optimizer masters are fp32. These tables are frozen and 196.6 B;
        that upcast is 393 GB and is the allocation that does not fit eight H20s.
        """
        if quantized_storage_dtype(self.config) is None:
            return super().float()
        return self

    @property
    def device(self) -> torch.device:
        """Compute and collective device (follows ``token_map`` when tables are CPU-offloaded)."""
        token_map = self._buffers.get("token_map")
        if token_map is not None:
            return token_map.device
        return super().device

    @property
    def compressed_pad_id(self) -> torch.Tensor:
        """The pad token in the *compressed* id space -- what a blocked lookback slot becomes."""
        return self.token_map[self.config.pad_token_id]

    def hash_ids(
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hash ids of the n-grams ending at each position: ``[B, L, n_layers, n_hash_cols]``.

        ``token_mask`` is ``False`` for positions that take no part in an n-gram
        (image spans); those become :data:`DEAD` so the lookback stops there.
        ``history`` / ``start_pos`` are the decode path -- see
        :meth:`compress` for what a caller carries across steps. ``position_ids``
        restarts at zero per sample in a packed micro-batch.
        """
        compressed = self.compress(input_ids, token_mask)
        tokens, blocked = gather_lookback(
            compressed, self.config.max_ngram_size, history, start_pos, position_ids=position_ids
        )
        tokens = torch.where(blocked, self.compressed_pad_id, tokens)
        return rolling_hash(tokens, self.multipliers, self.primes, self.offsets)

    def compress(self, input_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Token ids -> compressed ids, with image spans marked :data:`DEAD`.

        This is also what a caller stashes as ``history`` for the next decode
        step: the compressed form is what the lookback reads, so recompressing
        the prompt every step would be pure waste.
        """
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, DEAD))
        return compressed

    def lookup(self, hash_ids: torch.Tensor) -> List[torch.Tensor]:
        """Gather each engram layer's rows: one ``[B, L, n_hash_cols, head_dim]`` per layer.

        Vocab-parallel when the ``emb`` group is on. ``AllToAllEmbedding``
        flattens the ids internally, so the ``[B, L, n_hash_cols]`` shape needs
        no reshaping here, and the result is bit-identical to a full-table
        lookup.
        """
        return [table(hash_ids[:, :, layer_index]) for layer_index, table in enumerate(self.tables)]

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        cp_size = int(kwargs.pop("cp_size", 1) or 1)
        cp_rank = int(kwargs.pop("cp_rank", 0) or 0)
        del kwargs
        compressed = self.compress(input_ids, token_mask)
        tokens, blocked = gather_lookback(
            compressed, self.config.max_ngram_size, history, start_pos, position_ids=position_ids
        )
        tokens = torch.where(blocked, self.compressed_pad_id, tokens)
        hash_ids = rolling_hash(tokens, self.multipliers, self.primes, self.offsets)
        if cp_size > 1 and history is None and start_pos == 0:
            valid_len = hash_ids.shape[1]
            pad = -valid_len % cp_size
            full_len = valid_len + pad
            local_len = full_len // cp_size
            start = cp_rank * local_len
            end = start + local_len
            real_end = min(end, valid_len)
            local_hash_ids = hash_ids[:, start:real_end]
            ngram_embeds = self.lookup(local_hash_ids)
            if real_end < end:
                tail_pad = end - real_end
                ngram_embeds = [
                    torch.cat([rows, rows.new_zeros((rows.shape[0], tail_pad, *rows.shape[2:]))], dim=1)
                    for rows in ngram_embeds
                ]
        else:
            ngram_embeds = self.lookup(hash_ids)
        return {
            "ngram_embeds": ngram_embeds,
            "ngram_compressed": compressed,
        }
