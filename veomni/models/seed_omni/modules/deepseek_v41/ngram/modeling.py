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

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ......distributed.emb_parallel import AllToAllEmbedding, ShardedEmbedding
from ......distributed.parallel_state import get_parallel_state
from ......utils.device import is_nvidia_cuda_device
from .....checkpoint.convert import ConvertedCheckpointTensor
from ....utils.conversation import ConversationItem
from ...base.llm_packing import item_token_len
from ...module_modeling_base import PretrainedOmniModule
from ..dequant import dequant_fp8_blockwise
from ..quant_scheme import QUANT_BLOCK_SIZE, SCALE_DTYPE, quantized_storage_dtype
from .configuration import DeepseekV41NgramConfig


# ---------------------------------------------------------------------------
# Bucket layout & rolling hash
# ---------------------------------------------------------------------------

DEAD = -1
"""Cache sentinel for a position that takes no part in an n-gram (an image span)."""


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def find_next_prime(start: int, seen: set) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_bucket_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> list[list[list[int]]]:
    """Per ``[layer][n-gram size][head]`` bucket modulus.

    Drawn in order and never reused, which is what keeps the ranges disjoint:
    each (n-gram size, head) pair owns a contiguous slice of its layer's table.
    """
    primes: list[list[list[int]]] = []
    seen: set = set()
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def build_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """One odd multiplier per ``(layer, lookback)``, from a per-layer RNG.

    Seeding by ``layer_id`` is what makes the two engram layers hash the same
    n-gram to unrelated buckets. The bound keeps ``token_id * multiplier``
    inside int64.
    """
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_bucket_offsets(primes: list[list[list[int]]]) -> torch.Tensor:
    """Start row of each ``(n-gram size, head)`` bucket within its layer's table."""
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
    return torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))


def gather_lookback(
    compressed: torch.Tensor,
    max_ngram_size: int,
    history: torch.Tensor | None = None,
    start_pos: int = 0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack each position with the ``max_ngram_size - 1`` compressed ids before it.

    ``compressed`` is ``[B, L]`` in the compressed id space, already carrying
    :data:`DEAD` where the token is an image. ``position_ids`` (``[B, L]``,
    packed training) restarts at zero for each sample so lookback does not
    cross sample boundaries.

    Returns ``(tokens, blocked)``: ``tokens`` is ``[B, L, max_ngram_size]`` with
    blocked slots left as :data:`DEAD` for the caller to fill with its pad id,
    and ``blocked`` is ``[B, L, max_ngram_size]``.
    """
    batch, seqlen = compressed.shape
    window = compressed if history is None else torch.cat([history, compressed], dim=1)
    offset = 0 if history is None else history.shape[1]

    row_positions = torch.arange(start_pos, start_pos + seqlen, device=compressed.device).expand(batch, seqlen)
    sample_positions = row_positions if position_ids is None else position_ids.to(compressed.device)
    tokens, blocked_slots = [], []
    blocked = torch.zeros_like(row_positions, dtype=torch.bool)
    for shift in range(max_ngram_size):
        index = (row_positions - start_pos + offset - shift).clamp_min(0)
        source = window.gather(1, index)
        blocked = blocked | (sample_positions < shift) | (source == DEAD)
        tokens.append(source)
        blocked_slots.append(blocked)
    return torch.stack(tokens, dim=-1), torch.stack(blocked_slots, dim=-1)


def rolling_hash(
    tokens: torch.Tensor, multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Hash ids of the n-grams ending at each position.

    ``tokens`` ``[B, L, max_ngram_size]`` -> ``[B, L, n_layers, n_hash_cols]``.
    """
    products = tokens.unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


# ---------------------------------------------------------------------------
# Conversation carrier packing
# ---------------------------------------------------------------------------


def pack_ngram_inputs(
    conversations: list[list[ConversationItem]],
    device: torch.device,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(input_ids, token_mask, position_ids)`` over the packed row the backbone will see.

    All three are ``[1, total]``, and *total* is the full packed length **including
    image spans** -- not the text-only length the text encoder embeds.
    """
    ids: list[torch.Tensor] = []
    mask: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    build_on_cpu = is_nvidia_cuda_device(device)
    target_dev = torch.device("cpu") if build_on_cpu else device
    for sample in conversations:
        sample_len = 0
        for item in sample:
            if item.is_dummy:
                continue
            if item.type == "text":
                token_ids = item.meta.get("input_ids")
                if token_ids is None:
                    raise ValueError(
                        "deepseek_v41_ngram needs meta['input_ids'] on every text item: it hashes token ids, "
                        "and the text encoder overwrites `value` with embeddings. DeepseekV41ChatTemplate."
                        "tokenize stashes the copy -- a conversation built by another template will not have it."
                    )
                length = token_ids.numel()
                flat_ids = token_ids.reshape(-1)
                ids.append(flat_ids.cpu() if build_on_cpu and flat_ids.is_cuda else flat_ids.to(target_dev))
                mask.append(torch.ones(length, dtype=torch.bool, device=target_dev))
            else:
                length = item_token_len(item)
                ids.append(torch.full((length,), pad_token_id, dtype=torch.long, device=target_dev))
                mask.append(torch.zeros(length, dtype=torch.bool, device=target_dev))
            positions.append(torch.arange(sample_len, sample_len + length, dtype=torch.long, device=target_dev))
            sample_len += length
    cat_ids = torch.cat(ids).unsqueeze(0)
    cat_mask = torch.cat(mask).unsqueeze(0)
    cat_pos = torch.cat(positions).unsqueeze(0)
    if build_on_cpu:
        cat_ids = cat_ids.pin_memory().to(device, non_blocking=True)
        cat_mask = cat_mask.pin_memory().to(device, non_blocking=True)
        cat_pos = cat_pos.pin_memory().to(device, non_blocking=True)
    return cat_ids, cat_mask, cat_pos


# ---------------------------------------------------------------------------
# Quantized n-gram tables & row-padding checkpoint converters
# ---------------------------------------------------------------------------

_DEFAULT_ALL_TO_ALL_APPLY = AllToAllEmbedding.apply


def _quantized_all_to_all_lookup(
    group: dist.ProcessGroup | None,
    ids: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch ``ids`` once over ``group`` and return both ``(rows, scales)`` on ``ids.device``."""
    if AllToAllEmbedding.apply is not _DEFAULT_ALL_TO_ALL_APPLY:
        rows = AllToAllEmbedding.apply(group, ids, weight.view(torch.int8)).view(torch.float8_e4m3fn)
        scales = AllToAllEmbedding.apply(group, ids, weight_scale.view(torch.int8)).view(SCALE_DTYPE)
        return rows, scales

    emb_size = 1 if group is None else dist.get_world_size(group)
    emb_rank = 0 if group is None else dist.get_rank(group)
    sharded = emb_size > 1
    vocab_size_per_rank, row_dim = weight.shape
    scale_dim = weight_scale.shape[1]
    packed_dim = row_dim + scale_dim

    raw_shape = ids.shape
    input_flat = ids.reshape(-1)
    num_input_ids = input_flat.shape[0]
    vocab_size = vocab_size_per_rank * emb_size

    in_range = (input_flat >= 0) & (input_flat < vocab_size)
    id_rank = torch.where(in_range, input_flat // vocab_size_per_rank, 0)
    full_rank_index = torch.argsort(id_rank, stable=True)
    send_counts = torch.zeros(emb_size, dtype=torch.int64, device=id_rank.device)
    send_counts.scatter_add_(0, id_rank.to(torch.int64), torch.ones_like(id_rank, dtype=torch.int64))
    send_meta = torch.stack([send_counts, (~in_range).any().long().expand(emb_size)], dim=1)

    if sharded:
        recv_meta = torch.empty_like(send_meta)
        dist.all_to_all_single(recv_meta, send_meta, group=group)
    else:
        recv_meta = send_meta
    send_meta_list, recv_meta_list = torch.stack([send_meta, recv_meta]).tolist()
    if any(bad for _, bad in recv_meta_list):
        bad_ids = input_flat[~in_range]
        span = f"[{int(bad_ids.min())}, {int(bad_ids.max())}]" if bad_ids.numel() > 0 else "all in range"
        raise RuntimeError(
            f"AllToAllEmbedding: token ids must lie in [0, {vocab_size}); "
            f"some rank of the emb group is out of range (this rank's offending ids: {span})."
        )
    send_rank_count_list = [count for count, _ in send_meta_list]
    receive_rank_count_list = [count for count, _ in recv_meta_list]

    send_ids = input_flat[full_rank_index].contiguous()
    if sharded:
        recv_ids = torch.empty(sum(receive_rank_count_list), dtype=input_flat.dtype, device=input_flat.device)
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=receive_rank_count_list,
            input_split_sizes=send_rank_count_list,
            group=group,
        )
    else:
        recv_ids = send_ids

    local_indices = recv_ids - emb_rank * vocab_size_per_rank
    if weight.device != local_indices.device:
        local_indices = local_indices.to(weight.device)
    rows_local = F.embedding(local_indices, weight.view(torch.int8))
    scales_local = F.embedding(local_indices, weight_scale.view(torch.int8))
    packed_local = torch.cat([rows_local, scales_local], dim=-1)
    if packed_local.device != input_flat.device:
        packed_local = packed_local.to(input_flat.device, non_blocking=True)

    if sharded:
        packed_recv = torch.empty(num_input_ids, packed_dim, dtype=torch.int8, device=input_flat.device)
        dist.all_to_all_single(
            packed_recv,
            packed_local.contiguous(),
            output_split_sizes=send_rank_count_list,
            input_split_sizes=receive_rank_count_list,
            group=group,
        )
    else:
        packed_recv = packed_local

    output = torch.empty(num_input_ids, packed_dim, dtype=torch.int8, device=input_flat.device)
    if num_input_ids > 0:
        output[full_rank_index] = packed_recv
    rows = output[:, :row_dim].contiguous().view(torch.float8_e4m3fn).view(*raw_shape, row_dim)
    scales = output[:, row_dim:].contiguous().view(SCALE_DTYPE).view(*raw_shape, scale_dim)
    return rows, scales


def dequantize_rows(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Gathered fp8 rows plus their block scales -> bf16, for any leading shape."""
    columns = rows.shape[-1]
    dequantized = dequant_fp8_blockwise(rows.reshape(-1, columns), scales.reshape(-1, scales.shape[-1]))
    return dequantized.view(*rows.shape[:-1], columns)


class DeepseekV41NgramTable(ShardedEmbedding):
    """One n-gram memory table, optionally over the released fp8 rows."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_dtype: torch.dtype | None = None,
        device=None,
    ) -> None:
        if quant_dtype is None:
            super().__init__(num_embeddings, embedding_dim, device=device)
            return

        if embedding_dim % QUANT_BLOCK_SIZE:
            raise ValueError(
                f"A {embedding_dim}-wide row is not a whole number of {QUANT_BLOCK_SIZE}-wide scale "
                "blocks, so its scales would not line up with its columns."
            )
        super().__init__(
            num_embeddings,
            embedding_dim,
            _weight=torch.empty(num_embeddings, embedding_dim, dtype=quant_dtype, device=device),
            _freeze=True,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim // QUANT_BLOCK_SIZE,
                dtype=SCALE_DTYPE,
                device=device,
            ),
            requires_grad=False,
        )

    @property
    def is_quantized(self) -> bool:
        return getattr(self, "weight_scale", None) is not None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather ``ids``, vocab-parallel when the plan split this table over ``emb``."""
        if not self.is_quantized:
            return super().forward(ids)
        if not self._is_split():
            if self.weight.device != ids.device:
                ids_local = ids.to(self.weight.device)
                rows = F.embedding(ids_local, self.weight).to(ids.device, non_blocking=True)
                scales = F.embedding(ids_local, self.weight_scale).to(ids.device, non_blocking=True)
                return dequantize_rows(rows, scales)
            return dequantize_rows(F.embedding(ids, self.weight), F.embedding(ids, self.weight_scale))
        self._check_own_fsdp_unit()
        group = get_parallel_state().extra_parallel_group("emb")
        rows, scales = _quantized_all_to_all_lookup(group, ids, self.weight, self.weight_scale)
        return dequantize_rows(rows, scales)


_TABLE_TENSOR = re.compile(r"^tables\.(?P<index>\d+)\.(weight|weight_scale)$")


class NgramTablePadding:
    """Zero-pads the two tables' dim 0 up to the counts the module allocated."""

    def __init__(
        self,
        padded_num_embeddings: list[int],
        num_embeddings: list[int] | None = None,
        *,
        include_scale: bool = True,
    ) -> None:
        self.padded_num_embeddings = list(padded_num_embeddings)
        self.num_embeddings = list(num_embeddings) if num_embeddings is not None else list(self.padded_num_embeddings)
        self.include_scale = include_scale
        if len(self.num_embeddings) != len(self.padded_num_embeddings):
            raise ValueError(
                f"num_embeddings ({len(self.num_embeddings)}) and padded_num_embeddings "
                f"({len(self.padded_num_embeddings)}) must have the same length."
            )
        for index, (logical, padded) in enumerate(zip(self.num_embeddings, self.padded_num_embeddings)):
            if not 0 < logical <= padded:
                raise ValueError(f"tables.{index}: invalid logical/padded row counts {(logical, padded)}")

    def checkpoint_row_counts(self) -> dict[str, tuple[int, int]]:
        """Map each n-gram table parameter FQN to ``(logical_rows, padded_rows)`` for DCP and HF export cropping."""
        counts: dict[str, tuple[int, int]] = {}
        for index, (logical, padded) in enumerate(zip(self.num_embeddings, self.padded_num_embeddings)):
            if logical == padded:
                continue
            counts[f"tables.{index}.weight"] = (logical, padded)
            if self.include_scale:
                counts[f"tables.{index}.weight_scale"] = (logical, padded)
        return counts

    def can_handle(self, name: str) -> bool:
        return _TABLE_TENSOR.match(name) is not None

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = _TABLE_TENSOR.match(name)
        if match is None:
            return None

        index = int(match.group("index"))
        if index >= len(self.padded_num_embeddings):
            raise ValueError(
                f"{name} is table {index}, but this config declares only {len(self.padded_num_embeddings)} tables."
            )

        padded_rows = self.padded_num_embeddings[index]
        rows = tensor.shape[0]
        if rows > padded_rows:
            raise ValueError(
                f"{name} has {rows} rows, more than the {padded_rows} the module allocated. The config's "
                "`num_embeddings` describes a different table than this checkpoint holds, and every lookup "
                "would read a row belonging to another bucket."
            )
        if rows < padded_rows:
            padding = tensor.new_zeros((padded_rows - rows, *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        return ConvertedCheckpointTensor(name=name, tensor=tensor)

    def is_dim0_zero_pad(self, name: str) -> bool:
        return self.can_handle(name)

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        return []


class NgramTableDequant(NgramTablePadding):
    """Released fp8 rows and their scales -> the padded bf16 tables training allocates."""

    def __init__(
        self,
        padded_num_embeddings: list[int],
        num_embeddings: list[int] | None = None,
    ) -> None:
        super().__init__(padded_num_embeddings, num_embeddings=num_embeddings, include_scale=False)
        self._pending: dict[int, dict[str, torch.Tensor]] = {}

    def dim0_rowwise_companion(self, name: str) -> str | None:
        match = _TABLE_TENSOR.match(name)
        if match is None or match.group(2) != "weight":
            return None
        return f"tables.{match.group('index')}.weight_scale"

    def convert_rows(self, name: str, rows: torch.Tensor, companion_rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == 0:
            return torch.empty(rows.shape, dtype=torch.bfloat16)
        return dequantize_rows(rows, companion_rows)

    def is_dim0_zero_pad(self, name: str) -> bool:
        return self.can_handle(name) and name.endswith(".weight")

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = _TABLE_TENSOR.match(name)
        if match is None:
            return None
        index, part = int(match.group("index")), match.group(2)
        if part == "weight" and tensor.dtype != torch.float8_e4m3fn:
            return super().convert(name, tensor)
        pending = self._pending.setdefault(index, {})
        pending[part] = tensor
        if len(pending) < 2:
            return None
        del self._pending[index]
        weight = self.convert_rows(f"tables.{index}.weight", pending["weight"], pending["weight_scale"])
        return super().convert(f"tables.{index}.weight", weight)

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        if self._pending:
            unpaired = {index: sorted(parts) for index, parts in self._pending.items()}
            raise ValueError(f"n-gram tables missing their weight or weight_scale: {unpaired}")
        return []


def create_ngram_table_converter(model) -> NgramTablePadding | None:
    """The converter for a :class:`DeepseekV41Ngram`, or ``None`` if it needs none."""
    config = model.config
    if quantized_storage_dtype(config) is None:
        return NgramTableDequant(config.padded_num_embeddings, num_embeddings=config.num_embeddings)
    if list(config.padded_num_embeddings) == list(config.num_embeddings):
        return None
    return NgramTablePadding(
        config.padded_num_embeddings,
        num_embeddings=config.num_embeddings,
        include_scale=True,
    )


def padded_rows(rows: int, divisor: int) -> int:
    """``rows`` rounded up to a multiple of ``divisor``."""
    return math.ceil(rows / divisor) * divisor


# ---------------------------------------------------------------------------
# DeepseekV41Ngram OmniModule
# ---------------------------------------------------------------------------


class DeepseekV41Ngram(PretrainedOmniModule):
    """Two n-gram hash tables plus the hashing that addresses them."""

    config_class = DeepseekV41NgramConfig
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_ngram"
    main_input_name = "input_ids"
    _no_split_modules = ["DeepseekV41NgramTable"]
    supports_gradient_checkpointing = False

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: DeepseekV41Ngram):
        return create_ngram_table_converter(model)

    def __init__(self, config: DeepseekV41NgramConfig):
        super().__init__(config)
        config.validate_bucket_layout()
        self.config = config

        quant_dtype = quantized_storage_dtype(config)
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
        self.register_buffer("token_map", torch.zeros(config.vocab_size, dtype=torch.int64), persistent=True)
        self._ngram_cache = None
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)

    def freeze_model(self) -> None:
        if self.config.freeze:
            self.tables.requires_grad_(False)

    def float(self):
        if quantized_storage_dtype(self.config) is None:
            return super().float()
        return self

    @property
    def device(self) -> torch.device:
        token_map = self._buffers.get("token_map")
        if token_map is not None:
            return token_map.device
        return super().device

    @property
    def compressed_pad_id(self) -> torch.Tensor:
        return self.token_map[self.config.pad_token_id]

    def hash_ids(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        compressed = self.compress(input_ids, token_mask)
        tokens, blocked = gather_lookback(
            compressed, self.config.max_ngram_size, history, start_pos, position_ids=position_ids
        )
        tokens = torch.where(blocked, self.compressed_pad_id, tokens)
        return rolling_hash(tokens, self.multipliers, self.primes, self.offsets)

    def compress(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, DEAD))
        return compressed

    def lookup(self, hash_ids: torch.Tensor) -> list[torch.Tensor]:
        return [table(hash_ids[:, :, layer_index]) for layer_index, table in enumerate(self.tables)]

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
            real_end = max(start, min(end, valid_len))
            local_hash_ids = hash_ids[:, start:real_end]
            ngram_embeds = self.lookup(local_hash_ids)
            if real_end < end:
                ngram_embeds = [
                    torch.cat(
                        [rows, rows.new_zeros((rows.shape[0], local_len - rows.shape[1], *rows.shape[2:]))], dim=1
                    )
                    for rows in ngram_embeds
                ]
        else:
            ngram_embeds = self.lookup(hash_ids)
        return {
            "ngram_embeds": ngram_embeds,
            "ngram_compressed": compressed,
        }


__all__ = [
    "DEAD",
    "DeepseekV41Ngram",
    "DeepseekV41NgramTable",
    "NgramTableDequant",
    "NgramTablePadding",
    "build_bucket_offsets",
    "build_bucket_primes",
    "build_hash_multipliers",
    "create_ngram_table_converter",
    "dequantize_rows",
    "find_next_prime",
    "gather_lookback",
    "pack_ngram_inputs",
    "padded_rows",
    "rolling_hash",
]
