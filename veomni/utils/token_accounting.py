# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Topology-invariant token accounting for packed training batches.

Kernel sequence metadata may describe only the local sequence-parallel shard.
Business metrics instead need global counters captured before SP/CP slicing.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import torch

from .constants import IGNORE_INDEX
from .seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


TOKEN_ACCOUNTING_KEY = "_token_accounting"
METRIC_PREFIX = "token_accounting"


@dataclass(frozen=True)
class TokenAccounting:
    """Global token counters for one pre-shard micro-batch."""

    physical_window_tokens: int
    aligned_compute_tokens: int
    source_input_tokens: int
    loss_tokens: int
    num_documents: int
    max_document_length: int
    sum_document_len_squared: int = 0

    def validate_ordering(self) -> None:
        if not (
            0
            <= self.loss_tokens
            <= self.source_input_tokens
            <= self.aligned_compute_tokens
            <= self.physical_window_tokens
        ):
            raise ValueError(
                "token accounting ordering violated: "
                f"loss={self.loss_tokens} source={self.source_input_tokens} "
                f"aligned={self.aligned_compute_tokens} physical={self.physical_window_tokens}"
            )
        if self.num_documents < 0:
            raise ValueError(f"num_documents must be non-negative, got {self.num_documents}")

    def validate_training_step(self) -> None:
        """Fail closed when a step has schema-only or empty business counters."""
        self.validate_ordering()
        if self.loss_tokens <= 0 or self.num_documents <= 0:
            raise ValueError(
                "token accounting training-step gate violated: "
                f"loss={self.loss_tokens} source={self.source_input_tokens} "
                f"aligned={self.aligned_compute_tokens} physical={self.physical_window_tokens} "
                f"num_documents={self.num_documents}"
            )

    def to_dict(self) -> Dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TokenAccounting":
        return cls(
            physical_window_tokens=int(data["physical_window_tokens"]),
            aligned_compute_tokens=int(data["aligned_compute_tokens"]),
            source_input_tokens=int(data["source_input_tokens"]),
            loss_tokens=int(data["loss_tokens"]),
            num_documents=int(data["num_documents"]),
            max_document_length=int(data["max_document_length"]),
            sum_document_len_squared=int(data.get("sum_document_len_squared", 0)),
        )


def document_lengths_from_position_ids(position_ids: torch.Tensor) -> List[int]:
    """Return packed document lengths from global, pre-shard position IDs."""
    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]
    (cu_seq_lens, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
    return [int(length) for length in (cu_seq_lens[1:] - cu_seq_lens[:-1]).tolist()]


def summarize_document_lengths(lengths: Sequence[int]) -> Dict[str, int]:
    lengths = [int(length) for length in lengths]
    return {
        "source_input_tokens": int(sum(lengths)),
        "num_documents": len(lengths),
        "max_document_length": int(max(lengths, default=0)),
        "sum_document_len_squared": int(sum(length * length for length in lengths)),
    }


def count_loss_tokens(labels: torch.Tensor) -> int:
    """Count labels that participate in the loss denominator."""
    return int((labels != IGNORE_INDEX).sum().item())


def build_token_accounting(
    *,
    physical_window_tokens: int,
    aligned_compute_tokens: int,
    source_input_tokens: int,
    loss_tokens: int,
    num_documents: int,
    max_document_length: int,
    sum_document_len_squared: int = 0,
) -> TokenAccounting:
    stats = TokenAccounting(
        physical_window_tokens=int(physical_window_tokens),
        aligned_compute_tokens=int(aligned_compute_tokens),
        source_input_tokens=int(source_input_tokens),
        loss_tokens=int(loss_tokens),
        num_documents=int(num_documents),
        max_document_length=int(max_document_length),
        sum_document_len_squared=int(sum_document_len_squared),
    )
    stats.validate_ordering()
    return stats


def attach_token_accounting(
    batch: MutableMapping[str, Any],
    stats: Union[TokenAccounting, Mapping[str, Any]],
) -> None:
    batch[TOKEN_ACCOUNTING_KEY] = (
        stats.to_dict() if isinstance(stats, TokenAccounting) else {key: int(value) for key, value in stats.items()}
    )


def pop_token_accounting(batch: MutableMapping[str, Any]) -> Optional[TokenAccounting]:
    raw = batch.pop(TOKEN_ACCOUNTING_KEY, None)
    if raw is None:
        return None
    return raw if isinstance(raw, TokenAccounting) else TokenAccounting.from_mapping(raw)


def sum_accountings(items: Iterable[TokenAccounting]) -> TokenAccounting:
    physical = aligned = source = loss = documents = max_document = sum_squared = 0
    for item in items:
        physical += item.physical_window_tokens
        aligned += item.aligned_compute_tokens
        source += item.source_input_tokens
        loss += item.loss_tokens
        documents += item.num_documents
        max_document = max(max_document, item.max_document_length)
        sum_squared += item.sum_document_len_squared
    return TokenAccounting(
        physical_window_tokens=physical,
        aligned_compute_tokens=aligned,
        source_input_tokens=source,
        loss_tokens=loss,
        num_documents=documents,
        max_document_length=max_document,
        sum_document_len_squared=sum_squared,
    )


def metrics_from_totals(
    totals: TokenAccounting,
    *,
    delta_time: float,
    prefix: str = METRIC_PREFIX,
) -> Dict[str, float]:
    """Build W&B metrics from DP-reduced token totals for one step."""
    physical = float(totals.physical_window_tokens)
    aligned = float(totals.aligned_compute_tokens)
    source = float(totals.source_input_tokens)
    loss = float(totals.loss_tokens)
    seconds = float(delta_time)

    def tokens_per_second(tokens: float) -> float:
        return tokens / seconds / 1e6 if seconds > 0 else 0.0

    return {
        f"{prefix}/physical_window_tokens": physical,
        f"{prefix}/aligned_compute_tokens": aligned,
        f"{prefix}/source_input_tokens": source,
        f"{prefix}/loss_tokens": loss,
        f"{prefix}/num_documents": float(totals.num_documents),
        f"{prefix}/max_document_length": float(totals.max_document_length),
        f"{prefix}/sum_document_len_squared": float(totals.sum_document_len_squared),
        f"{prefix}/source_fill": source / physical if physical else 0.0,
        f"{prefix}/aligned_compute_fill": aligned / physical if physical else 0.0,
        f"{prefix}/loss_density": loss / source if source else 0.0,
        f"{prefix}/loss_fill": loss / physical if physical else 0.0,
        f"{prefix}/capacity_tokens_per_second(M)": tokens_per_second(physical),
        f"{prefix}/aligned_compute_tokens_per_second(M)": tokens_per_second(aligned),
        f"{prefix}/source_tokens_per_second(M)": tokens_per_second(source),
        f"{prefix}/loss_tokens_per_second(M)": tokens_per_second(loss),
    }
