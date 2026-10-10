"""Shared tensor contract for compact-index sparse attention."""

import torch


def _validate_qsa_inputs(query, key, value, selected_indices):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"query/key/value must be rank-4 tensors; got query={query.shape}, key={key.shape}, value={value.shape}"
        )
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match; got key={key.shape}, value={value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch and head dimensions must match; got query={query.shape}, key={key.shape}")
    if selected_indices.ndim != 3 or selected_indices.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError(
            "query batch and sequence dimensions must match selected_indices; "
            f"got query={query.shape}, selected_indices={selected_indices.shape}"
        )
    if not (query.device == key.device == value.device == selected_indices.device):
        raise ValueError("query/key/value/selected_indices must be on the same device")
    if any(size == 0 for size in query.shape) or key.shape[1] == 0:
        raise ValueError("query dimensions and key/value heads must be positive")
    if selected_indices.shape[-1] == 0:
        raise ValueError("selected_indices must contain at least one slot (use -1 for empty selections)")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(f"query heads ({query.shape[1]}) must be divisible by kv heads ({key.shape[1]})")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_indices must use int32 or int64, got {selected_indices.dtype}")
    if key.shape[2] == 0:
        raise ValueError("key/value must contain at least one token")
    if torch.any((selected_indices < -1) | (selected_indices >= key.shape[2])):
        raise ValueError(
            f"selected_indices must contain -1 padding or valid global KV token indices in [0, {key.shape[2]})"
        )
