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

"""Unit tests for the generic multimodal metadata helpers + collator protocol.

These cover the model-agnostic contract documented in
`.agents/knowledge/multimodal_metadata.md`:

  * per-sample emission (`per_sample_metadata`),
  * `position_id_func` return merging (`merge_position_id_returns`),
  * the collator → `metadata_collate_func` hook handoff.

The batch-level cu_seqlens / window cu_seqlens derivation is per-model and
lives in the patchgen-generated helpers — exercised by the model sync-gate
test (`tests/models/test_model_forward_no_implicit_sync.py`), not here.
"""

import os

import pytest
import torch


# Bootstrap a single-rank "process group" env so collator import doesn't crash
# on `get_parallel_state()`. Mirrors the pattern used by
# tests/models/test_model_forward_no_implicit_sync.py.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12357")


from veomni.data.mm_metadata import (  # noqa: E402
    merge_position_id_returns,
    per_sample_metadata,
)


# ── per_sample_metadata ─────────────────────────────────────────────────────


def test_per_sample_metadata_image_only():
    sample = {
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long),
    }
    md = per_sample_metadata(sample)
    assert md == {"image_grid_thw_list": [[1, 4, 4], [1, 2, 2]]}


def test_per_sample_metadata_image_and_video():
    sample = {
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
    }
    md = per_sample_metadata(sample)
    assert md == {"image_grid_thw_list": [[1, 4, 4]], "video_grid_thw_list": [[2, 2, 2]]}


def test_per_sample_metadata_text_only():
    """No grid_thw keys → empty output (no-op for text samples)."""
    assert per_sample_metadata({"input_ids": torch.zeros(8)}) == {}


# ── merge_position_id_returns ───────────────────────────────────────────────


def test_merge_position_id_returns_keeps_only_position_ids():
    """Only ``position_ids`` is propagated; ``rope_deltas`` (generation-only)
    is dropped — the training forward never reads it."""
    target = {}
    merge_position_id_returns(
        target,
        {"position_ids": torch.zeros(3, 5), "rope_deltas": torch.tensor([[2]])},
    )
    assert "position_ids" in target
    assert "rope_deltas" not in target


def test_merge_position_id_returns_position_ids_only():
    """Default position_ids path (text-only datasets) should pass through cleanly."""
    target = {}
    merge_position_id_returns(target, {"position_ids": torch.arange(5)})
    assert "position_ids" in target


def test_merge_position_id_returns_missing_position_ids_raises():
    with pytest.raises(KeyError, match="position_ids"):
        merge_position_id_returns({}, {"rope_deltas": torch.zeros(1, 1)})


# ── Collator → metadata_collate_func hook handoff ───────────────────────────


def _mm_sample(seq_len, image_grid_thw_list):
    return {
        "input_ids": torch.randint(0, 1000, (seq_len,), dtype=torch.long),
        "labels": torch.randint(0, 1000, (seq_len,), dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
        "image_grid_thw": torch.tensor(image_grid_thw_list, dtype=torch.long),
        "image_grid_thw_list": image_grid_thw_list,
    }


def test_packing_collator_invokes_metadata_hook_with_flattened_lists():
    """Non-SP PackingCollator flattens per-sample grid_thw lists across the
    batch and hands the packed batch + zero sp-pad to the model hook."""
    from veomni.data.data_collator import PackingCollator

    seen = {}

    def hook(batch, sp_pad):
        seen["image_grid_thw_list"] = batch["image_grid_thw_list"]
        seen["sp_pad"] = sp_pad
        batch["multimodal_metadata"] = {"image_grid_thw_list": batch.pop("image_grid_thw_list")}

    batch = PackingCollator(metadata_collate_func=hook)(
        [_mm_sample(16, [[1, 4, 4]]), _mm_sample(8, [[1, 2, 2], [2, 2, 2]])]
    )
    # Per-sample lists flattened across the batch before the hook sees them.
    assert seen["image_grid_thw_list"] == [[1, 4, 4], [1, 2, 2], [2, 2, 2]]
    assert seen["sp_pad"] == {"pixel_values": 0, "pixel_values_videos": 0}
    assert batch["multimodal_metadata"]["image_grid_thw_list"] == [[1, 4, 4], [1, 2, 2], [2, 2, 2]]


def test_packing_collator_no_hook_is_noop():
    """Without a hook (text models / third-party pipelines) the collator runs
    cleanly and produces no ``multimodal_metadata``."""
    from veomni.data.data_collator import PackingCollator

    sample = {
        "input_ids": torch.randint(0, 1000, (16,), dtype=torch.long),
        "labels": torch.randint(0, 1000, (16,), dtype=torch.long),
        "attention_mask": torch.ones(16, dtype=torch.long),
        "position_ids": torch.arange(16, dtype=torch.long),
    }
    batch = PackingCollator()([sample, sample])
    assert "multimodal_metadata" not in batch


# ── Nested-dict device move (BaseTrainer.preforward / dit_trainer.preforward) ────


def test_preforward_recurses_into_multimodal_metadata():
    """The trainer ``preforward`` must move tensors inside ``multimodal_metadata`` to device.

    Mirrors the recursion logic used by ``BaseTrainer.preforward`` and
    ``dit_trainer.preforward``: tensors recurse, dicts recurse, everything
    else (Python ints / lists / None) passes through.
    """

    def _to_device(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {k: _to_device(vv, device) for k, vv in v.items()}
        return v

    micro_batch = {
        "input_ids": torch.zeros(8, dtype=torch.long),
        "multimodal_metadata": {
            "vit_image_cu_seqlens": torch.tensor([0, 16], dtype=torch.int32),
            "vit_image_max_seqlen": 16,  # Python int → must pass through
            "image_grid_thw_list": [[1, 4, 4]],  # Python list → must pass through
        },
    }
    out = {k: _to_device(v, "cpu") for k, v in micro_batch.items()}
    md = out["multimodal_metadata"]
    assert isinstance(md, dict)
    assert isinstance(md["vit_image_cu_seqlens"], torch.Tensor)
    # Non-tensor values survive
    assert md["vit_image_max_seqlen"] == 16
    assert md["image_grid_thw_list"] == [[1, 4, 4]]


# ── Per-model hook: picklable + behavior-equivalent ─────────────────────────


def test_qwen3_vl_metadata_hook_is_picklable_and_correct():
    """The model's ``collate_multimodal_metadata`` hook must survive being
    shipped to DataLoader workers (pickle) and reproduce the temporal-unroll
    cu_seqlens + SP-pad tail the prior collator-side derivation produced."""
    import pickle

    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        collate_multimodal_metadata,
    )

    # Module-level function → picklable for multiprocessing DataLoader workers.
    hook = pickle.loads(pickle.dumps(collate_multimodal_metadata))

    # Temporal unroll: each (t, h, w) → t cu steps of h*w patches.
    batch = {"image_grid_thw_list": [[1, 4, 4], [1, 2, 2], [2, 2, 2]]}
    hook(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    md = batch["multimodal_metadata"]
    assert md["vit_image_cu_seqlens"].tolist() == [0, 16, 20, 24, 28]
    assert md["vit_image_cu_seqlens"].dtype == torch.int32
    assert md["vit_image_max_seqlen"] == 16
    assert "image_grid_thw_list" not in batch  # popped into the metadata dict

    # SP-pad tail: padded pixel rows become one synthetic trailing "image".
    batch = {"image_grid_thw_list": [[1, 4, 4]]}
    hook(batch, {"pixel_values": 20, "pixel_values_videos": 0})
    md = batch["multimodal_metadata"]
    assert md["vit_image_cu_seqlens"].tolist() == [0, 16, 36]
    assert md["vit_image_max_seqlen"] == 20

    # Text-only batch → no multimodal_metadata key.
    batch = {"input_ids": torch.zeros(8)}
    hook(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    assert "multimodal_metadata" not in batch
