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

"""Unit tests for the multimodal metadata collator protocol.

Covers the model-agnostic contract documented in
`.agents/knowledge/multimodal_metadata.md`:

  * the collator → ``metadata_collate_func`` hook handoff,
  * the trainer ``preforward`` nested-dict device move,
  * a model hook (``collate_multimodal_metadata``) being picklable and
    deriving the temporal-unroll cu_seqlens + SP-pad tail correctly.

The per-sample data transforms emit only the HF processor's ``*_grid_thw``
CPU tensors (no Python-list sidecar); the collator packs them via
``DataCollateInfo`` and the model hook does the ``.tolist()`` once at batch
time. The batch-level cu_seqlens / window cu_seqlens derivation is per-model
and lives in the patchgen-generated helpers — its sync behaviour is gated by
``tests/models/test_model_forward_no_implicit_sync.py``.
"""

import os

import torch


# Bootstrap a single-rank "process group" env so collator import doesn't crash
# on `get_parallel_state()`. Mirrors the pattern used by
# tests/models/test_model_forward_no_implicit_sync.py.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12357")


# ── Collator → metadata_collate_func hook handoff ───────────────────────────


def _mm_sample(seq_len, image_grid_thw_list):
    """A per-sample multimodal feature dict as the data transform emits it.

    Note there is no ``image_grid_thw_list`` sidecar — only the HF processor's
    ``image_grid_thw`` CPU tensor. The collator packs it; the model hook does
    the ``.tolist()``.
    """
    return {
        "input_ids": torch.randint(0, 1000, (seq_len,), dtype=torch.long),
        "labels": torch.randint(0, 1000, (seq_len,), dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
        "image_grid_thw": torch.tensor(image_grid_thw_list, dtype=torch.long),
    }


def test_packing_collator_packs_grid_thw_and_invokes_hook():
    """Non-SP PackingCollator packs per-sample ``image_grid_thw`` tensors
    (DataCollateInfo pack_dim=0) and hands the packed batch + zero sp-pad to
    the model hook."""
    from veomni.data.data_collator import PackingCollator

    seen = {}

    def hook(batch, sp_pad):
        seen["image_grid_thw"] = batch["image_grid_thw"]
        seen["sp_pad"] = sp_pad
        batch["multimodal_metadata"] = {"ok": True}

    batch = PackingCollator(metadata_collate_func=hook)(
        [_mm_sample(16, [[1, 4, 4]]), _mm_sample(8, [[1, 2, 2], [2, 2, 2]])]
    )
    # image_grid_thw packed across the batch: (1+2, 3) = 3 image rows.
    assert seen["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 2], [2, 2, 2]]
    assert seen["sp_pad"] == {"pixel_values": 0, "pixel_values_videos": 0}
    assert batch["multimodal_metadata"] == {"ok": True}


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
    cu_seqlens + SP-pad tail from the packed ``image_grid_thw`` tensor."""
    import pickle

    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        collate_multimodal_metadata,
    )

    # Module-level function → picklable for multiprocessing DataLoader workers.
    hook = pickle.loads(pickle.dumps(collate_multimodal_metadata))

    # Temporal unroll: each (t, h, w) → t cu steps of h*w patches. The hook
    # reads the packed ``image_grid_thw`` CPU tensor and .tolist()s it.
    batch = {"image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2], [2, 2, 2]], dtype=torch.long)}
    hook(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    md = batch["multimodal_metadata"]
    assert md["vit_image_cu_seqlens"].tolist() == [0, 16, 20, 24, 28]
    assert md["vit_image_cu_seqlens"].dtype == torch.int32
    assert md["vit_image_max_seqlen"] == 16
    assert md["image_grid_thw_list"] == [[1, 4, 4], [1, 2, 2], [2, 2, 2]]
    # image_grid_thw stays in the batch — Model.forward still needs it as the
    # HF get_image_features arg; the hook only reads it.
    assert "image_grid_thw" in batch

    # SP-pad tail: padded pixel rows become one synthetic trailing "image".
    batch = {"image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)}
    hook(batch, {"pixel_values": 20, "pixel_values_videos": 0})
    md = batch["multimodal_metadata"]
    assert md["vit_image_cu_seqlens"].tolist() == [0, 16, 36]
    assert md["vit_image_max_seqlen"] == 20

    # Text-only batch → no multimodal_metadata key.
    batch = {"input_ids": torch.zeros(8)}
    hook(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    assert "multimodal_metadata" not in batch
