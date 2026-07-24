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

"""P4: checkpoint manifest (same-topology gate) + full RNG snapshot (CPU-only)."""

import random
from types import SimpleNamespace

import torch

from veomni.checkpoint.checkpoint_manifest import (
    build_checkpoint_manifest,
    read_checkpoint_manifest,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from veomni.checkpoint.rng_state import restore_rng_state, snapshot_rng_state


def _parallel_state(**overrides):
    base = dict(
        world_size=8,
        dp_size=4,
        dp_replicate_size=1,
        dp_shard_size=4,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=2,
        extra_parallel_sizes={"ep": 4},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _manifest(ps=None, model_type="hunyuan_image_3_moe", extra=None, soft=None):
    return build_checkpoint_manifest(
        model_config={"model_type": model_type, "hidden_size": 4096},
        parallel_state=ps or _parallel_state(),
        extra_hashes=extra or {},
        soft_fields=soft or {},
    )


def test_identical_manifests_are_compatible():
    a = _manifest(extra={"bucket_scheduler": "abc"}, soft={"train_seed": 0})
    b = _manifest(extra={"bucket_scheduler": "abc"}, soft={"train_seed": 0})
    hard, soft = validate_checkpoint_manifest(a, b)
    assert hard == [] and soft == []


def test_mesh_change_is_hard_fail():
    saved = _manifest(_parallel_state(ulysses_size=2, dp_shard_size=4))
    current = _manifest(_parallel_state(ulysses_size=1, dp_shard_size=8))
    hard, _ = validate_checkpoint_manifest(saved, current)
    assert any("mesh" in r for r in hard)


def test_ep_size_change_is_hard_fail():
    saved = _manifest(_parallel_state(extra_parallel_sizes={"ep": 4}))
    current = _manifest(_parallel_state(extra_parallel_sizes={"ep": 8}))
    hard, _ = validate_checkpoint_manifest(saved, current)
    assert any("mesh" in r for r in hard)


def test_model_type_and_version_are_hard_fail():
    saved = _manifest(model_type="hunyuan_image_3_moe")
    current = _manifest(model_type="qwen3_moe")
    hard, _ = validate_checkpoint_manifest(saved, current)
    assert any("model_type" in r for r in hard)

    bumped = dict(saved)
    bumped["manifest_version"] = saved["manifest_version"] + 1
    hard, _ = validate_checkpoint_manifest(bumped, saved)
    assert any("manifest_version" in r for r in hard)


def test_extra_hash_change_is_hard_fail():
    saved = _manifest(extra={"bucket_scheduler": "policy-A", "component_policy": {"vae_encoder": "frozen"}})
    current = _manifest(extra={"bucket_scheduler": "policy-B", "component_policy": {"vae_encoder": "frozen"}})
    hard, _ = validate_checkpoint_manifest(saved, current)
    assert any("bucket_scheduler" in r for r in hard)


def test_bucket_indexer_fingerprint_is_hard_gated():
    """P4: a ``bucket_indexer_fingerprint`` drift MUST hard-fail — the sampler's
    cursors would otherwise index a different bucket partition on resume,
    silently corrupting the training. The manifest's generic ``extra_hashes``
    machinery handles this; the checkpoint callback wires
    ``self.trainer.bucket_indexer_fingerprint`` in via ``_manifest_identity``.
    """
    baseline_extra = {
        "bucket_scheduler": "policy-abc",
        "bucket_indexer_fingerprint": "aa11bb22cc33dd44",
    }
    saved = _manifest(extra=baseline_extra)

    # Same fingerprint → compatible.
    current_same = _manifest(extra=dict(baseline_extra))
    assert validate_checkpoint_manifest(saved, current_same) == ([], [])

    # Different fingerprint (dataset / policy / base_size / key drift) → hard fail.
    current_drift = _manifest(extra={**baseline_extra, "bucket_indexer_fingerprint": "ffffeeeeddddcccc"})
    hard, _ = validate_checkpoint_manifest(saved, current_drift)
    assert any("bucket_indexer_fingerprint" in r for r in hard), (
        f"expected bucket_indexer_fingerprint drift to hard-fail, got hard={hard}"
    )


def test_soft_field_change_is_warn_only():
    saved = _manifest(soft={"train_seed": 0})
    current = _manifest(soft={"train_seed": 7})
    hard, soft = validate_checkpoint_manifest(saved, current)
    assert hard == []
    assert any("train_seed" in r for r in soft)


def test_write_read_round_trip(tmp_path):
    manifest = _manifest(extra={"bucket_scheduler": "abc"})
    write_checkpoint_manifest(str(tmp_path), manifest)
    loaded = read_checkpoint_manifest(str(tmp_path))
    assert validate_checkpoint_manifest(manifest, loaded) == ([], [])


def test_missing_manifest_reads_none(tmp_path):
    assert read_checkpoint_manifest(str(tmp_path)) is None


def test_rng_snapshot_restore_reproduces_stream():
    torch.manual_seed(123)
    random.seed(123)
    state = snapshot_rng_state()
    first_torch = torch.rand(4)
    first_py = [random.random() for _ in range(4)]

    # Advance the generators, then restore and confirm the stream repeats.
    torch.rand(10)
    [random.random() for _ in range(10)]
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), first_torch)
    assert [random.random() for _ in range(4)] == first_py


def test_rng_snapshot_has_all_available_streams():
    state = snapshot_rng_state()
    assert "python" in state and "torch_cpu" in state
    # numpy is a hard dep of the repo, so it should be captured.
    assert "numpy" in state
