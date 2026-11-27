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

"""A bounded, replayable cyclic packer for deterministic map-style SFT data."""

import hashlib
import json
import pickle
import random
import re
import struct
from collections import deque
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data._utils.pin_memory import pin_memory

from ..utils.device import get_device_type
from ..utils.dist_utils import raise_if_any_rank_failed


_STATE_KIND = "veomni_cyclic_packing_v2"


class _StatelessMainCollator:
    """Expose only the native, stateless collator path to cyclic replay."""

    supports_stateless_cyclic_replay = True

    def __init__(self, collator):
        from . import data_collator

        if type(collator) is not data_collator.MainCollator:
            raise ValueError("cyclic_packed requires the native MainCollator")
        hook = collator.metadata_collate_func
        pure_hook = getattr(data_collator, "SerializablePureCollatorHook", None)
        if hook is not None and (pure_hook is None or type(hook) is not pure_hook):
            raise ValueError("cyclic_packed requires a certified stateless multimodal metadata hook")
        if hook is not None and not getattr(hook, "checkpoint_collator_identity", ""):
            raise ValueError("cyclic_packed requires a stable metadata hook identity")
        self._collator = collator
        self._configuration_identity()  # Validate the pipeline before changing its cyclic-only option.
        self._collator.preforward_pipeline[1].clone_fallback_tensors = True
        self.checkpoint_collator_identity = self._configuration_identity()

    def _configuration_identity(self):
        """Identify the supported native collator pipeline and its options."""
        from . import data_collator

        pipeline = self._collator.preforward_pipeline
        expected = [data_collator.PrecomputePositionIDsCollator, data_collator.PackingCollator]
        if len(pipeline) == 3:
            expected.append(data_collator.SequenceParallelCollator)
        if len(pipeline) != len(expected) or any(type(step) is not cls for step, cls in zip(pipeline, expected)):
            raise ValueError("cyclic_packed requires the standard MainCollator pipeline")
        for step in pipeline[1:]:
            if step.collate_infos is not self._collator.collate_infos:
                raise ValueError("cyclic_packed requires shared, unchanged collator metadata")
            if step.metadata_collate_func is not self._collator.metadata_collate_func:
                raise ValueError("cyclic_packed requires the certified metadata hook throughout collation")
        hook = self._collator.metadata_collate_func
        return _signature(
            {
                "infos": {key: asdict(value) for key, value in self._collator.collate_infos.items()},
                "pad_to_length": pipeline[1].pad_to_length,
                "clone_fallback_tensors": pipeline[1].clone_fallback_tensors,
                "seq_classification": pipeline[1].seq_classification,
                "sp_enabled": pipeline[1].sp_enabled,
                "sp_size": getattr(pipeline[-1], "sp_size", 1),
                "sp_seq_classification": getattr(pipeline[-1], "seq_classification", None),
                "hook": getattr(hook, "checkpoint_collator_identity", None),
            }
        )

    def __call__(self, features):
        if "loss_weights" not in self._collator.collate_infos and any(
            feature.get("loss_weights") is not None for feature in features
        ):
            raise ValueError("cyclic packing loss_weights require the weighted-loss collator")
        return self._collator(features)

    def validate_checkpoint_identity(self):
        """Reject a collator configuration that changed since construction."""
        if self._configuration_identity() != self.checkpoint_collator_identity:
            raise ValueError("cyclic packing collator configuration changed")


def _hash_part(digest, tag: bytes, value: bytes) -> None:
    """Append a tagged, length-delimited value to a hash."""
    digest.update(tag)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _hash_value(digest, value: Any) -> None:
    """Hash the complete CPU feature, including metadata and tensor bytes."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
            raise ValueError("cyclic packing requires ordinary CPU feature tensors")
        _hash_part(digest, b"T", str(value.dtype).encode())
        _hash_part(digest, b"S", json.dumps(list(value.shape), separators=(",", ":")).encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(memoryview(raw))
    elif value is None:
        _hash_part(digest, b"N", b"")
    elif type(value) is bool:
        _hash_part(digest, b"B", b"1" if value else b"0")
    elif type(value) is int:
        _hash_part(digest, b"I", str(value).encode())
    elif type(value) is float:
        _hash_part(digest, b"F", struct.pack(">d", value))
    elif isinstance(value, str):
        _hash_part(digest, b"U", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _hash_part(digest, b"Y", value)
    elif isinstance(value, (list, tuple)):
        _hash_part(digest, b"L" if isinstance(value, list) else b"P", str(len(value)).encode())
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
        _hash_part(digest, b"D", str(len(value)).encode())
        for key in sorted(value):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    else:
        raise TypeError(f"unsupported feature type for exact cyclic packing: {type(value).__name__}")


def _feature_hash(feature: dict) -> str:
    """Return a content hash for a feature or collated output."""
    digest = hashlib.sha256()
    _hash_value(digest, feature)
    return digest.hexdigest()


def _feature_logical_bytes(feature: Any) -> int:
    """Count logical tensor bytes independently of rank-local storage layout."""

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
                raise ValueError("cyclic packing requires ordinary CPU feature tensors")
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        if isinstance(value, (str, bytes)):
            return len(value)
        return 0

    return visit(feature)


def _signature(value: Any) -> str:
    """Hash a JSON-serializable configuration using canonical ordering."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rng_snapshot():
    """Detect changes to the process-global CPU random streams."""
    return random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state()


def _assert_rng_unchanged(before):
    """Reject transforms that consume Python, NumPy, or Torch CPU RNG."""
    if before[0] != random.getstate() or before[1] != pickle.dumps(np.random.get_state()):
        raise ValueError("cyclic packing does not support transforms that consume global RNG")
    if not torch.equal(before[2], torch.get_rng_state()):
        raise ValueError("cyclic packing does not support transforms that consume global RNG")


def _agree(error: Exception | None, signature: Any, group=None) -> None:
    """All participants report errors before comparing a small replay signature."""
    raise_if_any_rank_failed(error, "preparing cyclic packed data", group=group)
    if error is not None:
        raise error
    if not dist.is_initialized():
        return
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, signature, group=group)
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError("cyclic packed inputs differ across sequence-parallel ranks")


class CyclicPackedLoader:
    """Pack a finite map dataset in deterministic cycles with exact replay.

    A step ends at a dataset boundary, including a short tail step. Data is
    committed only after the optimizer and scheduler succeed. Checkpoints hold
    hashes and counters, not decoded image tensors: restore rereads the current
    cycle and verifies both emitted groups and the complete lookahead frontier.
    """

    _veomni_requires_optimizer_commit = True

    def __init__(
        self,
        dataset,
        collate_fn: Callable,
        *,
        max_seq_len: int,
        micro_batches_per_step: int,
        buffer_size: int,
        max_buffer_bytes: int,
        input_contract: str,
        source_fingerprint: str,
        deterministic: bool,
        train_steps: int,
        train_epochs: int,
        sp_size: int,
        step_getter: Callable[[], int],
        require_resume: bool = False,
        pin: bool = False,
        group=None,
    ):
        numbers = (
            max_seq_len,
            micro_batches_per_step,
            buffer_size,
            max_buffer_bytes,
            train_steps,
            train_epochs,
            sp_size,
        )
        if any(type(value) is not int or value <= 0 for value in numbers):
            raise ValueError("cyclic packing geometry and limits must be positive integers")
        if not isinstance(input_contract, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", input_contract) is None:
            raise ValueError("cyclic packing requires a sha256 input contract")
        if deterministic is not True:
            raise ValueError("cyclic packing requires a deterministic-input attestation")
        if not isinstance(source_fingerprint, str) or not source_fingerprint:
            raise ValueError("cyclic packing requires a source dataset fingerprint")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size != sp_size or (dist.is_initialized() and dist.get_world_size(group) != sp_size):
            raise ValueError("cyclic packing currently requires all ranks in one sequence-parallel group")
        if not callable(collate_fn) or not callable(step_getter):
            raise TypeError("cyclic packing requires a collator and trainer step getter")
        if getattr(collate_fn, "supports_stateless_cyclic_replay", False) is not True:
            raise ValueError("cyclic packing requires a certified stateless collator")
        collator_identity = getattr(collate_fn, "checkpoint_collator_identity", None)
        if not isinstance(collator_identity, str) or not collator_identity:
            raise ValueError("cyclic packing requires a stable collator identity")
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__") or len(dataset) <= 0:
            raise ValueError("cyclic packing requires a nonempty map-style dataset")
        self.dataset = dataset
        self.collate_fn = collate_fn
        self._step_getter = step_getter
        self._require_resume = require_resume
        self._restored = False
        # Match DataLoader's behavior on a CPU-only host: pinning is enabled
        # only when an accelerator is available.
        self._pin = bool(pin and torch.accelerator.is_available())
        self._group = group
        self._group_rank = dist.get_rank(group) if dist.is_initialized() else 0
        self._max_seq_len = max_seq_len
        self._micro_batches_per_step = micro_batches_per_step
        self._buffer_size = buffer_size
        self._max_buffer_bytes = max_buffer_bytes
        self._source_len = len(dataset)
        self._train_steps = train_steps
        self._train_epochs = train_epochs
        self._total_steps = train_steps * train_epochs
        self._identity = _signature(
            {
                "kind": _STATE_KIND,
                "input_contract": input_contract,
                "source_fingerprint": source_fingerprint,
                "collator_identity": collator_identity,
                "source_len": self._source_len,
                "max_seq_len": max_seq_len,
                "micro_batches_per_step": micro_batches_per_step,
                "buffer_size": buffer_size,
                "max_buffer_bytes": max_buffer_bytes,
                "train_steps": train_steps,
                "train_epochs": train_epochs,
                "sp_size": sp_size,
            }
        )
        self._pending_step = False
        self._poisoned = False
        self._committed_step = 0
        self._cycle = 0
        self._reset_cycle()

    def _reset_cycle(self):
        """Reset the replay frontier and hashes for a new data cycle."""
        self._source_index = 0
        self._pending = deque()
        self._lookahead = None
        self._buffer_bytes = 0
        self._cycle_groups = 0
        self._cycle_items = 0
        self._layout_hash = hashlib.sha256()
        self._output_hash = hashlib.sha256()

    def __len__(self):
        return self._train_steps

    def __iter__(self):
        return self

    def set_epoch(self, epoch: int) -> None:
        """Require the trainer epoch to match the committed step."""
        if self._pending_step or epoch != self._committed_step // self._train_steps or epoch >= self._train_epochs:
            raise ValueError("trainer epoch does not match the committed cyclic packing step")

    def _load_item(self):
        """Load and validate one deterministic source feature."""
        if self._source_index >= self._source_len:
            return None
        index = self._source_index
        rng_before = _rng_snapshot()
        item = self.dataset[index]
        _assert_rng_unchanged(rng_before)
        if isinstance(item, list) and len(item) == 1:
            item = item[0]
        if not isinstance(item, dict):
            raise ValueError("each cyclic packing dataset row must produce exactly one feature dict")
        tokens = item.get("input_ids")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ValueError("cyclic packing expects one-dimensional tensor input_ids")
        length = tokens.shape[0]
        if not 0 < length <= self._max_seq_len:
            raise ValueError("cyclic packing does not accept empty or overlength features")
        labels = item.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != length:
            raise ValueError("cyclic packing requires labels aligned with input_ids")
        for key in ("attention_mask", "loss_weights", "image_mask", "video_mask", "mm_token_type_ids"):
            value = item.get(key)
            if value is not None and (not isinstance(value, torch.Tensor) or value.shape != tokens.shape):
                raise ValueError(f"cyclic packing requires {key} aligned with input_ids")
        positions = item.get("position_ids")
        if positions is not None and (
            not isinstance(positions, torch.Tensor) or positions.ndim == 0 or positions.shape[-1] != length
        ):
            raise ValueError("cyclic packing requires position_ids aligned with input_ids")
        size = _feature_logical_bytes(item)
        digest = _feature_hash(item)
        self._source_index += 1
        return index, item, length, size, digest

    def _next_group(self):
        """Build the next packed group from the bounded replay buffer."""
        while len(self._pending) < self._buffer_size:
            if self._lookahead is None:
                self._lookahead = self._load_item()
            if self._lookahead is None:
                break
            if self._pending and self._buffer_bytes + self._lookahead[3] > self._max_buffer_bytes:
                break
            self._pending.append(self._lookahead)
            self._buffer_bytes += self._lookahead[3]
            self._lookahead = None
        if not self._pending:
            return None
        group, retained, length = [], deque(), 0
        while self._pending:
            item = self._pending.popleft()
            if length + item[2] <= self._max_seq_len:
                group.append(item)
                length += item[2]
                self._buffer_bytes -= item[3]
            else:
                retained.append(item)
        self._pending = retained
        if not group:
            raise RuntimeError("cyclic packing made no progress")
        layout = [(item[0], item[2], item[4]) for item in group]
        encoded = json.dumps(layout, separators=(",", ":")).encode()
        _hash_part(self._layout_hash, b"G", encoded)
        self._cycle_groups += 1
        self._cycle_items += len(group)
        return group

    def _cycle_complete(self):
        """Return whether the source and all buffered items are exhausted."""
        return self._source_index == self._source_len and not self._pending and self._lookahead is None

    def _collate_group(self, group):
        """Copy feature dictionaries and labels, then collate and hash the output."""
        # PackingCollator may mask a boundary label in-place. Copy the dict and
        # labels so repeated dataset reads cannot be mutated.
        features = []
        for _, feature, _, _, _ in group:
            item = dict(feature)
            if isinstance(item.get("labels"), torch.Tensor):
                item["labels"] = item["labels"].clone()
            features.append(item)
        rng_before = _rng_snapshot()
        output = self.collate_fn(features)
        _assert_rng_unchanged(rng_before)
        _hash_part(self._output_hash, b"C", _feature_hash(output).encode())
        return output

    def _frontier(self):
        """Hash the buffered and lookahead positions needed for exact replay."""

        def descriptor(item):
            return None if item is None else (item[0], item[2], item[3], item[4])

        return _signature(
            {
                "source_index": self._source_index,
                "buffer": [descriptor(item) for item in self._pending],
                "lookahead": descriptor(self._lookahead),
                "buffer_bytes": self._buffer_bytes,
            }
        )

    def __next__(self):
        """Prepare a training step without committing its data cursor."""
        error = None
        batches = []
        done = False
        local_supervised = False
        try:
            if self._poisoned:
                raise RuntimeError("cyclic packing cursor is invalid after a failed operation")
            if self._require_resume and not self._restored:
                raise RuntimeError("cyclic packing cannot resume without its matching loader checkpoint")
            if self._pending_step:
                raise RuntimeError("the preceding packed step has not been committed")
            if self._step_getter() not in (self._committed_step, self._committed_step + 1):
                raise RuntimeError("trainer and cyclic loader steps differ")
            if self._committed_step >= self._total_steps:
                done = True
            else:
                if self._cycle_complete():
                    self._cycle += 1
                    self._reset_cycle()
                for _ in range(self._micro_batches_per_step):
                    group = self._next_group()
                    if group is None:
                        break
                    batch = self._collate_group(group)
                    labels = batch.get("labels")
                    if not isinstance(labels, torch.Tensor):
                        raise ValueError("cyclic packing requires collated tensor labels")
                    weights = batch.get("loss_weights")
                    valid = labels != -100
                    if weights is not None:
                        if not isinstance(weights, torch.Tensor) or weights.shape != labels.shape:
                            raise ValueError("cyclic packing requires aligned collated loss_weights")
                        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                            raise ValueError("cyclic packing requires finite non-negative loss_weights")
                        valid &= weights > 0
                    local_supervised = local_supervised or bool(valid.any())
                    batches.append(batch)
                if not batches:
                    raise RuntimeError("cyclic packing yielded an empty step")
                if self._pin:
                    batches = pin_memory(batches)
        except Exception as exc:  # synchronize errors before peers enter model collectives
            error = exc
        try:
            if dist.is_initialized():
                device = "cpu" if dist.get_backend(self._group) == "gloo" else get_device_type()
                # Reuse the required supervision reduction to coordinate
                # errors and geometry without per-step content gathering.
                geometry = [len(batches)]
                for index in range(self._micro_batches_per_step):
                    if index < len(batches):
                        batch = batches[index]
                        geometry.extend(
                            (
                                batch["labels"].numel(),
                                int(batch.get("pixel_values") is not None),
                                int(batch.get("pixel_values_videos") is not None),
                            )
                        )
                    else:
                        geometry.extend((-1, -1, -1))
                status = torch.tensor(
                    [int(error is not None), int(local_supervised), int(done), int(not done)]
                    + geometry
                    + [-value for value in geometry],
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self._group)
                values = status.tolist()
                failed, supervised, any_done, any_not_done = values[:4]
                if failed:
                    raise error or RuntimeError("preparing cyclic packed data failed on another rank")
                if any_done and any_not_done:
                    raise RuntimeError("cyclic packing end-of-stream differs across sequence-parallel ranks")
                count = len(geometry)
                if any(a != -b for a, b in zip(values[4 : 4 + count], values[4 + count :])):
                    raise RuntimeError("cyclic packing geometry differs across sequence-parallel ranks")
                local_supervised = bool(supervised)
            elif error is not None:
                raise error
        except Exception:
            self._poisoned = True
            raise
        if done:
            raise StopIteration
        if not local_supervised:
            self._poisoned = True
            raise ValueError("cyclic packing refuses an optimizer step with no supervised tokens")
        self._pending_step = True
        return batches

    def validate_pending_step(self) -> None:
        """Check the cursor before the trainer's coordinated step result."""
        if not self._pending_step or self._step_getter() != self._committed_step + 1:
            self._poisoned = True
            raise RuntimeError("cannot commit cyclic data before its optimizer step")

    def commit_step(self) -> None:
        """Commit prepared data after the matching optimizer step succeeds."""
        self.validate_pending_step()
        self._committed_step += 1
        self._pending_step = False

    def validate_creation_consensus(self) -> None:
        """Agree on the immutable data contract once, before training."""
        _agree(None, self._identity, self._group)

    def validate_restore_consensus(self) -> None:
        """Agree on restored global data position before any rank can train."""
        error = None
        try:
            if self._poisoned or self._pending_step or self._step_getter() != self._committed_step:
                raise RuntimeError("invalid restored cyclic packing position")
            if self._require_resume and not self._restored:
                raise RuntimeError("cyclic packing resume cursor was not restored")
        except Exception as exc:
            error = exc
        _agree(
            error,
            (self._identity, self._committed_step, self._cycle, self._layout_hash.hexdigest(), self._frontier()),
            self._group,
        )

    def state_dict(self):
        """Snapshot the committed cursor, identity, and replay frontier."""
        if self._poisoned or self._pending_step or self._step_getter() != self._committed_step:
            raise RuntimeError("cannot snapshot an uncommitted cyclic packing step")
        if self._require_resume and not self._restored:
            raise RuntimeError("cyclic packing resume cursor was not restored")
        validate_identity = getattr(self.collate_fn, "validate_checkpoint_identity", None)
        if callable(validate_identity):
            validate_identity()
        return {
            "kind": _STATE_KIND,
            "identity": self._identity,
            "step": self._committed_step,
            "group_rank": self._group_rank,
            "cycle": self._cycle,
            "groups": self._cycle_groups,
            "items": self._cycle_items,
            "layout": self._layout_hash.hexdigest(),
            "output_layout": self._output_hash.hexdigest(),
            "frontier": self._frontier(),
        }

    def load_state_dict(self, state):
        """Replay and verify a committed cursor into a fresh loader."""
        if self._poisoned or self._pending_step or self._committed_step or self._cycle_groups:
            raise RuntimeError("cyclic loader must be fresh before restoring")
        self._poisoned = True
        validate_identity = getattr(self.collate_fn, "validate_checkpoint_identity", None)
        if callable(validate_identity):
            validate_identity()
        keys = {
            "kind",
            "identity",
            "step",
            "group_rank",
            "cycle",
            "groups",
            "items",
            "layout",
            "output_layout",
            "frontier",
        }
        if not isinstance(state, dict) or set(state) != keys or state.get("kind") != _STATE_KIND:
            raise ValueError("missing or unsupported cyclic packing checkpoint")
        if state.get("identity") != self._identity:
            raise ValueError("cyclic packing input contract or geometry differs")
        if (
            type(state["step"]) is not int
            or not 0 <= state["step"] <= self._total_steps
            or self._step_getter() != state["step"]
            or type(state["group_rank"]) is not int
            or state["group_rank"] != self._group_rank
            or type(state["cycle"]) is not int
            or not 0 <= state["cycle"] <= state["step"]
            or type(state["groups"]) is not int
            or not 0 <= state["groups"] <= self._source_len
            or type(state["items"]) is not int
            or not 0 <= state["items"] <= self._source_len
            or not isinstance(state["layout"], str)
            or not isinstance(state["output_layout"], str)
            or not isinstance(state["frontier"], str)
        ):
            raise ValueError("invalid cyclic packing checkpoint position")
        self._cycle = state["cycle"]
        self._reset_cycle()
        for _ in range(state["groups"]):
            group = self._next_group()
            if group is None:
                raise ValueError("cyclic packing checkpoint exceeds the data cycle")
            self._collate_group(group)
        if (
            self._cycle_items != state["items"]
            or self._layout_hash.hexdigest() != state["layout"]
            or self._output_hash.hexdigest() != state["output_layout"]
            or self._frontier() != state["frontier"]
        ):
            raise ValueError("cyclic packing data or buffered frontier changed on replay")
        self._committed_step = state["step"]
        self._restored = True
        self._poisoned = False
