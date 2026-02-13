import hashlib
from typing import Any, Callable, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.multisource_utils import parse_multisource_config
from .dataset import DATASET_REGISTRY, build_iterable_dataset


logger = logging.get_logger(__name__)


def _build_source_id(source: Any, source_name: Optional[str]) -> str:
    raw = f"{source_name or ''}|{source}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SimpleMultiSourceIterableDataset(IterableDataset):
    """
    A simple multi-source iterable dataset that supports sampling from multiple datasets with weights.

    TODO: [Performance] Implement Async Prefetching
    Currently, data fetching is synchronous. Inspiration from MosaicML Streaming:
    - Introduce a background thread (Prefetcher) to fetch samples from `_datasets` into a `queue.Queue`.
    - This decouples I/O (downloading/reading) from the training loop, preventing GPU starvation.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: int = 42,
        level: str = "sample",
        transforms: Optional[Union[Callable, Sequence[Callable]]] = None,
        sample_token_len_fn: Optional[Callable[[Any], float]] = None,
        source_names: Optional[Sequence[str]] = None,
        source_ids: Optional[Sequence[str]] = None,
        sharded: bool = False,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
        max_seq_len: Optional[int] = None,
        overlong_strategy: Literal["drop", "truncate"] = "drop",
    ) -> None:
        self._datasets = list(datasets)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._seed = seed
        self._level = level
        self._transforms = transforms
        self._sample_token_len_fn = sample_token_len_fn or self._default_sample_token_len
        self._source_names = list(source_names) if source_names is not None else None
        self._source_ids = list(source_ids) if source_ids is not None else []
        self._sharded = sharded
        self._stopping_strategy = stopping_strategy
        self._max_seq_len = max_seq_len
        self._overlong_strategy = overlong_strategy
        self._ds_num = len(self._datasets)
        if not self._source_ids:
            if self._source_names is not None:
                self._source_ids = [str(name) for name in self._source_names]
            else:
                self._source_ids = [f"source_{idx}" for idx in range(self._ds_num)]
        self._avg_len_sum = [0.0 for _ in range(self._ds_num)]
        self._avg_len_count = [0 for _ in range(self._ds_num)]
        self._global_sample_idx = 0
        self._random_state = np.random.RandomState(seed=self._seed)
        self._seeded_in_worker = False
        self._iters: List[Any] = []
        self._epoch = 0
        if self._weights.shape[0] != self._ds_num:
            raise ValueError("weights length must match datasets length")
        if self._source_names is not None and len(self._source_names) != self._ds_num:
            raise ValueError("source_names length must match datasets length")
        if len(self._source_ids) != self._ds_num:
            raise ValueError("source_ids length must match datasets length")
        if len(set(self._source_ids)) != self._ds_num:
            raise ValueError("source_ids must be unique")
        if self._level not in ("sample", "token"):
            raise ValueError("level must be 'sample' or 'token'")
        if self._stopping_strategy not in ("first_exhausted", "all_exhausted"):
            raise ValueError("stopping_strategy must be 'first_exhausted' or 'all_exhausted'")
        if self._overlong_strategy not in ("drop", "truncate"):
            raise ValueError("overlong_strategy must be 'drop' or 'truncate'")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number and update the random state seed."""
        self._epoch = epoch
        self._seeded_in_worker = False  # Force re-seeding in __iter__ with new epoch

    def __iter__(self):
        """
        TODO: [Performance] Integrate Background Prefetcher
        - Spawn a daemon thread that calls `_next_sample` and puts items into a `Queue`.
        - The main `__iter__` loop should simply `yield queue.get()`.
        """
        worker_info = get_worker_info()
        parallel_state = get_parallel_state()
        dp_rank = max(0, int(getattr(parallel_state, "dp_rank", 0)))
        dp_size = max(1, int(getattr(parallel_state, "dp_size", 1)))
        if not self._seeded_in_worker:
            worker_id = worker_info.id if worker_info is not None else 0
            base_seed = worker_info.seed if worker_info is not None else self._seed
            seed_seq = np.random.SeedSequence([base_seed, self._epoch, dp_rank, worker_id])
            current_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            self._random_state = np.random.RandomState(current_seed)
            self._seeded_in_worker = True
        self._iters = [iter(ds) for ds in self._datasets]
        self._exhausted = [False for _ in range(self._ds_num)]
        while True:
            ds_idx = self._random_state.choice(self._ds_num, p=self._runtime_weights())
            try:
                sample = self._next_sample(ds_idx)
            except StopIteration:
                return
            if sample is None:
                continue
            sample = self._attach_meta(sample, ds_idx)
            sample = self._apply_transforms(sample)
            if sample is None:
                continue
            sample = self._ensure_meta(sample, ds_idx)
            sample, token_len = self._maybe_apply_max_seq_len(sample)
            if token_len <= 0:
                continue
            if self._level == "token":
                self._avg_len_sum[ds_idx] += token_len
                self._avg_len_count[ds_idx] += 1
            self._global_sample_idx += 1
            if not self._sharded and self._global_sample_idx % dp_size != dp_rank:
                continue
            yield sample

    def _runtime_weights(self) -> np.ndarray:
        if self._level == "sample":
            weights = self._weights
        else:
            avg_lens = []
            for idx in range(self._ds_num):
                if self._avg_len_count[idx] > 0:
                    avg_lens.append(self._avg_len_sum[idx] / self._avg_len_count[idx])
                else:
                    avg_lens.append(1.0)
            weights = self._weights / np.asarray(avg_lens, dtype=np.float64)
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("sum of weights must be positive")
        return weights / total

    def _next_sample(self, ds_idx: int) -> Any:
        while True:
            try:
                return next(self._iters[ds_idx])
            except StopIteration:
                if self._stopping_strategy == "first_exhausted":
                    raise
                self._exhausted[ds_idx] = True
                if all(self._exhausted):
                    raise
                self._iters[ds_idx] = iter(self._datasets[ds_idx])

    def _apply_transforms(self, sample: Any) -> Any:
        if self._transforms is None:
            return sample
        transforms = self._transforms
        if not isinstance(transforms, list):
            if isinstance(transforms, Sequence):
                transforms = list(transforms)
            else:
                transforms = [transforms]
        if isinstance(sample, list):
            items = []
            for item in sample:
                out = item
                for transform in transforms:
                    out = transform(out)
                    if out is None:
                        break
                if out is None:
                    continue
                if isinstance(out, list):
                    items.extend(out)
                else:
                    items.append(out)
            return items if items else None
        out = sample
        for transform in transforms:
            out = transform(out)
            if out is None:
                return None
        return out

    def _attach_meta(self, sample: Any, ds_idx: int) -> Any:
        source_name = self._source_names[ds_idx] if self._source_names is not None else None
        if isinstance(sample, list):
            for item in sample:
                if isinstance(item, dict):
                    item["ds_idx"] = ds_idx
                    if source_name is not None:
                        item["source_name"] = source_name
            return sample
        if isinstance(sample, dict):
            sample["ds_idx"] = ds_idx
            if source_name is not None:
                sample["source_name"] = source_name
        return sample

    def _ensure_meta(self, sample: Any, ds_idx: int) -> Any:
        return self._attach_meta(sample, ds_idx)

    def _maybe_apply_max_seq_len(self, sample: Any) -> tuple[Any, float]:
        if sample is None:
            return None, 0.0
        token_len = self._sample_token_len_fn(sample)
        if self._max_seq_len is None:
            return sample, token_len
        if token_len <= 0:
            return sample, token_len
        if token_len <= self._max_seq_len:
            return sample, token_len
        if self._overlong_strategy == "drop":
            return None, 0.0
        truncated = self._truncate_sample(sample, self._max_seq_len)
        if truncated is None:
            return None, 0.0
        token_len = self._sample_token_len_fn(truncated)
        return truncated, token_len

    def _truncate_sample(self, sample: Any, max_seq_len: int) -> Any:
        if sample is None:
            return None
        if isinstance(sample, list):
            items = []
            for item in sample:
                truncated = self._truncate_sample(item, max_seq_len)
                if truncated is None:
                    continue
                items.append(truncated)
            return items if items else None
        if not isinstance(sample, dict):
            return sample
        for key in ("input_ids", "attention_mask", "labels", "position_ids", "token_type_ids"):
            if key in sample:
                sample[key] = self._truncate_value(sample[key], max_seq_len)
        return sample

    def _truncate_value(self, value: Any, max_seq_len: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[..., :max_seq_len]
        if isinstance(value, list):
            return value[:max_seq_len]
        return value

    def _default_sample_token_len(self, sample: Any) -> float:
        if sample is None:
            return 0
        if isinstance(sample, list):
            return float(sum(self._default_sample_token_len(item) for item in sample))
        if not isinstance(sample, dict):
            return 1.0
        if "attention_mask" in sample:
            attention_mask = sample["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                return float(attention_mask.sum().item())
            if isinstance(attention_mask, list):
                return float(sum(attention_mask))
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                return float(input_ids.numel())
            if isinstance(input_ids, list):
                return float(len(input_ids))
        return 1.0

    def state_dict(self) -> dict:
        dataset_states_by_id = {}
        for dataset, source_id in zip(self._datasets, self._source_ids):
            state_fn = getattr(dataset, "state_dict", None)
            getstate_fn = getattr(dataset, "__getstate__", None)
            if callable(state_fn):
                ds_state = state_fn()
            elif callable(getstate_fn):
                ds_state = getstate_fn()
            else:
                ds_state = None
            dataset_states_by_id[source_id] = ds_state
        avg_len_sum_by_id = {source_id: self._avg_len_sum[idx] for idx, source_id in enumerate(self._source_ids)}
        avg_len_count_by_id = {source_id: self._avg_len_count[idx] for idx, source_id in enumerate(self._source_ids)}
        return {
            "version": 2,
            "topology": {
                "source_ids": list(self._source_ids),
                "source_names": list(self._source_names) if self._source_names is not None else None,
                "weights": self._weights.tolist(),
                "level": self._level,
                "stopping_strategy": self._stopping_strategy,
                "max_seq_len": self._max_seq_len,
                "overlong_strategy": self._overlong_strategy,
                "sharded": self._sharded,
            },
            "runtime": {
                "random_state": self._random_state.get_state(),
                "avg_len_sum": avg_len_sum_by_id,
                "avg_len_count": avg_len_count_by_id,
                "global_sample_idx": self._global_sample_idx,
                "dataset_states": dataset_states_by_id,
            },
        }

    def load_state_dict(
        self,
        state: dict,
        reconcile_policy: Literal["strict", "allow_add", "allow_add_remove", "warn_only"] = "allow_add_remove",
    ) -> None:
        """
        TODO: [Resumption] Improve Deterministic Resumption & Elasticity
        Currently, we restore `random_state` but do not "fast-forward" the underlying iterators to the exact position.
        Inspiration from MosaicML Streaming:
        - Implement precise seeking or fast-forwarding (skipping) based on `global_sample_idx`.
        - Consider decoupling data sharding from physical node count (e.g., using `num_canonical_nodes`) to support elastic training (resuming on different number of GPUs).
        """
        if "topology" not in state or "runtime" not in state:
            raise ValueError("state_dict missing required keys: topology/runtime")
        runtime = state["runtime"]
        topology = state["topology"]
        if "source_ids" not in topology:
            raise ValueError("state_dict missing topology.source_ids")
        saved_source_ids = topology["source_ids"]
        added = []
        removed = []
        if saved_source_ids:
            saved_set = set(saved_source_ids)
            added = [source_id for source_id in self._source_ids if source_id not in saved_set]
            removed = [source_id for source_id in saved_source_ids if source_id not in set(self._source_ids)]
            if added or removed:
                if reconcile_policy == "strict":
                    raise ValueError(
                        f"source_ids mismatch: added={added} removed={removed} with policy={reconcile_policy}"
                    )
                if reconcile_policy == "allow_add" and removed:
                    raise ValueError(
                        f"source_ids removed not allowed: removed={removed} with policy={reconcile_policy}"
                    )
                if reconcile_policy == "warn_only":
                    logger.warning(
                        f"source_ids changed: added={added} removed={removed} with policy={reconcile_policy}"
                    )
        random_state = runtime["random_state"]
        self._random_state.set_state(random_state)
        avg_len_sum = runtime["avg_len_sum"]
        avg_len_count = runtime["avg_len_count"]
        if not isinstance(avg_len_sum, dict) or not isinstance(avg_len_count, dict):
            raise ValueError("runtime.avg_len_sum and runtime.avg_len_count must be dicts keyed by source_id")
        self._avg_len_sum = [float(avg_len_sum.get(source_id, 0.0)) for source_id in self._source_ids]
        self._avg_len_count = [int(avg_len_count.get(source_id, 0)) for source_id in self._source_ids]
        self._global_sample_idx = runtime.get("global_sample_idx", 0)
        dataset_states = runtime["dataset_states"]
        if not isinstance(dataset_states, dict):
            raise ValueError("runtime.dataset_states must be a dict keyed by source_id")
        dataset_states_by_id = dataset_states
        for dataset, source_id in zip(self._datasets, self._source_ids):
            ds_state = dataset_states_by_id.get(source_id)
            if ds_state is None:
                continue
            load_state_fn = getattr(dataset, "load_state_dict", None)
            setstate_fn = getattr(dataset, "__setstate__", None)
            if callable(load_state_fn):
                load_state_fn(ds_state)
            elif callable(setstate_fn):
                setstate_fn(ds_state)


@DATASET_REGISTRY.register("simple_multisource")
def build_simple_multisource_dataset(
    train_path: str,
    datasets_type: str = "iterable",
    namespace: Literal["train", "test"] = "train",
    transform: Optional[Callable] = None,
    seed: int = 42,
    level: str = "sample",
    sample_token_len_fn: Optional[Callable[[Any], float]] = None,
    split_by_node: bool = True,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    max_seq_len: Optional[int] = None,
    overlong_strategy: Literal["drop", "truncate"] = "drop",
    **kwargs: Any,
) -> IterableDataset:
    multisource_config = parse_multisource_config(train_path)
    schedule = multisource_config["schedule"]
    if len(schedule) != 1 or schedule[0].get("schedule_type") != "const":
        raise ValueError("simple_multisource only supports a single const schedule")
    if datasets_type != "iterable":
        raise ValueError("simple_multisource only supports iterable datasets")
    sources = multisource_config["sources"]
    source_names = multisource_config.get("names")
    weights = schedule[0]["weights"]
    source_names_for_id: List[Optional[str]] = (
        list(source_names) if source_names is not None else [None for _ in sources]
    )
    source_ids = [_build_source_id(source, source_name) for source, source_name in zip(sources, source_names_for_id)]
    datasets = [
        build_iterable_dataset(
            train_path=source,
            namespace=namespace,
            seed=seed,
            transform=None,
            split_by_node=split_by_node,
        )
        for source in sources
    ]
    return SimpleMultiSourceIterableDataset(
        datasets=datasets,
        weights=weights,
        seed=seed,
        level=level,
        transforms=transform,
        sample_token_len_fn=sample_token_len_fn,
        source_names=source_names,
        source_ids=source_ids,
        sharded=split_by_node,
        stopping_strategy=stopping_strategy,
        max_seq_len=max_seq_len,
        overlong_strategy=overlong_strategy,
    )
