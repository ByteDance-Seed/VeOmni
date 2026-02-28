import copy
from typing import Any, Callable, List, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging


logger = logging.get_logger(__name__)


class MultiSourceDataset(IterableDataset):
    """Multi-source dataset with weighted sampling.

    This dataset samples from multiple upstream iterable datasets according to a
    (possibly token-adjusted) weight distribution.

    It supports:
    - Per-epoch deterministic randomness (seeded by epoch, dp rank, and worker id).
    - Optional distributed sharding behavior controlled by ``sharded``.
    - Stopping strategies for how to behave when an upstream source is exhausted.
    - Optional refetch-index passthrough for checkpointing buffers by index.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: int = 42,
        level: str = "sample",
        sample_token_len_fn: Optional[Callable[[Any], float]] = None,
        source_names: Optional[Sequence[str]] = None,
        source_ids: Optional[Sequence[str]] = None,
        sharded: bool = False,
        stopping_strategy: Literal["first_exhausted", "all_exhausted", "never_exhausted"] = "first_exhausted",
        output_refetch_idx: bool = False,
    ) -> None:
        """Initialize a MultiSourceDataset.

        Args:
            datasets: Upstream iterable datasets (one per source).
            weights: Sampling weights aligned with ``datasets``.
            seed: Base random seed.
            level: Sampling level. ``sample`` uses ``weights`` directly; ``token`` reweights
                by the inverse of the running average token length per source.
            sample_token_len_fn: Function that returns the token length of a sample.
                If not provided, a default heuristic is used.
            source_names: Optional display names for each source (for meta fields).
            source_ids: Optional stable IDs for each source (used in checkpoint state).
            sharded: If False, performs deterministic modulo-based sharding by dp rank on
                the produced samples. If True, assumes upstream datasets already handle
                sharding/splitting.
            stopping_strategy:
                - ``first_exhausted``: Stop the whole dataset once any source is exhausted.
                - ``all_exhausted``: Restart an exhausted source until all sources are exhausted.
                - ``never_exhausted``: Always restart exhausted sources and never terminate.
            output_refetch_idx: If True, yields ``(sample, (source_id, refetch_idx))`` so that
                downstream components can checkpoint buffers by indices and reconstruct them.

        Raises:
            ValueError: If input arguments are invalid.
        """
        self._datasets = list(datasets)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._seed = seed
        self._level = level
        self._sample_token_len_fn = sample_token_len_fn or self._default_sample_token_len
        self._source_names = list(source_names) if source_names is not None else None
        self._source_ids = list(source_ids) if source_ids is not None else []
        self._sharded = sharded
        self._stopping_strategy = stopping_strategy
        self._ds_num = len(self._datasets)

        if not self._source_names:
            self._source_names = []
            for i, dataset in enumerate(self._datasets):
                if callable(getattr(dataset, "get_name", None)):
                    self._source_names.append(dataset.get_name())
                else:
                    self._source_names.append(f"source_{i}")

        if not self._source_ids:
            self._source_ids = copy.deepcopy(self._source_names)

        self._id2dataset = dict(zip(self._source_ids, self._datasets))
        self._avg_len_sum = [0.0 for _ in range(self._ds_num)]
        self._avg_len_count = [0 for _ in range(self._ds_num)]
        self._global_sample_idx = 0
        self._random_state = np.random.RandomState(seed=self._seed)
        self._iters: List[Any] = []
        self._epoch = 0
        self._exhausted = [False for _ in range(self._ds_num)]
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
        if self._stopping_strategy not in ("first_exhausted", "all_exhausted", "never_exhausted"):
            raise ValueError("stopping_strategy must be 'first_exhausted', 'all_exhausted', or 'never_exhausted'")

        parallel_state = get_parallel_state()
        self.dp_rank = max(0, int(getattr(parallel_state, "dp_rank", 0)))
        self.dp_size = max(1, int(getattr(parallel_state, "dp_size", 1)))

        self.output_refetch_idx = output_refetch_idx

        self._just_resumed = False

    @property
    def output_refetch_idx(self) -> bool:
        """Whether to yield refetch indices alongside samples."""
        return self._output_refetch_idx

    @output_refetch_idx.setter
    def output_refetch_idx(self, value: bool) -> None:
        """Enable or disable refetch-index output.

        When enabled, each upstream dataset must provide:
        - ``get_item(idx)`` to fetch a sample by index
        - ``output_refetch_idx`` attribute to switch yielding ``(sample, idx)``

        Args:
            value: True to enable refetch indices, False to disable.

        Raises:
            ValueError: If any upstream dataset cannot support refetch-by-index.
        """
        if value:
            for source_id, dataset in self._id2dataset.items():
                if not (callable(getattr(dataset, "get_item", None)) and hasattr(dataset, "output_refetch_idx")):
                    raise ValueError(
                        f"output_refetch_idx is True, but dataset '{source_id}' does not have "
                        f"get_item method or output_refetch_idx attribute to resume samples "
                        f"in buffers based on idx"
                    )
        self._output_refetch_idx = value
        for dataset in self._datasets:
            if hasattr(dataset, "output_refetch_idx"):
                setattr(dataset, "output_refetch_idx", value)

    def get_item(self, refetch_idx):
        """Fetch a single sample by its source ID and index within that source.

        This is used by downstream checkpoint/resume logic that stores buffer
        contents as ``(source_id, idx)`` pairs instead of full samples.

        Args:
            refetch_idx: A ``(source_id, idx)`` tuple. ``source_id`` identifies the
                sub-dataset, and ``idx`` is the 0-based index within that sub-dataset.

        Returns:
            The sample returned by the underlying sub-dataset.

        Raises:
            AttributeError: If the underlying sub-dataset does not provide an index-based fetch API.
        """
        source_id, idx = refetch_idx
        dataset = self._id2dataset[source_id]
        get_item_fn = getattr(dataset, "get_item", None)
        if callable(get_item_fn):
            return get_item_fn(idx)
        raise AttributeError(f"dataset '{source_id}' does not implement get_item")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic sampling.

        Args:
            epoch: Current epoch number.
        """
        self._epoch = epoch
        for dataset in self._datasets:
            set_epoch_fn = getattr(dataset, "set_epoch", None)
            if callable(set_epoch_fn):
                set_epoch_fn(epoch)

    def __iter__(self):
        """Iterate and yield samples from multiple sources.

        Yields:
            If ``output_refetch_idx`` is False, yields a sample (typically a dict).
            If ``output_refetch_idx`` is True, yields ``(sample, (source_id, refetch_idx))``.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if not self._just_resumed:
            seed_seq = np.random.SeedSequence([self._seed, self._epoch, self.dp_rank, worker_id])
            current_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            self._random_state = np.random.RandomState(current_seed)
            self._exhausted = [False for _ in range(self._ds_num)]
            self._avg_len_sum = [0.0 for _ in range(self._ds_num)]
            self._avg_len_count = [0 for _ in range(self._ds_num)]
            self._global_sample_idx = 0
        else:
            self._just_resumed = False

        self._iters = [iter(ds) for ds in self._datasets]
        while True:
            ds_idx = self._random_state.choice(self._ds_num, p=self._runtime_weights())
            try:
                sample = self._next_sample(ds_idx)
            except StopIteration:
                return
            if sample is None:
                continue

            if self._output_refetch_idx:
                sample, refetch_idx = sample[0], sample[1]

            sample = self._attach_meta(sample, ds_idx)
            token_len = self._sample_token_len_fn(sample)
            if token_len <= 0:
                continue
            if self._level == "token":
                self._avg_len_sum[ds_idx] += token_len
                self._avg_len_count[ds_idx] += 1
            self._global_sample_idx += 1
            if not self._sharded and self._global_sample_idx % self.dp_size != self.dp_rank:
                continue
            if self._output_refetch_idx:
                yield sample, (self._source_ids[ds_idx], refetch_idx)
            else:
                yield sample

    def _runtime_weights(self) -> np.ndarray:
        """Compute the per-source sampling probabilities for the current runtime state.

        Returns:
            A probability vector of shape ``(num_sources,)`` that sums to 1.

        Raises:
            ValueError: If the weight sum is non-positive.
        """
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
        """Fetch the next sample from a specific sub-dataset index.

        Args:
            ds_idx: Index of the sub-dataset to fetch from.

        Returns:
            The next sample from the chosen sub-dataset.

        Raises:
            StopIteration: When the dataset terminates under the configured stopping strategy.
        """
        while True:
            try:
                return next(self._iters[ds_idx])
            except StopIteration:
                if self._stopping_strategy == "first_exhausted":
                    raise
                if self._stopping_strategy == "all_exhausted":
                    self._exhausted[ds_idx] = True
                    if all(self._exhausted):
                        raise
                elif self._stopping_strategy == "never_exhausted":
                    self._exhausted[ds_idx] = True
                    if all(self._exhausted):
                        self._exhausted = [False for _ in range(self._ds_num)]
                self._iters[ds_idx] = iter(self._datasets[ds_idx])

    def _attach_meta(self, sample: Any, ds_idx: int) -> Any:
        """Attach per-source metadata fields onto a sample.

        Adds:
            - ``ds_idx``: the integer source index
            - ``source_name``: optional display name if provided

        Args:
            sample: A sample or list of samples.
            ds_idx: Source index for this sample.

        Returns:
            The updated sample (mutated in place when possible).
        """
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

    def _default_sample_token_len(self, sample: Any) -> float:
        """Default heuristic to estimate token length of a sample.

        Args:
            sample: A single sample or a list of samples.

        Returns:
            Estimated token length as a float.
        """
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
        """Return a checkpointable state dict for this dataset."""
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
        # save _exhausted state
        exhausted_by_id = {source_id: self._exhausted[idx] for idx, source_id in enumerate(self._source_ids)}
        return {
            "version": 0,
            "topology": {
                "source_ids": list(self._source_ids),
                "source_names": list(self._source_names) if self._source_names is not None else None,
                "weights": self._weights.tolist(),
                "level": self._level,
                "stopping_strategy": self._stopping_strategy,
                "sharded": self._sharded,
            },
            "runtime": {
                "random_state": self._random_state.get_state(),
                "avg_len_sum": avg_len_sum_by_id,
                "avg_len_count": avg_len_count_by_id,
                "exhausted": exhausted_by_id,
                "global_sample_idx": self._global_sample_idx,
                "dataset_states": dataset_states_by_id,
            },
        }

    def load_state_dict(
        self,
        state: dict,
        reconcile_policy: Literal["strict", "allow_add", "allow_add_remove", "warn_only"] = "allow_add_remove",
    ) -> None:
        """Restore state from a previous ``state_dict()``.

        Args:
            state: State dict previously produced by ``state_dict()``.
            reconcile_policy: Policy for handling source-id changes:
                - ``strict``: error on any added/removed source.
                - ``allow_add``: allow new sources but error on removed ones.
                - ``allow_add_remove``: allow both add and remove.
                - ``warn_only``: allow changes and log a warning.

        Raises:
            ValueError: If required state fields are missing or incompatible.
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
        if saved_source_ids is not None:
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

        # Ensure _exhausted is re-initialized for the current source count
        # This is important when sources are added/removed during checkpoint resume
        if "exhausted" in runtime and isinstance(runtime["exhausted"], dict):
            exhausted_dict = runtime["exhausted"]
            self._exhausted = [bool(exhausted_dict.get(source_id, False)) for source_id in self._source_ids]
        else:
            self._exhausted = [False for _ in range(self._ds_num)]

        self._just_resumed = True
