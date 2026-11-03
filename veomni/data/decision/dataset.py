"""Case datasets and padded sampling for the native stateful dataloader."""

import math
import random
from dataclasses import dataclass
from itertools import chain, islice

from torch.utils.data import Dataset, Sampler


def rank_batches(order, global_batch_size, world, rank):
    """Shard each global batch by rank, padding missing cases with ``None``."""
    for start in range(0, len(order), global_batch_size):
        batch = list(order[start : start + global_batch_size])
        batch.extend([None] * (global_batch_size - len(batch)))
        yield batch[rank::world]


@dataclass
class CaseEncoder:
    transform: object
    shuffle_request: object
    seed: int
    shuffle: bool = False

    def __call__(self, case, *, epoch):
        request = self.shuffle_request(case, self.seed, epoch) if self.shuffle else case.request
        return self.transform(request)


class DecisionDataset(Dataset):
    def __init__(self, cases, transform=None):
        self.cases = cases
        self.transform = transform
        self.epoch = 0

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        if index is None:
            return [None]
        case = self.cases[index]
        sample = self.transform(case, epoch=self.epoch) if self.transform is not None else case
        return [sample]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]


class DecisionSampler(Sampler):
    """Keep case shuffling and zero-weight tail padding outside the loader."""

    def __init__(self, dataset, global_batch_size, world, rank, seed):
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.world = world
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.offset = 0
        self._resume_offset = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.global_batch_size) * (self.global_batch_size // self.world)

    def __iter__(self):
        self.offset = self._resume_offset
        self._resume_offset = 0
        order = list(range(len(self.dataset)))
        random.Random(f"{self.seed}:{self.epoch}").shuffle(order)
        indices = chain.from_iterable(rank_batches(order, self.global_batch_size, self.world, self.rank))
        for index in islice(indices, self.offset, None):
            self.offset += 1
            yield index

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        if epoch != self.epoch:
            self.offset = self._resume_offset = 0
        self.epoch = epoch

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.offset = self._resume_offset = state["offset"]
