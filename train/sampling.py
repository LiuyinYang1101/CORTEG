"""
Multi-subject batch samplers for pooled ECoG training.

Both samplers operate on a ConcatDataset where subjects are laid out
contiguously: [subj0_trials ... subj1_trials ... subjN_trials].
"""
from __future__ import annotations

import random
from itertools import cycle
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class SubjectInterleavedSampler(Sampler):
    """Training sampler that interleaves batches across subjects.

    Yields batches in round-robin order: Subj0_batch, Subj1_batch, ...
    Smaller datasets are cycled so every subject contributes a batch
    every round, regardless of dataset size differences.

    Args:
        dataset_sizes: Number of trials for each subject.
        batch_size: Number of trials per batch.
        shuffle: If True, shuffle each subject's trials each epoch.
    """

    def __init__(self, dataset_sizes: List[int], batch_size: int, shuffle: bool = True):
        self.sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_subjects = len(dataset_sizes)
        self.offsets: List[int] = [0]
        for s in dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def __iter__(self) -> Iterator[List[int]]:
        iterators = []
        for i, size in enumerate(self.sizes):
            indices = list(range(size))
            if self.shuffle:
                random.shuffle(indices)
            batches = [indices[k:k + self.batch_size] for k in range(0, size, self.batch_size)]
            offset = self.offsets[i]
            batches = [[idx + offset for idx in b] for b in batches]
            iterators.append(batches)

        max_batches = max(len(it) for it in iterators)
        cycled = [cycle(it) for it in iterators]
        for _ in range(max_batches):
            for it in cycled:
                yield next(it)

    def __len__(self) -> int:
        max_batches = max((s + self.batch_size - 1) // self.batch_size for s in self.sizes)
        return max_batches * self.num_subjects


class SequentialPooledSampler(Sampler):
    """Validation / test sampler that keeps subjects strictly separated.

    Iterates sequentially: all Subj0 batches, then all Subj1 batches, etc.
    No batch ever crosses a subject boundary, which is required when
    subjects have different channel counts.

    Args:
        dataset_sizes: Number of trials for each subject.
        batch_size: Number of trials per batch.
    """

    def __init__(self, dataset_sizes: List[int], batch_size: int):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.offsets: List[int] = [0]
        for s in dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def __iter__(self) -> Iterator[List[int]]:
        for i, size in enumerate(self.dataset_sizes):
            offset = self.offsets[i]
            indices = range(offset, offset + size)
            for k in range(0, len(indices), self.batch_size):
                yield list(indices[k:k + self.batch_size])

    def __len__(self) -> int:
        return sum((s + self.batch_size - 1) // self.batch_size for s in self.dataset_sizes)
