# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - Batched Sampler

import numpy as np
import math
from torch.utils.data import BatchSampler, Sampler


class CustomRandomSampler(Sampler):
    """Random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.
    
    For instance, the 'feature' could be the image aspect-ratio.
    The index returned is a tuple (sample_idx, feat_idx).
    """

    def __init__(
        self,
        dataset,
        batch_size,
        pool_size,
        min_view_size,
        max_view_size,
        world_size,
        rank=0,
        warmup=1,
        drop_last=True,
        view_sizes=None,
    ):
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.view_sizes = None if view_sizes is None else np.asarray(view_sizes, dtype=np.int64)
        self.drop_last = drop_last
        if world_size is None:
            raise ValueError("world_size must be provided explicitly (got None).")
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(f"world_size must be a positive int, got {world_size} ({type(world_size)})")
        if not isinstance(rank, int) or not (0 <= rank < world_size):
            raise ValueError(f"rank must be int in [0, world_size). Got rank={rank}, world_size={world_size}")

        self.world_size = world_size
        self.rank = rank
        if self.view_sizes is not None:
            if self.view_sizes.ndim != 1 or self.view_sizes.size == 0:
                raise ValueError(f"view_sizes must be a non-empty 1D sequence, got {view_sizes}")
            if np.any(self.view_sizes < self.min_view_size) or np.any(self.view_sizes > self.max_view_size):
                raise ValueError(
                    f"view_sizes={self.view_sizes.tolist()} must be within "
                    f"[{self.min_view_size}, {self.max_view_size}]"
                )

        self.len_dataset = N = len(dataset)
        if self.drop_last:
            self.num_samples = N // world_size
        else:
            self.num_samples = int(math.ceil(N / world_size))
        self.total_size = self.num_samples * world_size

        self.epoch = None
        self.epochf = 0.0
        self.start_batch_idx = 0  # For fast resume: skip this many batches without loading data

    def __len__(self):
        # Number of samples this *rank* will yield
        return self.num_samples

    def set_epoch(self, epoch, start_batch_idx=0):
        """Set epoch for shuffling. Optionally set start_batch_idx to skip batches without loading data."""
        self.epoch = epoch
        self.start_batch_idx = start_batch_idx

    def __iter__(self):
        if self.epoch is None:
            raise ValueError(
                "Epoch number not set. Please call 'set_epoch(epoch)' before iterating."
            )

        seed = self.epoch + 788
        rng = np.random.default_rng(seed=seed)
        indices = np.arange(self.len_dataset)
        rng.shuffle(indices)

        # Distributed sharding (DDP): match torch DistributedSampler semantics.
        if not self.drop_last:
            padding_size = self.total_size - self.len_dataset
            if padding_size > 0:
                indices = np.concatenate([indices, indices[:padding_size]])
        else:
            # Truncate to total_size (<= len_dataset) so each rank has equal num_samples
            indices = indices[: self.total_size]

        # Subsample for this rank
        sample_idxs = indices[self.rank : self.total_size : self.world_size]
        if sample_idxs.shape[0] != self.num_samples:
            raise RuntimeError(
                f"Internal error: expected num_samples={self.num_samples} for rank={self.rank}, "
                f"but got {sample_idxs.shape[0]}. total_size={self.total_size}, len_dataset={self.len_dataset}"
            )

        n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        if self.pool_size > 1:
            p = np.ones(self.pool_size)
            p[: self.pool_size // 2] *= 2
            p = p / p.sum()
            _feat_idxs = rng.choice(self.pool_size, size=n_batches, p=p)
        else:
            _feat_idxs = rng.integers(self.pool_size, size=n_batches)
        _feat_idxs = np.broadcast_to(_feat_idxs[:, None], (n_batches, self.batch_size))
        _feat_idxs = _feat_idxs.ravel()[: self.num_samples]
        if self.view_sizes is not None:
            _view_idxs = rng.choice(self.view_sizes, size=n_batches)
        else:
            _view_idxs = rng.integers(self.min_view_size, self.max_view_size + 1, size=n_batches)
        _view_idxs = np.broadcast_to(_view_idxs[:, None], (n_batches, self.batch_size))
        _view_idxs = _view_idxs.ravel()[: self.num_samples]

        idxs = np.c_[sample_idxs, _feat_idxs, _view_idxs]
        
        # Fast resume: skip samples without loading data
        start_sample_idx = self.start_batch_idx * self.batch_size
        if start_sample_idx > 0:
            original_len = len(idxs)
            idxs = idxs[start_sample_idx:]
            print(f"[CustomRandomSampler] Fast resume: skipped {start_sample_idx} samples, "
                  f"remaining {len(idxs)}/{original_len} (batches: {len(idxs)//self.batch_size})")
        
        yield from (tuple(idx) for idx in idxs)


class BatchedRandomSampler(BatchSampler):
    """Batch sampler that groups indices from RandomSampler into batches."""

    def __init__(self, sampler: CustomRandomSampler, batch_size, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch, start_batch_idx=0):
        """Set epoch for shuffling. Optionally set start_batch_idx to skip batches without loading data."""
        self.sampler.set_epoch(epoch, start_batch_idx)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple
