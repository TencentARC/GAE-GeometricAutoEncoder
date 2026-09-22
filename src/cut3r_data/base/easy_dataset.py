# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - EasyDataset base class

import numpy as np
from cut3r_data.base.batched_sampler import BatchedRandomSampler, CustomRandomSampler


class EasyDataset:
    """A dataset that you can easily resize and combine.
    
    Examples:
        2 * dataset ==> duplicate each element 2x
        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)
        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass  # nothing to do by default

    def make_sampler(self, batch_size, shuffle=True, drop_last=True, world_size=1, rank=0, fixed_length=False):
        """Create a sampler for multi-resolution support.
        
        Note: Even when shuffle=False, we still need CustomRandomSampler for multi-resolution
        support. The 'shuffle' behavior is controlled by epoch seed for RandomSampler.
        For test mode (shuffle=False, fixed_length=True), we use fixed epoch=0 for determinism.
        """
        num_of_aspect_ratios = len(self._resolutions)
        num_of_views = self.num_views
        view_choices = getattr(self, "view_choices", None)
        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios,
            min(view_choices) if view_choices else (num_of_views if fixed_length else 4),
            num_of_views,
            world_size=world_size,
            rank=rank,
            warmup=1,
            drop_last=drop_last,
            view_sizes=view_choices,
        )
        return BatchedRandomSampler(sampler, batch_size, drop_last)


class MulDataset(EasyDataset):
    """Artificially augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[idx // self.multiplicator, other, another]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views

    @property
    def view_choices(self):
        return getattr(self.dataset, "view_choices", None)


class ResizedDataset(EasyDataset):
    """Artificially changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        # Return 0 if underlying dataset is empty (allows skipping missing datasets like hypersim)
        if len(self.dataset) == 0:
            return 0
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        if len(self.dataset) == 0:
            # Skip empty datasets (e.g., hypersim not available on this server)
            self._idxs_mapping = np.array([], dtype=np.int64)
            return
        rng = np.random.default_rng(seed=epoch + 777)
        perm = rng.permutation(len(self.dataset))
        shuffled_idxs = np.concatenate([perm] * (1 + (len(self) - 1) // len(self.dataset)))
        self._idxs_mapping = shuffled_idxs[: self.new_size]
        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert hasattr(self, "_idxs_mapping"), \
            "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[self._idxs_mapping[idx], other, another]
        else:
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views

    @property
    def view_choices(self):
        return getattr(self.dataset, "view_choices", None)


class CatDataset(EasyDataset):
    """Concatenation of several datasets."""

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = [d for d in datasets if len(d) > 0]
        self._cum_sizes = np.cumsum([len(d) for d in self.datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            idx, other, another = idx

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None and another is not None:
            new_idx = (new_idx, other, another)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions)
        return resolutions

    @property
    def num_views(self):
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views
        return num_views

    @property
    def view_choices(self):
        view_choices = getattr(self.datasets[0], "view_choices", None)
        for dataset in self.datasets[1:]:
            assert getattr(dataset, "view_choices", None) == view_choices, \
                "All concatenated datasets must share the same view_choices"
        return view_choices
