# CUT3R Data Package for GLD
# Self-contained dataset utilities, adapted from CUT3R/dust3r
# No external dependencies on CUT3R repository

import torch
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from accelerate import Accelerator

# Base classes
from cut3r_data.base import (
    EasyDataset,
    MulDataset,
    ResizedDataset,
    CatDataset,
    BatchedRandomSampler,
    CustomRandomSampler,
    BaseMultiViewDataset,
)

# Dataset implementations
from cut3r_data.datasets import (
    DL3DV_Multi,
    HyperSim_Multi,
    MVSSynth_Multi,
    RE10K_Multi,
    RE10K_Packed,
    RE10KLatent_Multi,
    TartanAir_Multi,
    OpenVid_Multi,
    OpenVidT2V_Multi,
    OSPIstockRGB_Multi,
    OSPLatent_Multi,
    ScanNetppLatent_Multi,
    ScanNetppRGB_Multi,
    T2ILatentFlat_Multi,
    VideoMetaScene_Multi,
)


def robust_collate(batch):
    """``default_collate`` that only stacks keys present in every sample.

    CatDataset mixes RE10K/DL3DV/OSP/etc. in one batch; view dict keys differ
    (OSP emits ``disable_plucker``, geometry sets do not). Plain collate raises
    ``KeyError`` when the first row is OSP.
    """
    from torch.utils.data.dataloader import default_collate

    elem = batch[0]
    if isinstance(elem, dict):
        common = set(elem.keys())
        for item in batch[1:]:
            common &= set(item.keys())
        return {
            k: robust_collate([d[k] for d in batch])
            for k in elem.keys()
            if k in common
        }
    if isinstance(elem, (list, tuple)):
        return [robust_collate(list(samples)) for samples in zip(*batch)]
    return default_collate(batch)


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    world_size: int = 1,
    rank: int = 0,
    fixed_length=False,
    collate_fn=None,
):
    """Create a data loader for the given dataset.
    
    Args:
        dataset: Dataset or string representation
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pin_mem: Whether to pin memory
        world_size: Number of distributed processes (DDP world size). Must be explicit.
        rank: Rank of the current process in [0, world_size). Must be explicit.
        fixed_length: If True, use fixed view count (for test mode)
        
    Returns:
        DataLoader instance
    """
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )
    return data_loader


__all__ = [
    # Base classes
    "EasyDataset",
    "MulDataset", 
    "ResizedDataset",
    "CatDataset",
    "BatchedRandomSampler",
    "CustomRandomSampler",
    "BaseMultiViewDataset",
    # Datasets
    "DL3DV_Multi",
    "HyperSim_Multi",
    "MVSSynth_Multi",
    "RE10K_Multi",
    "RE10K_Packed",
    "RE10KLatent_Multi",
    "TartanAir_Multi",
    "OpenVid_Multi",
    "OpenVidT2V_Multi",
    "OSPIstockRGB_Multi",
    "OSPLatent_Multi",
    "ScanNetppLatent_Multi",
    "ScanNetppRGB_Multi",
    "T2ILatentFlat_Multi",
    "VideoMetaScene_Multi",
    # Functions
    "get_data_loader",
    "robust_collate",
]
