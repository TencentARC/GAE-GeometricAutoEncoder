"""Streaming datasets for T2I / ImageNet co-training."""

__all__ = ["BLIP3OWebDataset", "build_blip3o_wds_loader"]


def __getattr__(name: str):
    if name in __all__:
        from .blip3o_wds import BLIP3OWebDataset, build_blip3o_wds_loader

        exports = {
            "BLIP3OWebDataset": BLIP3OWebDataset,
            "build_blip3o_wds_loader": build_blip3o_wds_loader,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
