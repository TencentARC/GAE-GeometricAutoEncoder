"""Shared resolution parsing and sampling helpers."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


def parse_resolution_list(
    resolution: Iterable[int | Sequence[int]] | int | None,
    *,
    image_size: int | None = None,
) -> list[tuple[int, int]]:
    """Parse config ``resolution`` into ``[(W, H), ...]``."""
    if resolution is None:
        if image_size is None:
            raise ValueError("resolution or image_size is required")
        if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
            if len(image_size) != 2:
                raise ValueError(f"image_size pair must have length 2, got {image_size!r}")
            return [(int(image_size[0]), int(image_size[1]))]
        side = int(image_size)
        return [(side, side)]
    if isinstance(resolution, int):
        side = int(resolution)
        return [(side, side)]

    out: list[tuple[int, int]] = []
    for item in resolution:
        if isinstance(item, int):
            side = int(item)
            out.append((side, side))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 2:
                raise ValueError(f"resolution pair must have length 2, got {item!r}")
            w, h = item
            out.append((int(w), int(h)))
        else:
            raise TypeError(f"unsupported resolution entry: {item!r}")
    if not out:
        raise ValueError("resolution list is empty")
    return out


def pick_resolution_index(rng: random.Random, pool_size: int) -> int:
    """Sample a resolution index with geometry-style weights."""
    if pool_size <= 1:
        return 0
    weights = [2 if i < pool_size // 2 else 1 for i in range(pool_size)]
    return rng.choices(range(pool_size), weights=weights, k=1)[0]
