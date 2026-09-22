# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - minimal subset of CUT3R/dust3r utilities

import os
import numpy as np
import PIL.Image
import cv2
import torch
import torchvision.transforms as tvf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class _FastImgNorm:
    """Drop-in replacement for ``Compose([ToTensor(), Normalize(ImageNet)])``.

    Per-sample profiling on BLIP3o JPEG (~48KB) showed torchvision's
    ToTensor+Normalize chain costing ~2.4ms — about 37% of the entire
    dataloader transform — because each step allocates an intermediate
    tensor. We collapse the four passes into a single fused numpy
    expression and convert once at the end. Measured speedup: ~5x per
    worker (43 → 221 samp/s on a single jpeg stream), with byte-level
    output equivalence (L_inf < 1e-6 vs the original Compose).

    Only RGB ``PIL.Image`` inputs use the fast path; anything else falls
    back to the original torchvision Compose to preserve exact behavior
    on grayscale / RGBA / non-PIL callers.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    _INV_255 = np.float32(1.0 / 255.0)

    def __init__(self) -> None:
        self._fallback = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._inv_std = (1.0 / self._STD).astype(np.float32)

    def __call__(self, img):
        if not (isinstance(img, PIL.Image.Image) and img.mode == "RGB"):
            return self._fallback(img)
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr * self._INV_255 - self._MEAN) * self._inv_std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


# Standard image normalization (ImageNet stats). Backed by a fused numpy
# implementation; see ``_FastImgNorm`` above. Public name is preserved so
# every existing import of ``ImgNorm`` keeps working unchanged.
ImgNorm = _FastImgNorm()


def imgnorm_to_unit(images: torch.Tensor) -> torch.Tensor:
    """Inverse of :data:`ImgNorm` — ImageNet-normalized tensor to ``[0, 1]``.

    Two pixel conventions coexist in this repo: multi-view batches reach the
    trainers as ``[0, 1]`` (``cut3r_adapter`` denormalizes ``gt_inp``), while
    T2I loaders hand over ``ImgNorm`` output directly. Codecs that apply their
    own ImageNet normalization, and the SD/Wan VAEs that want ``[-1, 1]``, need
    the ``[0, 1]`` form, so T2I callers convert here instead of each one
    reinventing the constants.

    Literal ImageNet constants on purpose: ``rae`` is absent on the SD/Wan
    backends and carries identity buffers on the RAEv2-DINOv3 wrapper.
    """
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def center_crop_resize_to(img: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
    """Center-crop to *width*/*height* aspect ratio, then resize to (W, H).

    Square targets delegate to :func:`short_side_center_crop_resize` so DA3
    square inputs stay on the same path as before. Non-square targets match
    geometry multi-view ``resolution=[(504,504), (504,378), ...]`` without
    stretching non-square sources.
    """
    target_w, target_h = int(width), int(height)
    if target_w == target_h:
        return short_side_center_crop_resize(img, target_w)

    src_w, src_h = img.size
    target_ar = target_w / target_h
    src_ar = src_w / src_h
    if src_ar > target_ar:
        new_w = int(round(src_h * target_ar))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(round(src_w / target_ar))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), PIL.Image.BICUBIC)
    return img


def short_side_center_crop_resize(img: PIL.Image.Image, resolution: int) -> PIL.Image.Image:
    """Resize so shorter side == resolution, then center-crop to square.

    Matches the DA3 / ImageNet preprocessing convention (square inputs at
    252 / 504 / etc.). Use this everywhere DA3 features are extracted, so
    inputs always live in the distribution DA3 was trained on. Direct
    ``img.resize((R, R))`` instead would stretch non-square sources
    (e.g. 4:3 photos in BLIP3o-Short) and pull DA3 off-distribution.
    """
    w, h = img.size
    s = min(w, h)
    if (w, h) != (s, s):
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), PIL.Image.BICUBIC)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
