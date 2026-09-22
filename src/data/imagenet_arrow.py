"""ImageNet HF Arrow loader for single-image T2I / Feature-VAE co-training."""

from __future__ import annotations

import importlib.util
import io
import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from cut3r_data.utils.image import ImgNorm, center_crop_resize_to
from cut3r_data.utils.resolution import parse_resolution_list, pick_resolution_index

logger = logging.getLogger(__name__)

IMAGENET_ARROW_NUM_SAMPLES = {
    "train": 1_281_167,
    "validation": 50_000,
    "test": 100_000,
}


def _load_imagenet_classes() -> list[str]:
    repo = Path(__file__).resolve().parents[2]
    candidates = [
        repo / "RAEv2" / "src" / "data" / "imagenet_classes.py",
        repo / "src" / "data" / "imagenet_classes.py",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("imagenet_classes", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.IMAGENET_CLASSES)
    from torchvision.datasets import ImageNet  # type: ignore

    return list(ImageNet.classes)


class _ImageNetArrowTransform:
    __slots__ = ("width", "height")

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = center_crop_resize_to(img, self.width, self.height)
        return ImgNorm(img)


class _ImageNetArrowUnitRangeTransform:
    __slots__ = ("width", "height", "_to_tensor")

    def __init__(self, width: int, height: int):
        from torchvision import transforms

        self.width = int(width)
        self.height = int(height)
        self._to_tensor = transforms.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = center_crop_resize_to(img, self.width, self.height)
        return self._to_tensor(img)


class ImageNetArrowIterableDataset(IterableDataset):
    """Iterate HF ImageNet Arrow shards as ``(image_tensor, caption)`` samples."""

    def __init__(
        self,
        *,
        root: str,
        split: str = "train",
        image_size: int = 504,
        resolutions: Sequence[tuple[int, int]] | None = None,
        batch_size: int = 1,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 2000,
        prompt_template: str | None = None,
        pixel_norm: str = "imagenet",
    ):
        self.root = Path(root)
        self.split = str(split)
        self.resolutions = parse_resolution_list(resolutions, image_size=image_size)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.prompt_template = prompt_template
        self.pixel_norm = str(pixel_norm).lower()
        if self.pixel_norm not in ("imagenet", "unit"):
            raise ValueError(f"pixel_norm must be 'imagenet' or 'unit'; got {pixel_norm!r}")
        self.epoch = 0
        self._class_names = _load_imagenet_classes() if prompt_template else None

        pattern = f"imagenet-1k-{self.split}-*.arrow"
        self.shards = sorted(self.root.glob(pattern))
        if not self.shards:
            raise RuntimeError(f"No ImageNet Arrow shards found: {self.root}/{pattern}")

        self.num_examples = self._estimate_size()
        res_str = ";".join(f"{w}x{h}" for w, h in self.resolutions)
        logger.info(
            "ImageNetArrow: %d shards split=%s root=%s ~%d samples resolutions=[%s]",
            len(self.shards),
            self.split,
            self.root,
            self.num_examples,
            res_str,
        )

    def _estimate_size(self) -> int:
        return IMAGENET_ARROW_NUM_SAMPLES.get(self.split, len(self.shards) * 4096)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_worker_shards(self) -> list[Path]:
        shards = list(self.shards)
        if self.shuffle_shards:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(shards)
        shards = shards[self.rank :: self.world_size]

        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        return shards

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + self.epoch * 7919 + worker_id)

        def raw_samples():
            if self.shuffle_buffer <= 1:
                yield from self._iter_raw_samples()
                return

            buffer: list[tuple[Image.Image, int | None]] = []
            for sample in self._iter_raw_samples():
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(sample)
                    continue
                j = rng.randrange(self.shuffle_buffer)
                out, buffer[j] = buffer[j], sample
                yield out
            rng.shuffle(buffer)
            yield from buffer

        def transformed_samples():
            samples_in_batch = 0
            current_transform: _ImageNetArrowTransform | _ImageNetArrowUnitRangeTransform | None = None
            transform_cls = (
                _ImageNetArrowUnitRangeTransform
                if self.pixel_norm == "unit"
                else _ImageNetArrowTransform
            )
            for img, label in raw_samples():
                if samples_in_batch == 0:
                    ar_idx = pick_resolution_index(rng, len(self.resolutions))
                    w, h = self.resolutions[ar_idx]
                    current_transform = transform_cls(w, h)
                assert current_transform is not None
                caption = ""
                if self.prompt_template and label is not None and self._class_names:
                    caption = self.prompt_template.format(class_name=self._class_names[label])
                yield current_transform(img), caption
                samples_in_batch += 1
                if samples_in_batch >= self.batch_size:
                    samples_in_batch = 0

        yield from transformed_samples()

    def _iter_raw_samples(self):
        for shard in self._rank_worker_shards():
            try:
                with pa.memory_map(str(shard), "r") as source:
                    reader = ipc.open_stream(source)
                    for batch in reader:
                        image_idx = batch.schema.get_field_index("image")
                        label_idx = batch.schema.get_field_index("label")
                        image_col = batch.column(image_idx)
                        label_col = (
                            batch.column(label_idx) if label_idx >= 0 else None
                        )
                        for idx in range(batch.num_rows):
                            obj = image_col[idx].as_py()
                            image_bytes = (
                                obj.get("bytes") if isinstance(obj, dict) else None
                            )
                            if not image_bytes:
                                continue
                            label = (
                                int(label_col[idx].as_py())
                                if label_col is not None
                                else None
                            )
                            try:
                                img = Image.open(io.BytesIO(image_bytes))
                            except Exception:
                                continue
                            yield img, label
            except Exception as e:
                logger.warning(
                    "Skipping unreadable ImageNet arrow shard %s: %s", shard, e
                )
                continue


class _ArrowLoaderWrapper:
    def __init__(
        self,
        dataset: ImageNetArrowIterableDataset,
        *,
        batch_size: int,
        num_workers: int,
        steps_per_epoch: int,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.steps_per_epoch = int(steps_per_epoch)
        self.pin_memory = bool(pin_memory)
        self._loader = self._build_loader()

    def _build_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self._loader = self._build_loader()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        it = iter(self._loader)
        for _ in range(self.steps_per_epoch):
            try:
                yield next(it)
            except StopIteration:
                self.dataset.set_epoch(self.dataset.epoch + 1)
                self._loader = self._build_loader()
                it = iter(self._loader)
                yield next(it)


def build_imagenet_arrow_loader(
    root: str,
    *,
    image_size: int | None = 504,
    resolutions: Sequence[tuple[int, int]] | None = None,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int = 0,
    split: str = "train",
    virtual_epoch_steps: Optional[int] = None,
    seed: int = 0,
    shuffle_buffer: int = 2000,
    prompt_template: str | None = None,
    pixel_norm: str = "imagenet",
) -> _ArrowLoaderWrapper:
    dataset = ImageNetArrowIterableDataset(
        root=root,
        split=split,
        image_size=504 if image_size is None else image_size,
        resolutions=resolutions,
        batch_size=batch_size,
        seed=seed,
        rank=rank,
        world_size=world_size,
        shuffle_buffer=shuffle_buffer,
        prompt_template=prompt_template,
        pixel_norm=pixel_norm,
    )
    if virtual_epoch_steps is None:
        virtual_epoch_steps = max(1, dataset.num_examples // (batch_size * world_size))
    return _ArrowLoaderWrapper(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        steps_per_epoch=virtual_epoch_steps,
    )


class MixedT2ILoader:
    """Mix multiple T2I loaders according to integer weights."""

    def __init__(self, loaders: Sequence[object], weights: Sequence[int], seed: int = 0):
        if len(loaders) != len(weights) or not loaders:
            raise ValueError("MixedT2ILoader requires matching non-empty loaders/weights")
        self.loaders = list(loaders)
        self.weights = [max(1, int(w)) for w in weights]
        self.seed = int(seed)
        self.epoch = 0
        self.steps_per_epoch = sum(len(loader) * w for loader, w in zip(self.loaders, self.weights))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for loader in self.loaders:
            loader.set_epoch(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        iters = [iter(loader) for loader in self.loaders]
        choices = [idx for idx, w in enumerate(self.weights) for _ in range(w)]
        for _ in range(self.steps_per_epoch):
            idx = rng.choice(choices)
            try:
                yield next(iters[idx])
            except StopIteration:
                self.loaders[idx].set_epoch(self.epoch + 1)
                iters[idx] = iter(self.loaders[idx])
                try:
                    yield next(iters[idx])
                except StopIteration:
                    continue
