"""BLIP3o data loader on top of ``s3torchconnector.S3IterableDataset``.

Backend selected by ``--data-backend s3torch`` (paired with the existing
``wds`` backend in ``blip3o_wds.py``). Implemented per platform v1.4.0
S3 cookbook §6.3 "模式 1 流式读 S3" — recommended primary path for
container-side dataset I/O.

Compared with the ``wds + pipe:s5cmd cat`` backend:

  * **CRT (AWS Common Runtime) multi-threaded prefetch** instead of one
    s5cmd subprocess per worker. Removes the ~3 s shard-switch
    cold-start that benchmarks (2026-06-06) attribute to s5cmd start +
    DNS + S3 first-byte latency.
  * **``enable_sharding=True``** lets the library split S3 objects
    across (DDP rank × DataLoader worker) automatically — replaces the
    hand-written stratified strided ``_rank_shards`` in ``blip3o_wds``.
  * **No subprocess fork**: data flows in-process via Rust CRT bindings.

Output contract is identical to ``blip3o_wds.build_blip3o_wds_loader``:
``(image_tensor (B, 3, H, W) float32, list[str] captions)``.

Tar layout assumption: each ``.tar`` shard packs WebDataset-style
sample groups, i.e. files share a basename and differ only in extension
(``00000.jpg``, ``00000.txt``, ``00000.json``). We extract jpg/png/webp
+ txt; everything else is dropped.

Usage::

    from data.blip3o_s3torch import build_blip3o_s3torch_loader
    loader = build_blip3o_s3torch_loader(
        data_dir="s3://your-bucket/BLIP3o",
        splits=["journeydb", "short", "long"],
        image_size=252, batch_size=128, num_workers=4,
        world_size=32, rank=0,
    )

The trainer flips backends with ``dataset.backend`` in YAML or the
``--data-backend`` CLI flag (see the T2I co-train path in ``scripts/train/train_flow.py``).
"""
from __future__ import annotations

import io
import logging
import os
import tarfile
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

from cut3r_data.utils.image import ImgNorm, short_side_center_crop_resize

logger = logging.getLogger(__name__)


# Same split → S3 sub-directory mapping as the wds backend so config keys
# can be swapped backend-to-backend without touching downstream code.
GLD_BLIP3O_SUBDIRS = {
    "journeydb": "BLIP3o-Pretrain-JourneyDB",
    "short": "BLIP3o-Pretrain-Short-Caption",
    "long": "BLIP3o-Pretrain-Long-Caption",
    "short-caption": "BLIP3o-Pretrain-Short-Caption",
    "long-caption": "BLIP3o-Pretrain-Long-Caption",
}

BLIP3O_NUM_SAMPLES = {
    "journeydb": 4_280_000,
    "short": 4_770_000,
    "long": 27_200_000,
    "short-caption": 4_770_000,
    "long-caption": 27_200_000,
}

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}


class _GLDDefaultTransform:
    """Top-level callable so DataLoader spawn workers can pickle it."""

    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = short_side_center_crop_resize(img, self.size)
        return ImgNorm(img)


@dataclass
class _SampleParts:
    """Accumulator for the (jpg, txt, ...) files belonging to one sample
    inside a single tar. WebDataset groups by the leading basename
    (everything before the first dot)."""
    base: str
    image_bytes: Optional[bytes] = None
    image_ext: Optional[str] = None
    caption: Optional[str] = None

    def is_complete(self) -> bool:
        return self.image_bytes is not None and self.caption is not None


def _stream_tar_to_samples(
    fileobj: io.IOBase,
    *,
    url_for_log: str = "<s3>",
) -> Iterator[Tuple[bytes, str, str]]:
    """Yield ``(image_bytes, image_ext, caption_str)`` tuples from a tar
    stream. Uses ``mode='r|*'`` so tarfile reads sequentially without
    seeking — critical for streaming readers where ``seek()`` either
    isn't supported or triggers a re-download.

    Robustness: any malformed sample (decode error / missing pair) is
    skipped with a debug log; we never raise out of the iterator
    because that would kill the whole shard.
    """
    current: Optional[_SampleParts] = None
    try:
        # mode='r|*' = streaming, auto-detect compression. tarfile will
        # call fileobj.read() repeatedly without seek().
        with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                if "." not in name:
                    continue
                base, ext = name.rsplit(".", 1)
                ext = ext.lower()
                # Sample boundary: WebDataset convention is that all
                # files for one sample share a basename. As soon as we
                # see a different basename, flush the previous one.
                if current is None or current.base != base:
                    if current is not None and current.is_complete():
                        yield current.image_bytes, current.image_ext, current.caption
                    current = _SampleParts(base=base)
                # Pull the member bytes. tf.extractfile returns a
                # FileLike that's only valid until the next member is
                # consumed, so we must read() now.
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if ext in _IMAGE_EXTS:
                    current.image_bytes = data
                    current.image_ext = ext
                elif ext == "txt":
                    try:
                        current.caption = data.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        current.caption = ""
                # other extensions (json, cls, ...) are ignored
            if current is not None and current.is_complete():
                yield current.image_bytes, current.image_ext, current.caption
    except (tarfile.TarError, OSError) as e:
        logger.warning("[s3torch] tar parse error in %s: %s", url_for_log, e)


class BLIP3OS3TorchDataset(IterableDataset):
    """IterableDataset over BLIP3o tar shards on S3, backed by
    ``s3torchconnector.S3IterableDataset`` (CRT prefetch).

    Concurrency model:
      * Outer ``S3IterableDataset(enable_sharding=True)`` distributes
        S3 objects across (rank × worker) automatically — relies on
        DDP env vars (``RANK`` / ``WORLD_SIZE``) being set before the
        DataLoader is created, plus ``num_workers`` being honored.
      * Inside each worker: streaming tar parse, with the CRT client
        already pre-fetching the next byte range while we decode the
        current one. There's no explicit shard prefetch in user code
        because the CRT client buffers internally.

    Ordering / shuffling:
      * S3IterableDataset has no shuffle; relative order is whatever
        S3 ListObjectsV2 returns (lexicographic). We rely on the
        ``shuffle_buffer`` reservoir below to randomize order at
        sample granularity, matching the wds backend's ``.shuffle()``.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        splits: Sequence[str],
        region: str = "us-west-2",
        image_size: int = 252,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        super().__init__()
        if not data_dir.startswith("s3://"):
            raise ValueError(
                f"s3torch backend requires an s3:// data_dir, got {data_dir!r}"
            )
        self.data_dir = data_dir.rstrip("/")
        self.splits = list(splits)
        self.region = region
        self.image_size = int(image_size)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.transform = transform or _GLDDefaultTransform(self.image_size)

        # Validate splits up-front so misconfigs fail at construction,
        # not 5 minutes into training.
        for sp in self.splits:
            if sp not in GLD_BLIP3O_SUBDIRS:
                raise ValueError(
                    f"Unknown BLIP3o split {sp!r}; valid keys: "
                    f"{sorted(set(GLD_BLIP3O_SUBDIRS))}"
                )

        # Per-split S3 prefixes. We open one S3IterableDataset per
        # prefix and chain them — chaining preserves
        # enable_sharding semantics because each child sharder sees
        # its own object list.
        self._prefixes = [
            f"{self.data_dir}/{GLD_BLIP3O_SUBDIRS[sp]}/" for sp in self.splits
        ]
        self._estimated_size = sum(
            BLIP3O_NUM_SAMPLES.get(sp, 1_000_000) for sp in self.splits
        )
        logger.info(
            "BLIP3OS3TorchDataset: %d split(s) %s, estimated %d samples",
            len(self._prefixes), self.splits, self._estimated_size,
        )

    @property
    def estimated_size(self) -> int:
        return self._estimated_size

    # ----- shard iteration ------------------------------------------------
    def _build_inner_datasets(self):
        """Construct the s3torchconnector iterables. Imported lazily so
        the module loads even when the package isn't installed (lets us
        keep the wds backend usable in environments without
        ``s3torchconnector``)."""
        import s3torchconnector as s3t

        inners = []
        for prefix in self._prefixes:
            inners.append(
                s3t.S3IterableDataset.from_prefix(
                    prefix,
                    region=self.region,
                    enable_sharding=True,
                )
            )
        return inners

    @staticmethod
    def _is_tar_object(obj) -> bool:
        # S3Reader has .key (full key minus bucket). Cheap path-only
        # filter: skip HuggingFace cache, gitattributes, etc.
        key = obj.key
        return key.endswith(".tar") and "/.cache/" not in key

    # ----- main iterator --------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        inners = self._build_inner_datasets()

        # Per-worker shuffle buffer keyed by (worker_id, epoch-ish seed)
        # so two workers don't produce identical orders.
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        worker_seed = self.seed + worker_id * 1009 + os.getpid()

        import random
        rng = random.Random(worker_seed)
        buffer: List[Tuple[torch.Tensor, str]] = []
        target_buf = max(1, self.shuffle_buffer)

        def _decode(image_bytes: bytes, caption: str) -> Optional[Tuple[torch.Tensor, str]]:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensor = self.transform(img)
            except Exception as e:
                logger.debug("[s3torch] image decode skipped: %s", e)
                return None
            return tensor, caption

        def _sample_stream() -> Iterator[Tuple[torch.Tensor, str]]:
            for inner in inners:
                for obj in inner:
                    if not self._is_tar_object(obj):
                        continue
                    for img_bytes, _ext, caption in _stream_tar_to_samples(
                        obj, url_for_log=obj.key
                    ):
                        decoded = _decode(img_bytes, caption)
                        if decoded is not None:
                            yield decoded

        for sample in _sample_stream():
            if len(buffer) < target_buf:
                buffer.append(sample)
                if len(buffer) < target_buf:
                    continue
                # First-fill done — shuffle once so the initial drain
                # isn't FIFO.
                rng.shuffle(buffer)
            # Reservoir: replace a random slot, return the displaced
            # sample. Keeps buffer full and order stochastic without
            # holding 2× memory.
            i = rng.randrange(len(buffer))
            out = buffer[i]
            buffer[i] = sample
            yield out

        # Drain whatever's left after the source is exhausted.
        rng.shuffle(buffer)
        for s in buffer:
            yield s


def _collate(batch):
    """Custom collate: stack image tensors, keep captions as a list[str]
    (matches wds backend's output contract)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions


class _S3TorchLoaderWrapper:
    """``set_epoch``/``__len__``-compatible wrapper. Mirrors the API of
    ``blip3o_wds._WDSLoaderWrapper`` so the trainer can swap backends
    without branching."""

    def __init__(
        self,
        dataset: BLIP3OS3TorchDataset,
        *,
        batch_size: int,
        num_workers: int,
        steps_per_epoch: int,
        pin_memory: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.steps_per_epoch = steps_per_epoch
        self.pin_memory = pin_memory
        self._epoch = 0
        self._loader = self._build_loader()

    def _build_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
            persistent_workers=self.num_workers > 0,
            # We do shuffling inside the dataset (per-worker reservoir);
            # IterableDataset doesn't accept shuffle=True anyway.
        )

    def set_epoch(self, epoch: int) -> None:
        # Reseed the dataset so next iterator gives a different order,
        # then rebuild the loader (cheap; persistent workers will
        # reload). With persistent_workers=True PyTorch reuses worker
        # processes, but a fresh __iter__ inside each worker still
        # creates a new shuffle rng + S3IterableDataset instance.
        self._epoch = int(epoch)
        self.dataset.seed = self.dataset.seed + 1000003 * self._epoch
        self._loader = self._build_loader()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        # Truncate to steps_per_epoch so the trainer's progress bar /
        # LR schedule references are well-defined; the underlying
        # dataset is essentially infinite.
        it = iter(self._loader)
        for _ in range(self.steps_per_epoch):
            yield next(it)


def build_blip3o_s3torch_loader(
    data_dir: str,
    splits: Union[str, Sequence[str]],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int = 0,
    region: str = "us-west-2",
    virtual_epoch_steps: Optional[int] = None,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    pin_memory: bool = False,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> _S3TorchLoaderWrapper:
    """Public factory. Mirrors ``build_blip3o_wds_loader`` signature.

    ``world_size``/``rank`` are NOT used for manual sharding here:
    s3torchconnector reads them from ``RANK``/``WORLD_SIZE`` env vars
    (set by torchrun) when ``enable_sharding=True``. We still accept
    them in the signature so the trainer can stay backend-agnostic;
    any deviation between caller-provided rank and the env-var rank is
    a configuration bug we'd want surfaced loudly:
    """
    if isinstance(splits, str):
        splits = [splits]
    splits = list(splits)

    # Surface DDP/env mismatches early — ``enable_sharding`` reads from
    # env, so a wrong env would silently produce duplicated data
    # without any error. Better to assert.
    env_rank = int(os.environ.get("RANK", "0"))
    env_ws = int(os.environ.get("WORLD_SIZE", "1"))
    if (env_rank, env_ws) != (rank, world_size):
        logger.warning(
            "[s3torch] caller rank/world_size (%d/%d) != env (%d/%d); "
            "s3torchconnector enable_sharding uses env values.",
            rank, world_size, env_rank, env_ws,
        )

    ds = BLIP3OS3TorchDataset(
        data_dir=data_dir,
        splits=splits,
        region=region,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        transform=transform,
    )

    if virtual_epoch_steps is None:
        virtual_epoch_steps = max(1, ds.estimated_size // (batch_size * world_size))

    return _S3TorchLoaderWrapper(
        dataset=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        steps_per_epoch=virtual_epoch_steps,
        pin_memory=pin_memory,
    )
