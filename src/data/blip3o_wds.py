"""BLIP3o WebDataset loader for T2I training (RAEv2 recipe).

Adapted from RAEv2's ``src/data/blip3o_wds_dataset.py``. Differences vs
upstream:

  * Maps the three split names (``journeydb`` / ``short`` / ``long``) to
    GLD's actual on-disk directories — ``BLIP3o-Pretrain-{JourneyDB,
    Short-Caption,Long-Caption}``.
  * Default resize to 252 (GLD latent grid 252 / 14 = 18) instead of 256.
  * Image preprocessing matches ``scripts/precompute_t2i_latents.py``
    EXACTLY (``short_side_center_crop_resize`` + ImageNet ``ImgNorm``)
    so the pre-computed latent_stats whitening file remains valid when
    the trainer runs DA3+FeatureVAE on the fly.
  * Returns ``(image_tensor, caption_str)`` so the trainer's online
    encode pipeline (DA3 + FeatureVAE for image, Qwen3 for text)
    consumes directly.
  * **Direct-from-S3 streaming via ``pipe:s5cmd cat``** — ``data_dir``
    starting with ``s3://`` lists shards through ``s5cmd ls`` and
    constructs URLs of the form
    ``pipe:s5cmd cat s3://bucket/key.tar`` so WDS reads the tar stream
    out of s5cmd's stdout. No FUSE, no /local-ssd staging, ~70
    samples/s/worker measured on this cluster (4 workers ⇒ ~280
    samples/s ⇒ 2.2 batch/s at batch=128 — completely overlaps GPU
    forward).

The trainer drives epoch transitions by calling ``set_epoch(e)`` on
the returned loader wrapper which recreates the WDS pipeline with a
new seed.
"""
from __future__ import annotations

import logging
import math
import os
import random
import shlex
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import webdataset as wds
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from cut3r_data.utils.image import ImgNorm, center_crop_resize_to
from cut3r_data.utils.resolution import parse_resolution_list, pick_resolution_index

logger = logging.getLogger(__name__)


# Maps user-facing split key → on-disk / S3 sub-directory.
GLD_BLIP3O_SUBDIRS = {
    "journeydb": "BLIP3o-Pretrain-JourneyDB",
    "short": "BLIP3o-Pretrain-Short-Caption",
    "long": "BLIP3o-Pretrain-Long-Caption",
    # ImageNet captions packed by scripts/pack_imagenet_t2i_wds.py.
    # Keep the tar layout identical to BLIP3o: <key>.jpg + <key>.txt.
    "imagenet": "ImageNet-1K-T2I",
    # ImageNet-1k class-name captions (scripts/pack_imagenet_classname_wds.py).
    "imagenet_class": "ImageNet-1K-ClassName",
    "short-caption": "BLIP3o-Pretrain-Short-Caption",
    "long-caption": "BLIP3o-Pretrain-Long-Caption",
    # Video-frame t2i shards produced by scripts/caption_scannetpp_clips.py
    # (--wds-out, --dataset-name). Same tar layout (<key>.jpg + <key>.txt) so
    # they stream through the exact same pipeline and can be freely mixed into
    # ``splits`` alongside BLIP3o and each other.
    "scannetpp": "BLIP3o-Pretrain-ScanNetpp",
    "dl3dv": "BLIP3o-Pretrain-DL3DV",
    "re10k": "BLIP3o-Pretrain-RE10K",
    "mvssynth": "BLIP3o-Pretrain-MVSSynth",
    # v2 dual-caption shards produced by scripts/caption_dual_v2.py. Tar
    # layout is <key>.jpg + <key>.json with json={"long":..., "short":...};
    # _decode_sample reads .json and picks long vs short per sample by
    # dual_caption_prob (see BLIP3OWebDataset).
    "scannetpp-v2": "BLIP3o-Pretrain-ScanNetpp-v2",
    "dl3dv-v2": "BLIP3o-Pretrain-DL3DV-v2",
    "re10k-v2": "BLIP3o-Pretrain-RE10K-v2",
    "mvssynth-v2": "BLIP3o-Pretrain-MVSSynth-v2",
    # i1 (zlab-princeton/i1) T2I sources packed by
    # scripts/pack_i1_to_blip3o_wds.py. Tar layout is <key>.jpg + <key>.json
    # where json={"captions": [c1, ...]}; _caption_from_sample picks one
    # caption uniformly at random per sample (matches i1's 1-of-N sampling).
    "i1-fluxreason": "i1-fluxreason",
    "i1-rendered_text": "i1-rendered_text",
    "i1-gptedit": "i1-gptedit",
    "i1-textatlas": "i1-textatlas",
    "i1-pexels": "i1-pexels",
    "i1-imagenet22k": "i1-imagenet22k",
    "i1-midjourneyv6": "i1-midjourneyv6",
    "i1-megalith10m": "i1-megalith10m",
    "i1-inaturalist": "i1-inaturalist",
    "i1-places365": "i1-places365",
    "i1-redcaps": "i1-redcaps",
    "i1-yfcc": "i1-yfcc",
}

# Approximate sample counts per split — used only to derive a "virtual
# epoch" step count. WDS itself is infinite under ``with_epoch``;
# getting these wrong only shifts the LR schedule's reference epoch
# length, not training validity.
BLIP3O_NUM_SAMPLES = {
    "journeydb": 4_280_000,
    "short": 4_770_000,
    "long": 27_200_000,
    "short-caption": 4_770_000,
    "long-caption": 27_200_000,
    # Video-frame t2i splits. These priors only size the virtual-epoch step
    # count when one is the sole split; the packer prints the exact image count
    # (= clips × frames_per_clip). Values below assume ~1 frame/clip:
    #   scannetpp ~1951 clips · dl3dv ~6997 · re10k ~74k · mvssynth 120.
    "scannetpp": 2_000,
    "dl3dv": 7_000,
    "re10k": 74_000,
    "mvssynth": 120,
    # v2 shards cover the same clip lists as v1 (× frames_per_clip), so the
    # estimates carry over unchanged.
    "scannetpp-v2": 2_000,
    "dl3dv-v2": 7_000,
    "re10k-v2": 74_000,
    "mvssynth-v2": 120,
    "imagenet": 1_280_000,
    "imagenet_class": 1_280_000,
    # i1 sources — full row counts from the i1-captions dataset card (upper
    # bounds; actual packed counts are lower after image decode/join drops).
    # Only used to size the virtual-epoch step count.
    "i1-fluxreason": 5_890_279,
    "i1-rendered_text": 11_977_816,
    "i1-gptedit": 1_553_575,
    "i1-textatlas": 5_396_890,
    "i1-pexels": 2_810_634,
    "i1-imagenet22k": 13_673_544,
    "i1-midjourneyv6": 1_240_185,
    "i1-megalith10m": 9_393_971,
    "i1-inaturalist": 4_813_543,
    "i1-places365": 7_221_597,
    "i1-redcaps": 4_817_431,
    "i1-yfcc": 97_945_286,
}


def _filter_valid_samples(sample):
    """Drop tuples whose first element (image) is None — required for
    ``wds.select`` because lambdas cannot be pickled across spawn workers."""
    return sample[0] is not None


def _identity_nodesplitter(urls):
    """Module-level no-op nodesplitter for ``wds.WebDataset``. We split
    by rank manually in :meth:`BLIP3OWebDataset._rank_shards`, so WDS's
    own node splitter must be a no-op."""
    return urls


def _filter_valid_batch(sample):
    images, _captions = sample
    return images is not None and len(images) > 0 and images[0] is not None


class _GLDDefaultTransform:
    """Top-level callable so DataLoader spawn workers can pickle it.
    Closures defined inside ``__init__`` would fail with
    ``AttributeError: Can't pickle local object`` on the first
    spawn-mode multi-worker iteration."""

    __slots__ = ("width", "height")

    def __init__(self, width: int, height: int | None = None):
        self.width = int(width)
        self.height = int(self.width if height is None else height)

    def __call__(self, img):
        img = center_crop_resize_to(img, self.width, self.height)
        return ImgNorm(img)


def _is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def _list_s3_tars(s3_dir: str) -> List[str]:
    """Return absolute ``s3://bucket/key.tar`` URIs for every tar under
    ``s3_dir`` (non-recursive). Uses ``s5cmd ls`` because boto3 isn't
    a hard dependency of this codebase and s5cmd is already required
    for the staging script."""
    s3_dir = s3_dir.rstrip("/") + "/"
    cmd = ["s5cmd", "ls", f"{s3_dir}*.tar"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(
            "s5cmd not found on PATH; required for s3:// data_dir mode"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"s5cmd ls failed for {s3_dir}: {e.stderr or e.stdout}"
        ) from e
    uris = []
    for line in out.splitlines():
        # ``2026/04/23 16:43:47   455290880  00000.tar``
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if name.endswith(".tar"):
            uris.append(s3_dir + name)
    return sorted(uris)


def _to_pipe_url(uri: str) -> str:
    """Convert a shard URI to the URL form WDS will open. Local paths
    pass through unchanged; ``s3://`` URIs become ``pipe:s5cmd cat ...``
    so WDS reads from s5cmd stdout (no FUSE, no staging).

    Robustness wrapping for the s3 case (added after observing watchdog
    timeouts where rank 0/2 stalled 4+ min in dataloader forward):

      1. ``timeout ${BLIP3O_S5CMD_TIMEOUT_SEC}`` — outer hard kill on the
         s5cmd subprocess if it hangs (e.g. one TCP connection wedges and
         retries never make progress). Default 300s; a 7.5 GB JourneyDB
         shard at 25 MB/s = 300s, so even at degraded throughput we don't
         time out under normal conditions, but a true hang gets killed
         well before the 15 min NCCL watchdog fires.

      2. ``s5cmd --retry-count 10 --log error cat`` — s5cmd's internal
         retry covers short transient errors (TCP RST, DNS, throttling)
         without re-spawning the whole subprocess. ``--log error``
         suppresses per-request INFO noise.

    When timeout/kill fires, ``wds.warn_and_continue`` handler skips the
    shard and the loader moves on — losing ~10K samples per skipped
    shard but keeping training alive.
    """
    if _is_s3_uri(uri):
        timeout_s = int(os.environ.get("BLIP3O_S5CMD_TIMEOUT_SEC", "300"))
        retries = int(os.environ.get("BLIP3O_S5CMD_RETRIES", "10"))
        return (
            f"pipe:timeout {timeout_s} "
            f"s5cmd --retry-count {retries} --log error "
            f"cat {shlex.quote(uri)}"
        )
    return uri


def _find_split_tars(root: str, subdir: str) -> List[str]:
    """Return tar URIs for ``<root>/<subdir>`` or ``[]`` if absent here.

    Works for both an ``s3://`` root (``s5cmd ls``) and a local/FUSE root
    (``glob('*.tar')``). Returns empty (not raise) when the subdir simply
    isn't under this root, so a caller can fall through to the next root.
    """
    if _is_s3_uri(root):
        split_dir = f"{root.rstrip('/')}/{subdir}"
        try:
            return _list_s3_tars(split_dir)
        except RuntimeError:
            return []
    split_dir = Path(root) / subdir
    if not split_dir.exists():
        return []
    tars = sorted(split_dir.glob("*.tar"))
    if subdir == "ImageNet-1K-T2I" or subdir == "ImageNet-1K-ClassName":
        tars = [p for p in tars if Path(str(p) + ".done").is_file()]
    return [str(p) for p in tars]


class _ResampledSplitShards(IterableDataset):
    """Infinite, deterministic-per-worker shard stream for one split.

    Split selection happens later in ``wds.RandomMix`` at *sample* granularity.
    Resampling shards here only makes every source infinite, including tiny
    geometry sources whose shard count is smaller than world size/worker count.
    """

    def __init__(
        self,
        urls: Sequence[str],
        *,
        seed: int,
        rank: int,
        split_index: int,
    ):
        super().__init__()
        if not urls:
            raise ValueError("_ResampledSplitShards requires at least one shard")
        self.urls = list(urls)
        self.seed = int(seed)
        self.rank = int(rank)
        self.split_index = int(split_index)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(
            self.seed
            + 1_000_003 * self.rank
            + 10_007 * worker_id
            + 104_729 * self.split_index
        )
        while True:
            yield {"url": _to_pipe_url(rng.choice(self.urls))}


class BLIP3OWebDataset:
    """Streaming WDS reader over one or more BLIP3o splits.

    ``data_dir`` may be:
      * a local filesystem path (FUSE mount or pre-staged copy on
        /local-ssd) — shards are discovered via ``glob('*.tar')``;
      * an ``s3://bucket/prefix`` URI — shards are discovered via
        ``s5cmd ls`` and read inline through ``pipe:s5cmd cat``;
      * a LIST of the above — each split's subdir is resolved by
        searching the roots in order (first hit wins). This lets one
        config read BLIP3o together with the Geometry video-frame
        splits when they live under different roots and the
        mountpoint-s3 FUSE forbids symlinking them into one dir.
    """

    def __init__(
        self,
        data_dir: Union[str, Sequence[str]],
        splits: Union[str, Sequence[str]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 252,
        resolutions: Optional[Sequence[tuple[int, int]]] = None,
        batch_size: int = 1,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        dual_caption_prob: float = 0.5,
        split_weights: Optional[Mapping[str, int]] = None,
        split_probs: Optional[Mapping[str, float]] = None,
    ):
        roots = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.data_dirs = [str(r).rstrip("/") for r in roots]
        self.data_dir = self.data_dirs[0]  # back-compat for callers/logging
        self.splits = [splits] if isinstance(splits, str) else list(splits)
        self.transform = transform
        self.resolutions = parse_resolution_list(resolutions, image_size=image_size)
        self.image_size = self.resolutions[0][0]
        self.batch_size = max(1, int(batch_size))
        self.mixed_resolution = len(self.resolutions) > 1
        self.shuffle_buffer = shuffle_buffer
        self._pipeline_epoch = 0
        self._resolution_rng: Optional[random.Random] = None
        self._resolution_rng_epoch: Optional[int] = None
        self.seed = seed
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self._is_s3 = any(_is_s3_uri(r) for r in self.data_dirs)
        # P(use long caption) on v2 dual-caption splits (those tar files carry
        # both a 60-120 word "long" and an 8-25 word "short" caption in a
        # single .json entry). 0.5 → 50/50 mix per sample, matching the
        # natural BLIP3o-Long / BLIP3o-Short balance. Ignored by v1 splits.
        self.dual_caption_prob = float(dual_caption_prob)
        # Per-worker RNG for dual-caption sampling. Lazy-initialised in
        # _decode_sample so DataLoader fork workers each get a distinct
        # PID-seeded stream (so two workers don't return the same long/short
        # choice for the same key, defeating the mix).
        self._dual_rng: Optional[random.Random] = None

        self._shard_uris: List[str] = []
        self._split_shard_uris: dict[str, List[str]] = {}
        self._active_splits: List[str] = []
        self._total_samples = 0
        split_weights = split_weights or {}
        if split_probs is not None and split_weights:
            raise ValueError("split_probs and split_weights are mutually exclusive")
        for sp in self.splits:
            subdir = GLD_BLIP3O_SUBDIRS.get(sp)
            if subdir is None:
                raise ValueError(
                    f"Unknown BLIP3o split {sp!r}; valid keys: "
                    f"{sorted(set(GLD_BLIP3O_SUBDIRS))}"
                )
            tars: List[str] = []
            for root in self.data_dirs:
                tars = _find_split_tars(root, subdir)
                if tars:
                    break
            if not tars:
                # i1-* splits are packed in stages (see pack_i1_to_blip3o_wds.py);
                # a listed-but-not-yet-packed i1 split should not crash training —
                # skip it with a warning until its shards land. Non-i1 splits are
                # expected to exist, so keep the hard error for those.
                if sp.startswith("i1-"):
                    print(
                        f"[blip3o] WARN: i1 split {sp!r} (subdir {subdir!r}) has no "
                        f"tar shards yet under {self.data_dirs}; skipping for now.",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"split {sp!r} (subdir {subdir!r}) has no tar shards under "
                    f"any root: {self.data_dirs}"
                )
            self._active_splits.append(sp)
            self._split_shard_uris[sp] = tars
            repeats = max(1, int(split_weights.get(sp, 1)))
            for _ in range(repeats):
                self._shard_uris.extend(tars)
            self._total_samples += repeats * BLIP3O_NUM_SAMPLES.get(sp, len(tars) * 3500)

        self._split_probs: Optional[dict[str, float]] = None
        if split_probs is not None:
            unknown = sorted(set(split_probs) - set(self.splits))
            missing = sorted(set(self._active_splits) - set(split_probs))
            if unknown or missing:
                raise ValueError(
                    "split_probs keys must match configured splits; "
                    f"unknown={unknown}, missing={missing}"
                )
            active_probs: dict[str, float] = {}
            for sp in self._active_splits:
                prob = float(split_probs[sp])
                if not math.isfinite(prob) or prob < 0.0:
                    raise ValueError(
                        f"split_probs[{sp!r}] must be finite and >=0, got {prob}"
                    )
                active_probs[sp] = prob
            prob_sum = sum(active_probs.values())
            if prob_sum <= 0.0:
                raise ValueError("split_probs must contain at least one positive value")
            self._split_probs = {
                sp: prob / prob_sum for sp, prob in active_probs.items() if prob > 0.0
            }

        self._num_shards = len(self._shard_uris)
        if self.transform is None and not self.mixed_resolution:
            w, h = self.resolutions[0]
            self.transform = _GLDDefaultTransform(w, h)
        _s3 = any(_is_s3_uri(r) for r in self.data_dirs)
        _local = any(not _is_s3_uri(r) for r in self.data_dirs)
        mode = "mixed" if (_s3 and _local) else ("s3-pipe" if _s3 else "local")
        res_str = ";".join(f"{w}x{h}" for w, h in self.resolutions)
        logger.info(
            "BLIP3OWebDataset: %d shards (%s), %s mode over %d root(s), ~%d samples "
            "resolutions=[%s]",
            self._num_shards,
            ", ".join(self.splits),
            mode,
            len(self.data_dirs),
            self._total_samples,
            res_str,
        )
        if self._split_probs is not None:
            logger.info(
                "BLIP3OWebDataset weighted per-sample mix: %s",
                ", ".join(
                    f"{sp}={100.0 * prob:.2f}%"
                    for sp, prob in self._split_probs.items()
                ),
            )

    @property
    def estimated_size(self) -> int:
        return self._total_samples

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def shard_uris(self) -> List[str]:
        return list(self._shard_uris)

    def _caption_from_sample(self, sample) -> str:
        cap_json = sample.get("json")
        if isinstance(cap_json, dict):
            if self._dual_rng is None:
                self._dual_rng = random.Random(
                    self.seed + self.rank * 100003 + os.getpid()
                )
            # i1-style multi-caption list: uniform random pick per sample.
            caps = cap_json.get("captions")
            if isinstance(caps, (list, tuple)):
                caps = [str(c).strip() for c in caps if c and str(c).strip()]
                if caps:
                    return self._dual_rng.choice(caps)
            long_c = (cap_json.get("long") or "").strip()
            short_c = (cap_json.get("short") or "").strip()
            if long_c and short_c:
                use_long = self._dual_rng.random() < self.dual_caption_prob
                return long_c if use_long else short_c
            return long_c or short_c
        caption_raw = sample.get("txt", b"")
        if isinstance(caption_raw, bytes):
            return caption_raw.decode("utf-8", errors="ignore").strip()
        return str(caption_raw).strip()

    def _decode_sample(self, sample, transform: Optional[_GLDDefaultTransform] = None):
        image = (
            sample.get("jpg")
            or sample.get("png")
            or sample.get("jpeg")
            or sample.get("webp")
        )
        caption = self._caption_from_sample(sample)
        active_transform = transform if transform is not None else self.transform
        if image is not None and active_transform is not None:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = active_transform(image)
        return image, caption

    def _decode_batch(self, batch):
        import torch

        if (
            self._resolution_rng is None
            or self._resolution_rng_epoch != self._pipeline_epoch
        ):
            self._resolution_rng = random.Random(
                self.seed
                + self._pipeline_epoch * 9973
                + self.rank * 100003
                + os.getpid()
            )
            self._resolution_rng_epoch = self._pipeline_epoch
        ar_idx = pick_resolution_index(self._resolution_rng, len(self.resolutions))
        w, h = self.resolutions[ar_idx]
        transform = _GLDDefaultTransform(w, h)
        images = []
        captions = []
        for sample in batch:
            image, caption = self._decode_sample(sample, transform=transform)
            if image is None:
                continue
            images.append(image)
            captions.append(caption)
        if not images:
            return None, [""]
        return torch.stack(images), captions

    def _rank_shards(self, epoch: int) -> List[str]:
        """Return WDS-ready URLs for this rank, deterministically
        shuffled by ``seed + epoch``. We split by rank ourselves rather
        than relying on ``wds.split_by_node`` so the loader behaves
        identically with or without DDP.

        Stratified strided assignment: split shards by source directory
        first, then take ``[rank::world_size]`` within each split.
        Rationale: BLIP3o mixes 3 splits with **very different shard
        sizes** — measured 2026-06-06: JourneyDB shards median 7.45 GB
        (n=419), Short-Caption 453 MB (n=601), Long-Caption 491 MB
        (n=2891). A single global stride still leaves some ranks with
        16 JDB shards and others with 7, producing 1.86x byte-volume
        skew → 2.4x throughput skew at training time.

        Stratified stride gives every rank ~13 JDB + ~19 Short + ~90
        Long shards, cutting per-rank byte-volume std/mean to <5%.
        """
        rng = random.Random(self.seed + epoch)
        # Bucket by source split (parent S3 prefix / local sub-dir).
        # Within each bucket we shuffle independently then stride —
        # this is what guarantees an even mix per rank regardless of
        # shard-size skew across splits.
        buckets: dict[str, List[str]] = {}
        for uri in self._shard_uris:
            key = uri.rsplit("/", 1)[0]
            buckets.setdefault(key, []).append(uri)

        my_shards: List[str] = []
        for key in sorted(buckets):  # deterministic ordering
            bucket = list(buckets[key])
            rng.shuffle(bucket)
            my_shards.extend(bucket[self.rank :: self.world_size])

        # Final shuffle so the per-epoch read order isn't always
        # "all JDB first, then Short, then Long" — keeps WDS's own
        # buffer well-mixed across splits.
        rng.shuffle(my_shards)
        return [_to_pipe_url(u) for u in my_shards]

    def create_pipeline(self, epoch: int = 0):
        self._pipeline_epoch = epoch
        if self._split_probs is not None:
            # Build one infinite stream per source and mix their decoded raw
            # samples with explicit probabilities. This makes probabilities
            # independent of shard count and average shard size.
            sources = []
            probs = []
            for split_index, (sp, prob) in enumerate(self._split_probs.items()):
                shard_source = _ResampledSplitShards(
                    self._split_shard_uris[sp],
                    seed=self.seed + 1_000_000_007 * epoch,
                    rank=self.rank,
                    split_index=split_index,
                )
                sources.append(wds.DataPipeline(
                    shard_source,
                    wds.tarfile_to_samples(handler=wds.warn_and_continue),
                ))
                probs.append(prob)

            stages = [wds.RandomMix(sources, probs=probs)]
            if self.shuffle_buffer > 0:
                stages.append(wds.shuffle(
                    self.shuffle_buffer,
                    initial=max(1, self.shuffle_buffer // 2),
                    seed=self.seed + epoch,
                ))
            stages.append(wds.decode("pil", handler=wds.warn_and_continue))
            if self.mixed_resolution:
                stages.extend([
                    wds.batched(
                        self.batch_size, collation_fn=list, partial=False,
                    ),
                    wds.map(self._decode_batch, handler=wds.warn_and_continue),
                    wds.select(_filter_valid_batch),
                ])
            else:
                stages.extend([
                    wds.map(self._decode_sample, handler=wds.warn_and_continue),
                    wds.select(_filter_valid_samples),
                ])
            return wds.DataPipeline(*stages)

        rank_shards = self._rank_shards(epoch)
        if not rank_shards:
            raise RuntimeError(
                f"rank {self.rank}/{self.world_size} got 0 shards "
                f"out of {len(self._shard_uris)}; reduce world_size or split count."
            )
        pipe = (
            wds.WebDataset(
                rank_shards,
                nodesplitter=_identity_nodesplitter,  # rank split done above
                shardshuffle=False,
                seed=self.seed + epoch,
                handler=wds.warn_and_continue,
            )
            .shuffle(self.shuffle_buffer, initial=max(1, self.shuffle_buffer // 2))
            .decode("pil", handler=wds.warn_and_continue)
        )
        if self.mixed_resolution:
            return (
                pipe
                .batched(self.batch_size, collation_fn=list, partial=False)
                .map(self._decode_batch, handler=wds.warn_and_continue)
                .select(_filter_valid_batch)
            )
        return (
            pipe
            .map(self._decode_sample, handler=wds.warn_and_continue)
            .select(_filter_valid_samples)
        )


class _WDSLoaderWrapper:
    """Thin wrapper exposing ``set_epoch`` + ``__len__`` to look like a
    map-style DataLoader. Recreates the WDS pipeline + WebLoader on
    epoch boundary to re-seed shuffling deterministically."""

    def __init__(
        self,
        pipeline: BLIP3OWebDataset,
        batch_size: int,
        num_workers: int,
        world_size: int,
        steps_per_epoch: int,
        pin_memory: bool = True,
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.steps_per_epoch = steps_per_epoch
        self.pin_memory = pin_memory
        self._loader = self._build_loader(epoch=0)

    def _build_loader(self, epoch: int):
        ds = self.pipeline.create_pipeline(epoch=epoch)
        loader_batch_size = None if self.pipeline.mixed_resolution else self.batch_size
        # Use fork (default) rather than spawn: spawn re-imports the
        # full project per worker (cut3r_data pulls cv2/numpy/etc.,
        # >60s cold-start), and workers never touch CUDA so fork is
        # safe.
        loader = wds.WebLoader(
            ds,
            batch_size=loader_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader.with_epoch(self.steps_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self._loader = self._build_loader(epoch=epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        return iter(self._loader)


def build_blip3o_wds_loader(
    data_dir: Union[str, Sequence[str]],
    splits: Union[str, Sequence[str]],
    *,
    image_size: int = 252,
    resolutions: Optional[Sequence[tuple[int, int]]] = None,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int = 0,
    virtual_epoch_steps: Optional[int] = None,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    transform: Optional[transforms.Compose] = None,
    dual_caption_prob: float = 0.5,
    split_weights: Optional[Mapping[str, int]] = None,
    split_probs: Optional[Mapping[str, float]] = None,
) -> _WDSLoaderWrapper:
    """Public factory used by the trainer.

    The loader yields ``(image_tensor (B, 3, H, W) float32 in [0,1],
    list[str] captions of length B)`` — caption batching is the
    default Python-list collation that WDS does for non-tensor fields.

    Args:
        data_dir: either a local path (FUSE / pre-staged NVMe) or an
            ``s3://bucket/prefix`` URI. The latter triggers direct
            ``pipe:s5cmd cat`` streaming and skips disk entirely.
        rank, world_size: every rank reads a disjoint shard subset
            (see :meth:`BLIP3OWebDataset._rank_shards`).
    """
    pipeline = BLIP3OWebDataset(
        data_dir=data_dir,
        splits=splits,
        transform=transform,
        image_size=image_size,
        resolutions=resolutions,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        dual_caption_prob=dual_caption_prob,
        split_weights=split_weights,
        split_probs=split_probs,
    )
    if virtual_epoch_steps is None:
        virtual_epoch_steps = max(1, pipeline.estimated_size // (batch_size * world_size))

    return _WDSLoaderWrapper(
        pipeline=pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=world_size,
        steps_per_epoch=virtual_epoch_steps,
    )
