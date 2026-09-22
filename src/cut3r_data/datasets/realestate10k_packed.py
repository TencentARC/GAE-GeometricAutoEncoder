# RealEstate10K — Packed (video.mp4 + meta.json + caption.txt) layout.
#
# Why a new class instead of extending RE10K_Multi: the existing class
# assumes per-frame PNGs (`{ts}.png` + `{ts}_cam.npz` flat or NeRF-style
# `transforms.json` nested). The "packed" preprocessed layout is fully
# different:
#
#     ROOT/{scene_id}/video.mp4        # all N frames, mp4 idx == meta.frames[i]
#     ROOT/{scene_id}/meta.json        # {scene_id, num_frames, fps,
#                                      #  height, width, caption,
#                                      #  frames:[{name, c2w, K}, ...]}
#     ROOT/{scene_id}/caption.txt      # mirror of meta.caption
#
# Caption is **already inline** in meta.json — no aggregated
# train_caption.json fetch needed. Returned per view dict so the cut3r
# adapter promotes it to batch["caption"].
#
# Camera convention: meta.json's `c2w` and `K` are interpreted as-is in
# OpenCV / COLMAP convention (same as RE10K_Multi's transforms.json
# branch, see its docstring). No OpenGL flip.
#
# Frame ↔ video index mapping: verified equal on the reference asset
# (ffprobe nb_read_frames == meta.num_frames for shard
# train/0000cc6d8b108390 → 218). We treat `mp4 frame i ↔ meta.frames[i]`
# as the contract; sanity-checked at scene-index time when first opened.
from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import os.path as osp
import pickle
import tempfile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset

_CAM_CACHE_MAX = 1024
# Parallelism for the one-time scene-index build (FUSE IO is the bottleneck;
# 64 threads ≈ 10× speedup vs serial on a 67k-scene RE10K_Packed train ROOT).
_INDEX_WORKERS = int(os.environ.get("RE10K_PACKED_INDEX_WORKERS", "64"))

# Known locations of compatible VideoMetaScene caches we can adopt to skip
# the 67k-meta.json cold scan. Keyed by ROOT basename ("train" / "test").
# Override with env var RE10K_PACKED_VMS_CACHE=/path/to/pkl (forces use of
# that file regardless of ROOT match).
# Populated only in the internal cluster. Public users rely on the local
# index built on first use (or RE10K_PACKED_VMS_CACHE=/path/to.pkl).
_VMS_CACHE_HINTS: dict[str, str] = {}


def _writable_cache_dir(primary: str) -> str | None:
    """Return a directory we can actually write to (primary first, then
    user cache fallback). Returns None if every candidate fails."""
    candidates = [
        primary,
        osp.join(os.path.expanduser("~"), ".cache", "cut3r", "re10k_packed"),
        osp.join(tempfile.gettempdir(), "cut3r_re10k_packed"),
    ]
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            probe = osp.join(c, ".__write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return c
        except OSError:
            continue
    return None


class RE10K_Packed(BaseMultiViewDataset):
    """RealEstate10K loader for the packed mp4 + meta.json preprocessed
    layout (see module docstring for exact paths / contracts).

    Args:
        ROOT: Directory containing scene subdirectories.
        min_interval, max_interval: frame index interval bounds passed
            to ``get_seq_from_start_id_novid`` (same semantics as
            RE10K_Multi).
        exclude_tail_frames: drop the last N mp4/meta frames from all
            sampling. Packed RE10K clips often stitch a foreign tail segment
            after the main scene; those frames are unusable for training.
        ref_view_sampling: Forwarded into each view dict so the trainer can
            honor config-level policies such as ``prefix_fl``.
        All other args forward to BaseMultiViewDataset.
    """

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.min_interval = kwargs.pop("min_interval", 1)
        self.max_interval = kwargs.pop("max_interval", 128)
        self.exclude_tail_frames = int(kwargs.pop("exclude_tail_frames", 0))
        self.ref_view_sampling = kwargs.pop("ref_view_sampling", "prefix")
        super().__init__(*args, **kwargs)
        self._cam_cache: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._load_data()

    def _required_cut_off(self, num_views: int | None = None) -> int:
        """Minimum scene length and exclusive tail for valid ``start_pos``.

        With fixed ``max_interval`` and ``allow_repeat=False``, a window of
        ``num_views`` needs physical frames
        ``start_pos + (num_views - 1) * max_interval < num_frames``.
        The old ``cut_off = num_views`` only guarded the anchor frame and let
        high ``start_pos`` values collapse into dense tail sampling (interval→1),
        which often crosses packed-mp4 stitch boundaries at scene ends.
        """
        if num_views is None:
            if getattr(self, "view_choices", None) is not None:
                num_views = max(self.view_choices)
            else:
                num_views = self.num_views
        span = max(0, int(num_views) - 1) * int(self.max_interval)
        return max(2, span + 1)

    def _usable_frame_count(self, num_frames: int) -> int:
        """Physical frames minus the packed-mp4 tail that must not be sampled."""
        return max(0, int(num_frames) - self.exclude_tail_frames)

    def _num_valid_starts(self, num_frames: int, num_views: int | None = None) -> int:
        usable = self._usable_frame_count(num_frames)
        cut_off = self._required_cut_off(num_views)
        if usable < cut_off:
            return 0
        return usable - cut_off + 1

    def usable_frame_ids(self, num_frames: int) -> list[int]:
        """Frame indices eligible for window sampling (excludes tail trim)."""
        n = self._usable_frame_count(num_frames)
        return list(range(n))

    def _index_cache_tag(self) -> str:
        view_choices = getattr(self, "view_choices", None)
        vc_key = "none" if view_choices is None else ",".join(str(v) for v in view_choices)
        key = (
            f"re10k_packed::{self.ROOT}::nv{self.num_views}"
            f"::vc{vc_key}::ar{self.allow_repeat}"
            f"::iv{self.min_interval}-{self.max_interval}"
            f"::xt{self.exclude_tail_frames}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_filename(self) -> str:
        return f".re10k_packed_index_{self._index_cache_tag()}.pkl"

    def _all_cache_paths(self) -> list[str]:
        """Every known location for this index (shared ROOT first).

        Multi-GPU / multi-node precompute can leave the same pickle under
        different dirs depending on per-process FUSE write probes. Readers
        must try them all so a corrupt node-local partial write does not
        mask a good shared copy on S3.
        """
        name = self._cache_filename()
        dirs: list[str] = []
        for d in (
            self.ROOT,
            osp.join(os.path.expanduser("~"), ".cache", "cut3r", "re10k_packed"),
            osp.join(tempfile.gettempdir(), "cut3r_re10k_packed"),
        ):
            if d not in dirs:
                dirs.append(d)
        return [osp.join(d, name) for d in dirs]

    def _writable_cache_path(self) -> str | None:
        target_dir = _writable_cache_dir(self.ROOT)
        if target_dir is None:
            return None
        return osp.join(target_dir, self._cache_filename())

    @staticmethod
    def _write_cache_atomic(path: str, payload: dict) -> bool:
        """Write pickle atomically so concurrent readers never see a partial file."""
        parent = osp.dirname(path)
        try:
            os.makedirs(parent, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".__idx_", suffix=".pkl", dir=parent)
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
                return True
            finally:
                if osp.exists(tmp):
                    os.remove(tmp)
        except OSError as e:
            print(f"[RE10K_Packed] Warning: could not write cache {path}: {e}")
            return False

    def _write_own_cache(
        self,
        scenes: list[str],
        scene_data: dict[str, dict],
        start_img_ids: list[tuple[str, int]],
    ) -> None:
        payload = {
            "scenes": scenes,
            "scene_data": scene_data,
            "start_img_ids": start_img_ids,
        }
        for path in self._all_cache_paths():
            if self._write_cache_atomic(path, payload):
                print(f"[RE10K_Packed] Wrote own cache to {path}")
                return

    @staticmethod
    def _scan_one_scene(args):
        """Per-scene IO worker (runs in a ThreadPoolExecutor).

        Returns a tuple (scene_name, scene_dir, meta_path, video_path,
        num_frames, caption) on success, or None if the scene should be
        skipped (missing files / bad meta / num_frames mismatch).
        """
        scene_name, ROOT, cut_off = args
        scene_dir = osp.join(ROOT, scene_name)
        meta_path = osp.join(scene_dir, "meta.json")
        video_path = osp.join(scene_dir, "video.mp4")
        if not (osp.exists(meta_path) and osp.exists(video_path)):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            num_frames = int(meta.get("num_frames", len(meta.get("frames", []))))
            if num_frames < cut_off:
                return None
            if "frames" not in meta or len(meta["frames"]) != num_frames:
                # Defensive: meta.frames length must match num_frames so
                # that mp4 frame idx == meta.frames idx.
                return None
            return (
                scene_name, scene_dir, meta_path, video_path, num_frames,
                str(meta.get("caption", "") or ""),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def _try_adopt_vms_cache(self, cut_off: int) -> bool:
        """Fast-path: adopt a pre-built VideoMetaScene cache that scanned
        the same RE10K_Packed ROOT. Saves the 67k-meta.json cold scan
        (which takes ~30-120 min vs <5 s of pkl load).

        Field translation:
            VMS: scene_data[s] = {video_path, meta_path, num_frames,
                                  rgb_resolution, inline_caption}
            Ours: scene_data[s] = {scene_dir, meta_path, video_path,
                                   num_frames, caption}

        ``start_img_ids`` is recomputed from ``num_frames`` + our own
        ``cut_off`` so that VMS having been built with a different
        ``num_views`` does not propagate stale start positions.

        Returns True on success (self.* populated), False if no
        compatible VMS cache found.
        """
        vms_path = os.environ.get("RE10K_PACKED_VMS_CACHE", "")
        if not vms_path:
            vms_path = _VMS_CACHE_HINTS.get(osp.basename(self.ROOT.rstrip("/")), "")
        if not vms_path or not osp.isfile(vms_path):
            return False

        print(f"[RE10K_Packed] Found compatible VMS cache → adopting: {vms_path}")
        try:
            with open(vms_path, "rb") as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"[RE10K_Packed] VMS cache load failed ({e}); falling back to scan.")
            return False

        if not (
            isinstance(d, dict)
            and "scenes" in d and "scene_data" in d
            and d["scenes"] and d["scene_data"]
        ):
            print("[RE10K_Packed] VMS cache shape unexpected; falling back to scan.")
            return False

        # ROOT-match sanity check on the first entry's video_path.
        first_sd = next(iter(d["scene_data"].values()))
        first_vp = first_sd.get("video_path", "")
        if not first_vp.startswith(self.ROOT.rstrip("/")):
            print(
                f"[RE10K_Packed] VMS cache ROOT mismatch "
                f"(VMS sample vp={first_vp!r}, our ROOT={self.ROOT!r}); "
                f"falling back to scan."
            )
            return False

        scenes: list[str] = []
        scene_data: dict[str, dict] = {}
        start_img_ids: list[tuple[str, int]] = []
        skipped_too_short = 0
        for scene_name in d["scenes"]:
            vsd = d["scene_data"].get(scene_name)
            if not vsd:
                continue
            num_frames = int(vsd.get("num_frames", 0))
            if self._num_valid_starts(num_frames) <= 0:
                skipped_too_short += 1
                continue
            meta_path = vsd["meta_path"]
            scenes.append(scene_name)
            scene_data[scene_name] = {
                "scene_dir": osp.dirname(meta_path),
                "meta_path": meta_path,
                "video_path": vsd["video_path"],
                "num_frames": num_frames,
                # Fall back through caption key aliases — VMS uses
                # `inline_caption`, our own format uses `caption`.
                "caption": str(
                    vsd.get("caption", vsd.get("inline_caption", "")) or ""
                ),
            }
            num_start = self._num_valid_starts(num_frames)
            for si in range(num_start):
                start_img_ids.append((scene_name, si))

        self.scenes = scenes
        self.scene_data = scene_data
        self.start_img_ids = start_img_ids
        self.invalid_scenes = {s: False for s in scenes}
        print(
            f"[RE10K_Packed] Adopted {len(scenes)} scenes, "
            f"{len(start_img_ids)} start positions "
            f"(skipped {skipped_too_short} with num_frames<{cut_off})."
        )

        # Persist as our own cache so subsequent runs skip even the VMS
        # adoption step.
        self._write_own_cache(scenes, scene_data, start_img_ids)
        return True

    def _load_data(self):
        for cache in self._all_cache_paths():
            if not osp.isfile(cache):
                continue
            print(f"[RE10K_Packed] Loading cached index from {cache} ...")
            try:
                with open(cache, "rb") as f:
                    d = pickle.load(f)
                if not (
                    isinstance(d, dict)
                    and "scenes" in d
                    and "scene_data" in d
                    and "start_img_ids" in d
                ):
                    raise ValueError("unexpected cache shape")
            except (EOFError, pickle.UnpicklingError, KeyError, TypeError, ValueError) as e:
                print(f"[RE10K_Packed] Cache corrupt at {cache} ({e}); trying next.")
                continue
            self.scenes = d["scenes"]
            self.scene_data = d["scene_data"]
            self.start_img_ids = d["start_img_ids"]
            self.invalid_scenes = {s: False for s in self.scenes}
            print(
                f"[RE10K_Packed] {len(self.scenes)} scenes, "
                f"{len(self.start_img_ids)} start positions ready."
            )
            return

        cut_off = self._required_cut_off()
        min_raw_frames = cut_off + self.exclude_tail_frames
        if self._try_adopt_vms_cache(cut_off):
            return
        entries = sorted(
            e for e in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, e))
        )
        print(
            f"[RE10K_Packed] Building scene index for {self.ROOT} "
            f"({len(entries)} entries, {_INDEX_WORKERS} parallel IO workers, first time only) ..."
        )
        # Parallel scan: FUSE meta.json reads are IO-bound. Threads (not
        # processes) → no fork/pickle overhead and FUSE concurrency is
        # the real win. tqdm wraps the iterator so the user sees ETA.
        args_iter = ((s, self.ROOT, min_raw_frames) for s in entries)
        scan_results: list = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_INDEX_WORKERS,
        ) as ex:
            for r in tqdm(
                ex.map(self._scan_one_scene, args_iter),
                total=len(entries), desc="Indexing RE10K_Packed", smoothing=0.05,
            ):
                if r is not None:
                    scan_results.append(r)

        # Deterministic output order regardless of thread completion order.
        scan_results.sort(key=lambda r: r[0])

        scenes: list[str] = []
        scene_data: dict[str, dict] = {}
        start_img_ids: list[tuple[str, int]] = []
        for scene_name, scene_dir, meta_path, video_path, num_frames, caption in scan_results:
            if self._num_valid_starts(num_frames) <= 0:
                continue
            scenes.append(scene_name)
            scene_data[scene_name] = {
                "scene_dir": scene_dir,
                "meta_path": meta_path,
                "video_path": video_path,
                "num_frames": num_frames,
                "caption": caption,
            }
            num_start = self._num_valid_starts(num_frames)
            for si in range(num_start):
                start_img_ids.append((scene_name, si))

        self.scenes = scenes
        self.scene_data = scene_data
        self.start_img_ids = start_img_ids
        self.invalid_scenes = {s: False for s in scenes}

        cache = self._writable_cache_path()
        if cache and self._write_cache_atomic(
            cache,
            {
                "scenes": scenes,
                "scene_data": scene_data,
                "start_img_ids": start_img_ids,
            },
        ):
            print(f"[RE10K_Packed] Cached index to {cache}")

        print(
            f"[RE10K_Packed] {len(scenes)} scenes, "
            f"{len(start_img_ids)} start positions ready."
        )

    def __len__(self):
        return len(self.start_img_ids)

    def _load_cameras(self, scene_name: str):
        """Return (c2w (N,4,4), K (N,3,3)) for the scene, LRU-cached."""
        if scene_name in self._cam_cache:
            self._cam_cache.move_to_end(scene_name)
            return self._cam_cache[scene_name]

        sd = self.scene_data[scene_name]
        with open(sd["meta_path"]) as f:
            meta = json.load(f)
        frames = meta["frames"]
        c2w = np.stack(
            [np.asarray(fr["c2w"], dtype=np.float32) for fr in frames], axis=0,
        )  # (N, 4, 4)
        K = np.stack(
            [np.asarray(fr["K"], dtype=np.float32) for fr in frames], axis=0,
        )  # (N, 3, 3)

        # Replace any NaN/Inf with identity so base-class pose-norm filter
        # picks them up (rather than crashing later in the model).
        bad = ~(np.isfinite(c2w).reshape(c2w.shape[0], -1).all(axis=1))
        if bad.any():
            for i in np.where(bad)[0]:
                c2w[i] = np.eye(4, dtype=np.float32)

        self._cam_cache[scene_name] = (c2w, K)
        while len(self._cam_cache) > _CAM_CACHE_MAX:
            self._cam_cache.popitem(last=False)
        return self._cam_cache[scene_name]

    def _read_video_frames(self, video_path: str, frame_indices: list[int]):
        """Decode a set of frames at the given indices from one mp4.

        Opens the video once, iterates sequentially, and yields PIL images
        in the order of the (sorted) input indices. Returns a dict
        ``{idx: PIL.Image}``. Raises ``IOError`` if any requested index
        is unreachable.
        """
        # Sort once to do a single forward pass through the mp4. Random
        # access via CAP_PROP_POS_FRAMES is unreliable across codecs
        # (re10k_packed is H.264 with B-frames where seeks land on the
        # nearest keyframe, decoding decoded extra frames anyway). A
        # single sequential walk + grab-only on skipped frames is the
        # fast & correct path.
        wanted = sorted(set(int(i) for i in frame_indices))
        if not wanted:
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"cv2.VideoCapture failed to open {video_path}")
        try:
            out: dict[int, Image.Image] = {}
            cur = 0
            wi = 0
            target = wanted[wi]
            while wi < len(wanted):
                if cur < target:
                    # grab() is decode-only-as-needed (faster than read())
                    if not cap.grab():
                        raise IOError(
                            f"video ended before frame {target} (last cur={cur}) "
                            f"in {video_path}"
                        )
                    cur += 1
                    continue
                ret, frame = cap.read()
                if not ret:
                    raise IOError(
                        f"cv2 read failed at frame {target} in {video_path}"
                    )
                out[target] = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                cur += 1
                wi += 1
                if wi < len(wanted):
                    target = wanted[wi]
            return out
        finally:
            cap.release()

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        scene_name, start_pos = self.start_img_ids[idx]

        for _ in range(max_retries):
            while self.invalid_scenes.get(scene_name, False):
                idx = int(rng.integers(low=0, high=len(self.start_img_ids)))
                scene_name, start_pos = self.start_img_ids[idx]

            sd = self.scene_data[scene_name]
            total = sd["num_frames"]
            all_ids = self.usable_frame_ids(total)
            if getattr(self, "view_choices", None) is not None:
                # Per-draw V: respect interval span, not just anchor count.
                max_start = self._num_valid_starts(total, num_views) - 1
                if max_start < 0:
                    idx = int(rng.integers(low=0, high=len(self.start_img_ids)))
                    scene_name, start_pos = self.start_img_ids[idx]
                    continue
                start_pos = int(rng.integers(low=0, high=max_start + 1))

            pos = self.get_seq_from_start_id_novid(
                num_views, start_pos, all_ids, rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=1,  # keep temporal order
            )
            if pos is None:
                idx = int(rng.integers(low=0, high=len(self.start_img_ids)))
                scene_name, start_pos = self.start_img_ids[idx]
                continue

            try:
                c2w_all, K_all = self._load_cameras(scene_name)
                frames = self._read_video_frames(sd["video_path"], list(pos))
            except (OSError, IOError, KeyError, ValueError):
                self.invalid_scenes[scene_name] = True
                idx = int(rng.integers(low=0, high=len(self.start_img_ids)))
                scene_name, start_pos = self.start_img_ids[idx]
                continue

            views = []
            for pi in pos:
                rgb = frames.get(int(pi))
                if rgb is None:
                    self.invalid_scenes[scene_name] = True
                    break
                c2w = c2w_all[pi]
                K = K_all[pi].copy()
                depthmap = np.zeros(
                    (rgb.size[1], rgb.size[0], 3), dtype=np.uint8,
                )  # placeholder; matches RE10K_Multi's zeros_like(rgb)

                rgb_arr, depthmap, K_scaled = self._crop_resize_if_necessary(
                    rgb, depthmap, K, resolution, rng=rng, info=pi,
                )

                views.append(dict(
                    img=rgb_arr,
                    camera_pose=c2w.astype(np.float32),
                    camera_intrinsics=K_scaled.astype(np.float32),
                    frame_idx=int(pi),
                    caption=sd["caption"],
                    ref_view_sampling=self.ref_view_sampling,
                    is_t2v=False,
                    # Carry the *actual* scene identity so downstream consumers
                    # (caption/precompute) don't have to look up via the outer
                    # base-class idx, which can be stale when _get_views
                    # internally retries due to invalid_scenes / IOError.
                    scene_id=str(scene_name),
                    start_pos=int(start_pos),
                ))

            if len(views) == num_views:
                return views

            idx = int(rng.integers(low=0, high=len(self.start_img_ids)))
            scene_name, start_pos = self.start_img_ids[idx]

        raise RuntimeError(
            f"[RE10K_Packed] Could not find valid scene after {max_retries} retries."
        )
