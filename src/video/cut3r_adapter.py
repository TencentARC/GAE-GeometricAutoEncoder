"""
CUT3R DataLoader Adapter for gld pipeline.

Converts CUT3R batch format (List[Dict]) to gld format (Dict).
Handles:
1. View reordering based on ref_view_sampling
2. ImageNet denormalization to [0, 1]
3. Intrinsics format conversion (3x3) -> [fx, fy, cx, cy]
"""
import torch
from typing import List, Dict, Any, Optional
import hashlib


def convert_cut3r_batch(
    cut3r_batch: List[Dict],
    cond_num: int,
    ref_view_sampling: str = "prefix"
) -> Dict[str, Any]:
    """
    Convert CUT3R batch format to gld format.
    
    CUT3R returns: batch = [view0_dict, view1_dict, ...]
    gld expects: batch = {'gt_inp': (B,V,C,H,W), 'fxfycxcy': (B,V,4), ...}
    
    Args:
        cut3r_batch: List of view dicts from CUT3R DataLoader.
            Each dict contains:
                - 'img': (B, C, H, W) - ImageNet normalized
                - 'camera_pose': (B, 4, 4) - c2w matrix
                - 'camera_intrinsics': (B, 3, 3) - intrinsic matrix
                - 'idx': tuple - (sample_idx, ar_idx, view_idx)
        cond_num: Number of reference (conditioning) views.
        ref_view_sampling: How to select reference views.
            - "prefix": First cond_num views are references.
            - "interpolate": First and last views are references (cond_num=2).
            - "random": Randomly select cond_num views as references.
    
    Returns:
        Dict with keys:
            - 'gt_inp': (B, V, C, H, W) in [0, 1] range
            - 'fxfycxcy': (B, V, 4) intrinsics [fx, fy, cx, cy]
            - 'c2w': (B, V, 4, 4) camera-to-world extrinsics
            - 'video_id': str identifier
            - 'frame_indices': (B, V) per-slot ordinal frame index after
              reorder (equals ``order`` from ``_get_view_order``; prefix →
              ``0..V-1``, interpolate → ``[0, V-1, 1, …]``)
            - 'physical_frame_indices': (B, V) original dataset frame ids
              after reorder when the dataset provides ``frame_idx``; otherwise
              ``None`` (consumers then skip interval-based pose scaling).
    """
    V = len(cut3r_batch)
    B = cut3r_batch[0]['img'].shape[0]
    
    # 1. Determine view order based on ref_view_sampling.
    #    Per-dataset override: if views carry ref_view_sampling (e.g. OpenVid→"prefix"),
    #    use it instead of the global config value.
    sample_rvs = cut3r_batch[0].get('ref_view_sampling', None)
    if sample_rvs is not None:
        if hasattr(sample_rvs, '__getitem__') and not isinstance(sample_rvs, str):
            sample_rvs = sample_rvs[0]  # collated tensor/tuple → first element
        # Only a *concrete* per-sample mode (e.g. OpenVid→"prefix") forces an
        # override. A high-level policy like "prefix_fl" is resolved by the
        # caller per-step (regime + cond decoupled), so don't clobber the
        # concrete value the caller already passed.
        if isinstance(sample_rvs, str) and sample_rvs not in ("", "prefix_fl"):
            ref_view_sampling = sample_rvs
    # High-level policies (e.g. "prefix_fl") may reach here from either the
    # caller or a per-sample override; _get_view_order only knows concrete
    # modes, so resolve once before dispatching (identity for concrete modes).
    ref_view_sampling = resolve_ref_view_sampling(cond_num, ref_view_sampling)
    order = _get_view_order(V, cond_num, ref_view_sampling, cut3r_batch)
    
    # Reorder views
    reordered = [cut3r_batch[i] for i in order]
    
    # 2. Stack images: List[(B,C,H,W)] → (B,V,C,H,W)
    imgs = torch.stack([v['img'] for v in reordered], dim=1)  # (B,V,C,H,W)
    
    # 3. Denormalize ImageNet → [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(imgs)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(imgs)
    gt_inp = imgs * std + mean
    gt_inp = gt_inp.clamp(0, 1)
    
    # 4. Convert intrinsics (B,3,3) → (B,4) for each view, then stack
    fxfycxcy_list = []
    for v in reordered:
        K = v['camera_intrinsics']  # (B, 3, 3)
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        fxfycxcy_list.append(torch.stack([fx, fy, cx, cy], dim=-1))  # (B, 4)
    fxfycxcy = torch.stack(fxfycxcy_list, dim=1)  # (B, V, 4)
    
    # 5. Stack poses (camera_pose is c2w)
    c2w = torch.stack([v['camera_pose'] for v in reordered], dim=1)  # (B, V, 4, 4)
    
    # 6. Frame indices used by Temporal RoPE: keep ordinal clip slots, not
    # absolute video frame ids, to preserve the existing temporal phase.
    frame_indices = torch.tensor([order], dtype=torch.long).expand(B, -1)

    # physical_frame_indices: real dataset frame ids, used downstream to scale
    # pose translation by the true frame interval. Stays None when the dataset
    # does not provide 'frame_idx' (e.g. OpenVid T2V) so consumers fall back to
    # "no scaling" instead of mistaking ordinal slots for physical frames.
    physical_frame_indices = None
    if all('frame_idx' in v for v in reordered):
        physical_list = []
        for v in reordered:
            frame_idx = v['frame_idx']
            if torch.is_tensor(frame_idx):
                physical_list.append(frame_idx.long())
            else:
                physical_list.append(torch.full((B,), int(frame_idx), dtype=torch.long))
        physical_frame_indices = torch.stack(physical_list, dim=1)
    
    # 7. Preserve CUT3R provenance (critical for debugging / reproducibility)
    # Each view dict contains 'idx' = (sample_idx, ar_idx, view_idx) as tensors.
    cut3r_idx = [v.get('idx', None) for v in reordered]
    
    # 8. Detect per-sample ref_view_sampling override
    sample_rvs = reordered[0].get('ref_view_sampling', '')
    if isinstance(sample_rvs, (list, tuple)):
        sample_rvs = sample_rvs[0] if sample_rvs else ''

    # 9. Detect T2V mode (OpenVid: no real camera pose). Keep per-sample
    # tensors intact for mixed RE10K/OpenVid batches; the trainer masks Plücker
    # per row when only part of the batch is T2V.
    is_t2v = reordered[0].get('is_t2v', False)

    # 9b. disable_plucker — T2V dataset 可以显式 emit False（保留 Plücker，
    #     用于 camera-aware T2V，如 OSP iStock）。OpenVid 等 identity-pose
    #     T2V 不 emit 此字段 → 训练脚本 .get() 默认 True，旧行为不变。
    disable_plucker = reordered[0].get('disable_plucker', True)
    if isinstance(disable_plucker, torch.Tensor):
        # Per-sample bool tensor (B,) — pass through (训练脚本会按 per-sample 处理)
        disable_plucker = disable_plucker
    else:
        disable_plucker = bool(disable_plucker)

    # 9. Extract scene-level captions (if present in view dicts).
    #    caption is per-scene (same for all views), so take from first view.
    #    Result: List[str] of length B, or None if no captions available.
    caption = None
    if 'caption' in reordered[0]:
        # reordered[0]['caption'] is a list of B strings (batched by dataloader)
        # or a single string. Handle both.
        raw_cap = reordered[0]['caption']
        if isinstance(raw_cap, (list, tuple)):
            caption = list(raw_cap)  # List[str] length B
        elif isinstance(raw_cap, str):
            caption = [raw_cap] * B
        # If caption is None for some samples, keep None

    # 10. Carry per-sample scene identity if the dataset provided it (e.g.
    #     RE10K_Packed). Required by the offline caption/latent precompute
    #     pipeline to label records with the *actual* scene used, not the
    #     stale outer base-class idx.
    scene_ids = None
    start_positions = None
    raw_scene = reordered[0].get('scene_id', None)
    if raw_scene is not None:
        scene_ids = list(raw_scene) if isinstance(raw_scene, (list, tuple)) else [raw_scene] * B
        scene_ids = [str(s) for s in scene_ids]
    raw_start = reordered[0].get('start_pos', None)
    if raw_start is not None:
        if torch.is_tensor(raw_start):
            start_positions = [int(x) for x in raw_start.reshape(-1).tolist()]
        elif isinstance(raw_start, (list, tuple)):
            start_positions = [int(x) for x in raw_start]
        else:
            start_positions = [int(raw_start)] * B

    return {
        'gt_inp': gt_inp,           # (B, V, C, H, W) in [0,1]
        'fxfycxcy': fxfycxcy,       # (B, V, 4)
        'c2w': c2w,                 # (B, V, 4, 4)
        'video_id': 'cut3r_batch',
        'frame_indices': frame_indices,
        'physical_frame_indices': physical_frame_indices,
        'cut3r_idx': cut3r_idx,     # List[V] of CUT3R (sample_idx, ar_idx, view_idx)
        'caption': caption,         # List[str] length B, or None
        'scene_ids': scene_ids,        # Optional List[str] length B
        'start_positions': start_positions,  # Optional List[int] length B
        'is_t2v': is_t2v,                  # bool or (B,) tensor: True if T2V data
        'disable_plucker': disable_plucker,  # bool or (B,) tensor: zero Plücker?
        'ref_view_sampling': sample_rvs,  # str: per-sample override or ''
    }


def _get_view_order(
    num_views: int,
    cond_num: int,
    ref_view_sampling: str,
    batch: Optional[List[Dict]] = None
) -> List[int]:
    """
    Determine view reordering so that first cond_num views are references.
    
    Args:
        num_views: Total number of views.
        cond_num: Number of reference views.
        ref_view_sampling: Sampling strategy.
        batch: Original batch (used for deterministic hashing in 'random' mode).
    
    Returns:
        List of indices representing the new view order.
    """
    V = num_views
    
    if ref_view_sampling == "prefix":
        # First cond_num views are condition views (RGB provided to the model).
        #
        # NOTE: this is independent from PRoPE's *geometric* reference view,
        # which is fixed to the **last** view (`index = -1`) inside
        # `batch_sample_rays` / training-loop ProPE normalization. So the
        # condition view (prefix) and the geometric origin (suffix) are
        # different views by design — see `src/utils/camera/camera.py` docstring.
        return list(range(V))
    
    elif ref_view_sampling == "interpolate":
        # First and last views are references
        if cond_num != 2:
            raise ValueError(
                f"ref_view_sampling='interpolate' requires cond_num=2, got {cond_num}"
            )
        # Order: [first, last, middle views...]
        return [0, V - 1] + list(range(1, V - 1))
    
    elif ref_view_sampling == "random":
        # Randomly select reference views (deterministic based on batch)
        if cond_num < 1:
            # cond_num=0: no ref views, order does not matter
            return list(range(num_views))
        
        # Create deterministic seed from batch info
        if batch is not None and 'idx' in batch[0]:
            batch_hash = hashlib.md5(str(batch[0]['idx']).encode()).hexdigest()
            seed = int(batch_hash[:8], 16)
        else:
            seed = 0
        
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(V, generator=g).tolist()
        ref_pos = sorted(perm[:cond_num])
        tgt_pos = [i for i in range(V) if i not in set(ref_pos)]
        return ref_pos + tgt_pos
    
    else:
        raise ValueError(f"Unknown ref_view_sampling: {ref_view_sampling}")


def compute_view_order(
    num_views: int,
    cond_num: int,
    ref_view_sampling: str = "prefix",
) -> List[int]:
    """Public wrapper around ``_get_view_order`` (no batch needed for prefix/interpolate)."""
    effective = resolve_ref_view_sampling(cond_num, ref_view_sampling)
    return _get_view_order(num_views, cond_num, effective)


def resolve_ref_view_sampling(
    cond_num: int,
    ref_view_sampling: str,
    *,
    fl_when_cond: int = 2,
) -> str:
    """Map a high-level sampling policy to a concrete ``ref_view_sampling`` mode.

    ``prefix_fl`` (first+last / 首尾帧):
        * ``cond_num == fl_when_cond`` (default 2) → ``interpolate`` (frames 0 & V-1)
        * otherwise → ``prefix`` (first ``cond_num`` frames)
    """
    if ref_view_sampling == "prefix_fl":
        return "interpolate" if cond_num == fl_when_cond else "prefix"
    return ref_view_sampling


def is_cut3r_batch(batch: Any) -> bool:
    """
    Check if the batch is from CUT3R DataLoader.
    
    CUT3R returns List[Dict], gld returns Dict.
    """
    return isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict)
