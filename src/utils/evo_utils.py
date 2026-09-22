"""Trajectory evaluation utilities using the evo library.

Provides ATE and RPE computation with Sim3 (Umeyama) alignment,
matching the evaluation protocol of eval_cam.sh.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def get_tum_poses(c2w_list):
    """Convert list of (4,4) c2w matrices to TUM trajectory format.

    Returns:
        np.ndarray of shape (N, 8): [timestamp tx ty tz qx qy qz qw]
    """
    rows = []
    for i, c2w in enumerate(c2w_list):
        c2w = np.asarray(c2w, dtype=np.float64)
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        rows.append([float(i), t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
    return np.array(rows, dtype=np.float64)


def eval_metrics(pred_tum, gt_tum, seq="scene", filename=None, correct_scale=True):
    """Compute ATE and RPE using evo library (Umeyama alignment).

    Args:
        pred_tum: (N, 8) TUM format [ts tx ty tz qx qy qz qw]
        gt_tum:   (N, 8) TUM format
        seq: sequence name (for logging only)
        filename: unused, kept for API compatibility
        correct_scale: True → Sim3 (rotation+translation+scale) alignment;
            False → SE3 (rotation+translation only), keeping the estimator's
            native scale so residual scale error shows up in ATE/RPEt.

    Returns:
        ate_rmse, rpe_trans_rmse, rpe_rot_rmse (degrees)
    """
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import metrics, sync

    def _to_traj(tum):
        return PoseTrajectory3D(
            positions_xyz=tum[:, 1:4],
            orientations_quat_wxyz=tum[:, [7, 4, 5, 6]],  # evo expects wxyz
            timestamps=tum[:, 0],
        )

    traj_pred = _to_traj(pred_tum)
    traj_gt = _to_traj(gt_tum)

    traj_gt, traj_pred = sync.associate_trajectories(traj_gt, traj_pred)

    # Umeyama alignment — align() modifies traj_pred in-place. correct_scale
    # toggles Sim3 (with scale) vs SE3 (rotation+translation only).
    traj_pred.align(traj_gt, correct_scale=correct_scale)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_gt, traj_pred))
    ate_rmse = ape.get_statistic(metrics.StatisticsType.rmse)

    # RPE (translation + rotation)
    rpe_t = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=1, delta_unit=metrics.Unit.frames, all_pairs=True,
    )
    rpe_t.process_data((traj_gt, traj_pred))
    rpe_trans_rmse = rpe_t.get_statistic(metrics.StatisticsType.rmse)

    rpe_r = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg,
        delta=1, delta_unit=metrics.Unit.frames, all_pairs=True,
    )
    rpe_r.process_data((traj_gt, traj_pred))
    rpe_rot_rmse = rpe_r.get_statistic(metrics.StatisticsType.rmse)

    return ate_rmse, rpe_trans_rmse, rpe_rot_rmse
