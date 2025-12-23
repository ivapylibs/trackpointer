# trackpointer/trackpointer/hand_model.py
from __future__ import annotations

import numpy as np
import cv2


# Use ONLY rigid-ish palm anchors for PnP (avoid fingertips).
# MediaPipe indices: 0 wrist, 5/9/13/17 MCP knuckles.
PALM_PNP_IDXS = np.array([0, 5, 9, 13, 17], dtype=int)


def canonical_palm_points() -> np.ndarray:
    """
    Returns (N,3) canonical 3D positions for the subset of MediaPipe landmarks
    in PALM_PNP_IDXS, in a hand-local coordinate frame.

    Units are arbitrary but consistent (treat as 'cm' for sanity).
    NOTE: Only includes rigid-ish palm points suitable for solvePnP.
    """
    # Simple flat palm model:
    # wrist near origin, MCPs forming a rough arc in the palm plane.
    pts = np.array([
        [0.0, 0.0, 0.0],   # 0: wrist
        [2.0, 3.0, 0.0],   # 5: index MCP
        [3.0, 4.0, 0.0],   # 9: middle MCP
        [4.0, 3.5, 0.0],   # 13: ring MCP
        [5.0, 3.0, 0.0],   # 17: pinky MCP
    ], dtype=np.float32)
    return pts


def canonical_extra_points() -> dict[int, np.ndarray]:
    """
    Canonical 3D points for non-rigid landmarks that we may want to query
    (e.g., for a "pick" definition), but that we DO NOT feed into solvePnP.
    """
    # These are only used to define a canonical "pick" location/direction
    # after we have a stable palm pose (R,t) from PALM_PNP_IDXS.
    return {
        4: np.array([0.5, 2.0, -1.0], dtype=np.float32),  # thumb tip (slightly out of plane)
        8: np.array([2.0, 7.0,  0.0], dtype=np.float32),  # index tip
    }


def solve_hand_pnp_from_landmarks(
    landmarks_norm: np.ndarray,
    image_shape: tuple[int, int],
    K: np.ndarray | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    landmarks_norm: (21, 3) or (21, 2) normalized in [0,1] coords.
    image_shape: (H, W).

    Returns:
        success: bool
        R: (3,3) rotation matrix in camera frame or None
        t: (3,1) translation vector in camera frame or None
    """
    H, W = image_shape[:2]

    L = np.asarray(landmarks_norm, dtype=np.float32)
    if L.ndim != 2 or L.shape[0] <= PALM_PNP_IDXS.max():
        return False, None, None

    # Use only the xy components (assumed normalized [0,1])
    L_xy = L[:, :2]

    # Build 2D image points (pixels) for selected indices
    img_pts = []
    for idx in PALM_PNP_IDXS:
        x_norm, y_norm = L_xy[idx]
        if not np.isfinite(x_norm) or not np.isfinite(y_norm):
            return False, None, None
        u = float(x_norm) * W
        v = float(y_norm) * H
        img_pts.append([u, v])

    img_pts = np.asarray(img_pts, dtype=np.float32)  # (N,2)

    # Corresponding 3D canonical points (same N, same order)
    obj_pts = canonical_palm_points()                # (N,3)

    if K is None:
        # Approximate intrinsics: focal proportional to image size
        f = float(max(H, W)) * 1.2
        cx = W / 2.0
        cy = H / 2.0
        K = np.array([[f, 0.0, cx],
                      [0.0, f, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    # Use explicit zero distortion for consistency
    distCoeffs = np.zeros((5, 1), dtype=np.float32)

    # Run PnP (ITERATIVE is fine; can add refinement later)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return False, None, None

    R, _ = cv2.Rodrigues(rvec)  # (3,3)
    tvec = tvec.reshape(3, 1)   # (3,1)

    return True, R, tvec


def compute_pick_pose_camera(
    landmarks_norm: np.ndarray,
    image_shape: tuple[int, int],
    K: np.ndarray | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    High-level helper:

    Uses a stable palm pose from solvePnP (rigid palm anchors),
    then defines a canonical "pick" point as the midpoint of canonical
    thumb tip (4) and canonical index tip (8), transformed into camera frame.

    Returns:
        success: bool
        pick_cam: (3,) position in camera frame or None
        axis_cam: (3,) direction in camera frame or None
    """
    success, R, t = solve_hand_pnp_from_landmarks(landmarks_norm, image_shape, K)
    if not success or R is None or t is None:
        return False, None, None

    # Canonical pick definition (NOT used for PnP correspondences)
    extras = canonical_extra_points()
    if 4 not in extras or 8 not in extras:
        return False, None, None

    thumb_c = extras[4]
    index_c = extras[8]
    pick_c = 0.5 * (thumb_c + index_c)

    # Canonical axis definition from palm anchors (rigid)
    palm_pts = canonical_palm_points()
    # In canonical_palm_points order: [0, 5, 9, 13, 17]
    wrist_c = palm_pts[0]
    mid_mcp_c = palm_pts[2]  # corresponds to 9: middle MCP
    axis_c = mid_mcp_c - wrist_c
    axis_c_norm = float(np.linalg.norm(axis_c))
    if axis_c_norm > 1e-6:
        axis_c = axis_c / axis_c_norm
    else:
        axis_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Transform into camera frame: X_cam = R * X_canon + t
    pick_cam = (R @ pick_c.reshape(3, 1) + t).reshape(3)
    axis_cam = (R @ axis_c.reshape(3, 1)).reshape(3)

    # Normalize axis_cam for safety
    axis_cam_norm = float(np.linalg.norm(axis_cam))
    if axis_cam_norm > 1e-6:
        axis_cam = axis_cam / axis_cam_norm

    return True, pick_cam, axis_cam
