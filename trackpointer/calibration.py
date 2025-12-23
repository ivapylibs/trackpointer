# trackpointer/trackpointer/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_int_pair(x: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return default
    try:
        return (int(x[0]), int(x[1]))
    except (TypeError, ValueError):
        return default


def _as_int_list(x: Any, default: List[int]) -> List[int]:
    if x is None:
        return list(default)
    if not isinstance(x, (list, tuple, np.ndarray)):
        return list(default)
    out: List[int] = []
    for v in x:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return list(default)
    return out


def _dist_to_vec5(dist: Any) -> np.ndarray:
    """
    Normalize distortion coefficients into (5,1) float32:
      - if missing -> zeros
      - if length 4 -> pad with 0
      - if length >=5 -> take first 5
    """
    if dist is None:
        return np.zeros((5, 1), dtype=np.float32)

    if isinstance(dist, np.ndarray):
        d = dist.flatten().tolist()
    elif isinstance(dist, (list, tuple)):
        d = list(dist)
    else:
        return np.zeros((5, 1), dtype=np.float32)

    vals: List[float] = []
    for v in d:
        f = _as_float(v)
        if f is None:
            f = 0.0
        vals.append(f)

    if len(vals) == 4:
        vals.append(0.0)
    if len(vals) < 5:
        vals = vals + [0.0] * (5 - len(vals))

    vals = vals[:5]
    return np.asarray(vals, dtype=np.float32).reshape(5, 1)


def _default_palm_object_points_cm() -> np.ndarray:
    """
    Default canonical palm model in 'cm' units.
    Order corresponds to default indices [0,5,9,13,17].
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],   # 0 wrist
            [2.0, 3.0, 0.0],   # 5 index MCP
            [3.0, 4.0, 0.0],   # 9 middle MCP
            [4.0, 3.5, 0.0],   # 13 ring MCP
            [5.0, 3.0, 0.0],   # 17 pinky MCP
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class MediaPoseCalibration:
    """
    Parses and holds camera + PnP + pick-related configuration.

    Expected cfg keys (all optional):
      camera:
        intrinsics: {fx, fy, cx, cy}  # pixels
        dist_coeffs: [k1,k2,p1,p2,k3]  # optional
      pnp:
        enabled: bool
        landmark_indices: [0,5,9,13,17]
        object_points: [[x,y,z], ...]  # must match landmark_indices length
        units: "cm" | "m" | etc (informational, you decide usage)
      pick:
        enabled: bool
        pinch_landmarks: [4,8]
        axis_pair: [0,9]
        pinch_ratio_thresh: float
    """

    # camera intrinsics (None means "use approximate from image_shape")
    fx: Optional[float]
    fy: Optional[float]
    cx: Optional[float]
    cy: Optional[float]
    dist_coeffs: np.ndarray  # (5,1)

    # pnp settings
    pnp_enabled: bool
    pnp_indices: Tuple[int, ...]
    object_points: np.ndarray  # (N,3) float32
    pnp_units: str

    # pick settings
    pick_enabled: bool
    pinch_landmarks: Tuple[int, int]
    axis_pair: Tuple[int, int]
    pinch_ratio_thresh: float

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "MediaPoseCalibration":
        camera = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
        intr = camera.get("intrinsics", {}) if isinstance(camera, dict) else {}

        fx = _as_float(intr.get("fx", None))
        fy = _as_float(intr.get("fy", None))
        cx = _as_float(intr.get("cx", None))
        cy = _as_float(intr.get("cy", None))
        dist = _dist_to_vec5(camera.get("dist_coeffs", None))

        pnp = cfg.get("pnp", {}) if isinstance(cfg, dict) else {}
        pnp_enabled = bool(pnp.get("enabled", True))

        default_idxs = [0, 5, 9, 13, 17]
        idxs = _as_int_list(pnp.get("landmark_indices", None), default_idxs)
        pnp_indices = tuple(idxs)

        units = pnp.get("units", "cm")
        if not isinstance(units, str):
            units = "cm"

        obj_pts_raw = pnp.get("object_points", None)
        if obj_pts_raw is None:
            obj_pts = _default_palm_object_points_cm()
        else:
            obj_pts = np.asarray(obj_pts_raw, dtype=np.float32)
            if obj_pts.ndim != 2 or obj_pts.shape[1] != 3:
                raise ValueError(
                    f"pnp.object_points must be Nx3; got shape {obj_pts.shape}"
                )

        if len(pnp_indices) != int(obj_pts.shape[0]):
            raise ValueError(
                f"pnp.landmark_indices length ({len(pnp_indices)}) must match "
                f"pnp.object_points rows ({obj_pts.shape[0]})."
            )

        pick = cfg.get("pick", {}) if isinstance(cfg, dict) else {}
        pick_enabled = bool(pick.get("enabled", True))
        pinch_landmarks = _as_int_pair(pick.get("pinch_landmarks", None), (4, 8))
        axis_pair = _as_int_pair(pick.get("axis_pair", None), (0, 9))

        thresh = _as_float(pick.get("pinch_ratio_thresh", None))
        if thresh is None:
            thresh = 0.25

        return MediaPoseCalibration(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist,
            pnp_enabled=pnp_enabled,
            pnp_indices=pnp_indices,
            object_points=obj_pts.astype(np.float32, copy=False),
            pnp_units=units,
            pick_enabled=pick_enabled,
            pinch_landmarks=pinch_landmarks,
            axis_pair=axis_pair,
            pinch_ratio_thresh=float(thresh),
        )

    def has_intrinsics(self) -> bool:
        return (
            self.fx is not None
            and self.fy is not None
            and self.cx is not None
            and self.cy is not None
        )

    def get_K(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Returns camera matrix K (3x3 float32).
        If fx/fy/cx/cy are provided, use them.
        Otherwise fall back to the same approximate K strategy you used before.
        """
        H, W = image_shape[:2]

        if self.has_intrinsics():
            fx = float(self.fx)  # type: ignore[arg-type]
            fy = float(self.fy)  # type: ignore[arg-type]
            cx = float(self.cx)  # type: ignore[arg-type]
            cy = float(self.cy)  # type: ignore[arg-type]
        else:
            f = float(max(H, W)) * 1.2
            fx = f
            fy = f
            cx = W / 2.0
            cy = H / 2.0

        return np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def summary(self, image_shape: Optional[Tuple[int, int]] = None) -> str:
        src = "yaml" if self.has_intrinsics() else "approx"
        k_str = f"(source={src}"
        if image_shape is not None:
            K = self.get_K(image_shape)
            k_str += f", fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f})"
        else:
            k_str += ")"

        return (
            "MediaPoseCalibration: "
            f"PnP(enabled={self.pnp_enabled}, idxs={list(self.pnp_indices)}, units={self.pnp_units}), "
            f"K={k_str}, "
            f"Pick(enabled={self.pick_enabled}, pinch={self.pinch_landmarks}, axis={self.axis_pair}, "
            f"thresh={self.pinch_ratio_thresh})"
        )
