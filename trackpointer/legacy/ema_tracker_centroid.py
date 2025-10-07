# trackpointer/trackpointer/ema_tracker_centroid.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Track:
    id: str
    label: str
    score: float
    landmarks: np.ndarray
    palm: Optional[np.ndarray]
    fingers: Optional[np.ndarray]
    centroid: Optional[np.ndarray]   # (2,) normalized xy or None
    present: bool

class EMA:
    def __init__(self, alpha=0.4):
        self.alpha, self.prev = alpha, None
    def __call__(self, x):
        if self.prev is None or not isinstance(self.prev, np.ndarray) or self.prev.shape != x.shape or not np.all(np.isfinite(self.prev)):
            self.prev = x
        else:
            self.prev = self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev
    def reset(self): self.prev = None

class EMAHandTracker:
    def __init__(self, alpha=0.4):
        self.filters = {"left": EMA(alpha), "right": EMA(alpha)}
        self._palm_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)

    def update(self, dets) -> List[Track]:
        out: List[Track] = []
        seen = set()

        for d in dets:
            k = d.label  # "left" or "right"
            if d.present:
                seen.add(k)
                sm = self.filters[k](d.landmarks)  # smooth all 21x3
                # recompute palm/fingers from smoothed landmarks for temporal consistency
                palm = sm[self._palm_idx, :]
                fingers = np.delete(sm, self._palm_idx, axis=0)

                # centroid: prefer detector-provided; else from smoothed palm; else None
                cen = getattr(d, "centroid", None)
                if cen is None and np.all(np.isfinite(palm)):
                    cen = palm[:, :2].mean(axis=0)

                out.append(Track(k, k, d.score, sm, palm, fingers, cen, True))
            else:
                # absent: pass through NaNs, keep centroid None
                out.append(Track(k, k, np.nan, d.landmarks, d.palm, d.fingers, None, False))

        # reset filters for hands not seen this frame
        for k in ("left", "right"):
            if k not in seen:
                self.filters[k].reset()

        return out
