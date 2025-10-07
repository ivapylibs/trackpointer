# trackpointer/trackpointer/ema_tracker.py
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Track:
    id: str           # "left" or "right"
    label: str
    score: float
    landmarks: np.ndarray
    palm: np.ndarray | None
    fingers: np.ndarray | None
    #centroid: np.ndarray | None
    present: bool

class EMA:
    def __init__(self, alpha=0.4): self.alpha, self.prev = alpha, None
    def __call__(self, x):
        if self.prev is None or not np.all(np.isfinite(self.prev)):
            self.prev = x
        else:
            self.prev = self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev
    def reset(self): self.prev = None

class EMAHandTracker:
    def __init__(self, alpha=0.4):
        self.filters = {"left": EMA(alpha), "right": EMA(alpha)}

    def update(self, dets) -> List[Track]:
        out = []
        seen = set()
        for d in dets:
            k = d.label
            if d.present:
                seen.add(k)
                sm = self.filters[k](d.landmarks)
                out.append(Track(k, k, d.score, sm, d.palm, d.fingers, True))
            else:
                out.append(Track(k, k, np.nan, d.landmarks, d.palm, d.fingers, False))
        for k in ("left", "right"):
            if k not in seen:
                self.filters[k].reset()
        return out
