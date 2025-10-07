from dataclasses import dataclass
from typing import List
import numpy as np
from detector.detector.types import HandOutput
from .ema import EMA

@dataclass
class HandTrack:
    label: str
    present: bool
    score: float
    landmarks: np.ndarray  # (21,3)

class HandLandmarksTracker:
    def __init__(self, alpha=0.4):
        self.filters = {"left": EMA(alpha), "right": EMA(alpha)}
    def update(self, dets: List[HandOutput]) -> List[HandTrack]:
        out = []
        seen = set()
        for d in dets:
            f = self.filters[d.label]
            if d.present:
                seen.add(d.label)
                sm = f(d.landmarks)
                out.append(HandTrack(d.label, True, d.score, sm))
            else:
                f.reset()
                out.append(HandTrack(d.label, False, np.nan, d.landmarks))
        for k in ("left","right"):
            if k not in seen:
                self.filters[k].reset()
        return out
