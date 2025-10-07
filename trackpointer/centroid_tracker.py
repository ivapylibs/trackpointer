from dataclasses import dataclass
from typing import List
import numpy as np
from detector.detector.types import HandOutput
from detector.detector.feats import palm_from_landmarks, centroid_from_palm
from .ema import EMA

@dataclass
class CentroidTrack:
    label: str
    present: bool
    score: float
    centroid: np.ndarray  # (2,)

class CentroidTracker:
    def __init__(self, alpha=0.4):
        self.filters = {"left": EMA(alpha), "right": EMA(alpha)}
    def update(self, dets: List[HandOutput]) -> List[CentroidTrack]:
        out = []
        seen = set()
        for d in dets:
            f = self.filters[d.label]
            if d.present:
                seen.add(d.label)
                cen = centroid_from_palm(palm_from_landmarks(d.landmarks))
                sm = f(cen)  # EMA over (2,)
                out.append(CentroidTrack(d.label, True, d.score, sm))
            else:
                f.reset()
                out.append(CentroidTrack(d.label, False, np.nan, np.array([np.nan, np.nan], np.float32)))
        for k in ("left","right"):
            if k not in seen:
                self.filters[k].reset()
        return out
