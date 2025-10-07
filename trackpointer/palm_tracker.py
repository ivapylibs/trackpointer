from dataclasses import dataclass
from typing import List
import numpy as np
from detector.detector.types import HandOutput
from detector.detector.feats import palm_from_landmarks
from .ema import EMA

@dataclass
class PalmTrack:
    label: str
    present: bool
    score: float
    palm: np.ndarray  # (6,3)

class PalmTracker:
    def __init__(self, alpha=0.4):
        self.filters = {"left": EMA(alpha), "right": EMA(alpha)}
    def update(self, dets: List[HandOutput]) -> List[PalmTrack]:
        out = []
        seen = set()
        for d in dets:
            f = self.filters[d.label]
            if d.present:
                seen.add(d.label)
                palm = palm_from_landmarks(d.landmarks)
                sm = f(palm)
                out.append(PalmTrack(d.label, True, d.score, sm))
            else:
                f.reset()
                out.append(PalmTrack(d.label, False, np.nan, np.full((6,3), np.nan, np.float32)))
        for k in ("left","right"):
            if k not in seen:
                self.filters[k].reset()
        return out
