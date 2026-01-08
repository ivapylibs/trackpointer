from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from detector.types import HandOutput

@dataclass
class CentroidTrack:
    label: str
    present: bool
    score: float
    centroid: Optional[np.ndarray]   # (2,) or (3,) normalized, or None

class CentroidTracker:
    """Association-only. No filtering—smoothing is done in perceiver."""
    def __init__(self):
        pass

    def update(self, dets: List[HandOutput]) -> List[CentroidTrack]:
        out: List[CentroidTrack] = []
        for d in dets:
            if d.present and isinstance(d.centroid, np.ndarray):
                out.append(CentroidTrack(d.label, True, d.score, d.centroid))
            else:
                out.append(CentroidTrack(d.label, False, np.nan, d.centroid))
        return out
