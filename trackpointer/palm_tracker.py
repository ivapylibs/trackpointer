from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from detector.types import HandOutput

@dataclass
class PalmTrack:
    label: str
    present: bool
    score: float
    palm: Optional[np.ndarray]   # (6,3) or None

class PalmTracker:
    """Association-only. No filtering—smoothing is done in perceiver."""
    def __init__(self):
        pass

    def update(self, dets: List[HandOutput]) -> List[PalmTrack]:
        out: List[PalmTrack] = []
        for d in dets:
            if d.present and isinstance(d.palm, np.ndarray):
                out.append(PalmTrack(d.label, True, d.score, d.palm))
            else:
                out.append(PalmTrack(d.label, False, np.nan, d.palm))
        return out
