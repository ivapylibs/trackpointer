# trackpointer/trackpointer/hand_tracker.py
from dataclasses import dataclass
from typing import List
import numpy as np
from detector.detector.types import HandOutput

@dataclass
class HandTrack:
    label: str
    present: bool
    score: float
    landmarks: np.ndarray  # (21,3)

class HandLandmarksTracker:
    """
    Association-only tracker:
    - No EMA (filtering now lives in perceiver).
    - Does not compute or change features; just forwards detector outputs.
    """
    def __init__(self):
        pass

    def update(self, dets: List[HandOutput]) -> List[HandTrack]:
        out: List[HandTrack] = []
        for d in dets:
            if d.present:
                out.append(HandTrack(d.label, True, d.score, d.landmarks))
            else:
                out.append(HandTrack(d.label, False, np.nan, d.landmarks))
        return out
