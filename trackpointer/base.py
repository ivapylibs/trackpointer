from abc import ABC, abstractmethod
from typing import List
import numpy as np
from detector.detector.types import HandOutput

class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[HandOutput]):
        ...
