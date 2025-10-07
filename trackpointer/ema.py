import numpy as np

class EMA:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev = None
    def __call__(self, x):
        x = np.asarray(x)
        if self.prev is None or self.prev.shape != x.shape or not np.all(np.isfinite(self.prev)):
            self.prev = x
            return x
        self.prev = self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev
    def reset(self):
        self.prev = None
