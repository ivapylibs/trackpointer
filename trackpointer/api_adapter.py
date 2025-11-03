# trackpointer/trackpointer/api_adapter.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Phase-1 types
from perceiver.perceiver.types import Detections, Tracks

# Reuse your HandOutput structure so we can feed existing trackers unchanged.
# (These come from your detector implementation used in the demo.)
from detector.detector.mediapipe_hands import HandOutput, make_hand_nan

# Your concrete trackers
from trackpointer.trackpointer.hand_tracker import HandLandmarksTracker
from trackpointer.trackpointer.palm_tracker import PalmTracker
from trackpointer.trackpointer.centroid_tracker import CentroidTracker


# Map tracker type -> payload key we will propagate into Tracks.items
_FEATURE_KEY: Dict[str, str] = {
    "hand": "landmarks",
    "palm": "palm",
    "centroid": "centroid",
}


class APITrackpointerAdapter:
    """
    Adapter to expose legacy trackers under the new API:

        update(detections: Detections, timestamp: Optional[float]) -> Tracks

    Internally:
    - Builds [left, right] HandOutput list from Detections (filling missing with NaNs)
    - Calls your existing tracker.update(hands) unchanged
    - Packs the tracker outputs into a Tracks object with one primary payload key
    """

    def __init__(self, tracker_type: str = "hand", **tracker_kwargs: Any) -> None:
        if tracker_type not in _FEATURE_KEY:
            raise ValueError(f"Unknown tracker_type '{tracker_type}'. Valid: {list(_FEATURE_KEY.keys())}")
        self.tracker_type = tracker_type
        self.payload_key = _FEATURE_KEY[tracker_type]

        # Instantiate the concrete tracker exactly as you do in the demo class
        if tracker_type == "hand":
            self._trk = HandLandmarksTracker(**tracker_kwargs)
        elif tracker_type == "palm":
            self._trk = PalmTracker(**tracker_kwargs)
        else:
            self._trk = CentroidTracker(**tracker_kwargs)

        # ---- add this alias so update() and debug lines can use a single name
        self.inner = self._trk

        # (optional) one-time sanity print
        # print("[APITrackpointerAdapter] inner =", type(self.inner).__name__)

    # --- New API ---
    def update(self, detections: Detections, timestamp: Optional[float] = None) -> Tracks:
        """
        Convert Detections -> [HandOutput, HandOutput], call legacy tracker.update(),
        then pack result into Tracks(items=[...], meta={...}).
        """
        # 1) Build [left, right] HandOutput from detections (normalized coordinates)
        hands_lr: List[HandOutput] = self._detections_to_hands_lr(detections)

        # 2) Call your existing tracker method (kept unchanged)
        tracks_out = self.inner.update(hands_lr)

        # 3) Convert the tracker outputs into our Tracks phase-1 type
        items: List[Dict[str, Any]] = []

        # `tracks_out` is whatever your tracker currently returns in the demo.
        # In the demo you iterate over `tracks` (likely a list of HandOutput-like objects).
        for t in tracks_out:
            label = getattr(t, "label", None)
            present = bool(getattr(t, "present", False))

            # ID: if your tracker provides a stable ID, prefer it; else fallback to label
            tid = getattr(t, "id", None)
            if tid is None:
                tid = label if label is not None else "track"

            payload = getattr(t, self.payload_key, None)

            # Only add present tracks that have a valid payload
            if not present or payload is None:
                continue

            # Ensure numpy array payloads are well-formed
            if isinstance(payload, np.ndarray) and (not np.isfinite(payload).all()):
                continue  # skip NaN payloads

            item: Dict[str, Any] = {
                "id": tid,
                "label": label,
                self.payload_key: payload,
            }
            # If the tracker exposed score/confidence, pass it through
            score = getattr(t, "score", None)
            if score is not None:
                try:
                    item["score"] = float(score)
                except Exception:
                    pass

            items.append(item)

        meta = {
            "timestamp": timestamp,
            "tracker": self.tracker_type,
            "num_tracks": len(items),
        }
        if items:
            print("TRK_CLASS:", type(self.inner).__name__)
            print("DICT_KEYS_0:", list(items[0].keys()))
        return Tracks(items=items, meta=meta)

    def reset(self) -> None:
        """Reset the underlying tracker if it exposes a reset; otherwise reinit."""
        if hasattr(self._trk, "reset") and callable(self._trk.reset):
            self._trk.reset()
        else:
            # Fall back to re-instantiation if no explicit reset is available
            self.__init__(self.tracker_type)

    # --- Helpers --------------------------------------------------------

    def _detections_to_hands_lr(self, detections: Detections) -> List[HandOutput]:
        """
        Create a stable (left, right) HandOutput list from Detections.
        Missing side is filled with a NaN hand so legacy tracker.update(hands) stays happy.
        """
        left = make_hand_nan("left")
        right = make_hand_nan("right")

        # Sort by score descending; pick top two
        def _score(it: Dict[str, Any]) -> float:
            try:
                return float(it.get("score", 0.0))
            except Exception:
                return 0.0

        items_sorted = sorted(detections.items, key=_score, reverse=True)  # type: ignore[attr-defined]

        top = items_sorted[:2] if items_sorted is not None else []

        # Fill left/right slots
        for it in top:
            lab = (it.get("label") or "").lower()
            present = True
            score = float(it.get("score", 0.0))

            # Collect candidate arrays (may be None)
            landmarks = it.get("landmarks", None)
            palm      = it.get("palm", None)
            fingers   = it.get("fingers", None)
            centroid  = it.get("centroid", None)

            # Build HandOutput object
            h = HandOutput(
                label=lab if lab in ("left", "right") else ("left" if left.present is False else "right"),
                present=present,
                score=score,
                landmarks=self._as_array(landmarks, shape=(21, 3), fill=np.nan),
                palm=self._as_array(palm, shape=(6, 3), fill=np.nan),
                fingers=self._as_array(fingers, shape=(15, 3), fill=np.nan),
                centroid=self._as_centroid(centroid),
            )

            if h.label == "left":
                left = h
            else:
                right = h

        return [left, right]

    @staticmethod
    def _as_array(x: Any, shape: Tuple[int, int], fill: float) -> np.ndarray:
        """
        Ensure x is a float32 numpy array with given shape; if None or wrong shape,
        return an array of fill values (NaNs).
        """
        if isinstance(x, np.ndarray) and x.shape == shape:
            return x.astype(np.float32, copy=False)
        # Accept list-like and try reshape if size matches
        try:
            arr = np.asarray(x, dtype=np.float32)
            if arr.shape == shape:
                return arr
        except Exception:
            pass
        return np.full(shape, fill, dtype=np.float32)

    @staticmethod
    def _as_centroid(x: Any) -> Optional[np.ndarray]:
        """
        Normalize centroid to shape (2,) float32 if possible; otherwise None.
        Accepts (2,), (1,2), or (2,1).
        """
        if x is None:
            return None
        try:
            arr = np.asarray(x, dtype=np.float32)
            if arr.shape == (2,):
                return arr
            if arr.shape == (1, 2):
                return arr.reshape(2,)
            if arr.shape == (2, 1):
                return arr.reshape(2,)
        except Exception:
            pass
        return None
