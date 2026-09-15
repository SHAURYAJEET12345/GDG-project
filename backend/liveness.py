"""
liveness.py — Eye Aspect Ratio (EAR) based blink detection for liveness verification.

Uses dlib's 68-point facial landmark predictor to compute per-eye EAR and
detect genuine blinks, preventing spoofing via static photos or replay videos.
"""

from __future__ import annotations

from collections import deque

import cv2
import dlib  # type: ignore[import-untyped]
import numpy as np
from scipy.spatial import distance as dist

# ---------------------------------------------------------------------------
# Landmark indices (iBUG 68-point model)
# ---------------------------------------------------------------------------
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# EAR threshold: eyes below this value are considered "closed"
EAR_THRESHOLD = 0.25
# Minimum consecutive frames the eye must be below threshold to count as a blink
EAR_CONSEC_FRAMES = 2
# Number of blinks required to confirm liveness
REQUIRED_BLINKS = 2


def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio from 6 (x, y) landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Guard against division-by-zero (degenerate face rect)
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


def landmarks_to_np(shape: dlib.full_object_detection, dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
    """
    Convert a dlib full_object_detection shape to a (68, 2) NumPy array.

    Uses float64 (not float32) to avoid precision loss in euclidean distance
    calculations used by _eye_aspect_ratio.
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


class LivenessDetector:
    """
    Stateful blink-based liveness detector.

    Usage
    -----
    detector = LivenessDetector(predictor_path="shape_predictor_68_face_landmarks.dat")
    for frame in video_frames:
        result = detector.process_frame(frame)
        if result["is_live"]:
            break
    """

    def __init__(
        self,
        predictor_path: str = "shape_predictor_68_face_landmarks.dat",
        required_blinks: int = REQUIRED_BLINKS,
        ear_threshold: float = EAR_THRESHOLD,
        ear_consec_frames: int = EAR_CONSEC_FRAMES,
    ) -> None:
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.required_blinks  = required_blinks
        self.ear_threshold    = ear_threshold
        self.ear_consec_frames = ear_consec_frames

        # Rolling EAR history for temporal smoothing
        self._ear_history: deque[float] = deque(maxlen=5)
        self._consec_below: int  = 0
        self._blink_count:  int  = 0
        self._is_live:      bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all state — call between different users / sessions."""
        self._ear_history.clear()
        self._consec_below = 0
        self._blink_count  = 0
        self._is_live      = False

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Analyse a single BGR frame and update liveness state.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (e.g., decoded from webcam base64 payload).

        Returns
        -------
        dict with keys:
            faces_found  : int
            ear          : float | None   (smoothed EAR value, None if no face)
            blink_count  : int
            is_live      : bool
            message      : str
        """
        result: dict = {
            "faces_found": 0,
            "ear":         None,
            "blink_count": self._blink_count,
            "is_live":     self._is_live,
            "message":     "No face detected",
        }

        # Short-circuit once liveness is confirmed
        if self._is_live:
            result["message"] = "Liveness confirmed ✓"
            return result

        if frame is None or frame.size == 0:
            result["message"] = "Empty frame received"
            return result

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        result["faces_found"] = len(rects)

        if not rects:
            return result

        # Process only the largest detected face
        face_rect = max(rects, key=lambda face: face.width() * face.height())
        shape     = self.predictor(gray, face_rect)
        coords    = landmarks_to_np(shape)

        left_eye  = coords[LEFT_EYE_IDX]
        right_eye = coords[RIGHT_EYE_IDX]

        left_ear  = _eye_aspect_ratio(left_eye)
        right_ear = _eye_aspect_ratio(right_eye)
        ear       = (left_ear + right_ear) / 2.0

        self._ear_history.append(ear)
        smoothed_ear = float(np.mean(self._ear_history))
        result["ear"] = round(smoothed_ear, 4)

        # Blink detection: count a blink when EAR rises back above threshold
        # after being below it for at least ear_consec_frames frames.
        if smoothed_ear < self.ear_threshold:
            self._consec_below += 1
        else:
            if self._consec_below >= self.ear_consec_frames:
                self._blink_count += 1
            self._consec_below = 0

        result["blink_count"] = self._blink_count

        if self._blink_count >= self.required_blinks:
            self._is_live      = True
            result["is_live"]  = True
            result["message"]  = "Liveness confirmed ✓"
        else:
            remaining = self.required_blinks - self._blink_count
            result["message"] = f"Please blink {remaining} more time(s) to verify"

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def draw_debug(frame: np.ndarray, result: dict) -> np.ndarray:
        """Overlay EAR and blink info onto a frame (for debugging only)."""
        overlay = frame.copy()
        color   = (0, 255, 0) if result["is_live"] else (0, 165, 255)
        cv2.putText(
            overlay,
            f"EAR: {result.get('ear', 'N/A')}  Blinks: {result['blink_count']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            result["message"],
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
        return overlay
