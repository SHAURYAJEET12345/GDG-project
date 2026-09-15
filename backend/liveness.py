"""
liveness.py — Eye Aspect Ratio (EAR) based blink detection for liveness verification.

Uses dlib's 68-point facial landmark predictor to compute per-eye EAR and
detect genuine blinks, preventing spoofing via static photos or replay videos.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import cv2
import dlib

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
    # Vertical distances
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return float(ear)


def landmarks_to_np(shape, dtype=np.float32) -> np.ndarray:
    """Convert dlib shape object to (68, 2) NumPy array."""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


class LivenessDetector:
    """
    Stateful blink-based liveness detector.

    Usage
    -----
    detector = LivenessDetector()
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
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.required_blinks = required_blinks
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames

        # Rolling history for smoothing EAR
        self._ear_history: deque = deque(maxlen=5)
        self._consec_below: int = 0
        self._blink_count: int = 0
        self._is_live: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset blink counter (call between sessions / users)."""
        self._ear_history.clear()
        self._consec_below = 0
        self._blink_count = 0
        self._is_live = False

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Analyse a single BGR frame and update liveness state.

        Returns
        -------
        dict with keys:
            faces_found  : int
            ear          : float | None
            blink_count  : int
            is_live      : bool
            message      : str
        """
        result = {
            "faces_found": 0,
            "ear": None,
            "blink_count": self._blink_count,
            "is_live": self._is_live,
            "message": "No face detected",
        }

        if self._is_live:
            result["message"] = "Liveness confirmed ✓"
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        result["faces_found"] = len(rects)

        if not rects:
            return result

        # Use the largest detected face
        rect = max(rects, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, rect)
        coords = landmarks_to_np(shape)

        left_eye  = coords[LEFT_EYE_IDX]
        right_eye = coords[RIGHT_EYE_IDX]

        left_ear  = _eye_aspect_ratio(left_eye)
        right_ear = _eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        self._ear_history.append(ear)
        smoothed_ear = float(np.mean(self._ear_history))
        result["ear"] = round(smoothed_ear, 4)

        if smoothed_ear < self.ear_threshold:
            self._consec_below += 1
        else:
            if self._consec_below >= self.ear_consec_frames:
                self._blink_count += 1
            self._consec_below = 0

        result["blink_count"] = self._blink_count

        if self._blink_count >= self.required_blinks:
            self._is_live = True
            result["is_live"] = True
            result["message"] = "Liveness confirmed ✓"
        else:
            remaining = self.required_blinks - self._blink_count
            result["message"] = f"Please blink {remaining} more time(s) to verify"

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def draw_debug(frame: np.ndarray, result: dict) -> np.ndarray:
        """Overlay EAR and blink info onto frame (for debugging)."""
        overlay = frame.copy()
        color = (0, 255, 0) if result["is_live"] else (0, 165, 255)
        cv2.putText(
            overlay,
            f"EAR: {result.get('ear', 'N/A')}  Blinks: {result['blink_count']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        cv2.putText(
            overlay,
            result["message"],
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        return overlay
