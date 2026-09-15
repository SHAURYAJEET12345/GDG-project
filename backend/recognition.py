"""
recognition.py — Face encoding, loading, and real-time identification.

Workflow
--------
1.  At startup, load all known face images from `known_faces/` (JPEG/PNG).
    File names (without extension) are treated as person names.
2.  For each incoming frame, locate face bounding boxes, compute 128-d
    encodings, compare against the known set, and return the best match.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

# Default tolerance: lower → stricter matching (0.4–0.6 recommended)
DEFAULT_TOLERANCE = 0.5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
class FaceDatabase:
    """Holds all known encodings and their labels."""

    def __init__(self) -> None:
        self.encodings: list[np.ndarray] = []
        self.names: list[str] = []

    def __len__(self) -> int:
        return len(self.names)

    def add(self, name: str, encoding: np.ndarray) -> None:
        self.names.append(name)
        self.encodings.append(encoding)

    def is_empty(self) -> bool:
        return len(self.names) == 0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_known_faces(known_faces_dir: str | Path) -> FaceDatabase:
    """
    Scan *known_faces_dir* for images and build a FaceDatabase.

    File naming convention:
        John_Doe.jpg   →  name = "John Doe"  (underscores become spaces)
        alice.png      →  name = "alice"
    """
    db = FaceDatabase()
    known_faces_dir = Path(known_faces_dir)

    if not known_faces_dir.exists():
        logger.warning("known_faces directory not found: %s", known_faces_dir)
        return db

    for img_path in sorted(known_faces_dir.iterdir()):
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        name = img_path.stem.replace("_", " ")
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                logger.warning("No face found in %s — skipping", img_path.name)
                continue
            if len(encodings) > 1:
                logger.warning(
                    "%s contains %d faces; using only the first.",
                    img_path.name, len(encodings),
                )
            db.add(name, encodings[0])
            logger.info("Loaded face for '%s' from %s", name, img_path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", img_path.name, exc)

    logger.info("Face database loaded: %d known person(s)", len(db))
    return db


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def identify_faces(
    frame_bgr: np.ndarray,
    db: FaceDatabase,
    tolerance: float = DEFAULT_TOLERANCE,
    model: str = "hog",          # "hog" (fast, CPU) or "cnn" (accurate, GPU)
    upsample: int = 1,
) -> list[dict]:
    """
    Detect and identify all faces in *frame_bgr*.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image from OpenCV.
    db : FaceDatabase
        Known faces database.
    tolerance : float
        Match threshold. Lower = stricter.
    model : str
        Detection model ("hog" or "cnn").
    upsample : int
        Upsample factor for small-face detection.

    Returns
    -------
    list of dicts:
        {
            "name":       str,           # "Unknown" if no match
            "confidence": float,         # 0.0–1.0  (1 - distance)
            "location":   tuple,         # (top, right, bottom, left)
            "matched":    bool,
        }
    """
    # face_recognition expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize for speed (optional — keeps original coords via scale factor)
    scale = 0.5
    small = cv2.resize(frame_rgb, (0, 0), fx=scale, fy=scale)

    locations = face_recognition.face_locations(small, number_of_times_to_upsample=upsample, model=model)
    encodings = face_recognition.face_encodings(small, locations)

    results: list[dict] = []

    for enc, loc in zip(encodings, locations):
        top, right, bottom, left = loc
        # Scale back to original resolution
        orig_loc = (
            int(top / scale), int(right / scale),
            int(bottom / scale), int(left / scale),
        )

        name = "Unknown"
        confidence = 0.0
        matched = False

        if not db.is_empty():
            distances = face_recognition.face_distance(db.encodings, enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = round(max(0.0, 1.0 - best_dist), 3)

            if best_dist <= tolerance:
                name = db.names[best_idx]
                matched = True

        results.append(
            {
                "name": name,
                "confidence": confidence,
                "location": orig_loc,
                "matched": matched,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Registration helper (used by /register endpoint)
# ---------------------------------------------------------------------------
def register_face(
    frame_bgr: np.ndarray,
    name: str,
    known_faces_dir: str | Path,
    db: FaceDatabase,
) -> dict:
    """
    Extract the first face from *frame_bgr*, save it to *known_faces_dir*,
    and hot-add it to the in-memory *db*.

    Returns dict with "success" bool and "message" str.
    """
    known_faces_dir = Path(known_faces_dir)
    known_faces_dir.mkdir(parents=True, exist_ok=True)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(frame_rgb)

    if not encodings:
        return {"success": False, "message": "No face detected in the provided image."}

    enc = encodings[0]
    safe_name = name.strip().replace(" ", "_")
    dest = known_faces_dir / f"{safe_name}.jpg"

    # Save image (BGR is fine for JPEG)
    cv2.imwrite(str(dest), frame_bgr)

    db.add(name.strip(), enc)
    logger.info("Registered new face: '%s' → %s", name, dest)

    return {"success": True, "message": f"Face registered for '{name}'."}
