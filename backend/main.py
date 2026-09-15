"""
main.py — FastAPI application for the Smart Attendance System.

Routes
------
GET    /                       Health check
POST   /register               Register a new face (base64 image + name)
POST   /verify                 Liveness + recognition pipeline (base64 frame)
POST   /verify/reset-liveness  Reset blink counter for a new session
GET    /attendance/records     Return all attendance logs
DELETE /attendance/clear       Clear all attendance records

Environment Variables
---------------------
KNOWN_FACES_DIR : path to reference face images  (default: ./known_faces)
PREDICTOR_PATH  : dlib 68-pt landmark .dat file  (default: ./shape_predictor_68_face_landmarks.dat)
DB_PATH         : SQLite file path               (default: ./database.db)
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy import text

try:
    # Package mode: uvicorn backend.main:app  (PYTHONPATH=/app)
    from backend.recognition import FaceDatabase, load_known_faces, identify_faces, register_face
    from backend.liveness import LivenessDetector
except ImportError:
    # Local dev mode: python main.py  (cwd = backend/)
    from recognition import FaceDatabase, load_known_faces, identify_faces, register_face  # type: ignore
    from liveness import LivenessDetector  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("attendance")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR = Path(os.getenv("KNOWN_FACES_DIR", "./known_faces"))
PREDICTOR_PATH  = os.getenv("PREDICTOR_PATH", "./shape_predictor_68_face_landmarks.dat")
DB_PATH         = os.getenv("DB_PATH", "./database.db")

# ---------------------------------------------------------------------------
# Database setup (SQLite via SQLAlchemy core)
# ---------------------------------------------------------------------------
engine = sa.create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

metadata = sa.MetaData()
attendance_table = sa.Table(
    "attendance",
    metadata,
    sa.Column("id",         sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("name",       sa.String,  nullable=False),
    sa.Column("timestamp",  sa.String,  nullable=False),
    sa.Column("confidence", sa.Float,   nullable=False),
    sa.Column("status",     sa.String,  nullable=False),  # "present" | "unknown" | "spoof"
)
# NOTE: metadata.create_all() is called inside lifespan() so the correct
# DB_PATH env-var is guaranteed to be resolved before the engine is used.

# ---------------------------------------------------------------------------
# Shared in-memory state
# ---------------------------------------------------------------------------
face_db: FaceDatabase = FaceDatabase()
liveness_detector: LivenessDetector | None = None

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; nothing to clean on shutdown."""
    global face_db, liveness_detector

    logger.info("Loading known faces from: %s", KNOWN_FACES_DIR)
    face_db = load_known_faces(KNOWN_FACES_DIR)

    # Create DB tables now that env-vars are confirmed loaded
    metadata.create_all(engine)
    logger.info("Database ready at: %s", DB_PATH)

    logger.info("Initialising liveness detector (predictor: %s)", PREDICTOR_PATH)
    try:
        liveness_detector = LivenessDetector(predictor_path=PREDICTOR_PATH)
        logger.info("Liveness detector ready.")
    except Exception as exc:          # dlib raises generic Exception on missing .dat
        logger.error("Failed to init liveness detector: %s", exc)
        liveness_detector = None

    yield   # application runs here

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Attendance System",
    description="Liveness detection + face recognition based attendance logging",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FramePayload(BaseModel):
    image:      str           # base64-encoded JPEG/PNG frame
    session_id: str = "default"


class RegisterPayload(BaseModel):
    image: str                # base64-encoded JPEG/PNG
    name:  str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_b64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 (or data-URI) image string to a BGR OpenCV array."""
    if "," in b64_str:                          # strip "data:image/jpeg;base64,"
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image from base64 payload.")
    return frame


def _log_attendance(name: str, confidence: float, status: str) -> None:
    """Insert an attendance record, deduplicating within the same minute."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with engine.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT id FROM attendance "
                "WHERE name=:name AND timestamp LIKE :prefix"
            ),
            {"name": name, "prefix": ts[:16] + "%"},
        ).fetchone()
        if not existing:
            conn.execute(
                attendance_table.insert().values(
                    name=name, timestamp=ts,
                    confidence=confidence, status=status,
                )
            )
            logger.info("Logged attendance: %s [%s] %.2f", name, status, confidence)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health():
    return {
        "status":           "ok",
        "known_faces":      len(face_db),
        "liveness_detector": liveness_detector is not None,
    }


@app.post("/register", tags=["Registration"])
async def register(payload: RegisterPayload):
    """Register a new face into the system from a base64 image."""
    try:
        frame = _decode_b64_image(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    result = register_face(frame, payload.name, KNOWN_FACES_DIR, face_db)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.post("/verify", tags=["Verification"])
async def verify(payload: FramePayload):
    """
    Full pipeline for a single webcam frame:
      1. Decode base64 image
      2. Liveness check (EAR blink counting)
      3. Face recognition
      4. Log attendance to SQLite
    """
    try:
        frame = _decode_b64_image(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # --- Liveness ---
    liveness_result: dict = {
        "faces_found": 0,
        "ear":         None,
        "blink_count": 0,
        "is_live":     False,
        "message":     "Liveness detector not available",
    }
    if liveness_detector is not None:
        liveness_result = liveness_detector.process_frame(frame)

    # --- Recognition ---
    recognition_results = identify_faces(frame, face_db)

    # --- Attendance logging ---
    logged: list[str] = []
    if liveness_result["is_live"] and recognition_results:
        for r in recognition_results:
            status = "present" if r["matched"] else "unknown"
            _log_attendance(r["name"], r["confidence"], status)
            logged.append(r["name"])
    elif not liveness_result["is_live"] and recognition_results:
        # Possible spoof — log for auditing but don't mark present
        for r in recognition_results:
            _log_attendance(r.get("name", "Unknown"), r["confidence"], "spoof")

    return {
        "liveness":    liveness_result,
        "recognition": recognition_results,
        "logged":      logged,
        "frame_time":  datetime.utcnow().isoformat(),
    }


@app.post("/verify/reset-liveness", tags=["Verification"])
async def reset_liveness():
    """Reset the blink counter — call between different users."""
    if liveness_detector is not None:
        liveness_detector.reset()
    return {"message": "Liveness detector reset."}


@app.get("/attendance/records", tags=["Attendance"])
async def get_records(limit: int = 100):
    """Fetch the latest attendance records (newest first)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, name, timestamp, confidence, status "
                "FROM attendance ORDER BY id DESC LIMIT :limit"
            ),
            {"limit": limit},
        ).fetchall()
    return [
        {
            "id":         r[0],
            "name":       r[1],
            "timestamp":  r[2],
            "confidence": round(r[3], 3),
            "status":     r[4],
        }
        for r in rows
    ]


@app.delete("/attendance/clear", tags=["Attendance"])
async def clear_records():
    """Delete all attendance logs."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM attendance"))
    return {"message": "All attendance records cleared."}


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
