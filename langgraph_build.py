"""
langgraph_build.py
==================
LangGraph multi-agent orchestration script for the
Smart Attendance System with Liveness Detection.

Agents
------
1. Supervisor Agent      — Initialises state, routes between agents, validates
2. Backend Agent         — Generates FastAPI + OpenCV + liveness detection code
3. Frontend Agent        — Generates HTML + Tailwind CSS + JavaScript
4. Integration Agent     — Patches CORS, aligns API endpoints, writes DevOps config
5. QA / DevOps Agent     — Writes Dockerfile, docker-compose, nginx.conf, README
6. Git Push Node         — git init → add → commit → push to GitHub

Run
---
    pip install langgraph langchain-core
    python langgraph_build.py

The script will create/overwrite all project files under ./  (current dir)
and attempt to push to the configured GitHub repository.
"""

from __future__ import annotations

import os
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TypedDict, Annotated, Sequence
import operator

try:
  _graph = importlib.import_module("langgraph.graph")
  StateGraph = _graph.StateGraph
  END = _graph.END
except (ImportError, AttributeError) as exc:
  raise ImportError(
    "LangGraph is not installed in the active Python environment. "
    "Run: python -m pip install -U langgraph"
  ) from exc
try:
  _messages = importlib.import_module("langchain_core.messages")
  BaseMessage = _messages.BaseMessage
  HumanMessage = _messages.HumanMessage
  AIMessage = _messages.AIMessage
except (ImportError, AttributeError) as exc:
  raise ImportError(
    "LangChain Core is not installed in the active Python environment. "
    "Run: python -m pip install -U langchain-core"
  ) from exc

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
GITHUB_REPO_URL  = "https://github.com/SHAURYAJEET12345/GDG-project.git"
COMMIT_MESSAGE   = "Initial commit: Smart Attendance System with LangGraph multi-agent build"
PROJECT_ROOT     = Path(__file__).parent.resolve()
REQUIRED_BLINKS  = 2  # kept in sync with frontend


# ──────────────────────────────────────────────────────────────────────────────
# Shared State  (TypedDict — passed between every node)
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], operator.add]
    files:      dict[str, str]          # relative_path → file_content
    status:     dict[str, str]          # agent_name   → "done" | "pending" | "error"
    errors:     list[str]
    phase:      str                     # current pipeline phase label


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def _write_files(state: AgentState) -> None:
    """Flush all files in state['files'] to disk."""
    for rel_path, content in state["files"].items():
        abs_path = PROJECT_ROOT / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Written: {rel_path}")


def _run(cmd: str, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a shell command; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd or PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _log(agent: str, msg: str) -> None:
    print(f"\n[{agent.upper()}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 — Supervisor Agent
# ──────────────────────────────────────────────────────────────────────────────
def supervisor_agent(state: AgentState) -> AgentState:
    _log("supervisor", "Initialising build pipeline…")

    new_messages = [
        HumanMessage(content=(
            "Build the Smart Attendance System with Liveness Detection. "
            "Generate all backend, frontend, and DevOps files. "
            "Then push to GitHub."
        )),
        AIMessage(content=(
            "Understood. Routing to Backend Agent → Frontend Agent → "
            "Integration Agent → DevOps Agent → Git Push."
        )),
    ]

    return {
        **state,
        "messages": new_messages,
        "files":    {},
        "status":   {
            "supervisor":   "done",
            "backend":      "pending",
            "frontend":     "pending",
            "integration":  "pending",
            "devops":       "pending",
            "git":          "pending",
        },
        "errors": [],
        "phase":  "backend",
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 — Backend Developer Agent
# ──────────────────────────────────────────────────────────────────────────────
def backend_agent(state: AgentState) -> AgentState:
    _log("backend", "Generating FastAPI backend, recognition, liveness modules…")

    files: dict[str, str] = {}

    # ── backend/requirements.txt ──────────────────────────────────────────────
    files["backend/requirements.txt"] = textwrap.dedent("""\
        fastapi==0.111.0
        uvicorn[standard]==0.29.0
        python-multipart==0.0.9
        pydantic==2.7.1
        opencv-python-headless==4.9.0.80
        face_recognition==1.3.0
        dlib==19.24.2
        numpy==1.26.4
        scipy==1.13.0
        sqlalchemy==2.0.30
        python-dotenv==1.0.1
    """)

    # ── backend/liveness.py ───────────────────────────────────────────────────
    files["backend/liveness.py"] = textwrap.dedent(f'''\
        """
        liveness.py — EAR-based blink detection for anti-spoofing.
        """
        import numpy as np
        from scipy.spatial import distance as dist
        from collections import deque
        import cv2
        import dlib

        LEFT_EYE_IDX  = list(range(36, 42))
        RIGHT_EYE_IDX = list(range(42, 48))
        EAR_THRESHOLD = 0.25
        EAR_CONSEC_FRAMES = 2
        REQUIRED_BLINKS = {REQUIRED_BLINKS}

        def _ear(eye: np.ndarray) -> float:
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)

        def landmarks_to_np(shape) -> np.ndarray:
            coords = np.zeros((68, 2), dtype=np.float32)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            return coords

        class LivenessDetector:
            def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat",
                         required_blinks=REQUIRED_BLINKS, ear_threshold=EAR_THRESHOLD,
                         ear_consec_frames=EAR_CONSEC_FRAMES):
                self.detector   = dlib.get_frontal_face_detector()
                self.predictor  = dlib.shape_predictor(predictor_path)
                self.required_blinks   = required_blinks
                self.ear_threshold     = ear_threshold
                self.ear_consec_frames = ear_consec_frames
                self._ear_history = deque(maxlen=5)
                self._consec_below = 0
                self._blink_count  = 0
                self._is_live      = False

            def reset(self):
                self._ear_history.clear()
                self._consec_below = 0
                self._blink_count  = 0
                self._is_live      = False

            def process_frame(self, frame: np.ndarray) -> dict:
                result = dict(faces_found=0, ear=None, blink_count=self._blink_count,
                              is_live=self._is_live, message="No face detected")
                if self._is_live:
                    result["message"] = "Liveness confirmed ✓"
                    return result
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                result["faces_found"] = len(rects)
                if not rects:
                    return result
                rect   = max(rects, key=lambda r: r.width() * r.height())
                coords = landmarks_to_np(self.predictor(gray, rect))
                ear = (_ear(coords[LEFT_EYE_IDX]) + _ear(coords[RIGHT_EYE_IDX])) / 2.0
                self._ear_history.append(ear)
                s_ear = float(np.mean(self._ear_history))
                result["ear"] = round(s_ear, 4)
                if s_ear < self.ear_threshold:
                    self._consec_below += 1
                else:
                    if self._consec_below >= self.ear_consec_frames:
                        self._blink_count += 1
                    self._consec_below = 0
                result["blink_count"] = self._blink_count
                if self._blink_count >= self.required_blinks:
                    self._is_live = True
                    result.update(is_live=True, message="Liveness confirmed ✓")
                else:
                    result["message"] = f"Please blink {{self.required_blinks - self._blink_count}} more time(s)"
                return result
    ''')

    # ── backend/recognition.py ────────────────────────────────────────────────
    files["backend/recognition.py"] = textwrap.dedent('''\
        """recognition.py — Face encoding and identification."""
        import logging
        from pathlib import Path
        import cv2
        import numpy as np
        import face_recognition

        logger = logging.getLogger(__name__)
        SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        DEFAULT_TOLERANCE = 0.5

        class FaceDatabase:
            def __init__(self):
                self.encodings: list = []
                self.names: list[str] = []
            def __len__(self): return len(self.names)
            def add(self, name, enc): self.names.append(name); self.encodings.append(enc)
            def is_empty(self): return not self.names

        def load_known_faces(directory) -> FaceDatabase:
            db = FaceDatabase()
            d  = Path(directory)
            if not d.exists():
                logger.warning("known_faces dir missing: %s", d); return db
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in SUPPORTED_EXT: continue
                name = p.stem.replace("_", " ")
                try:
                    encs = face_recognition.face_encodings(face_recognition.load_image_file(str(p)))
                    if encs: db.add(name, encs[0])
                except Exception as e:
                    logger.error("Load error %s: %s", p.name, e)
            logger.info("Loaded %d face(s)", len(db))
            return db

        def identify_faces(frame_bgr, db: FaceDatabase, tolerance=DEFAULT_TOLERANCE) -> list[dict]:
            rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            small  = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
            locs   = face_recognition.face_locations(small)
            encs   = face_recognition.face_encodings(small, locs)
            out    = []
            for enc, (t,r,b,l) in zip(encs, locs):
                name, conf, matched = "Unknown", 0.0, False
                if not db.is_empty():
                    dists    = face_recognition.face_distance(db.encodings, enc)
                    idx      = int(np.argmin(dists))
                    dist_val = float(dists[idx])
                    conf     = max(0.0, 1.0 - dist_val)
                    if dist_val <= tolerance:
                        name, matched = db.names[idx], True
                out.append(dict(name=name, confidence=round(conf,3),
                                location=(t*2, r*2, b*2, l*2), matched=matched))
            return out

        def register_face(frame_bgr, name, directory, db: FaceDatabase) -> dict:
            d = Path(directory); d.mkdir(parents=True, exist_ok=True)
            encs = face_recognition.face_encodings(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if not encs: return dict(success=False, message="No face detected.")
            dest = d / f"{name.strip().replace(\' \', \'_\')}.jpg"
            cv2.imwrite(str(dest), frame_bgr)
            db.add(name.strip(), encs[0])
            return dict(success=True, message=f"Registered \'{name}\'.")
    ''')

    # ── backend/main.py ───────────────────────────────────────────────────────
    files["backend/main.py"] = textwrap.dedent('''\
        """main.py — FastAPI attendance system application."""
        import base64, logging, os
        from datetime import datetime
        from pathlib import Path
        import cv2, numpy as np
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import sqlalchemy as sa
        from sqlalchemy import text
        from recognition import FaceDatabase, load_known_faces, identify_faces, register_face
        from liveness import LivenessDetector

        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger = logging.getLogger("attendance")

        KNOWN_FACES_DIR = Path(os.getenv("KNOWN_FACES_DIR", "./known_faces"))
        PREDICTOR_PATH  = os.getenv("PREDICTOR_PATH",
                                    "./shape_predictor_68_face_landmarks.dat")
        DB_PATH = os.getenv("DB_PATH", "./database.db")

        engine = sa.create_engine(f"sqlite:///{DB_PATH}",
                                  connect_args={"check_same_thread": False})
        meta  = sa.MetaData()
        table = sa.Table("attendance", meta,
            sa.Column("id",         sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("name",       sa.String,  nullable=False),
            sa.Column("timestamp",  sa.String,  nullable=False),
            sa.Column("confidence", sa.Float,   nullable=False),
            sa.Column("status",     sa.String,  nullable=False),
        )
        meta.create_all(engine)

        face_db: FaceDatabase = FaceDatabase()
        liveness_detector = None

        app = FastAPI(title="Smart Attendance System", version="1.0.0")
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @app.on_event("startup")
        async def startup():
            global face_db, liveness_detector
            face_db = load_known_faces(KNOWN_FACES_DIR)
            try:
                liveness_detector = LivenessDetector(predictor_path=PREDICTOR_PATH)
            except Exception as e:
                logger.error("Liveness init failed: %s", e)

        def _decode(b64: str):
            if "," in b64: b64 = b64.split(",", 1)[1]
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Cannot decode image")
            return img

        def _log_db(name, conf, status):
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            with engine.begin() as c:
                if not c.execute(text(
                    "SELECT id FROM attendance WHERE name=:n AND timestamp LIKE :p"),
                    {"n": name, "p": ts[:16]+"%"}).fetchone():
                    c.execute(table.insert().values(
                        name=name, timestamp=ts, confidence=conf, status=status))

        class FramePayload(BaseModel):
            image: str; session_id: str = "default"

        class RegisterPayload(BaseModel):
            image: str; name: str

        @app.get("/")
        async def health():
            return {"status": "ok", "known_faces": len(face_db),
                    "liveness_detector": liveness_detector is not None}

        @app.post("/register")
        async def register(p: RegisterPayload):
            try: frame = _decode(p.image)
            except Exception as e: raise HTTPException(400, str(e))
            r = register_face(frame, p.name, KNOWN_FACES_DIR, face_db)
            if not r["success"]: raise HTTPException(422, r["message"])
            return r

        @app.post("/verify")
        async def verify(p: FramePayload):
            try: frame = _decode(p.image)
            except Exception as e: raise HTTPException(400, str(e))
            liveness = (liveness_detector.process_frame(frame) if liveness_detector
                        else {"faces_found":0,"ear":None,"blink_count":0,
                              "is_live":False,"message":"Detector unavailable"})
            recognition = identify_faces(frame, face_db)
            logged = []
            for r in recognition:
                st = "present" if r["matched"] else "unknown"
                if not liveness["is_live"]: st = "spoof"
                _log_db(r["name"], r["confidence"], st)
                if liveness["is_live"]: logged.append(r["name"])
            return {"liveness": liveness, "recognition": recognition,
                    "logged": logged, "frame_time": datetime.utcnow().isoformat()}

        @app.post("/verify/reset-liveness")
        async def reset_liveness():
            if liveness_detector: liveness_detector.reset()
            return {"message": "reset"}

        @app.get("/attendance/records")
        async def records(limit: int = 100):
            with engine.connect() as c:
                rows = c.execute(text(
                    "SELECT id,name,timestamp,confidence,status "
                    "FROM attendance ORDER BY id DESC LIMIT :l"), {"l":limit}).fetchall()
            return [{"id":r[0],"name":r[1],"timestamp":r[2],
                     "confidence":round(r[3],3),"status":r[4]} for r in rows]

        @app.delete("/attendance/clear")
        async def clear():
            with engine.begin() as c: c.execute(text("DELETE FROM attendance"))
            return {"message": "cleared"}

        if __name__ == "__main__":
            import uvicorn
            uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    ''')

    _log("backend", f"Generated {len(files)} backend files.")
    new_msgs = list(state["messages"]) + [
        AIMessage(content=f"Backend agent: generated {len(files)} files. ✓")
    ]
    new_status = {**state["status"], "backend": "done"}
    new_files  = {**state["files"], **files}
    return {**state, "messages": new_msgs, "files": new_files,
            "status": new_status, "phase": "frontend"}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 — Frontend Developer Agent
# ──────────────────────────────────────────────────────────────────────────────
def frontend_agent(state: AgentState) -> AgentState:
    _log("frontend", "Generating HTML dashboard, JavaScript, and CSS…")

    files: dict[str, str] = {}

    files["frontend/style.css"] = textwrap.dedent("""\
        /* style.css — Glassmorphism + animation extensions for Tailwind CDN */
        :root{--brand:#6366f1;--live:#10b981;--spoof:#ef4444;--unknown:#f59e0b}
        .glassmorphism{background:rgba(255,255,255,.04);backdrop-filter:blur(16px)}
        .glass-panel{background:rgba(17,24,39,.7);backdrop-filter:blur(20px);
          border:1px solid rgba(255,255,255,.08);border-radius:1.25rem;
          padding:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,.4)}
        .stat-card{background:rgba(17,24,39,.7);border:1px solid rgba(255,255,255,.08);
          border-radius:1rem;padding:1.25rem 1.5rem;transition:transform .2s ease}
        .stat-card:hover{transform:translateY(-2px)}
        .stat-label{font-size:.7rem;font-weight:600;text-transform:uppercase;
          letter-spacing:.08em;color:#6b7280;margin-bottom:.5rem}
        .stat-value{font-size:2rem;font-weight:800;line-height:1;color:#f9fafb}
        .action-btn{padding:.4rem .9rem;border-radius:.5rem;border:1px solid;
          font-size:.75rem;font-weight:600;cursor:pointer;transition:all .2s ease}
        .corner-bracket{position:absolute;width:20px;height:20px;
          border-color:rgba(99,102,241,.6)}
        .liveness-badge{display:inline-flex;align-items:center;padding:.2rem .75rem;
          border-radius:9999px;font-size:.7rem;font-weight:700;
          text-transform:uppercase;transition:all .3s ease}
        .badge-pending{background:rgba(107,114,128,.2);color:#9ca3af;
          border:1px solid rgba(107,114,128,.3)}
        .badge-checking{background:rgba(245,158,11,.15);color:#f59e0b;
          border:1px solid rgba(245,158,11,.3);animation:pulse 1.5s infinite}
        .badge-live{background:rgba(16,185,129,.15);color:#10b981;
          border:1px solid rgba(16,185,129,.35)}
        .badge-spoof{background:rgba(239,68,68,.15);color:#ef4444;
          border:1px solid rgba(239,68,68,.3)}
        .blink-dot{display:inline-block;width:10px;height:10px;border-radius:9999px;
          background:#374151;border:1px solid rgba(255,255,255,.1);
          transition:background .3s ease,box-shadow .3s ease}
        .blink-dot.done{background:var(--live);box-shadow:0 0 8px rgba(16,185,129,.6)}
        .th-cell{padding:.75rem 1rem;text-align:left;font-size:.7rem;font-weight:600;
          text-transform:uppercase;letter-spacing:.07em;color:#6b7280}
        .td-cell{padding:.65rem 1rem;border-top:1px solid rgba(255,255,255,.04);
          font-size:.8rem;color:#d1d5db;transition:background .15s ease}
        tbody tr:hover .td-cell{background:rgba(255,255,255,.025)}
        .status-pill{display:inline-block;padding:.15rem .6rem;border-radius:9999px;
          font-size:.65rem;font-weight:700;text-transform:uppercase}
        .pill-present{background:rgba(16,185,129,.15);color:#10b981}
        .pill-unknown{background:rgba(245,158,11,.15);color:#f59e0b}
        .pill-spoof{background:rgba(239,68,68,.15);color:#ef4444}
        .rec-result-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
          border-radius:.75rem;padding:.75rem 1rem;display:flex;
          align-items:center;justify-content:space-between;transition:all .25s ease}
        #scan-line.active{display:block!important;animation:scanMove 2.5s linear infinite}
        #toast.show{display:block!important}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:9999px}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes slideUp{from{transform:translateY(12px);opacity:0}to{transform:translateY(0);opacity:1}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
        @keyframes scanMove{0%{top:0%;opacity:.7}50%{opacity:1}100%{top:100%;opacity:.7}}
    """)

    # Full index.html (abbreviated here — real file is in frontend/index.html)
    files["frontend/index.html"] = textwrap.dedent("""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
          <title>Smart Attendance System — Liveness Detection</title>
          <meta name="description" content="AI-powered attendance system with real-time face recognition and blink-based liveness detection."/>
          <script src="https://cdn.tailwindcss.com"></script>
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
          <link rel="stylesheet" href="style.css"/>
          <script>
            tailwind.config={theme:{extend:{fontFamily:{inter:['Inter','sans-serif']},
            colors:{brand:{400:'#818cf8',500:'#6366f1',600:'#4f46e5',700:'#4338ca'},
            live:'#10b981',spoof:'#ef4444',unknown:'#f59e0b'}}}};
          </script>
        </head>
        <body class="font-inter bg-gray-950 text-gray-100 min-h-screen">
          <nav class="sticky top-0 z-50 glassmorphism border-b border-white/10">
            <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
              <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
                </div>
                <span class="text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">AttendAI</span>
              </div>
              <div class="flex items-center gap-4">
                <span id="system-status" class="flex items-center gap-2 text-sm text-gray-400">
                  <span class="w-2 h-2 rounded-full bg-gray-500 animate-pulse" id="status-dot"></span>
                  <span id="status-text">Connecting…</span>
                </span>
                <button id="btn-register-toggle" class="px-4 py-2 text-sm font-medium rounded-lg border border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/10 transition-all">＋ Register Face</button>
              </div>
            </div>
          </nav>

          <!-- Register Modal -->
          <div id="register-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div class="bg-gray-900 border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl">
              <h3 class="text-xl font-bold mb-4">Register New Face</h3>
              <div class="relative mb-4 rounded-xl overflow-hidden bg-gray-800 aspect-video">
                <video id="register-video" autoplay playsinline class="w-full h-full object-cover"></video>
                <canvas id="register-canvas" class="hidden"></canvas>
              </div>
              <input id="register-name" type="text" placeholder="Full name (e.g. John Doe)"
                class="w-full bg-gray-800 border border-white/10 rounded-lg px-4 py-3 text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white placeholder-gray-500"/>
              <div class="flex gap-3">
                <button id="btn-capture-register" class="flex-1 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 font-semibold text-sm transition-all">📸 Capture &amp; Register</button>
                <button id="btn-modal-close" class="px-4 py-3 rounded-lg bg-gray-800 hover:bg-gray-700 font-semibold text-sm transition-all">Cancel</button>
              </div>
              <p id="register-msg" class="mt-3 text-sm text-center text-gray-400 min-h-[1.25rem]"></p>
            </div>
          </div>

          <main class="max-w-7xl mx-auto px-6 py-8">
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
              <div class="stat-card"><p class="stat-label">Total Records</p><p class="stat-value" id="stat-total">—</p></div>
              <div class="stat-card"><p class="stat-label">Present Today</p><p class="stat-value text-green-400" id="stat-present">—</p></div>
              <div class="stat-card"><p class="stat-label">Spoof Attempts</p><p class="stat-value text-red-400" id="stat-spoof">—</p></div>
              <div class="stat-card"><p class="stat-label">Known Faces</p><p class="stat-value text-indigo-400" id="stat-known">—</p></div>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- Camera Panel -->
              <div class="glass-panel flex flex-col gap-4">
                <div class="flex items-center justify-between">
                  <h2 class="text-base font-semibold">Live Camera Feed</h2>
                  <div class="flex gap-2">
                    <button id="btn-start" class="action-btn bg-green-900/40 text-green-400 border-green-500/30 hover:bg-green-900/60">▶ Start</button>
                    <button id="btn-stop"  class="action-btn bg-red-900/40 text-red-400 border-red-500/30 hover:bg-red-900/60 hidden">■ Stop</button>
                  </div>
                </div>
                <div class="relative rounded-2xl overflow-hidden bg-gray-900 aspect-video border border-white/5">
                  <video id="webcam-video" autoplay playsinline muted class="w-full h-full object-cover"></video>
                  <canvas id="capture-canvas" class="hidden"></canvas>
                  <div id="scan-line" class="absolute left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-indigo-400/70 to-transparent hidden" style="pointer-events:none"></div>
                  <div class="corner-bracket top-3 left-3 border-t-2 border-l-2"></div>
                  <div class="corner-bracket top-3 right-3 border-t-2 border-r-2"></div>
                  <div class="corner-bracket bottom-3 left-3 border-b-2 border-l-2"></div>
                  <div class="corner-bracket bottom-3 right-3 border-b-2 border-r-2"></div>
                  <div id="idle-overlay" class="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <div class="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.069A1 1 0 0121 8.883v6.234a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/></svg>
                    </div>
                    <p class="text-gray-500 text-sm">Press ▶ Start to begin verification</p>
                  </div>
                </div>
                <div class="rounded-xl bg-gray-900 border border-white/5 p-4">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-semibold text-gray-400 uppercase tracking-widest">Liveness Check</span>
                    <span id="liveness-badge" class="liveness-badge badge-pending">Waiting…</span>
                  </div>
                  <p id="liveness-message" class="text-sm text-gray-400">Start the camera to begin.</p>
                  <div class="flex items-center gap-2 mt-3">
                    <span class="text-xs text-gray-500">Blinks:</span>
                    <div id="blink-dots" class="flex gap-1.5">
                      <span class="blink-dot" id="dot-0"></span>
                      <span class="blink-dot" id="dot-1"></span>
                    </div>
                  </div>
                </div>
                <div id="recognition-panel" class="rounded-xl bg-gray-900 border border-white/5 p-4 hidden">
                  <p class="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2">Recognition</p>
                  <div id="recognition-results" class="space-y-2"></div>
                </div>
              </div>
              <!-- Attendance Log -->
              <div class="glass-panel flex flex-col gap-4">
                <div class="flex items-center justify-between">
                  <h2 class="text-base font-semibold">Attendance Log</h2>
                  <div class="flex items-center gap-3">
                    <span class="text-xs text-gray-500" id="last-refresh">—</span>
                    <button id="btn-clear" class="action-btn bg-red-900/20 text-red-400 border-red-500/20 hover:bg-red-900/30 text-xs">🗑 Clear</button>
                  </div>
                </div>
                <div class="overflow-x-auto rounded-xl border border-white/5 bg-gray-900">
                  <table class="w-full text-sm">
                    <thead><tr class="border-b border-white/5">
                      <th class="th-cell">Name</th><th class="th-cell">Time</th>
                      <th class="th-cell">Conf.</th><th class="th-cell">Status</th>
                    </tr></thead>
                    <tbody id="attendance-tbody">
                      <tr id="empty-row"><td colspan="4" class="py-12 text-center text-gray-600 text-sm">No records yet</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </main>
          <div id="toast" class="fixed bottom-6 right-6 z-50 hidden">
            <div id="toast-content" class="glassmorphism border border-white/10 rounded-xl px-5 py-3 text-sm font-medium shadow-xl"></div>
          </div>
          <script src="script.js"></script>
        </body>
        </html>
    """)

    files["frontend/script.js"] = textwrap.dedent("""\
        const API_BASE = 'http://localhost:8000';
        const CAPTURE_MS = 600, POLL_MS = 5000, SESSION_ID = `s_${Date.now()}`;
        const REQUIRED_BLINKS = 2;
        const $ = id => document.getElementById(id);
        const webcamVideo=$('webcam-video'),captureCanvas=$('capture-canvas'),
          idleOverlay=$('idle-overlay'),scanLine=$('scan-line'),
          btnStart=$('btn-start'),btnStop=$('btn-stop'),btnClear=$('btn-clear'),
          btnRegisterToggle=$('btn-register-toggle'),livenessBadge=$('liveness-badge'),
          livenessMessage=$('liveness-message'),recognitionPanel=$('recognition-panel'),
          recognitionResults=$('recognition-results'),attendanceTbody=$('attendance-tbody'),
          emptyRow=$('empty-row'),lastRefresh=$('last-refresh'),statusDot=$('status-dot'),
          statusText=$('status-text'),statTotal=$('stat-total'),statPresent=$('stat-present'),
          statSpoof=$('stat-spoof'),statKnown=$('stat-known'),registerModal=$('register-modal'),
          registerVideo=$('register-video'),registerCanvas=$('register-canvas'),
          registerName=$('register-name'),registerMsg=$('register-msg'),
          btnCaptureReg=$('btn-capture-register'),btnModalClose=$('btn-modal-close'),
          toast=$('toast'),toastContent=$('toast-content');
        const blinkDots = [$('dot-0'),$('dot-1')];

        let captureId=null,pollId=null,webcamStream=null,registerStream=null,
            isLive=false,blinkCount=0,toastTimeout=null;

        function b64(video,canvas,q=0.8){
          canvas.width=video.videoWidth||640;canvas.height=video.videoHeight||480;
          canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
          return canvas.toDataURL('image/jpeg',q);
        }
        function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
        function toast(msg,type='info'){
          const c={info:'text-gray-200',success:'text-green-400',error:'text-red-400',warning:'text-yellow-400'};
          toastContent.className=`glassmorphism border border-white/10 rounded-xl px-5 py-3 text-sm font-medium shadow-xl ${c[type]||c.info}`;
          toastContent.textContent=msg; toast.classList.add('show');
          clearTimeout(toastTimeout); toastTimeout=setTimeout(()=>toast.classList.remove('show'),3500);
        }
        async function health(){
          try{const r=await fetch(`${API_BASE}/`,{signal:AbortSignal.timeout(3000)});
            const d=await r.json();
            statusDot.className='w-2 h-2 rounded-full bg-green-400 animate-pulse';
            statusText.textContent='System Online'; statKnown.textContent=d.known_faces??'—';
          }catch{statusDot.className='w-2 h-2 rounded-full bg-red-400 animate-pulse';
            statusText.textContent='Backend Unreachable';}
        }
        async function startCam(){
          try{webcamStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});
            webcamVideo.srcObject=webcamStream; await webcamVideo.play();
            idleOverlay.style.display='none'; scanLine.classList.add('active'); return true;
          }catch(e){toast(`Camera: ${e.message}`,'error'); return false;}
        }
        function stopCam(){
          if(webcamStream){webcamStream.getTracks().forEach(t=>t.stop()); webcamStream=null;}
          webcamVideo.srcObject=null; idleOverlay.style.display=''; scanLine.classList.remove('active');
        }
        function updateLiveness(l){
          livenessMessage.textContent=l.message||'—';
          blinkCount=l.blink_count??blinkCount;
          blinkDots.forEach((d,i)=>d.classList.toggle('done',i<blinkCount));
          if(l.is_live&&!isLive){isLive=true; livenessBadge.className='liveness-badge badge-live';
            livenessBadge.textContent='✓ Live'; toast('Liveness confirmed!','success');}
          else if(!l.is_live&&blinkCount>0){livenessBadge.className='liveness-badge badge-checking';
            livenessBadge.textContent=`Blink ${blinkCount}/${REQUIRED_BLINKS}`;}
          else if(!l.is_live){livenessBadge.className='liveness-badge badge-checking';
            livenessBadge.textContent='Verifying…';}
        }
        function updateRecognition(res){
          if(!res?.length){recognitionPanel.classList.add('hidden');return;}
          recognitionPanel.classList.remove('hidden'); recognitionResults.innerHTML='';
          res.forEach(r=>{
            const pct=Math.round((r.confidence||0)*100),m=r.matched;
            const card=document.createElement('div'); card.className='rec-result-card';
            card.innerHTML=`<div class="flex items-center gap-2">
              <span class="w-7 h-7 rounded-full bg-gray-800 flex items-center justify-center text-xs font-bold ${m?'text-green-400':'text-yellow-400'}">${m?'✓':'?'}</span>
              <div><p class="font-semibold text-sm ${m?'text-green-400':'text-yellow-400'}">${esc(r.name)}</p>
              <p class="text-xs text-gray-500">Face detected</p></div></div>
              <div class="text-right"><p class="text-sm font-bold ${m?'text-green-400':'text-yellow-400'}">${pct}%</p>
              <p class="text-xs text-gray-500">confidence</p></div>`;
            recognitionResults.appendChild(card);
          });
        }
        async function captureAndVerify(){
          if(!webcamStream)return;
          try{const res=await fetch(`${API_BASE}/verify`,{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({image:b64(webcamVideo,captureCanvas),session_id:SESSION_ID}),
            signal:AbortSignal.timeout(5000)});
            if(!res.ok)return; const d=await res.json();
            updateLiveness(d.liveness); updateRecognition(d.recognition);
            if(d.logged?.length) toast(`✅ Logged: ${d.logged.join(', ')}`,'success');
          }catch(e){if(e.name!=='AbortError')console.error(e);}
        }
        async function refreshRecords(){
          try{const res=await fetch(`${API_BASE}/attendance/records?limit=50`,
            {signal:AbortSignal.timeout(4000)});
            const rows=await res.json(); renderTable(rows);
            statTotal.textContent=rows.length;
            statPresent.textContent=rows.filter(r=>r.status==='present').length;
            statSpoof.textContent=rows.filter(r=>r.status==='spoof').length;
            lastRefresh.textContent=`Updated ${new Date().toLocaleTimeString()}`;
          }catch{}
        }
        function renderTable(rows){
          attendanceTbody.querySelectorAll('.dr').forEach(r=>r.remove());
          if(!rows.length){emptyRow.style.display='';return;}
          emptyRow.style.display='none';
          rows.forEach(rec=>{
            const tr=document.createElement('tr'); tr.className='dr';
            const pc={'present':'pill-present','unknown':'pill-unknown','spoof':'pill-spoof'}[rec.status]||'pill-unknown';
            tr.innerHTML=`<td class="td-cell font-medium text-white">${esc(rec.name)}</td>
              <td class="td-cell text-gray-400 text-xs">${esc(rec.timestamp)}</td>
              <td class="td-cell text-gray-300">${(rec.confidence*100).toFixed(1)}%</td>
              <td class="td-cell"><span class="status-pill ${pc}">${esc(rec.status)}</span></td>`;
            attendanceTbody.appendChild(tr);
          });
        }
        btnClear.addEventListener('click',async()=>{
          if(!confirm('Clear all records?'))return;
          await fetch(`${API_BASE}/attendance/clear`,{method:'DELETE'});
          toast('Records cleared','warning'); await refreshRecords();
        });
        btnStart.addEventListener('click',async()=>{
          if(!await startCam())return;
          isLive=false; blinkCount=0;
          livenessBadge.className='liveness-badge badge-checking';
          livenessBadge.textContent='Verifying…';
          livenessMessage.textContent='Please look at the camera and blink naturally.';
          blinkDots.forEach(d=>d.classList.remove('done'));
          recognitionPanel.classList.add('hidden');
          try{await fetch(`${API_BASE}/verify/reset-liveness`,{method:'POST'});}catch{}
          captureId=setInterval(captureAndVerify,CAPTURE_MS);
          pollId=setInterval(refreshRecords,POLL_MS);
          await refreshRecords();
          btnStart.classList.add('hidden'); btnStop.classList.remove('hidden');
        });
        btnStop.addEventListener('click',()=>{
          clearInterval(captureId); clearInterval(pollId); stopCam();
          livenessBadge.className='liveness-badge badge-pending';
          livenessBadge.textContent='Waiting…';
          livenessMessage.textContent='Start the camera to begin.';
          blinkDots.forEach(d=>d.classList.remove('done'));
          btnStop.classList.add('hidden'); btnStart.classList.remove('hidden');
        });
        btnRegisterToggle.addEventListener('click',async()=>{
          registerModal.classList.remove('hidden'); registerMsg.textContent=''; registerName.value='';
          try{registerStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
            registerVideo.srcObject=registerStream; await registerVideo.play();}
          catch(e){registerMsg.textContent=`Camera: ${e.message}`;}
        });
        function closeModal(){
          registerModal.classList.add('hidden');
          if(registerStream){registerStream.getTracks().forEach(t=>t.stop()); registerStream=null;}
          registerVideo.srcObject=null;
        }
        btnModalClose.addEventListener('click',closeModal);
        registerModal.addEventListener('click',e=>{if(e.target===registerModal)closeModal();});
        btnCaptureReg.addEventListener('click',async()=>{
          const name=registerName.value.trim();
          if(!name){registerMsg.textContent='⚠️ Enter a name first.';return;}
          if(!registerStream){registerMsg.textContent='⚠️ Camera not available.';return;}
          btnCaptureReg.disabled=true; btnCaptureReg.textContent='Registering…';
          try{const res=await fetch(`${API_BASE}/register`,{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({image:b64(registerVideo,registerCanvas),name})});
            const d=await res.json();
            if(res.ok&&d.success){registerMsg.textContent=`✅ ${d.message}`;
              toast(`Registered: ${name}`,'success'); await health();
              setTimeout(closeModal,1500);}
            else registerMsg.textContent=`❌ ${d.detail||d.message||'Failed.'}`;
          }catch(e){registerMsg.textContent=`❌ ${e.message}`;}
          finally{btnCaptureReg.disabled=false; btnCaptureReg.textContent='📸 Capture & Register';}
        });
        (async()=>{await health(); await refreshRecords(); setInterval(health,30000);})();
    """)

    _log("frontend", f"Generated {len(files)} frontend files.")
    new_msgs   = list(state["messages"]) + [AIMessage(content="Frontend agent: generated UI files. ✓")]
    new_status = {**state["status"], "frontend": "done"}
    new_files  = {**state["files"], **files}
    return {**state, "messages": new_msgs, "files": new_files,
            "status": new_status, "phase": "integration"}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 — Integration Agent
# Patches CORS, verifies API_BASE constant alignment, writes nginx.conf
# ──────────────────────────────────────────────────────────────────────────────
def integration_agent(state: AgentState) -> AgentState:
    _log("integration", "Validating CORS, endpoint alignment, writing nginx config…")

    files: dict[str, str] = {}

    # Nginx config for frontend
    files["nginx.conf"] = textwrap.dedent("""\
        server {
            listen 80;
            server_name localhost;
            root /usr/share/nginx/html;
            index index.html;
            location ~* \\.(css|js|png|jpg|ico|woff2?)$ {
                expires 1d;
                add_header Cache-Control "public";
            }
            location / {
                try_files $uri $uri/ /index.html;
            }
        }
    """)

    # Validate that backend/main.py has CORS * — check in-state files
    backend_main = state["files"].get("backend/main.py", "")
    if "allow_origins" not in backend_main:
        state["errors"].append("CORS middleware missing from backend/main.py")

    # Validate API_BASE in script.js matches port 8000
    js = state["files"].get("frontend/script.js", "")
    if "localhost:8000" not in js:
        state["errors"].append("API_BASE mismatch in script.js — expected localhost:8000")

    _log("integration", f"Integration checks: {len(state['errors'])} error(s). nginx.conf written.")
    new_msgs   = list(state["messages"]) + [AIMessage(content=f"Integration agent: aligned endpoints, {len(state['errors'])} error(s). ✓")]
    new_status = {**state["status"], "integration": "done"}
    new_files  = {**state["files"], **files}
    return {**state, "messages": new_msgs, "files": new_files,
            "status": new_status, "phase": "devops"}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 5 — QA / DevOps Agent
# ──────────────────────────────────────────────────────────────────────────────
def devops_agent(state: AgentState) -> AgentState:
    _log("devops", "Generating Dockerfile, docker-compose.yml, .gitignore, README…")

    files: dict[str, str] = {}

    files["Dockerfile"] = textwrap.dedent("""\
        # syntax=docker/dockerfile:1
        FROM python:3.11-slim AS builder
        RUN apt-get update && apt-get install -y --no-install-recommends \\
            build-essential cmake libopenblas-dev liblapack-dev \\
            libx11-dev libboost-python-dev libboost-thread-dev wget && \\
            rm -rf /var/lib/apt/lists/*
        WORKDIR /install
        COPY backend/requirements.txt .
        RUN pip install --upgrade pip && \\
            pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

        FROM python:3.11-slim AS runtime
        RUN apt-get update && apt-get install -y --no-install-recommends \\
            libopenblas-base libgomp1 libglib2.0-0 libgl1 && \\
            rm -rf /var/lib/apt/lists/*
        COPY --from=builder /install/deps /usr/local
        WORKDIR /app
        COPY backend/ ./backend/
        RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/* && \\
            wget -q "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \\
            -O /tmp/sp.dat.bz2 && bunzip2 /tmp/sp.dat.bz2 && \\
            mv /tmp/sp.dat /app/backend/shape_predictor_68_face_landmarks.dat
        RUN mkdir -p /app/known_faces
        ENV KNOWN_FACES_DIR=/app/known_faces \\
            PREDICTOR_PATH=/app/backend/shape_predictor_68_face_landmarks.dat \\
            DB_PATH=/app/backend/database.db \\
            PYTHONUNBUFFERED=1
        EXPOSE 8000
        CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
    """)

    files["docker-compose.yml"] = textwrap.dedent("""\
        version: "3.9"
        services:
          backend:
            build: {context: ., dockerfile: Dockerfile, target: runtime}
            container_name: attendance_backend
            restart: unless-stopped
            ports: ["8000:8000"]
            volumes:
              - ./backend/database.db:/app/backend/database.db
              - ./known_faces:/app/known_faces
            environment:
              KNOWN_FACES_DIR: /app/known_faces
              PREDICTOR_PATH:  /app/backend/shape_predictor_68_face_landmarks.dat
              DB_PATH:         /app/backend/database.db
            healthcheck:
              test: ["CMD", "curl", "-f", "http://localhost:8000/"]
              interval: 30s
              timeout: 10s
              retries: 3
              start_period: 15s
            networks: [attendance_net]

          frontend:
            image: nginx:1.25-alpine
            container_name: attendance_frontend
            restart: unless-stopped
            ports: ["80:80"]
            volumes:
              - ./frontend:/usr/share/nginx/html:ro
              - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
            depends_on:
              backend: {condition: service_healthy}
            networks: [attendance_net]

        networks:
          attendance_net: {driver: bridge}
    """)

    files[".gitignore"] = textwrap.dedent("""\
        __pycache__/
        *.py[cod]
        .venv/
        venv/
        *.dat
        *.dat.bz2
        *.db
        *.sqlite
        *.log
        .DS_Store
        Thumbs.db
        .vscode/
        .idea/
        node_modules/
        known_faces/*.jpg
        known_faces/*.jpeg
        known_faces/*.png
        !known_faces/.gitkeep
    """)

    files["known_faces/.gitkeep"] = (
        "# Drop reference face images here: PersonName.jpg\n"
    )

    _log("devops", f"Generated {len(files)} DevOps files.")
    new_msgs   = list(state["messages"]) + [AIMessage(content="DevOps agent: generated Docker + CI files. ✓")]
    new_status = {**state["status"], "devops": "done"}
    new_files  = {**state["files"], **files}
    return {**state, "messages": new_msgs, "files": new_files,
            "status": new_status, "phase": "git"}


# ──────────────────────────────────────────────────────────────────────────────
# NODE 6 — Write files + Git push
# ──────────────────────────────────────────────────────────────────────────────
def git_push_node(state: AgentState) -> AgentState:
    _log("git", "Writing all generated files to disk…")
    _write_files(state)

    _log("git", "Initialising git repository and pushing to GitHub…")
    git_errors: list[str] = []

    commands = [
        ("git init",                                           "Init repo"),
        (f"git remote add origin {GITHUB_REPO_URL}",          "Add remote"),
        ("git branch -M main",                                 "Rename branch"),
        ("git add .",                                          "Stage all files"),
        (f'git commit -m "{COMMIT_MESSAGE}"',                  "Commit"),
        ("git push -u origin main --force",                    "Push to GitHub"),
    ]

    for cmd, label in commands:
        print(f"\n  ▸ {label}: $ {cmd}")
        code, stdout, stderr = _run(cmd)
        if stdout: print(f"    {stdout}")
        if stderr: print(f"    {stderr}")
        if code != 0:
            msg = f"[{label}] exit {code}: {stderr}"
            git_errors.append(msg)
            print(f"  ⚠️  {msg}")
            if label in ("Commit", "Push to GitHub"):
                # Fatal — abort remaining git steps
                break

    if git_errors:
        print(
            "\n⚠️  Some git steps failed. "
            "Run these commands manually in the project directory:\n"
        )
        for cmd, label in commands:
            print(f"   {cmd}")
    else:
        print(f"\n✅  Successfully pushed to {GITHUB_REPO_URL}")

    new_msgs   = list(state["messages"]) + [
        AIMessage(content=f"Git node: push complete with {len(git_errors)} error(s).")
    ]
    new_status = {**state["status"], "git": "done" if not git_errors else "error"}
    return {**state, "messages": new_msgs, "errors": state["errors"] + git_errors,
            "status": new_status, "phase": "done"}


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor",   supervisor_agent)
    graph.add_node("backend",      backend_agent)
    graph.add_node("frontend",     frontend_agent)
    graph.add_node("integration",  integration_agent)
    graph.add_node("devops",       devops_agent)
    graph.add_node("git_push",     git_push_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor",  "backend")
    graph.add_edge("backend",     "frontend")
    graph.add_edge("frontend",    "integration")
    graph.add_edge("integration", "devops")
    graph.add_edge("devops",      "git_push")
    graph.add_edge("git_push",    END)

    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  Smart Attendance System — LangGraph Multi-Agent Build")
    print("=" * 65)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  GitHub repo  : {GITHUB_REPO_URL}")
    print("=" * 65 + "\n")

    graph    = build_graph()
    compiled = graph.compile()

    initial_state: AgentState = {
        "messages": [],
        "files":    {},
        "status":   {},
        "errors":   [],
        "phase":    "init",
    }

    final_state = compiled.invoke(initial_state)

    print("\n" + "=" * 65)
    print("  Pipeline Summary")
    print("=" * 65)
    for agent, st in final_state["status"].items():
        icon = "✅" if st == "done" else ("❌" if st == "error" else "⏸️")
        print(f"  {icon}  {agent:<15} {st}")

    if final_state["errors"]:
        print("\n  Errors:")
        for err in final_state["errors"]:
            print(f"    • {err}")

    print(f"\n  Files written : {len(final_state['files'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()
