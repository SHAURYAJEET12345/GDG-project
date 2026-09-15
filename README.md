# Smart Attendance System with Liveness Detection

> **AI-powered attendance tracking** using real-time face recognition and blink-based liveness detection — preventing spoofing from printed photos or replay videos.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (User)                       │
│  ┌───────────────────┐     HTTP / JSON      ┌────────────┐  │
│  │   Frontend (Nginx) │ ◄────────────────► │  Backend   │  │
│  │  HTML + Tailwind   │                     │  FastAPI   │  │
│  │  script.js         │   POST /verify      │  OpenCV    │  │
│  │  getUserMedia()    │ ──────────────────► │  dlib EAR  │  │
│  └───────────────────┘                     │  face_rec  │  │
│                                             │  SQLite    │  │
│                                             └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| 🎥 **Live Webcam Stream** | `getUserMedia` captures frames in-browser |
| 👁️ **Liveness Detection** | EAR (Eye Aspect Ratio) blink counting via dlib 68-pt landmarks |
| 🧠 **Face Recognition** | `face_recognition` library (dlib-based HOG + 128-d embeddings) |
| 📋 **Attendance Logging** | SQLite via SQLAlchemy; auto-deduplicates within the same minute |
| 🚫 **Anti-Spoofing** | Flags static-photo/video attacks as `spoof` in the log |
| ➕ **Face Registration** | Live webcam capture → `/register` endpoint |
| 🐳 **Docker Ready** | Multi-stage build; single `docker-compose up` |

---

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- Git

### 1 — Clone the repository
```bash
git clone https://github.com/SHAURYAJEET12345/GDG-project.git
cd GDG-project
```

### 2 — Add reference face images
Drop JPEG images into `known_faces/`. File name = person's name.

```
known_faces/
├── John_Doe.jpg       →  "John Doe"
├── Alice_Smith.jpg    →  "Alice Smith"
└── ...
```

> **Tip:** You can also use the in-app **Register Face** button to capture faces live.

### 3 — Build & run
```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend dashboard | http://localhost |
| Backend API docs   | http://localhost:8000/docs |

### 4 — Mark attendance
1. Open `http://localhost` in your browser
2. Click **▶ Start**
3. Allow camera access
4. **Blink twice** when prompted — liveness confirmed
5. Your name and timestamp appear in the Attendance Log

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + system info |
| `POST` | `/register` | Register a face (base64 image + name) |
| `POST` | `/verify` | Liveness + recognition pipeline |
| `POST` | `/verify/reset-liveness` | Reset blink counter for new session |
| `GET` | `/attendance/records` | Fetch attendance log (latest 100) |
| `DELETE` | `/attendance/clear` | Clear all records |

### POST `/verify` payload
```json
{
  "image": "<base64-encoded JPEG>",
  "session_id": "optional-string"
}
```

### POST `/verify` response
```json
{
  "liveness": {
    "faces_found": 1,
    "ear": 0.312,
    "blink_count": 2,
    "is_live": true,
    "message": "Liveness confirmed ✓"
  },
  "recognition": [
    {
      "name": "John Doe",
      "confidence": 0.87,
      "location": [100, 300, 250, 150],
      "matched": true
    }
  ],
  "logged": ["John Doe"],
  "frame_time": "2025-06-01T12:34:56.789"
}
```

---

## Directory Structure

```
gdg-project/
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── recognition.py       # Face loading & identification
│   ├── liveness.py          # EAR blink detection
│   ├── requirements.txt     # Python deps
│   └── database.db          # SQLite attendance log (auto-created)
│
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── script.js            # Webcam handler & API client
│   └── style.css            # Custom CSS (glassmorphism + animations)
│
├── known_faces/             # Drop reference images here
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Service orchestration
├── nginx.conf               # Nginx config for frontend
├── langgraph_build.py       # LangGraph multi-agent build script
└── README.md
```

---

## Liveness Detection — How it Works

```
Frame → dlib face detector → 68 facial landmarks
    → Extract LEFT_EYE[36:42] + RIGHT_EYE[42:48]
    → Compute EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)
    → EAR < 0.25 for ≥ 2 frames → blink detected
    → 2 blinks confirmed → is_live = True
```

The EAR naturally drops near zero during a genuine blink but stays constant in a static image or video replay, making it an effective anti-spoofing signal.

---

## Local Development (without Docker)

> **Note:** Requires CMake, C++ compiler, and dlib pre-built for Windows. Recommended to use Docker instead.

```bash
# Backend
cd backend
pip install -r requirements.txt

# Download dlib landmark model
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat backend/

python main.py

# Frontend — open in browser directly
# Or use any static server, e.g.:
python -m http.server 3000 --directory frontend
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KNOWN_FACES_DIR` | `./known_faces` | Path to reference face images |
| `PREDICTOR_PATH` | `./shape_predictor_68_face_landmarks.dat` | dlib model path |
| `DB_PATH` | `./database.db` | SQLite database path |

---

## LangGraph Multi-Agent Build

This project was scaffolded by a **LangGraph multi-agent orchestration script** (`langgraph_build.py`). The workflow:

```
Supervisor → Backend Agent → Frontend Agent → Integration Agent → DevOps Agent → Git Push
```

To regenerate all project files from scratch:
```bash
pip install langgraph langchain-core
python langgraph_build.py
```

---

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, OpenCV, face_recognition, dlib, SQLAlchemy
- **Frontend:** HTML5, Tailwind CSS, Vanilla JS
- **Container:** Docker, Docker Compose, Nginx
- **Orchestration:** LangGraph, Python

---

## License

MIT — see [LICENSE](LICENSE)
