# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Builder
# Compile dlib and install heavy Python deps with build tools present.
# This stage is discarded; only installed packages are copied forward.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build deps (cmake needed by dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements and install into a prefix we can COPY later
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Runtime
# Lean final image: only runtime libs + our app code.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime shared libs required by OpenCV & dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/deps /usr/local

WORKDIR /app

# Copy backend source
COPY backend/ ./backend/

# Download dlib shape predictor (68-point model) at build time so the
# container is self-contained (no internet needed at runtime).
RUN wget -q \
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \
    -O /tmp/sp.dat.bz2 && \
    bunzip2 /tmp/sp.dat.bz2 && \
    mv /tmp/sp.dat /app/backend/shape_predictor_68_face_landmarks.dat

# Create directories for persistent data
RUN mkdir -p /app/known_faces

# Environment defaults (can be overridden via docker-compose env)
ENV KNOWN_FACES_DIR=/app/known_faces \
    PREDICTOR_PATH=/app/backend/shape_predictor_68_face_landmarks.dat \
    DB_PATH=/app/backend/database.db \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
