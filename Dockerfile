# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Builder
# • Installs build tools (cmake, boost, etc.) needed to compile dlib
# • Installs all Python packages into /install/deps
# • Downloads the dlib 68-point shape predictor model
# The builder image is discarded; only its outputs are copied to runtime.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build deps
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
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Install Python packages into a relocatable prefix
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

# Download dlib shape predictor in builder so the runtime image needs no wget/bzip2
RUN wget -q \
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \
    -O /tmp/sp.dat.bz2 && \
    bunzip2 /tmp/sp.dat.bz2 && \
    mv /tmp/sp.dat /install/shape_predictor_68_face_landmarks.dat


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Runtime
# Lean final image: no build tools, no wget, no cmake.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime shared libs required by OpenCV & dlib + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install/deps /usr/local

WORKDIR /app

# Copy backend source code
COPY backend/ ./backend/

# Copy the dlib model from builder (no wget needed in runtime)
COPY --from=builder /install/shape_predictor_68_face_landmarks.dat \
    /app/backend/shape_predictor_68_face_landmarks.dat

# Create data directory for SQLite DB (separate from source code)
RUN mkdir -p /app/data /app/known_faces

# Environment defaults — override via docker-compose environment section
ENV KNOWN_FACES_DIR=/app/known_faces \
    PREDICTOR_PATH=/app/backend/shape_predictor_68_face_landmarks.dat \
    DB_PATH=/app/data/database.db \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
