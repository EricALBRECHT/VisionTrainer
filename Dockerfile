# Vision Trainer — CPU image
# Future GPU: switch base to an NVIDIA CUDA + PyTorch image and enable
# `gpus: all` in docker-compose (see comments in docker-compose.yml).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONTRAINER_DATA_DIR=/app/data \
    YOLO_CONFIG_DIR=/app/data/ultralytics \
    TMPDIR=/app/data/tmp \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# OpenCV / video / Ultralytics runtime libs (CPU).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying the full app (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

COPY app ./app
COPY .streamlit ./.streamlit

RUN mkdir -p /app/data/datasets /app/data/runs /app/data/ultralytics /app/data/tmp

# Run as root so Docker Desktop bind mounts (./data) remain writable on Windows/macOS.
# Do not use --privileged; NVIDIA GPU can be added later via Compose `gpus: all`.

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
