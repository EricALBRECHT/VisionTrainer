# Vision Trainer — NVIDIA GPU / GTX 1060 Pascal

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

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# GTX 1060 = architecture Pascal.
# On force les wheels PyTorch CUDA 12.6.
RUN pip install --no-cache-dir \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu126

COPY pyproject.toml README.md ./
COPY src ./src

# Installe VisionTrainer + Streamlit + Ultralytics + Pillow + PyYAML etc.
RUN pip install --no-cache-dir .

COPY app ./app
COPY .streamlit ./.streamlit

RUN mkdir -p \
    /app/data/datasets \
    /app/data/runs \
    /app/data/ultralytics \
    /app/data/tmp

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
