# MoS Analyzer Universal Container
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# Install system libraries for the Gmsh OpenCASCADE kernel and OpenGL (VTK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgomp1 \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt requirements-image3d-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-image

# Copy project files
COPY mos ./mos
COPY cad_models ./cad_models
COPY tools ./tools
COPY docs ./docs
COPY pyproject.toml README.md ./

EXPOSE 8080

CMD ["python", "-m", "mos.app", "--server", "--port", "8080", "--host", "0.0.0.0"]
