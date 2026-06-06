# Base image 
# Slim Python 3.11 — keeps the image small (~200 MB final)
FROM python:3.11-slim

# Build argument 
# Override at build time:  docker build --build-arg HF_MODEL_NAME=user/repo .
ARG HF_MODEL_NAME=pp2711/imdb-distilbert-sentiment
ENV HF_MODEL=${HF_MODEL_NAME}

# System dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Working directory 
WORKDIR /app

# Python dependencies 
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/inference.py .

# Runtime 
# INPUT_TEXT and HF_TOKEN are passed via -e at docker run time
CMD ["python", "inference.py"]
