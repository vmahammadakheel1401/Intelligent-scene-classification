# Small CPU-only image for the Flask app.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest.
COPY requirements.txt .
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# App code, templates, static files, model and model_meta.json (see .dockerignore for what is left out).
COPY . .

EXPOSE 8000
# 2 workers x 4 threads; each worker holds one copy of the model in memory.
CMD ["gunicorn", "--workers=2", "--threads=4", "--timeout=120", "--bind", "0.0.0.0:8000", "app:app"]
