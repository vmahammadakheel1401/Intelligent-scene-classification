# Small CPU-only image for the Flask app (ONNX Runtime, no PyTorch).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# App code, templates, static files, model.onnx and model_meta.json (see .dockerignore for what is left out).
COPY . .

EXPOSE 8000
# 2 workers x 4 threads; each worker holds one copy of the model in memory.
CMD ["gunicorn", "--workers=2", "--threads=4", "--timeout=120", "--bind", "0.0.0.0:8000", "app:app"]
