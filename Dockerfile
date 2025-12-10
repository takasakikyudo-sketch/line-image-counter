# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Google credentials will be provided via env var pointing to file path
ENV GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_key.json

EXPOSE 10000
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--workers", "2", "--threads", "4"]
