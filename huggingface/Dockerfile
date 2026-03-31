FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (for frontend build)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Build frontend (copy only what's needed for npm install first)
COPY frontend/package.json frontend/package-lock.json /app/frontend/
WORKDIR /app/frontend
RUN npm ci --no-audit --no-fund
COPY frontend/ /app/frontend/
RUN npm run build

# Copy backend files (after frontend build to maximize cache reuse)
WORKDIR /app
COPY app.py model.py sightengine.py /app/
COPY *.pkl /app/

# Hugging Face Spaces default port
EXPOSE 7860

CMD ["python", "app.py"]
