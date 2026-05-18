
FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch — keeps image ~1.5GB instead of 6GB
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# ChromaDB persists here — mount this volume to survive restarts
VOLUME ["/app/data/chromadb"]

ENV PYTHONUNBUFFERED=1
ENV CHROMA_PERSIST_DIR=/app/data/chromadb
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
