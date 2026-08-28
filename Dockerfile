# CPU-only torch. The default wheel bundles CUDA libraries and is ~2.5GB; the
# cpu index is ~200MB and this never touches a GPU.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Bake BOTH models into the image. Downloading at boot makes the first request
# take ~8s and makes startup depend on HuggingFace being up -- and the runtime
# user cannot write to /models anyway, so a lazy download fails with EACCES the
# first time Layer 2 runs. Bake both or the semantic layer 500s.
COPY embedder.py normalize.py config.py ./
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
               SentenceTransformer('all-MiniLM-L6-v2'); \
               CrossEncoder('cross-encoder/quora-distilroberta-base')" \
    && chmod -R a+rX /models

# Never reach for the network at runtime: fail loudly on a missing model
# instead of hanging on a download that cannot be written anyway.
ENV HF_HUB_OFFLINE=1

COPY . .

# Non-root, and a writable spot for the sqlite file. NOTE: container
# filesystems are ephemeral -- mount a volume at /data on your host, or the
# cache starts empty on every redeploy.
RUN useradd -u 10001 -m app && mkdir -p /data && chown app /data
USER app
ENV CACHE_DB=/data/cache.db

EXPOSE 8000
# Hosts inject $PORT; default to 8000 for `docker run` locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
