# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# ---- dependencies ----
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---- model weights baked into the image (container runs fully offline) ----
FROM deps AS model
ARG MODEL_ID=Qwen/Qwen3.5-0.8B
RUN /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/model', \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])"

# ---- runtime ----
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MODEL_PATH=/models/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --from=model /models /models
COPY app ./app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
