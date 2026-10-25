# local-inference-x86-05b

Local CPU inference of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on PyTorch,
served over FastAPI and packaged as a `linux/amd64` Docker image. Model weights are baked into the image,
so the container works offline.

## Run

```bash
docker run --rm -p 8000:8000 ghcr.io/stackthrower/local-inference-x86-05b:latest
```

Or build locally:

```bash
docker build --platform linux/amd64 -t qwen-local .
docker run --rm -p 8000:8000 qwen-local
```

Without Docker:

```bash
uv sync
uv run uvicorn app.main:app --port 8000
```

## API

Swagger UI: http://localhost:8000/docs

`GET /health` — readiness check.

`POST /chat`:

```bash
curl localhost:8000/chat -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Привет! Кто ты?"}
  ],
  "max_new_tokens": 256,
  "temperature": 0.7
}'
```

Pass `"stream": true` to get Server-Sent Events (`data: {"content": "..."}` chunks, ending with `data: [DONE]`).
`temperature: 0` switches to greedy decoding.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/model` in image | Local path or HF model id |
| `TORCH_NUM_THREADS` | `0` (torch default) | CPU threads for inference |

Build arg `MODEL_ID` selects a different model to bake in, e.g. `--build-arg MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct`.

## CI

`.github/workflows/docker.yml` builds the image on every push and PR and pushes it to
`ghcr.io/<owner>/<repo>` on pushes to `main`/`master` and `v*` tags (tags: branch, semver, sha, `latest`).
