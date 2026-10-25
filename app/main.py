import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.model import ChatModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    stream: bool = False


class ChatResponse(BaseModel):
    model: str
    message: Message
    prompt_tokens: int
    completion_tokens: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ChatModel()
    yield


app = FastAPI(title="Qwen local inference", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model": request.app.state.model.model_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    model: ChatModel = request.app.state.model
    messages = [m.model_dump() for m in req.messages]
    params = dict(max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_p=req.top_p)

    if req.stream:
        def events():
            for chunk in model.stream(messages, **params):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        # A sync generator is iterated in a threadpool, so the event loop is not blocked.
        return StreamingResponse(events(), media_type="text/event-stream")

    result = await run_in_threadpool(model.generate, messages, **params)
    return ChatResponse(
        model=model.model_id,
        message=Message(role="assistant", content=result["content"]),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )
