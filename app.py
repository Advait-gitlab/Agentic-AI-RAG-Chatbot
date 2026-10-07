import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from fastapi.responses import FileResponse

from rag.graph import build_graph

state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["graph"] = build_graph()
    yield


app = FastAPI(title="Agentic AI eBook RAG Chatbot", lifespan=lifespan)

class ChatRequest(BaseModel):
    question: str = Field(min_length=3)


class ContextChunk(BaseModel):
    rank: int
    page: int
    text: str
    rerank_score: float
    vector_score: float


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    contexts: list[ContextChunk]

#@app.get("/", include_in_schema=False)
#def root():
   # return RedirectResponse(url="/docs")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    out = state["graph"].invoke({"question": req.question})
    contexts = [
        ContextChunk(
            rank=i,
            page=c["page"],
            text=c["text"],
            rerank_score=round(c["rerank_score"], 4),
            vector_score=round(c["vector_score"], 4),
        )
        for i, c in enumerate(out.get("contexts", []), 1)
    ]
    return ChatResponse(
        answer=out["answer"],
        grounded=out["grounded"],
        confidence=out["confidence"],
        contexts=contexts,
    )