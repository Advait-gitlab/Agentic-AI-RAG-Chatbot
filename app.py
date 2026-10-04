from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.graph import build_graph

state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["graph"] = build_graph()
    yield


app = FastAPI(title="Agentic AI eBook RAG Chatbot", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}