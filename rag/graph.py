from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone

from . import config


class RAGState(TypedDict, total=False):
    question: str
    candidates: list[dict]  
    contexts: list[dict]    
    answer: str
    grounded: bool
    confidence: float


def build_graph():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.INDEX_NAME)

    def retrieve(state: RAGState) -> RAGState:
        q = pc.inference.embed(
            model=config.EMBED_MODEL,
            inputs=[state["question"]],
            parameters={"input_type": "query", "truncate": "END"},
        )
        res = index.query(vector=q[0].values, top_k=config.TOP_K, include_metadata=True)
        cands = [
            {
                "id": m.id,
                "text": m.metadata["text"],
                "page": int(m.metadata["page"]),
                "vector_score": float(m.score),
            }
            for m in res.matches
        ]
        return {"candidates": cands}

    def rerank(state: RAGState) -> RAGState:
        cands = state["candidates"]
        if not cands:
            return {"contexts": []}
        rr = pc.inference.rerank(
            model=config.RERANK_MODEL,
            query=state["question"],
            documents=[{"id": c["id"], "text": c["text"]} for c in cands],
            rank_fields=["text"],
            top_n=config.RERANK_TOP_N,
            return_documents=False,
        )
        contexts = []
        for item in rr.data:
            c = dict(cands[item.index])
            c["rerank_score"] = float(item.score)
            contexts.append(c)
        return {"contexts": contexts}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("retrieve", END)
    return g.compile()