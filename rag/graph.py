from openai import OpenAI

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone

from . import config

NOT_FOUND = "NOT_IN_DOCUMENT"
REFUSAL = "I can't answer that from the Agentic AI eBook, so I won't guess."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages from the Agentic AI eBook.
Rules:
- Use only facts stated in the passages. Never use outside knowledge.
- Cite passages inline like [1], [2].
- If the passages contain relevant information, answer with what they say, even if it only covers part of the question, and state what is not covered.
- Only if the passages contain nothing relevant to the question, reply with exactly: {NOT_FOUND}
- State only what the passages say. Do not add comparisons, implications or contrasts that the passages do not state.
- Be concise and direct."""

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
    llm = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=5, timeout=60)

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

    def gate(state: RAGState) -> str:
        ctx = state.get("contexts") or []
        if not ctx or ctx[0]["rerank_score"] < config.MIN_RELEVANCE:
            return "refuse"
        return "generate"

    def generate(state: RAGState) -> RAGState:
        passages = "\n\n".join(
            f"[{i}] (page {c['page']}) {c['text']}" for i, c in enumerate(state["contexts"], 1)
        )
        resp = llm.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{passages}\n\nQuestion: {state['question']}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if NOT_FOUND in text:
            return {"answer": REFUSAL, "grounded": False, "confidence": 0.0}
        return {
            "answer": text,
            "grounded": True,
            "confidence": round(state["contexts"][0]["rerank_score"], 4),
        }

    def refuse(state: RAGState) -> RAGState:
        return {"answer": REFUSAL, "grounded": False, "confidence": 0.0}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("generate", generate)
    g.add_node("refuse", refuse)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", gate, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()