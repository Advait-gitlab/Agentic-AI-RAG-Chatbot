import os

import requests

API = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

QUERIES = [
    "What is agentic AI?",
    "What are the challenges of orchestrating complex agentic systems?",
    "What is a multi-agent system and how does it differ from a single-agent system?",
    "Why is testing and validation important before deploying agentic AI?",
    "How should an organization assess its readiness before adopting agentic AI?",
    "Who won the 2022 FIFA World Cup?"
]

os.makedirs("docs", exist_ok=True)
lines = ["# Sample queries\n"]

for q in QUERIES:
    r = requests.post(API, json={"question": q}, timeout=120)
    r.raise_for_status()
    d = r.json()
    lines.append(f"## Q: {q}\n")
    lines.append(f"**Answer:** {d['answer']}\n")
    lines.append(f"**Grounded:** {d['grounded']} | **Confidence:** {d['confidence']}\n")
    lines.append("**Retrieved chunks:**\n")
    for c in d["contexts"]:
        snippet = c["text"][:200].replace("\n", " ")
        lines.append(f"- p.{c['page']} (rerank {c['rerank_score']}, vector {c['vector_score']}): {snippet}...")
    lines.append("")
    print(f"done: {q}")

with open("docs/sample_queries.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote docs/sample_queries.md")