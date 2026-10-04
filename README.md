# Agentic AI eBook RAG Chatbot

A chatbot that answers questions only from the
[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). If the answer is not in the
eBook, it says so instead of guessing.

Built with Python, LangGraph, Pinecone (vector database, embeddings and reranking) and FastAPI.
The LLM is Groq `openai/gpt-oss-20b`; any OpenAI-compatible provider works.

## What you need

- Python 3.10 or newer
- A free [Pinecone](https://app.pinecone.io) API key
- A free [Groq](https://console.groq.com) API key

## Run it (about 5 minutes)

### 1. Get the code and install

```
git clone https://github.com/<your-github-username>/Agentic-AI-RAG-Chatbot.git
cd Agentic-AI-RAG-Chatbot
python -m venv .venv
```

Activate the virtual environment:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Then install:

```
pip install -r requirements.txt
```

### 2. Add your keys

Copy `.env.example` to a new file named `.env`:

- Windows: `copy .env.example .env`
- macOS/Linux: `cp .env.example .env`

Open `.env` and fill in two values:

```
PINECONE_API_KEY=<your Pinecone key>
LLM_API_KEY=<your Groq key>
```

Leave the other lines as they are.

### 3. Load the eBook into Pinecone (once)

```
python ingest.py
```

This downloads the PDF, splits it into chunks, embeds them and stores them in Pinecone.
Success looks like this (the last line shows 111 vectors):

```
111 chunks from PDF
embedded and upserted 96/111
embedded and upserted 111/111
DescribeIndexStatsResponse(dimension=1024, total_vector_count=111, ...)
```

### 4. Start the server

```
uvicorn app:app --reload
```

Leave this terminal open. When you see `Uvicorn running on http://127.0.0.1:8000`, it is ready.

### 5. Ask a question

1. Open **http://127.0.0.1:8000** in your browser. It opens the interactive API page.
2. Click the green **POST /chat** bar to expand it.
3. Click **Try it out** (top right of that section).
4. In the **Request body** box, replace the text with your question:
```
   {"question": "What is agentic AI?"}
```
5. Click the blue **Execute** button.
6. Scroll down to **Server response**. The answer is in the **Response body**, in the `answer` field.

Prefer the command line? In a second terminal:

- Windows PowerShell:
```
  Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body '{"question": "What is agentic AI?"}' | ConvertTo-Json -Depth 5
```
- macOS/Linux:
```
  curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"question": "What is agentic AI?"}'
```

### 6. Read the response

| Field | Meaning |
|---|---|
| `answer` | The final answer. `[1]`, `[2]` point to the chunks listed in `contexts` (by their `rank`). |
| `grounded` | `true` if answered from the eBook, `false` if the question was refused. |
| `confidence` | Relevance score from 0 to 1 of the best retrieved chunk. It is 0 for refused questions. |
| `contexts` | The retrieved chunks: `rank`, `page`, `text`, `rerank_score`, `vector_score`. |

To see a refusal, ask something unrelated, for example
`{"question": "Who won the 2022 FIFA World Cup?"}`. You should get `grounded: false`.

### 7. Run the sample queries (optional)

With the server still running, open a second terminal in the project folder, activate the virtual
environment, and run:

```
python scripts/sample_queries.py
```

This asks six questions and writes the results to `docs/sample_queries.md`.
Pre-generated results are already in that file.

## Architecture

```
PDF -> pypdf -> text splitter (1000 chars, 150 overlap)
    -> Pinecone embeddings (multilingual-e5-large, 1024-d) -> Pinecone index
```

## How it works

```
Question -> retrieve: embed the question, fetch the 8 closest chunks from Pinecone
         -> rerank:   a reranker model re-scores them and keeps the best 4
         -> gate:     best score below 0.10 -> refuse, no LLM call
                      otherwise -> generate an answer from the 4 chunks only
                      LLM says NOT_IN_DOCUMENT -> refuse
```

- **Ingestion** (`ingest.py`): PDF pages are split into 1000-character chunks with 150 characters of
  overlap. Each chunk is embedded with Pinecone's `multilingual-e5-large` model (1024 dimensions)
  and stored with its page number and text. Chunk IDs are fixed, so re-running does not create
  duplicates.

- **Pipeline** (`rag/graph.py`): a LangGraph graph with four nodes, `retrieve`, `rerank`,
  `generate` and `refuse`, and a conditional edge that picks between the last two.

- **Why rerank:** vector similarity scores were close for on-topic and off-topic questions
  (about 0.85 to 0.90 vs 0.70). Reranker scores separated them clearly (0.94 to 0.998 vs about
  0.001), so the gate and the confidence use the reranker score.

- **Staying inside the eBook:** the prompt allows only facts from the retrieved chunks and
  requires citations, a relevance gate refuses weak matches before the LLM is called, and the LLM
  must answer `NOT_IN_DOCUMENT` when the chunks hold nothing relevant.

- **Confidence** measures how relevant the retrieved evidence is. It is not the probability that
  the answer is correct.

- **API** (`app.py`): FastAPI with `POST /chat` and `GET /health`.

## Project files

```
app.py                      API server
ingest.py                   PDF -> chunks -> embeddings -> Pinecone
rag/config.py               settings
rag/graph.py                LangGraph pipeline
scripts/sample_queries.py   runs the sample questions
docs/sample_queries.md      sample questions with real outputs
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `401` or "invalid API key" | Check the keys in `.env` for typos or extra spaces, then restart uvicorn. |
| `404 model not found` | Set `LLM_MODEL` in `.env` to a model your Groq account can use, then restart uvicorn. |
| `429` | The free LLM quota is used up. Wait, or switch provider or model in `.env`. |
| `.env` changes have no effect | Stop uvicorn with Ctrl+C and start it again. |
| `ModuleNotFoundError` | The virtual environment is not active, or `pip install -r requirements.txt` was skipped. |

## Known limitations

- Tables in the PDF are flattened by text extraction (for example the challenges table on
  page 36), so the link between each row and column is lost.
- The relevance threshold (`MIN_RELEVANCE`, default 0.10) was chosen from a small set of test
  questions.
- Free-tier LLM limits can cause temporary errors; the API returns a clear 429 or 503 message
  when that happens.