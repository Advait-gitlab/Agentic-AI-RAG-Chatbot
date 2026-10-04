import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("PINECONE_INDEX", "agentic-ai-ebook")

EMBED_MODEL = "multilingual-e5-large"  
EMBED_DIM = 1024
RERANK_MODEL = "bge-reranker-v2-m3"  

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.10"))

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Ebook-Agentic-AI.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150