import time

from pinecone import Pinecone, ServerlessSpec

import os
import re

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config


def download_pdf() -> str:
    path = os.path.abspath(config.PDF_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {config.PDF_URL}")
        r = requests.get(config.PDF_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def load_chunks(path: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if len(text) < 40:  
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({"id": f"p{page_no}-c{i}", "text": piece, "page": page_no})
    return chunks

def ensure_index(pc: Pinecone):
    existing = {i["name"] for i in pc.list_indexes()}
    if config.INDEX_NAME not in existing:
        print(f"Creating index {config.INDEX_NAME}")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    while not pc.describe_index(config.INDEX_NAME).status["ready"]:
        time.sleep(2)
    return pc.Index(config.INDEX_NAME)

def main():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    chunks = load_chunks(download_pdf())
    print(f"{len(chunks)} chunks from PDF")
    index = ensure_index(pc)
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()