"""
chatbot/vectorstore.py

Implements the RAG grounding layer (Task 3 requirement: "must be grounded
either via RAG against documents you provide, or structured context
injection"). This was an empty file in the starter template.

Responsibilities:
  - Split uploaded/preloaded document text into overlapping chunks
  - Embed chunks locally (sentence-transformers, zero API cost)
  - Store chunks in a persistent ChromaDB collection
  - Retrieve the top-k most relevant chunks for a given question
"""

import os
import chromadb
from chromadb.utils import embedding_functions

from config import DATABASE_FOLDER

COLLECTION_NAME = "enterprise_docs"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks
TOP_K = 4

_client = None
_collection = None
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embed_fn


def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(DATABASE_FOLDER, exist_ok=True)
        _client = chromadb.PersistentClient(path=DATABASE_FOLDER)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_get_embed_fn()
        )
    return _collection


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple sliding-window chunker. Splits on paragraph boundaries where
    possible so chunks stay readable and citable."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the end of the previous one
            current = (current[-overlap:] + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def index_document(doc_id: str, text: str):
    """Chunk + embed + store a document under doc_id. Re-indexing the same
    doc_id replaces its previous chunks (so re-uploading a file updates it)."""
    collection = _get_collection()

    # Remove any existing chunks for this document first
    try:
        collection.delete(where={"source": doc_id})
    except Exception:
        pass

    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(question: str, k: int = TOP_K):
    """Return the top-k most relevant chunks (with source + distance) for a question."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=min(k, collection.count()))
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
        })
    return hits


def clear_all():
    """Wipe the whole collection (useful when testing)."""
    global _collection
    client = chromadb.PersistentClient(path=DATABASE_FOLDER)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
