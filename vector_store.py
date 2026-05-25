"""Chroma 向量库封装"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from models import get_embeddings_model
from utils import get_logger

logger = get_logger("medagent.vectorstore")

_store_instance = None
_emb_instance = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PERSIST_DIR = os.path.join(BASE_DIR, "data", "db")

def _get_embeddings() -> Embeddings:
    global _emb_instance
    if _emb_instance is None:
        _emb_instance = get_embeddings_model()
    return _emb_instance

def get_vector_store(persist_dir: str | None = None) -> Chroma:
    global _store_instance
    if _store_instance is None:
        if persist_dir is None:
            persist_dir = DEFAULT_PERSIST_DIR
        else:
            if not os.path.isabs(persist_dir):
                persist_dir = os.path.join(BASE_DIR, persist_dir)

        persist_dir = os.path.abspath(persist_dir)
        Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

        _store_instance = Chroma(
            persist_directory=persist_dir,
            embedding_function=_get_embeddings(),
        )

        count = _store_instance._collection.count()
        logger.info(f"Initialized at {persist_dir}, collection count: {count}")

    return _store_instance

def add_documents_to_store(
    documents: list[Document],
    persist_dir: str | None = None,
    batch_size: int = 5000,
) -> Chroma:
    store = get_vector_store(persist_dir)

    total = len(documents)
    if total == 0:
        logger.warning("No documents to add")
        return store

    count_before = store._collection.count()
    logger.debug(f"Collection count before: {count_before}")

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        ids = [f"doc_{count_before + i + j}" for j in range(len(batch))]
        store.add_documents(batch, ids=ids)
        logger.info(f"Added {min(i + batch_size, total)} / {total} document chunks")

    count_after = store._collection.count()
    logger.info(f"Collection count after: {count_after} (new: {count_after - count_before})")

    return store

def similarity_search(
    query: str,
    k: int = 5,
    score_threshold: float | None = None,
    persist_dir: str | None = None,
) -> list[str]:
    store = get_vector_store(persist_dir)

    count = store._collection.count()
    if count == 0:
        logger.warning("Collection is empty, cannot retrieve")
        return []

    results = store.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        logger.info(f"No documents retrieved for: '{query[:30]}...'")
        return []

    threshold = score_threshold if score_threshold is not None else 0.45
    filtered = [doc.page_content for doc, score in results if score > threshold]

    logger.info(f"Retrieved '{query[:30]}...' -> {len(results)} raw, {len(filtered)} after threshold {threshold}")
    for i, (doc, score) in enumerate(results):
        flag = "✓" if score > threshold else "✗"
        logger.debug(f"  [{flag}] score={score:.3f}: {doc.page_content[:60]}...")

    if not filtered and results:
        filtered = [doc.page_content for doc, score in results[:2]]
        logger.info(f"Threshold filtered to empty, fallback to top-{len(filtered)}")

    return filtered