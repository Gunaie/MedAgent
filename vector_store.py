"""Chroma 向量库封装"""

import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

from models import embed_texts
from utils import get_logger

logger = get_logger("medagent.vectorstore")

_client = None
_collection = None

DEFAULT_PERSIST_DIR = "./data/db"

def _get_persist_dir(persist_dir: str | None = None) -> str:
    if persist_dir:
        return os.path.abspath(persist_dir)
    env_dir = os.getenv("CHROMA_PERSIST_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.abspath("./data/db")

def _get_client(persist_dir: str | None = None):
    global _client
    if _client is None:
        persist_dir = _get_persist_dir(persist_dir)
        Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

        # FIX: 双重保险，确保遥测被禁用（环境变量已在 app.py 中设置，此处再次确认）
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

        _client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
            is_persistent=True,
        ))
        logger.info(f"Chroma client initialized at {persist_dir}")

    return _client

def _get_collection(name: str = "documents", persist_dir: str | None = None):
    global _collection
    if _collection is None:
        client = _get_client(persist_dir)
        _collection = client.get_or_create_collection(name=name)
        logger.info(f"Collection '{name}' loaded, count: {_collection.count()}")

    return _collection

def add_documents_to_store(
    documents: list,
    persist_dir: str | None = None,
    batch_size: int = 10,
):
    collection = _get_collection(persist_dir=persist_dir)

    total = len(documents)
    if total == 0:
        logger.warning("No documents to add")
        return

    count_before = collection.count()
    logger.debug(f"Collection count before: {count_before}")

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]

        texts = []
        ids = []

        for j, doc in enumerate(batch):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if not content or len(content.strip()) < 2:
                continue

            texts.append(content.strip())
            ids.append(f"doc_{count_before + i + j}")

        if not texts:
            continue

        logger.debug(f"Processing batch {i//batch_size + 1}, texts: {len(texts)}")

        try:
            embeddings = embed_texts(texts)
            logger.debug(f"Got {len(embeddings)} embeddings")

            collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids
            )
            logger.info(f"Added {min(i + len(texts), total)} / {total} document chunks")

        except Exception as e:
            logger.error(f"Failed batch {i//batch_size + 1}: {e}")
            raise

    count_after = collection.count()
    logger.info(f"Collection count after: {count_after} (new: {count_after - count_before})")

def similarity_search(
    query: str,
    k: int = 5,
    score_threshold: float | None = None,
    persist_dir: str | None = None,
) -> list[str]:
    """相似度检索"""
    collection = _get_collection(persist_dir=persist_dir)

    count = collection.count()
    if count == 0:
        logger.warning("Collection is empty, cannot retrieve")
        return []

    try:
        query_embeddings = embed_texts([query])
        query_embedding = query_embeddings[0]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    if not results or not results["documents"]:
        logger.info(f"No documents retrieved for: '{query[:30]}...'")
        return []

    documents = results["documents"][0]
    distances = results["distances"][0] if "distances" in results else [0] * len(documents)

    # FIX: 默认阈值 0.3（更宽松），支持自定义
    threshold = score_threshold if score_threshold is not None else 0.3

    filtered = []
    for doc, dist in zip(documents, distances):
        score = 1.0 - min(dist, 1.0)
        logger.debug(f"  score={score:.3f}: {doc[:60]}...")
        if score > threshold:
            filtered.append(doc)

    logger.info(f"Retrieved '{query[:30]}...' -> {len(documents)} raw, {len(filtered)} after threshold {threshold}")

    # FIX: 如果过滤后为空，强制返回 top-2（避免"未找到"）
    if not filtered and documents:
        filtered = documents[:2]
        logger.info(f"Threshold filtered to empty, fallback to top-{len(filtered)}")

    return filtered