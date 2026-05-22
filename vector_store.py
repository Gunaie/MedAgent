"""Chroma 向量库封装（优化版）"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from models import get_embeddings_model

# 全局单例，避免每次查询重复初始化
_store_instance = None
_emb_instance = None


def _get_embeddings() -> Embeddings:
    """Embedding 模型单例"""
    global _emb_instance
    if _emb_instance is None:
        _emb_instance = get_embeddings_model()
    return _emb_instance


def get_vector_store(persist_dir: str | None = None) -> Chroma:
    """获取或创建持久化 Chroma 向量库（单例模式）"""
    global _store_instance
    if _store_instance is None:
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/db")
        Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)

        _store_instance = Chroma(
            persist_directory=persist_dir,
            embedding_function=_get_embeddings(),
        )
        print(f"[VectorStore] Initialized at {persist_dir}")
    return _store_instance


def add_documents_to_store(
        documents: list[Document],
        persist_dir: str | None = None,
        batch_size: int = 5000,  # ← FIX: Chroma 最大 batch size 约 5461，留安全余量
) -> Chroma:
    """批量添加文档到向量库（自动分批，避免 batch size 溢出）"""
    store = get_vector_store(persist_dir)

    total = len(documents)
    if total == 0:
        print("没有文档需要添加")
        return store

    # 分批处理，避免一次性传入过多文档导致 Chroma 报错
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        store.add_documents(batch)
        print(f"[VectorStore] 已添加 {min(i + batch_size, total)} / {total} 个文档片段")

    print(f"共向量化 {total} 个文档片段，存储于 {persist_dir or './data/db'}")
    return store


def similarity_search(
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        persist_dir: str | None = None,
) -> list[str]:
    """
    相似度检索，返回过滤后的文本内容列表
    """
    store = get_vector_store(persist_dir)
    results = store.similarity_search_with_relevance_scores(query, k=k)

    # 相关性分数越大越相似（范围通常为 0~1）
    filtered = [doc.page_content for doc, score in results if score > score_threshold]

    # 兜底：如果阈值过滤后为空，至少保留最相似的前2个，避免 Agent 只能走搜索
    if not filtered and results:
        filtered = [doc.page_content for doc, score in results[:2]]

    return filtered