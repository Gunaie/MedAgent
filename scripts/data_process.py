"""文档向量化处理"""

import os
from glob import glob

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import add_documents_to_store


def load_and_split_documents(data_dir: str = "../data/inputs") -> list:
    """加载并分割文档"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    documents = []
    for file_path in glob(os.path.join(data_dir, "*.*")):
        loader = None
        if file_path.endswith(".csv"):
            loader = CSVLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")

        if loader:
            documents.extend(loader.load_and_split(text_splitter))

    return documents


def build_vector_database(data_dir: str = "../data/inputs", persist_dir: str = "../data/db"):
    """构建向量数据库"""
    docs = load_and_split_documents(data_dir)
    if not docs:
        print("⚠️ 未找到文档，请检查数据目录")
        return

    print(f"📄 共加载 {len(docs)} 个文档片段，准备向量化...")
    add_documents_to_store(docs, persist_dir)


if __name__ == "__main__":
    build_vector_database()