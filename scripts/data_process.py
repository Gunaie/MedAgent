"""文档向量化处理"""

import os
import sys
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import add_documents_to_store, PROJECT_ROOT


def load_and_split_documents(data_dir: str | None = None) -> list:
    """加载并分割文档"""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "inputs")

    print(f"[DataProcess] 扫描目录: {data_dir}")
    print(f"[DataProcess] 匹配到的文件: {glob(os.path.join(data_dir, '*.*'))}")

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
            print(f"[DataProcess] 加载: {os.path.basename(file_path)}")
            try:
                docs = loader.load_and_split(text_splitter)
                documents.extend(docs)
                print(f"[DataProcess]   -> 分割为 {len(docs)} 个片段")
            except Exception as e:
                print(f"[DataProcess]   -> 加载失败: {e}")

    return documents


def build_vector_database(data_dir: str | None = None, persist_dir: str | None = None):
    """构建向量数据库"""
    docs = load_and_split_documents(data_dir)
    if not docs:
        print("⚠️ 未找到文档，请检查数据目录")
        return

    print(f"📄 共加载 {len(docs)} 个文档片段，准备向量化...")
    add_documents_to_store(docs, persist_dir)


if __name__ == "__main__":
    build_vector_database()