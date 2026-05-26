"""文档向量化处理"""

import os
import sys
from glob import glob

# FIX: 强制使用本文件所在目录推导项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 如果本文件在项目根目录，PROJECT_ROOT = SCRIPT_DIR
# 如果本文件在 scripts/，PROJECT_ROOT = 上级目录
if os.path.basename(SCRIPT_DIR) == 'scripts':
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

sys.path.insert(0, PROJECT_ROOT)

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import add_documents_to_store

print(f"[DEBUG] Script directory: {SCRIPT_DIR}")
print(f"[DEBUG] Project root: {PROJECT_ROOT}")

def load_and_split_documents(data_dir: str | None = None) -> list:
    """加载并分割文档"""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "inputs")

    print(f"[DataProcess] 扫描目录: {data_dir}")
    files = glob(os.path.join(data_dir, "*.*"))
    print(f"[DataProcess] 匹配到的文件: {files}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    documents = []
    for file_path in files:
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

    # FIX: 强制传入绝对路径
    if persist_dir is None:
        persist_dir = os.path.join(PROJECT_ROOT, "data", "db")

    print(f"[DEBUG] Vector store path: {persist_dir}")
    add_documents_to_store(docs, persist_dir)


if __name__ == "__main__":
    build_vector_database()