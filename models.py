"""LLM 与 Embedding 模型配置"""

import os
import requests
from typing import List

from langchain_openai import ChatOpenAI
from utils import get_env, get_logger

logger = get_logger("medagent.models")

def get_llm_model(model_name: str = None) -> ChatOpenAI:
    """获取阿里云 DashScope LLM"""
    if model_name is None:
        model_name = get_env("DASHSCOPE_MODEL", "qwen3.6-plus")

    api_key = get_env("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY not set")
        raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    logger.debug(f"Using model: {model_name}")
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
        max_tokens=1500,
    )

def get_embeddings_model():
    """获取 Embedding 模型（兼容 langchain 接口）"""
    from langchain_openai import OpenAIEmbeddings

    api_key = get_env("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY not set")
        raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    logger.debug("Using text-embedding-v2 embedding model")
    return OpenAIEmbeddings(
        model="text-embedding-v2",
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    原生 Embedding 调用，绕过 langchain 的封装问题
    直接调用 DashScope API，确保参数格式正确
    """
    api_key = get_env("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set")

    # DashScope Embedding API
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 分批处理，每批最多 10 条
    all_embeddings = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        payload = {
            "model": "text-embedding-v2",
            "input": {
                "texts": batch  # 确保是字符串列表
            }
        }

        print(f"[DEBUG] Embedding API batch {i//batch_size + 1}, texts: {len(batch)}")
        print(f"[DEBUG] First text: {batch[0][:30]}...")

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"[DEBUG] API status: {response.status_code}")

            if response.status_code != 200:
                print(f"[ERROR] API error: {response.text}")
                raise ValueError(f"API returned {response.status_code}")

            data = response.json()
            print(f"[DEBUG] API response keys: {data.keys()}")

            if "output" in data and "embeddings" in data["output"]:
                embeddings = [e["embedding"] for e in data["output"]["embeddings"]]
                all_embeddings.extend(embeddings)
                print(f"[DEBUG] Got {len(embeddings)} embeddings")
            else:
                print(f"[ERROR] Unexpected response: {data}")
                raise ValueError("Invalid response format")

        except Exception as e:
            print(f"[ERROR] Embedding API error: {e}")
            raise  # 不再吞掉异常

    return all_embeddings