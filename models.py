"""模型工厂：统一创建 LLM 和 Embedding 模型"""

import os
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
import dashscope

from utils import get_env, cached_llm_call, cached_embedding_call


api_key = get_env("DASHSCOPE_API_KEY")
model_name = get_env("LLM_MODEL", "qwen3.5-flash")
embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-v4")

print(f"[DEBUG] Model: {model_name}")


class CachedChatOpenAI(ChatOpenAI):
    """带缓存的 ChatOpenAI"""

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, list):
            key = str([m.get("content", "") if isinstance(m, dict) else str(m) for m in input])
        elif hasattr(input, 'to_messages'):
            key = str(input.to_messages())
        else:
            key = str(input)
        return cached_llm_call(key, super().invoke, input, config, **kwargs)


class DashScopeEmbedding(Embeddings):
    """自定义阿里云 DashScope Embedding（分批处理 + 缓存）"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_key = "|".join(batch)

            def do_embed():
                print(f"[DEBUG] Embedding batch {i//batch_size + 1}, size: {len(batch)}")
                resp = dashscope.TextEmbedding.call(
                    model=self.model,
                    input=batch,
                    api_key=self.api_key,
                )
                if resp.status_code != 200:
                    raise Exception(f"Embedding error: {resp.message}")
                return [item["embedding"] for item in resp.output["embeddings"]]

            batch_embeddings = cached_embedding_call(batch_key, do_embed)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_llm_model():
    return CachedChatOpenAI(
        model=model_name,
        temperature=float(get_env("TEMPERATURE", "0")),
        max_tokens=int(get_env("MAX_TOKENS", "1000")),
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=15,          # 超时报错不卡死
        max_retries=1,
    )


def get_embeddings_model():
    """获取阿里云 DashScope Embedding 模型"""
    return DashScopeEmbedding(
        model=embedding_model,
        api_key=api_key,
    )