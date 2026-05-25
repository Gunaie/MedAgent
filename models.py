"""LLM 与 Embedding 模型配置"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import get_env, get_logger

logger = get_logger("medagent.models")

def get_llm_model(model_name: str = None) -> ChatOpenAI:
    """获取阿里云 DashScope LLM"""
    if model_name is None:
        model_name = get_env("DASHSCOPE_MODEL", "qwen3.5-flash")

    api_key = get_env("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY not set")
        raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    logger.debug(f"Using model: {model_name}")
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=1500,
    )

def get_embeddings_model() -> OpenAIEmbeddings:
    """获取阿里云 DashScope Embedding"""
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