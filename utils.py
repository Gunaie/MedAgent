"""通用工具函数"""

import os
from dotenv import load_dotenv
from diskcache import Cache

# 加载环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(env_path)

# 创建缓存目录
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化缓存
cache = Cache(CACHE_DIR)


def get_env(key: str, default: str = "") -> str:
    """安全获取环境变量"""
    return os.getenv(key, default)


def cached_llm_call(key: str, func, *args, **kwargs):
    """缓存 LLM 调用结果"""
    cache_key = f"llm:{key}"
    result = cache.get(cache_key)
    if result is not None:
        print(f"[CACHE] Hit: {cache_key}")
        return result

    result = func(*args, **kwargs)
    cache.set(cache_key, result, expire=3600)  # 缓存1小时
    return result


def cached_embedding_call(key: str, func, *args, **kwargs):
    """缓存 Embedding 调用结果"""
    cache_key = f"emb:{hash(key)}"
    result = cache.get(cache_key)
    if result is not None:
        print(f"[CACHE] Hit: {cache_key}")
        return result

    result = func(*args, **kwargs)
    cache.set(cache_key, result, expire=86400)  # 缓存24小时
    return result


if __name__ == '__main__':
    print(f"Cache dir: {CACHE_DIR}")
    print(f"API Key loaded: {'Yes' if get_env('DASHSCOPE_API_KEY') else 'No'}")