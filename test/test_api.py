import os
os.environ.setdefault("DASHSCOPE_API_KEY", "你的key")  # ← 填你的

from models import get_llm_model

print("开始测试 LLM 连通性...")
llm = get_llm_model()

try:
    resp = llm.invoke("你好，请用中文回答'测试成功'")
    print(f"响应内容: {resp.content[:50]}")
    print("✅ API 正常，问题出在 Agent 层")
except Exception as e:
    print(f"❌ API 异常: {type(e).__name__}: {e}")