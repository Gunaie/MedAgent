# debug_model.py
import os
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("测试 models.py 的 LLM 配置")
print("=" * 60)

# 1. 环境变量
api_key = os.getenv("DASHSCOPE_API_KEY")
model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
print(f"API Key: {api_key[:10]}... (长度{len(api_key) if api_key else 0})")
print(f"Model: {model}")

# 2. 测试 models.py 的 get_llm_model
print("\n[1] 测试 get_llm_model()")
try:
    from models import get_llm_model
    llm = get_llm_model()
    print(f"  LLM 对象创建成功: {llm}")
    print(f"  model: {llm.model_name if hasattr(llm, 'model_name') else llm.model}")
    print(f"  openai_api_base: {llm.openai_api_base if hasattr(llm, 'openai_api_base') else 'N/A'}")
    print(f"  openai_api_key: {'已设置' if hasattr(llm, 'openai_api_key') and llm.openai_api_key else '❌ 未设置!'}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# 3. 直接调用测试
print("\n[2] 直接调用 LLM")
try:
    from models import get_llm_model
    llm = get_llm_model()
    result = llm.invoke("你好")
    print(f"  成功: {result.content[:50]}...")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# 4. 查看 service.py 里的 NER 调用
print("\n[3] 查看 service.py 中的 NER 代码")
import service
import inspect
source = inspect.getsource(service)
# 找到 LLM NER 相关代码
for i, line in enumerate(source.split('\n'), 1):
    if 'NER' in line or 'dashscope' in line.lower() or 'Generation' in line or 'url' in line.lower():
        print(f"  service.py:{i}: {line.strip()}")