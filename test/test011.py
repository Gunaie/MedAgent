from models import get_llm_model

llm = get_llm_model()
print(f"当前模型: {llm.model_name}")  # 应输出 qwen3.6-plus

# 测试调用
response = llm.invoke("你好")
print(response.content)