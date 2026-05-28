# debug_chat.py
import traceback
from service import ChatService

service = ChatService()

test_queries = [
    "你好",                           # generic
    "鼻炎是一种什么病",                # kg
    "寻医问药网的客服电话是多少",       # retrieval
    "感冒吃什么药好",                  # kg
]

for q in test_queries:
    print(f"\n{'='*50}")
    print(f"提问: {q}")
    try:
        result = service.chat(q, history=[])
        print(f"结果: {result[:100]}...")
    except Exception as e:
        print(f"❌ 异常: {e}")
        traceback.print_exc()