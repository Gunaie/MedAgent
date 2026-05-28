import traceback
from service import ChatService

service = ChatService(session_id="debug_test")

test_queries = [
    "你好",                           # generic
    "鼻炎是一种什么病",                # kg
    "寻医问药网的客服电话是多少",       # retrieval
    "感冒吃什么药好",                  # kg
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"提问: {q}")
    try:
        result = service.answer(q)
        print(f"结果: {result}")
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")
        traceback.print_exc()