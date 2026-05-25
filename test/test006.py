from service import _classify_intent

test_cases = [
    ("你好", "generic"),
    ("我叫张三，症状是头痛", "kg"),           # 旧代码会误判为 generic
    ("刀郎最近有什么新专辑？", "search"),      # 旧代码示例污染，新代码应 search
    ("寻医问药网客服电话", "retrieval"),
    ("鼻炎吃什么药", "kg"),
    ("高血压怎么预防", "kg"),
    ("谢谢", "generic"),
    ("这是什么病", "kg"),
    ("我肚子疼怎么办", "kg"),                  # 无实体但症状描述模式命中
]

for query, expected in test_cases:
    result = _classify_intent(query)
    status = "✅" if result == expected else "❌"
    print(f"{status} '{query}' -> {result} (expected: {expected})")