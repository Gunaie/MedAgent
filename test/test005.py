from app import _service_cache, get_service
import time

# 1. 创建会话
s1 = get_service("test_001")
print(f"创建后缓存大小: {len(_service_cache)}")

# 2. 再创建一个
s2 = get_service("test_002")
print(f"创建第二个后: {len(_service_cache)}")

# 3. 读取已存在的（不应增加）
s3 = get_service("test_001")
print(f"读取已存在后: {len(_service_cache)}")

# 4. 验证是同一个对象
print(f"同一对象: {s1 is s3}")

# 5. 模拟大量会话（测试 LRU 淘汰）
for i in range(1010):
    get_service(f"bulk_{i:04d}")
print(f"批量创建后: {len(_service_cache)}")  # 应 <= 1000