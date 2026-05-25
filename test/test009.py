from vector_store import similarity_search

# 测试 1：企业内部文档查询（假设已导入文档）
results = similarity_search("寻医问药网客服电话", k=5)
print(f"\n最终返回: {len(results)} 条")

# 测试 2：阈值过滤为空时的降级
results = similarity_search("完全不相关的查询xyz123", k=5)
print(f"\n降级测试: {len(results)} 条")