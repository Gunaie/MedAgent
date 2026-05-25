from tools import _search_all, cached_search

# 测试 1：通用问题（应走 DuckDuckGo 或 Bing）
results = _search_all("Python 最新版本是多少", num_results=3)
print(f"通用搜索: {len(results)} 条")
for r in results:
    print(f"  [{r['source']}] {r['title'][:30]}")

# 测试 2：带地域的问题（测试中文搜索）
results = _search_all("北京今天天气", num_results=3)
print(f"\n天气搜索: {len(results)} 条")

# 测试 3：缓存命中
results2 = cached_search("Python 最新版本是多少")
print(f"\n缓存测试: {len(results2)} 条")