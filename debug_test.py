import os
import sys
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("MedAgent 诊断脚本")
print("=" * 60)

# 1. 检查环境变量
print("\n[1] 环境变量检查")
api_key = os.getenv("DASHSCOPE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_pass = os.getenv("NEO4J_PASSWORD", "password123")
print(f"  DASHSCOPE_API_KEY: {'已设置' if api_key else '❌ 未设置'}")
print(f"  NEO4J_URI: {neo4j_uri}")
print(f"  NEO4J_USER: {neo4j_user}")
print(f"  NEO4J_PASSWORD: {'已设置' if neo4j_pass else '❌ 未设置'}")

# 2. 测试 Neo4j 连接 + 查询
print("\n[2] Neo4j 连接与查询测试")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    with driver.session() as session:
        result = session.run("MATCH (n:Disease {name:'鼻炎'}) RETURN n.name AS name LIMIT 1")
        record = result.single()
        if record:
            print(f"  ✅ 图谱查询成功: {record['name']}")
        else:
            print(f"  ⚠️ 连接成功，但找不到'鼻炎'节点（可能数据未完全导入或名称不匹配）")
    driver.close()
except Exception as e:
    print(f"  ❌ Neo4j 失败: {e}")

# 3. 测试 DashScope LLM
print("\n[3] DashScope LLM 测试")
try:
    import dashscope
    dashscope.api_key = api_key
    from dashscope import Generation
    response = Generation.call(
        model="qwen-turbo",  # 或你配置的模型
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=10
    )
    if response.status_code == 200:
        print(f"  ✅ LLM 正常: {response.output.text[:20]}...")
    else:
        print(f"  ❌ LLM 返回错误: {response.status_code} - {response.message}")
except Exception as e:
    print(f"  ❌ LLM 失败: {e}")

# 4. 测试实体字典加载
print("\n[4] 实体字典 (Trie) 测试")
try:
    import json
    with open("data/entity_dict.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    print(f"  ✅ 实体字典加载成功，共 {len(entities)} 个实体")
    if "鼻炎" in entities:
        print(f"  ✅ '鼻炎' 在字典中")
    else:
        print(f"  ⚠️ '鼻炎' 不在字典中（名称可能不同，如'急性鼻炎'等）")
except Exception as e:
    print(f"  ❌ 实体字典失败: {e}")

# 5. 测试 Chroma 向量库
print("\n[5] Chroma 向量库测试")
try:
    import chromadb
    client = chromadb.PersistentClient(path="data/db")
    collections = client.list_collections()
    print(f"  ✅ Chroma 连接成功，Collections: {[c.name for c in collections]}")
except Exception as e:
    print(f"  ❌ Chroma 失败: {e}")

# 6. 直接测试 service.py / agent.py 链路（如果可能）
print("\n[6] 业务层链路测试")
try:
    from service import classify_intent, extract_entities
    intent, scores = classify_intent("鼻炎是一种什么病")
    print(f"  意图分类结果: {intent}, 分数: {scores}")
    entities = extract_entities("鼻炎是一种什么病")
    print(f"  实体抽取结果: {entities}")
except Exception as e:
    print(f"  ❌ 业务层导入/运行失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断结束。请把上面的 ❌ 和 ⚠️ 结果贴给我。")
print("=" * 60)