from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
auth = ("neo4j", "12345678")  # ← 这里必须是 12345678，不能是别的

with GraphDatabase.driver(uri, auth=auth) as driver:
    driver.verify_connectivity()
    print("✅ Neo4j 连接成功！")