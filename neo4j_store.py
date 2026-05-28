from dotenv import load_dotenv
load_dotenv()

"""Neo4j 医疗知识图谱封装"""

import os
from typing import List, Dict, Set
from neo4j import GraphDatabase

from utils import get_env, get_logger

logger = get_logger("medagent.neo4j")

ALLOWED_LABELS: Set[str] = {
    "Disease", "Drug", "Symptom", "Department", "Check",
    "Food", "Cureway", "Dishes", "Category"
}
ALLOWED_RELATIONS: Set[str] = {
    "DISEASE_SYMPTOM", "DISEASE_DRUG", "DISEASE_DEPARTMENT",
    "DISEASE_CHECK", "DISEASE_CUREWAY", "DISEASE_CATEGORY",
    "DISEASE_ACOMPANY", "DISEASE_DO_EAT", "DISEASE_NOT_EAT",
    "DISEASE_DISHES"
}

class MedicalGraphStore:
    """医疗知识图谱存储"""

    def __init__(self):
        uri = get_env("NEO4J_URI", "bolt://localhost:7687")
        user = get_env("NEO4J_USER", "neo4j")
        password = get_env("NEO4J_PASSWORD", "password123")
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_acquisition_timeout=5,
            connection_timeout=5,
            max_transaction_retry_time=5,
        )
        self._existing_rels: Set[str] = set()
        self._drug_rels: Set[str] = set()
        self._refresh_schema()

    def _refresh_schema(self):
        """探测数据库中实际存在的关系类型"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                self._existing_rels = {r["relationshipType"] for r in result}
                self._drug_rels = {
                    r for r in self._existing_rels
                    if any(k in r.upper() for k in ["TREAT", "DRUG", "HAS_DRUG", "RELEVANT"])
                }
                logger.info(f"Neo4j 实际关系类型: {self._existing_rels}")
                logger.info(f"Neo4j 药物相关关系: {self._drug_rels}")
        except Exception as e:
            logger.warning(f"Schema 探测失败: {e}")

    def close(self):
        self.driver.close()

    def search_entities(self, name: str, limit: int = 5) -> List[Dict]:
        cypher = """
        MATCH (n:Disease|Symptom|Drug)
        WHERE n.name CONTAINS $name
        RETURN n.name as name, labels(n)[0] as type, coalesce(n.desc, '') as desc
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=name, limit=limit)
            return [dict(record) for record in result]

    def get_subgraph(self, entity_name: str, depth: int = 1, limit: int = 10) -> List[Dict]:
        if depth != 1:
            depth = 1
        results = []
        for label in ["Disease", "Symptom", "Drug"]:
            cypher = f"""
            MATCH path = (n:{label} {{name: $name}})-[r*1..1]-(m)
            RETURN [node in nodes(path) | {{name: node.name, type: labels(node)[0]}}] as nodes,
                   [rel in relationships(path) | {{type: type(rel), from: startNode(rel).name, to: endNode(rel).name}}] as rels,
                   length(path) as hops
            LIMIT $limit
            """
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, name=entity_name, limit=limit)
                    batch = [dict(record) for record in result]
                    results.extend(batch)
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.warning(f"Subgraph query failed for {label}/{entity_name}: {e}")
                continue
        return results[:limit]

    def query_disease_drugs(self, disease_name: str) -> List[Dict]:
        results = []
        cypher_out = """
        MATCH (d:Disease {name: $name})-[r:DISEASE_DRUG]->(drug:Drug)
        RETURN drug.name as name, type(r) as rel_type, coalesce(drug.desc, '') as desc
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(cypher_out, name=disease_name)
            for r in result:
                results.append({"name": r["name"], "rel_type": r["rel_type"], "desc": r["desc"] or ""})
        if not results:
            cypher_in = """
            MATCH (d:Disease {name: $name})-[r:DISEASE_DRUG]-(drug:Drug)
            RETURN drug.name as name, type(r) as rel_type, coalesce(drug.desc, '') as desc
            LIMIT 10
            """
            with self.driver.session() as session:
                result = session.run(cypher_in, name=disease_name)
                for r in result:
                    results.append({"name": r["name"], "rel_type": r["rel_type"], "desc": r["desc"] or ""})
        return results

    def query_drug_diseases(self, drug_name: str) -> List[Dict]:
        results = []
        cypher = """
        MATCH (d:Disease)-[r:DISEASE_DRUG]->(drug:Drug {name: $name})
        RETURN d.name as name, type(r) as rel_type
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=drug_name)
            for r in result:
                results.append({"name": r["name"], "rel_type": r["rel_type"]})
        return results

    def query_by_template(self, template_key: str, entity_name: str, fuzzy: bool = False) -> str:
        from config import GRAPH_TEMPLATE

        tpl = GRAPH_TEMPLATE.get(template_key)
        if not tpl:
            return ""

        slot = tpl["slots"][0]
        cypher = tpl["cypher"]

        if fuzzy and not (2 <= len(entity_name) <= 15):
            return ""

        try:
            with self.driver.session() as session:
                result = session.run(cypher, {slot: entity_name}, timeout=3.0)
                record = result.single()

                if not record and fuzzy:
                    fuzzy_cypher = (
                        cypher.replace(f"n.name = ${slot}", f"n.name CONTAINS ${slot}")
                        .replace(f"d.name = ${slot}", f"d.name CONTAINS ${slot}")
                        .replace(f"m.name = ${slot}", f"m.name CONTAINS ${slot}")
                        .replace(f"drug.name = ${slot}", f"drug.name CONTAINS ${slot}")
                    )
                    result = session.run(fuzzy_cypher, {slot: entity_name}, timeout=3.0)
                    record = result.single()

                if not record:
                    return ""

                answer = tpl["answer"]
                for key, value in record.items():
                    if isinstance(value, list):
                        value_str = "、".join(value) if value else "暂无记录"
                    else:
                        value_str = value or "暂无记录"
                    answer = answer.replace(f"{{{key}}}", value_str)
                answer = answer.replace(f"{{{slot}}}", entity_name)
                return answer
        except Exception as e:
            logger.warning(f"Template query failed ({template_key}, {entity_name}, fuzzy={fuzzy}): {e}")
            return ""

    def query_symptom_diseases(self, symptom_name: str) -> List[str]:
        cypher = """
        MATCH (s:Symptom {name: $name})<-[:DISEASE_SYMPTOM]-(d:Disease)
        RETURN d.name as name, coalesce(d.desc, '') as desc
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=symptom_name)
            return [r['name'] for r in result]

    def query_disease_symptoms(self, disease_name: str) -> List[str]:
        cypher = """
        MATCH (d:Disease {name: $name})-[:DISEASE_SYMPTOM]->(s:Symptom)
        RETURN s.name as symptom
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=disease_name)
            return [r["symptom"] for r in result]

    def get_cross_entity_info(self, entity_a: str, entity_b: str) -> str:
        cypher = """
        MATCH path = shortestPath((a)-[*1..3]-(b))
        WHERE a.name = $a AND b.name = $b
        RETURN [node in nodes(path) | node.name] as path_names,
               [rel in relationships(path) | type(rel)] as path_rels
        """
        with self.driver.session() as session:
            result = session.run(cypher, a=entity_a, b=entity_b)
            record = result.single()
            if not record:
                return "未找到关联"
            nodes = record["path_names"]
            rels = record["path_rels"]
            desc = nodes[0]
            for i, rel in enumerate(rels):
                desc += f" --[{rel}]--> {nodes[i+1]}"
            return desc

    def get_entity_type(self, name: str) -> str:
        cypher = """
        MATCH (n) WHERE n.name = $name
        RETURN labels(n)[0] as type LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=name)
            record = result.single()
            return record["type"] if record else "Unknown"

    def get_entity_types(self, names: List[str]) -> Dict[str, str]:
        if not names:
            return {}
        cypher = """
        UNWIND $names as name
        MATCH (n) WHERE n.name = name
        RETURN n.name as name, labels(n)[0] as type
        """
        with self.driver.session() as session:
            result = session.run(cypher, names=names)
            return {r["name"]: r["type"] for r in result}

    def load_all_entities(self) -> set:
        cypher = """
        MATCH (n:Disease|Symptom|Drug) WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name as name
        """
        with self.driver.session() as session:
            result = session.run(cypher)
            return {r["name"] for r in result}

_medical_graph = None

def get_medical_graph() -> MedicalGraphStore:
    global _medical_graph
    if _medical_graph is None:
        _medical_graph = MedicalGraphStore()
    return _medical_graph