import os
import re
import json
import logging
from typing import List, Dict
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_MODEL
from langchain_community.chat_models import ChatTongyi

logger = logging.getLogger(__name__)

_entity_dict = set()
_entity_trie = None
_entity_dict_loading = False
_entity_dict_load_failed = False

# ========== FIX 6: 改进 Trie 实体提取，增加调试日志 ==========
def _extract_entities_fast(text: str) -> list[str]:
    global _entity_trie
    if _entity_trie is None:
        _load_entity_dict()
    if _entity_trie is None:
        logger.warning("Trie is None, cannot extract")
        return []
    
    text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text)
    if not text:
        return []
    
    found = set()
    n = len(text)
    i = 0
    while i < n:
        node = _entity_trie
        j = i
        last_match = None
        while j < n and text[j] in node:
            node = node[text[j]]
            if _TRIE_END in node:
                last_match = j
            j += 1
        if last_match is not None:
            entity = text[i:last_match + 1]
            found.add(entity)
            i = last_match + 1
        else:
            i += 1
    
    result = sorted(found, key=lambda x: len(x), reverse=True)
    if result:
        logger.info(f"Fast NER: '{text[:30]}...' -> {result}")
    return result


def _llm_ner_extract(query: str) -> list[str]:
    llm = get_llm()
    prompt = f"""请从以下用户查询中提取所有医疗实体（疾病、症状、药物等），以JSON数组格式返回，不要添加任何解释。

用户查询：{query}

输出格式：["实体1", "实体2", ...]"""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            entities = json.loads(match.group(0))
            if isinstance(entities, list):
                valid = [e for e in entities if isinstance(e, str) and len(e) >= 2]
                logger.info(f"LLM NER: {valid}")
                return valid
    except Exception as e:
        logger.error(f"LLM NER error: {e}")
    return []


def _extract_entities(query: str) -> list[str]:
    _load_entity_dict()
    entities = _extract_entities_fast(query)
    if entities:
        return entities
    logger.info(f"Trie miss, fallback to LLM NER: {query}")
    return _llm_ner_extract(query)


def _is_disease(entity: str) -> bool:
    if not _entity_dict:
        return True
    return entity in _entity_dict


def _is_symptom(entity: str) -> bool:
    return True


def _is_medicine(entity: str) -> bool:
    return True


from models import get_llm_model
def get_llm():
    return get_llm_model()


def get_medical_graph():
    return MedicalGraph()


class MedicalGraph:
    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def query(self, cypher: str, parameters: dict = None) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [record.data() for record in result]

    def query_disease_symptoms(self, disease_name: str) -> List[str]:
        cypher = """
        MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name AS symptom
        """
        result = self.query(cypher, {"name": disease_name})
        symptoms = [r["symptom"] for r in result if r.get("symptom")]
        if not symptoms:
            fuzzy = self.search_entities(disease_name, limit=1)
            if fuzzy and fuzzy[0]["name"] != disease_name:
                result = self.query(cypher, {"name": fuzzy[0]["name"]})
                symptoms = [r["symptom"] for r in result if r.get("symptom")]
        return symptoms

    def query_disease_drugs(self, disease_name: str) -> List[str]:
        cypher = """
        MATCH (d:Disease {name: $name})-[:DISEASE_DRUG]->(drug:Drug)
        RETURN drug.name AS drug
        """
        result = self.query(cypher, {"name": disease_name})
        return [r["drug"] for r in result if r.get("drug")]

    def query_disease_treatments(self, disease_name: str) -> List[str]:
        cypher = """
        MATCH (d:Disease {name: $name})-[:TREATED_WITH]->(t:Treatment)
        RETURN t.name AS treatment
        """
        result = self.query(cypher, {"name": disease_name})
        return [r["treatment"] for r in result if r.get("treatment")]

    def get_entity_info(self, entity_name: str) -> str:
        result = self.query(f"MATCH (n) WHERE n.name = '{entity_name}' RETURN n LIMIT 1")
        if result and len(result) > 0:
            node = result[0]['n']
            properties = dict(node)
            return ", ".join([f"{k}: {v}" for k, v in properties.items() if v and k != 'name'])
        return ""

    def search_entities(self, keyword: str, limit: int = 5) -> List[Dict]:
        cypher = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        RETURN n.name AS name, labels(n) AS labels
        LIMIT $limit
        """
        return self.query(cypher, {"keyword": keyword, "limit": limit})

    def query_relationship(self, entity1: str, entity2: str) -> str:
        cypher = """
        MATCH (a {name: $e1})-[r]->(b {name: $e2})
        RETURN type(r) AS rel
        LIMIT 1
        """
        result = self.query(cypher, {"e1": entity1, "e2": entity2})
        if result:
            return result[0].get("rel", "")
        return ""

    def query_by_template(self, template_key: str, entity: str) -> str:
        templates = {
            "symptom": "MATCH (d:Disease {name: $entity})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS name",
            "drug": "MATCH (d:Disease {name: $entity})-[:DISEASE_DRUG]->(drug:Drug) RETURN drug.name AS name",
            "treatment": "MATCH (d:Disease {name: $entity})-[:TREATED_WITH]->(t:Treatment) RETURN t.name AS name",
            "check": "MATCH (d:Disease {name: $entity})-[:NEEDS_CHECK]->(c:Checkup) RETURN c.name AS name",
            "department": "MATCH (d:Disease {name: $entity})-[:BELONGS_TO]->(dep:Department) RETURN dep.name AS name",
            "prevent": "MATCH (d:Disease {name: $entity})-[:PREVENTED_BY]->(p:Prevention) RETURN p.name AS name",
            "cause": "MATCH (d:Disease {name: $entity})-[:CAUSED_BY]->(c:Cause) RETURN c.name AS name",
            "disease": "MATCH (d:Disease {name: $entity}) RETURN d.desc AS name",
            "complication": "MATCH (d:Disease {name: $entity})-[:HAS_COMPLICATION]->(c:Complication) RETURN c.name AS name",
        }
        if template_key not in templates:
            return ""
        cypher = templates[template_key]
        result = self.query(cypher, {"entity": entity})
        items = [r["name"] for r in result if r.get("name")]
        if not items:
            return ""
        if template_key == "symptom":
            return f"【{entity}】的症状包括：{'、'.join(items)}。"
        elif template_key == "drug":
            return f"【{entity}】的常用药物包括：{'、'.join(items)}。"
        elif template_key == "treatment":
            return f"【{entity}】的治疗方法包括：{'、'.join(items)}。"
        elif template_key == "check":
            return f"【{entity}】的相关检查包括：{'、'.join(items)}。"
        elif template_key == "department":
            return f"【{entity}】建议就诊科室：{'、'.join(items)}。"
        elif template_key == "prevent":
            return f"【{entity}】的预防措施包括：{'、'.join(items)}。"
        elif template_key == "cause":
            return f"【{entity}】的病因包括：{'、'.join(items)}。"
        elif template_key == "disease":
            return f"【{entity}】：{items[0]}"
        elif template_key == "complication":
            return f"【{entity}】的并发症包括：{'、'.join(items)}。"
        return ""


_TRIE_END = "__end__"


def _load_entity_dict():
    global _entity_dict, _entity_trie, _entity_dict_loading, _entity_dict_load_failed
    if _entity_trie is not None:
        return
    if _entity_dict_load_failed:
        return
    _entity_dict_loading = True
    try:
        graph = get_medical_graph()
        diseases = graph.query("MATCH (n:Disease) RETURN n.name AS name LIMIT 200")
        symptoms = graph.query("MATCH (n:Symptom) RETURN n.name AS name LIMIT 200")
        drugs = graph.query("MATCH (n:Drug) RETURN n.name AS name LIMIT 100")
        for d in diseases:
            name = d["name"].strip()
            if name:
                _entity_dict.add(name)
        for s in symptoms:
            name = s["name"].strip()
            if name:
                _entity_dict.add(name)
        for dr in drugs:
            name = dr["name"].strip()
            if name:
                _entity_dict.add(name)
        _entity_trie = {}
        for word in _entity_dict:
            node = _entity_trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node[_TRIE_END] = True
        logger.info(f"Entity dict loaded: {len(_entity_dict)} entities")
    except Exception as e:
        logger.error(f"Failed to load entity dict: {e}")
        _entity_dict_load_failed = True
    finally:
        _entity_dict_loading = False


# ========== FIX 7: 改进模板快速路径，增加调试和兜底 ==========
def _try_template_fast_path(graph, query: str) -> str | None:
    if _entity_trie is None:
        logger.warning("Trie not loaded in fast path")
        return None

    entities = _extract_entities_fast(query)
    if not entities:
        logger.info(f"Fast extraction miss for '{query}', trying LLM NER fallback")
        entities = _llm_ner_extract(query)
        if not entities:
            logger.warning(f"No entities for: {query}")
            return None

    q = query.lower()
    intent_configs = [
        ("symptom", ["症状", "表现", "征象", "有什么感觉", "不舒服", "不适", "难受", "异常"]),
        ("drug", ["药", "药物", "用药", "吃什么药", "用什么药", "治疗药物", "药品", "处方"]),
        ("treatment", ["治疗", "怎么治", "怎么办", "如何治", "治疗方案", "疗法", "处理", "医治"]),
        ("check", ["检查", "做什么检查", "怎么检查", "诊断", "确诊", "化验", "拍片", "B超", "CT", "核磁"]),
        ("department", ["科室", "挂什么科", "哪个科", "去什么科", "看什么科", "门诊", "部门"]),
        ("prevent", ["预防", "怎么预防", "如何避免", "防止", "防范", "阻止"]),
        ("cause", ["病因", "原因", "怎么引起", "为什么会", "因为什么", "由于什么", "导致", "引发"]),
        ("disease", ["是什么", "定义", "简介", "介绍", "什么叫", "什么是", "概述"]),
        ("complication", ["并发症", "并发", "合并症", "合并", "会引起什么", "导致什么", "继发"]),
    ]

    for template_key, keywords in intent_configs:
        if any(kw in q for kw in keywords):
            for entity in entities[:2]:
                result = graph.query_by_template(template_key, entity)
                if result and "暂无记录" not in result and "未找到" not in result:
                    logger.info(f"Fast path OK: {template_key}/{entity}")
                    return result
                
                fuzzy = graph.search_entities(entity, limit=1)
                if fuzzy and fuzzy[0]["name"] != entity:
                    result = graph.query_by_template(template_key, fuzzy[0]["name"])
                    if result and "暂无记录" not in result and "未找到" not in result:
                        logger.info(f"Fast path OK (fuzzy): {template_key}/{fuzzy[0]['name']}")
                        return result
    
    logger.info(f"Fast path miss: {query}")
    return None


def _format_symptom_response(disease: str, symptoms: list) -> str:
    if not symptoms:
        return f"知识图谱中暂无【{disease}】的症状信息。"
    return f"【{disease}】的症状包括：{'、'.join(symptoms)}。"


def _format_drug_response(disease: str, drugs: list) -> str:
    if not drugs:
        return f"知识图谱中暂无【{disease}】的药物信息。"
    return f"【{disease}】的常用药物包括：{'、'.join(drugs)}。"


def _format_treatment_response(disease: str, treatments: list) -> str:
    if not treatments:
        return f"知识图谱中暂无【{disease}】的治疗方法。"
    return f"【{disease}】的治疗方法包括：{'、'.join(treatments)}。"


# ========== FIX 8: 改进 KG 上下文构建，按意图查关系而非仅节点属性 ==========
def _build_kg_context(graph, entities, query):
    context_parts = []
    q = query.lower()
    
    is_symptom = any(kw in q for kw in ["症状", "表现", "征象", "不适", "感觉", "难受", "异常"])
    is_drug = any(kw in q for kw in ["药", "药物", "用药", "处方"])
    is_treatment = any(kw in q for kw in ["治疗", "怎么治", "疗法", "方案", "处理", "医治"])
    is_cause = any(kw in q for kw in ["病因", "原因", "引起", "导致", "为什么"])
    is_prevent = any(kw in q for kw in ["预防", "防止", "避免", "防范"])
    is_check = any(kw in q for kw in ["检查", "诊断", "确诊", "化验", "拍片", "B超", "CT", "核磁"])
    is_department = any(kw in q for kw in ["科室", "挂什么科", "哪个科", "门诊", "部门"])
    is_complication = any(kw in q for kw in ["并发症", "并发", "合并", "继发", "引起"])
    
    for entity in entities[:2]:
        if is_symptom:
            items = graph.query_disease_symptoms(entity)
            label = "的症状包括"
        elif is_drug:
            items = graph.query_disease_drugs(entity)
            label = "的常用药物包括"
        elif is_treatment:
            items = graph.query_disease_treatments(entity)
            label = "的治疗方法包括"
        elif is_cause:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = "的病因/原因"
        elif is_prevent:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = "的预防方法"
        elif is_check:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = "的相关检查"
        elif is_department:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = "建议就诊科室"
        elif is_complication:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = "的并发症"
        else:
            info = graph.get_entity_info(entity)
            items = [info] if info else []
            label = ""
        
        if items:
            if label:
                context_parts.append(f"【{entity}】{label}：{', '.join(items)}。")
            else:
                context_parts.append(f"【{entity}】：{items[0]}")
        else:
            context_parts.append(f"【{entity}】：知识图谱中暂无相关信息。")
    
    return "\n".join(context_parts)


# ========== FIX 9: 重写 kg_query_func，严格防幻觉 ==========
@tool
def kg_query_func(query: str) -> str:
    """用于回答疾病、症状、药物之间的医学关联关系，基于医疗知识图谱"""
    graph = get_medical_graph()

    # 1. 模板快速路径
    fast_result = _try_template_fast_path(graph, query)
    if fast_result:
        return fast_result

    # 2. 实体提取
    entities = _extract_entities(query)
    
    # FIX: 兜底提取
    if not entities:
        chinese_segments = re.findall(r'[\u4e00-\u9fff]{2,10}', query)
        for seg in sorted(set(chinese_segments), key=len, reverse=True):
            fuzzy = graph.search_entities(seg, limit=1)
            if fuzzy:
                entities = [fuzzy[0]["name"]]
                logger.info(f"Fallback entity: '{seg}' -> '{entities[0]}'")
                break
    
    if not entities:
        return "未识别到医疗实体，无法查询知识图谱。请尝试使用更具体的疾病名称，如'鼻炎'、'高血压'等。"

    q = query.lower()
    has_symptom = any(kw in q for kw in ["症状", "表现", "征象", "有什么感觉", "不舒服", "不适", "难受", "异常"])
    has_medicine = any(kw in q for kw in ["药", "药物", "用药", "吃什么药", "用什么药", "治疗药物", "药品", "处方"])
    has_treatment = any(kw in q for kw in ["治疗", "怎么治", "怎么办", "如何治", "治疗方案", "疗法", "处理", "医治"])
    has_check = any(kw in q for kw in ["检查", "做什么检查", "怎么检查", "诊断", "确诊", "化验", "拍片", "B超", "CT", "核磁"])
    has_department = any(kw in q for kw in ["科室", "挂什么科", "哪个科", "去什么科", "看什么科", "门诊", "部门"])
    has_prevent = any(kw in q for kw in ["预防", "怎么预防", "如何避免", "防止", "防范", "阻止"])
    has_cause = any(kw in q for kw in ["病因", "原因", "怎么引起", "为什么会", "因为什么", "由于什么", "导致", "引发"])
    has_define = any(kw in q for kw in ["是什么", "定义", "简介", "介绍", "什么叫", "什么是", "概述"])
    has_complication = any(kw in q for kw in ["并发症", "并发", "合并症", "合并", "会引起什么", "导致什么", "继发"])

    disease_entities = [e for e in entities if _is_disease(e)]
    symptom_entities = [e for e in entities if _is_symptom(e)]
    medicine_entities = [e for e in entities if _is_medicine(e)]

    # 症状查询
    if has_symptom and disease_entities:
        all_symptoms = []
        for ent in disease_entities:
            symptoms = graph.query_disease_symptoms(ent)
            if symptoms:
                all_symptoms.extend(symptoms)
        if all_symptoms:
            return _format_symptom_response(disease_entities[0], all_symptoms)
        return f"知识图谱中暂无【{disease_entities[0]}】的症状信息。"

    # 药物查询
    if has_medicine and disease_entities:
        all_drugs = []
        for ent in disease_entities:
            drugs = graph.query_disease_drugs(ent)
            if drugs:
                all_drugs.extend(drugs)
        if all_drugs:
            return _format_drug_response(disease_entities[0], all_drugs)
        return f"知识图谱中暂无【{disease_entities[0]}】的药物信息。"

    # 治疗查询
    if has_treatment and disease_entities:
        all_treatments = []
        for ent in disease_entities:
            treatments = graph.query_disease_treatments(ent)
            if treatments:
                all_treatments.extend(treatments)
        if all_treatments:
            return _format_treatment_response(disease_entities[0], all_treatments)
        return f"知识图谱中暂无【{disease_entities[0]}】的治疗方法。"

    # 定义查询
    if has_define and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的详细介绍。"

    # 病因查询
    if has_cause and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】的病因/原因：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的病因信息。"

    # 预防查询
    if has_prevent and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】的预防方法：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的预防信息。"

    # 检查查询
    if has_check and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】的相关检查：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的检查信息。"

    # 科室查询
    if has_department and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】建议就诊科室：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的科室信息。"

    # 并发症查询
    if has_complication and disease_entities:
        info = graph.get_entity_info(disease_entities[0])
        if info:
            return f"【{disease_entities[0]}】的并发症：{info}"
        return f"知识图谱中暂无【{disease_entities[0]}】的并发症信息。"

    # 关系查询
    if disease_entities and symptom_entities:
        relations = []
        for d in disease_entities:
            for s in symptom_entities:
                rel = graph.query_relationship(d, s)
                if rel:
                    relations.append(f"【{d}】与【{s}】的关系：{rel}")
        if relations:
            return "\n".join(relations)

    # ========== FIX 10: LLM 兜底严格约束，禁止幻觉 ==========
    kg_context = _build_kg_context(graph, entities, query)
    
    # 如果图谱上下文为空，直接拒绝，不走 LLM
    if not kg_context.strip():
        return f"知识图谱中未找到与【{'、'.join(entities)}】相关的信息。请尝试使用更标准的医学术语提问。"
    
    prompt = PromptTemplate.from_template("""\
你是一位严谨的医疗知识图谱问答助手。请**严格根据**下方提供的医疗知识图谱信息回答用户问题。
如果图谱信息不足以回答问题，请明确回答"根据当前知识图谱，暂无该问题的相关信息"，**绝对不要**使用你自己的医学知识补充、猜测或编造。

图谱信息：
{kg_context}

用户问题：{query}

要求：
1. 只基于上述图谱信息回答，100字以内。
2. 如果图谱信息为空或不相关，必须回答"根据当前知识图谱，暂无该问题的相关信息"。
3. 禁止提及图谱中未出现的任何疾病、药物或症状。
""")
    llm = get_llm()
    chain = prompt | llm
    return chain.invoke({"kg_context": kg_context, "query": query})


@tool
def search_func(query: str) -> str:
    """通过搜索引擎回答通用类问题"""
    return f"搜索工具暂未实现。用户查询：{query}"


@tool
def retrieval_func(query: str) -> str:
    """基于企业内部文档回答寻医问药网相关问题"""
    return f"文献检索工具暂未实现。用户查询：{query}"


@tool
def generic_chat_func(query: str) -> str:
    """解答通用领域的知识，例如打招呼等问题"""
    return "您好，我是医疗知识图谱助手，可以帮您查询疾病、症状、药物等医学信息。请提出具体的医学问题。"