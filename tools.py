"""Agent 工具函数"""

import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import List, Dict

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from models import get_llm_model
from vector_store import similarity_search
from config import GENERIC_PROMPT_TPL, RETRIEVAL_PROMPT_TPL, SEARCH_PROMPT_TPL
from neo4j_store import get_medical_graph

# ==================== 实体词典 ===================

_entity_dict: set[str] = set()
_entity_trie = None
_ENTITY_CACHE_FILE = "./data/entity_dict.json"
_entity_dict_loading = False
_entity_dict_load_failed = False


class _TrieNode:
    __slots__ = ['children', 'is_end', 'word']
    def __init__(self):
        self.children: Dict[str, '_TrieNode'] = {}
        self.is_end = False
        self.word = None


class _EntityTrie:
    def __init__(self):
        self.root = _TrieNode()
    def add(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.word = word
    def search(self, text: str) -> List[str]:
        found = []
        i = 0
        n = len(text)
        while i < n:
            node = self.root
            j = i
            last_match = None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    last_match = node.word
                j += 1
            if last_match:
                found.append(last_match)
                i += len(last_match)
            else:
                i += 1
        return found


def _load_entity_dict():
    global _entity_dict, _entity_trie, _entity_dict_loading, _entity_dict_load_failed
    if _entity_trie is not None:
        return
    if _entity_dict_load_failed:
        return
    if _entity_dict_loading:
        return

    _entity_dict_loading = True
    try:
        if os.path.exists(_ENTITY_CACHE_FILE):
            try:
                with open(_ENTITY_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _entity_dict = set(data)
                    _entity_trie = _EntityTrie()
                    for w in _entity_dict:
                        _entity_trie.add(w)
                    print(f"[KG] Loaded {len(_entity_dict)} entities from local cache (Trie)")
                    return
            except Exception as e:
                print(f"[KG] Local cache read failed: {e}")

        try:
            graph = get_medical_graph()
            cypher = """
                MATCH (n:Disease|Symptom|Drug) 
                WHERE n.name IS NOT NULL 
                RETURN DISTINCT n.name as name 
                LIMIT 5000
            """
            with graph.driver.session() as session:
                result = session.run(cypher)
                _entity_dict = {r["name"] for r in result if r["name"]}
                _entity_trie = _EntityTrie()
                for w in _entity_dict:
                    _entity_trie.add(w)
                print(f"[KG] Loaded {len(_entity_dict)} entities from Neo4j (Trie)")
                os.makedirs(os.path.dirname(_ENTITY_CACHE_FILE), exist_ok=True)
                with open(_ENTITY_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(_entity_dict), f, ensure_ascii=False)
        except Exception as e:
            print(f"[KG] Neo4j entity load failed (will not retry): {e}")
            _entity_dict_load_failed = True
    finally:
        _entity_dict_loading = False


# ==================== 工具 ====================

@tool
def generic_func(query: str) -> str:
    """可以解答通用领域的知识，例如打招呼，问你是谁等问题"""
    prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
    chain = prompt | get_llm_model() | StrOutputParser()
    return chain.invoke({"query": query})


@tool
def retrieval_func(query: str) -> str:
    """用于回答寻医问药网相关问题，基于企业内部文档"""
    contexts = similarity_search(query, k=3, score_threshold=0.6)
    truncated = []
    for ctx in contexts:
        if len(ctx) > 200:
            ctx = ctx[:200] + "..."
        truncated.append(ctx)
    prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT_TPL)
    chain = prompt | get_llm_model() | StrOutputParser()
    return chain.invoke({
        "query": query,
        "context": "\n\n".join(truncated) if truncated else "没有查到",
    })


# 搜索失败时由大模型自身知识兜底
@tool
def search_func(query: str) -> str:
    """通过搜索引擎回答通用类问题（非医疗问题）。搜索失败时由大模型自身知识回答。"""
    search_results = baidu_search(query, num_results=5, timeout=3.0)

    if search_results:
        formatted_results = "\n\n".join([
            f"标题：{r['title']}\n摘要：{r['abstract']}"
            for r in search_results[:3]
        ])
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        chain = prompt | get_llm_model() | StrOutputParser()
        return chain.invoke({
            "query": query,
            "query_result": formatted_results,
        })

    # 搜索无结果：直接让 LLM 回答（非医疗问题的最终兜底）
    print(f"[SEARCH] No results for: {query}, falling back to LLM")
    fallback_prompt = """请用中文简洁回答用户问题。如果问题涉及医疗诊断，请回复"抱歉，我无法提供医疗诊断建议，请咨询专业医生。"。
用户问题：{query}
"""
    prompt = PromptTemplate.from_template(fallback_prompt)
    chain = prompt | get_llm_model() | StrOutputParser()
    return chain.invoke({"query": query})


# ==================== 快速格式化（零 LLM）====================

def _format_drug_response(disease_name: str, drugs: List[Dict], drug_entities: List[str]) -> str:
    parts = []
    if drugs:
        drug_names = [d["name"] for d in drugs[:5]]
        parts.append(f"根据图谱，{disease_name}可用：{', '.join(drug_names)}。")
    else:
        parts.append(f"图谱中未记录{disease_name}的相关药物。")

    graph = get_medical_graph()
    for drug_name in drug_entities:
        diseases = graph.query_drug_diseases(drug_name)
        if diseases:
            disease_names = [d["name"] for d in diseases[:3]]
            parts.append(f"{drug_name}用于治疗：{', '.join(disease_names)}。")
        else:
            parts.append(f"图谱中未记录{drug_name}与{disease_name}的治疗关系。")

    parts.append("以上信息仅供参考，具体用药请遵医嘱。")
    return "".join(parts)


def _format_symptom_response(disease_name: str, symptoms: List[str]) -> str:
    if symptoms:
        return f"{disease_name}的症状包括：{', '.join(symptoms[:5])}。"
    return f"图谱中未记录{disease_name}的详细症状。"


# ==================== 实体抽取 ====================

def _llm_ner_extract(query: str) -> list[str]:
    """LLM 兜底实体抽取：仅当 Trie 未命中时调用一次"""
    prompt_text = f"""从以下用户输入中抽取医疗实体，仅输出 JSON 格式，不要任何其他内容。
JSON 格式要求：{{"disease": ["疾病名"], "symptom": ["症状名"], "drug": ["药品名"]}}
如果没有任何医疗实体，三个字段全部留空列表 []。

用户输入：{query}

输出："""

    try:
        llm = get_llm_model()
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{{.*?\}}", content, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        entities = []
        for key in ("disease", "symptom", "drug"):
            entities.extend(data.get(key, []))
        cleaned = [e.strip() for e in entities if isinstance(e, str) and len(e) > 1 and not e.isdigit()]
        return cleaned
    except Exception as e:
        print(f"[NER] LLM 兜底抽取失败: {e}")
        return []


def _extract_entities_fast(query: str) -> list[str]:
    """【Fast 档】仅 Trie 匹配，不触发 LLM"""
    if _entity_trie is None:
        return []
    found = _entity_trie.search(query)
    seen = set()
    unique = []
    for e in found:
        if e not in seen:
            seen.add(e)
            unique.append(e)
            if len(unique) >= 3:
                break
    return unique


def _extract_entities(query: str) -> list[str]:
    """【Full 档】Trie 匹配 + LLM 兜底"""
    _load_entity_dict()
    entities = _extract_entities_fast(query)
    if entities:
        return entities
    print(f"[NER] Trie 未命中，启动 LLM 兜底: {query}")
    return _llm_ner_extract(query)


# ==================== 模板快速路径 ====================

# FIX: 过滤掉返回"暂无记录"的伪命中
def _try_template_fast_path(graph, query: str) -> str | None:
    """模板快速路径：Trie 实体 + 意图关键词，零 LLM"""
    if _entity_trie is None:
        return None

    entities = _extract_entities_fast(query)
    if not entities:
        return None

    q = query.lower()
    intent_configs = [
        ("symptom", ["症状", "表现", "征象", "有什么感觉", "不舒服"]),
        ("cure_way", ["药", "治疗", "怎么治", "吃什么", "服用", "疗法", "能治"]),
        ("cause", ["引起", "病因", "原因", "为什么", "导致的"]),
        ("desc", ["是什么", "什么叫", "什么是", "的定义", "是一种什么"]),
        ("check", ["检查", "化验", "拍片", "做哪些", "查什么"]),
        ("department", ["科室", "挂什么科", "看什么科", "哪个科", "门诊"]),
        ("cured_prob", ["治好", "治愈率", "预后", "几率", "能好吗"]),
        ("indications", ["治什么病", "适应症", "治哪些", "能治", "可以吃", "能吃", "管用"]),
    ]

    for template_key, keywords in intent_configs:
        if any(kw in q for kw in keywords):
            for entity in entities[:2]:
                result = graph.query_by_template(template_key, entity)
                # FIX: 过滤空结果的伪命中
                if result and "暂无记录" not in result:
                    print(f"[KG] 模板精确命中: {template_key}, 实体: {entity}")
                    return result
                result = graph.query_by_template(template_key, entity, fuzzy=True)
                if result and "暂无记录" not in result:
                    print(f"[KG] 模板模糊命中: {template_key}, 实体: {entity}")
                    return result
    return None


# ==================== 知识图谱工具（修复版）====================

@tool
def kg_query_func(query: str) -> str:
    """用于回答疾病、症状、药物之间的医学关联关系，基于医疗知识图谱"""
    graph = get_medical_graph()

    # 1. 模板快速路径（零 LLM）
    fast_result = _try_template_fast_path(graph, query)
    if fast_result:
        return fast_result

    # 2. 实体抽取（Trie + LLM 兜底）
    entities = _extract_entities(query)
    if not entities:
        return "未识别到医疗实体，无法查询知识图谱。"

    # 3. 区分疾病和药物
    disease_entities = []
    drug_entities = []
    for ent in entities:
        try:
            symptoms = graph.query_disease_symptoms(ent)
            if symptoms:
                disease_entities.append(ent)
                continue
        except Exception:
            pass
        try:
            diseases = graph.query_drug_diseases(ent)
            if diseases:
                drug_entities.append(ent)
                continue
        except Exception:
            pass
        # 无法判断，默认当疾病
        disease_entities.append(ent)

    # 4. 判断意图
    q = query.lower()
    has_drug_intent = any(k in q for k in ["药", "吃什么", "用药", "治疗", "能吃", "服用", "可以吃"])
    has_symptom_intent = any(k in q for k in ["症状", "表现", "征象", "有什么"])

    # 当同时有疾病和药品，且问"能否/可以吃"时，查药品适应症并交叉验证
    if disease_entities and drug_entities and any(k in q for k in ["可以吃", "能吃", "能用", "管用", "适合"]):
        drug_name = drug_entities[0]
        disease_name = disease_entities[0]
        indications = graph.query_by_template("indications", drug_name)
        if indications:
            if disease_name in indications:
                return f"【{disease_name}】可以吃{drug_name}。{indications} 具体用药请遵医嘱。"
            else:
                # FIX: 不列出所有无关适应症，只简短说明
                return f"【{disease_name}】图谱未明确记录{drug_name}可用于治疗{disease_name}。具体用药请遵医嘱。"
        else:
            return f"图谱中未记录{drug_name}的适应症信息，无法确认是否适用于{disease_name}。具体用药请遵医嘱。"

    # 快速路径 — 明确用药查询，支持"只有疾病"或"只有药品"
    if has_drug_intent:
        if disease_entities:
            all_drugs = []
            for ent in disease_entities:
                drugs = graph.query_disease_drugs(ent)
                if drugs:
                    all_drugs.extend(drugs)
            if all_drugs or drug_entities:
                print("[KG] 快速路径: 疾病药物查询，跳过 LLM")
                return _format_drug_response(disease_entities[0], all_drugs, drug_entities)
        elif drug_entities:
            # 只有药品实体，查适应症
            indications_results = []
            for ent in drug_entities:
                ind = graph.query_by_template("indications", ent)
                if ind and "暂无记录" not in ind:
                    indications_results.append(ind)
            if indications_results:
                print("[KG] 快速路径: 药品适应症查询，跳过 LLM")
                return "\n".join(indications_results)

    if has_symptom_intent and disease_entities:
        all_symptoms = []
        for ent in disease_entities:
            symptoms = graph.query_disease_symptoms(ent)
            if symptoms:
                all_symptoms.extend(symptoms)
        if all_symptoms:
            print("[KG] 快速路径: 症状查询，跳过 LLM")
            return _format_symptom_response(disease_entities[0], all_symptoms)

    # 5. 复杂查询：收集上下文，LLM 整理兜底
    contexts = []
    max_context_items = 6

    for ent in entities[:2]:
        try:
            subgraphs = graph.get_subgraph(ent, depth=1, limit=2)
            if subgraphs:
                for sg in subgraphs:
                    rels = sg.get("rels", [])
                    for rel in rels[:1]:
                        contexts.append(f"{rel['from']} -> {rel['type']} -> {rel['to']}")
                        if len(contexts) >= max_context_items:
                            break
                    if len(contexts) >= max_context_items:
                        break
        except Exception as e:
            print(f"[KG] Subgraph query failed for {ent}: {e}")

    if has_drug_intent:
        for ent in disease_entities:
            try:
                drugs = graph.query_disease_drugs(ent)
                if drugs:
                    drug_str = ", ".join([d["name"] for d in drugs[:3]])
                    contexts.append(f"【{ent}】相关药物：{drug_str}")
            except Exception as e:
                print(f"[KG] Drug query failed for {ent}: {e}")
        for ent in drug_entities:
            try:
                diseases = graph.query_drug_diseases(ent)
                if diseases:
                    dis_str = ", ".join([d["name"] for d in diseases[:3]])
                    contexts.append(f"【{ent}】用于治疗：{dis_str}")
            except Exception as e:
                print(f"[KG] Drug-disease query failed for {ent}: {e}")

    if has_symptom_intent:
        for ent in disease_entities:
            try:
                symptoms = graph.query_disease_symptoms(ent)
                if symptoms:
                    contexts.append(f"【{ent}】症状：{', '.join(symptoms[:5])}")
            except Exception as e:
                print(f"[KG] Symptom query failed for {ent}: {e}")

    if not contexts:
        return "知识图谱中未找到相关信息。"

    # 6. LLM 整理（兜底）
    prompt = PromptTemplate.from_template("""\
请根据以下医疗知识图谱信息，用简洁语言回答用户问题（100字以内）。不要编造。

图谱信息：
{kg_context}

用户问题：{query}
""")
    chain = prompt | get_llm_model() | StrOutputParser()
    try:
        return chain.invoke({
            "query": query,
            "kg_context": "\n".join(contexts[:6]),
        })
    except Exception as e:
        print(f"[KG] LLM 整理失败: {e}")
        return "图谱信息：" + "；".join(contexts[:3])


# ==================== 搜索 ====================

# FIX: 超时放宽到 3 秒
def baidu_search(query: str, num_results: int = 10, timeout: float = 3.0) -> list[dict]:
    """百度搜索（网页抓取）"""
    try:
        search_queries = [query, f"{query} 官方"]
        all_results = []
        for sq in search_queries[:2]:
            url = f"https://www.baidu.com/s?wd={quote(sq)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Referer": "https://www.baidu.com/",
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.encoding = 'utf-8'
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            selectors = ['.result', '.c-container', '[tpl]', '.c-result']
            containers = []
            for selector in selectors:
                containers = soup.select(selector)
                if containers:
                    break
            for container in containers:
                title_elem = (
                    container.select_one('h3 a') or
                    container.select_one('.t a') or
                    container.select_one('a[data-click]') or
                    container.select_one('a')
                )
                abstract_elem = (
                    container.select_one('.content-right_8Zs40') or
                    container.select_one('.c-abstract') or
                    container.select_one('.content-right') or
                    container.select_one('p')
                )
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                    if len(title) > 5 and (len(abstract) > 20 or '电话' in title or '客服' in title):
                        result = {"title": title, "abstract": abstract}
                        if not any(r['title'] == title for r in all_results):
                            all_results.append(result)
                            if len(all_results) >= num_results:
                                break
            if len(all_results) >= num_results:
                break
        print(f"[SEARCH] Baidu found {len(all_results)} results for: {query}")
        return all_results
    except requests.exceptions.Timeout:
        print(f"[SEARCH] Baidu timeout (> {timeout}s), skipping...")
        return []
    except Exception as e:
        print(f"[SEARCH] Baidu error: {e}")
        return []


_search_cache: Dict[str, tuple] = {}
_CACHE_TTL = 300


def cached_search(query: str, search_func_name: str = "baidu") -> list[dict]:
    """带缓存 + TTL 的搜索"""
    cache_key = f"{search_func_name}:{query}"
    now = time.time()
    if cache_key in _search_cache:
        result, ts = _search_cache[cache_key]
        if now - ts < _CACHE_TTL:
            print(f"[CACHE] Search hit: {cache_key}")
            return result
    if search_func_name == "baidu":
        results = baidu_search(query)
    else:
        results = []
    _search_cache[cache_key] = (results, now)
    return results


# ==================== 启动预加载 ====================
try:
    _load_entity_dict()
except Exception as e:
    print(f"[KG] Startup entity preload skipped: {e}")