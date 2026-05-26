"""Agent 工具函数"""

import os
import json
import time
import re
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import List, Dict

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from models import get_llm_model
from vector_store import similarity_search
from config import GENERIC_PROMPT_TPL, RETRIEVAL_PROMPT_TPL, SEARCH_PROMPT_TPL, SEARCH_CONFIG
from neo4j_store import get_medical_graph
from utils import get_logger

logger = get_logger("medagent.tools")

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
                _essential_entities = {"鼻炎", "高血压", "感冒", "糖尿病"}
                if _essential_entities.issubset(_entity_dict):
                    _entity_trie = _EntityTrie()
                    for w in _entity_dict:
                        _entity_trie.add(w)
                    logger.info(f"Loaded {len(_entity_dict)} entities from local cache (Trie)")
                    return
                else:
                    missing = _essential_entities - _entity_dict
                    logger.info(f"Cache incomplete (missing {missing}), refreshing from Neo4j...")
            except Exception as e:
                logger.warning(f"Local cache read failed: {e}")

        try:
            graph = get_medical_graph()
            cypher = """
            MATCH (n:Disease|Symptom|Drug)
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name as name
            ORDER BY n.name
            """
            with graph.driver.session() as session:
                result = session.run(cypher)
                _entity_dict = {r["name"] for r in result if r["name"]}
                _entity_trie = _EntityTrie()
                for w in _entity_dict:
                    _entity_trie.add(w)
                logger.info(f"Loaded {len(_entity_dict)} entities from Neo4j (Trie)")
                sample = sorted(list(_entity_dict))[:15]
                logger.debug(f"Sample entities: {sample}")
                os.makedirs(os.path.dirname(_ENTITY_CACHE_FILE), exist_ok=True)
                with open(_ENTITY_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(_entity_dict), f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Neo4j entity load failed (will not retry): {e}")
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
    # FIX: 降低阈值到 0.3，增加 k 到 5
    contexts = similarity_search(query, k=5, score_threshold=0.3)

    if not contexts:
        return "抱歉，暂时没有找到相关信息。"

    truncated = []
    for ctx in contexts:
        if len(ctx) > 200:
            ctx = ctx[:200] + "..."
        truncated.append(ctx)

    prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT_TPL)
    chain = prompt | get_llm_model() | StrOutputParser()
    return chain.invoke({
        "query": query,
        "context": "\n\n".join(truncated),
    })

@tool
def search_func(query: str) -> str:
    """通过搜索引擎回答通用类问题（非医疗问题）。搜索失败时由大模型自身知识回答。"""
    search_results = cached_search(query)

    if search_results:
        formatted_results = "\n\n".join([
            f"[{r.get('source', 'web').upper()}] {r['title']}\n{r['abstract']}"
            for r in search_results[:3]
        ])
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        chain = prompt | get_llm_model() | StrOutputParser()
        return chain.invoke({
            "query": query,
            "query_result": formatted_results,
        })

    logger.info(f"No search results for: {query}, falling back to LLM")
    fallback_prompt = """请用中文简洁回答用户问题。如果问题涉及医疗诊断，请回复"抱歉，我无法提供医疗诊断建议，请咨询专业医生。"。
用户问题：{query}
"""
    prompt = PromptTemplate.from_template(fallback_prompt)
    chain = prompt | get_llm_model() | StrOutputParser()
    return chain.invoke({"query": query})

# ==================== 快速格式化 ====================

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
    prompt_text = f"""从以下用户输入中抽取医疗实体，仅输出 JSON 格式，不要任何其他内容。
JSON 格式要求：{{"disease": ["疾病名"], "symptom": ["症状名"], "drug": ["药品名"]}}
如果没有任何医疗实体，三个字段全部留空列表 []。

用户输入：{query}

输出："""

    try:
        llm = get_llm_model()
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"LLM raw output: {content[:200]}")
        match = re.search(r"\{{.*?\}}", content, re.DOTALL)
        if not match:
            logger.debug("No JSON found in LLM output")
            return []
        data = json.loads(match.group())
        raw_entities = []
        for key in ("disease", "symptom", "drug"):
            raw_entities.extend(data.get(key, []))
        logger.debug(f"LLM extracted raw: {raw_entities}")
        return _map_to_graph_entities(raw_entities)
    except Exception as e:
        logger.warning(f"LLM NER fallback failed: {e}")
        return []

def _fuzzy_search_entity(name: str, limit: int = 5) -> List[str]:
    graph = get_medical_graph()
    results = graph.search_entities(name, limit=limit)
    candidates = []
    for r in results:
        candidate = r["name"]
        if name in candidate or candidate in name:
            candidates.append(candidate)
    return candidates

def _map_to_graph_entities(raw_entities: List[str]) -> List[str]:
    _load_entity_dict()
    mapped = []
    for raw in raw_entities:
        raw = raw.strip()
        if len(raw) <= 1 or raw.isdigit():
            continue

        if raw in _entity_dict:
            mapped.append(raw)
            continue

        fuzzy_matches = _fuzzy_search_entity(raw, limit=3)
        if fuzzy_matches:
            best = fuzzy_matches[0]
            logger.info(f"Entity alias mapping: '{raw}' -> '{best}'")
            mapped.append(best)
            continue

        logger.debug(f"Entity not in graph: '{raw}'")

    seen = set()
    unique = []
    for e in mapped:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique

def _extract_entities_fast(query: str) -> list[str]:
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

    if not unique:
        graph = get_medical_graph()
        segments = re.findall(r'[\u4e00-\u9fff]+', query)
        candidates = []
        for seg in segments:
            for length in range(min(10, len(seg)), 1, -1):
                for i in range(len(seg) - length + 1):
                    candidates.append(seg[i:i+length])

        seen_cands = set()
        for cand in candidates:
            if cand in seen_cands:
                continue
            seen_cands.add(cand)

            fuzzy_results = graph.search_entities(cand, limit=3)
            for r in fuzzy_results:
                name = r["name"]
                if name not in seen:
                    if cand in name or name in cand:
                        seen.add(name)
                        unique.append(name)
                        logger.info(f"Fast-path fuzzy match: '{cand}' -> '{name}'")
                        if len(unique) >= 3:
                            break
            if len(unique) >= 3:
                break

    return unique

def _extract_entities(query: str) -> list[str]:
    _load_entity_dict()
    entities = _extract_entities_fast(query)
    if entities:
        return entities
    logger.info(f"Trie miss, starting LLM NER fallback: {query}")
    return _llm_ner_extract(query)

# ==================== 模板快速路径 ====================

def _try_template_fast_path(graph, query: str) -> str | None:
    if _entity_trie is None:
        return None

    entities = _extract_entities_fast(query)
    if not entities:
        return None

    q = query.lower()
    intent_configs = [
        ("symptom", ["症状", "表现", "征象", "有什么感觉", "不舒服"]),
        ("cure_way", ["药", "吃什么", "服用", "药物", "吃药", "用药"]),
        ("cure_method", ["治疗", "怎么治", "疗法", "手术", "能治", "医治"]),
        ("cause", ["引起", "病因", "原因", "为什么", "导致的"]),
        ("desc", ["是什么", "什么叫", "什么是", "的定义", "是一种什么"]),
        ("check", ["检查", "化验", "拍片", "做哪些", "查什么"]),
        ("department", ["科室", "挂什么科", "看什么科", "哪个科", "门诊"]),
        ("cured_prob", ["治好", "治愈率", "预后", "几率", "能好吗"]),
        ("indications", ["治什么病", "适应症", "治哪些", "能吃", "管用", "适用"]),
        ("prevent", ["预防", "防止", "避免"]),
    ]

    for template_key, keywords in intent_configs:
        if any(kw in q for kw in keywords):
            for entity in entities[:2]:
                result = graph.query_by_template(template_key, entity)
                if result and "暂无记录" not in result:
                    logger.info(f"Template exact hit: {template_key}, entity: {entity}")
                    return result
                result = graph.query_by_template(template_key, entity, fuzzy=True)
                if result and "暂无记录" not in result:
                    logger.info(f"Template fuzzy hit: {template_key}, entity: {entity}")
                    return result
    return None

# ==================== 知识图谱工具 ====================

@tool
def kg_query_func(query: str) -> str:
    """用于回答疾病、症状、药物之间的医学关联关系，基于医疗知识图谱"""
    graph = get_medical_graph()

    fast_result = _try_template_fast_path(graph, query)
    if fast_result:
        return fast_result

    entities = _extract_entities(query)
    if not entities:
        return "未识别到医疗实体，无法查询知识图谱。"

    entity_types = graph.get_entity_types(entities)
    disease_entities = [e for e in entities if entity_types.get(e) == "Disease"]
    drug_entities = [e for e in entities if entity_types.get(e) == "Drug"]
    symptom_entities = [e for e in entities if entity_types.get(e) == "Symptom"]

    untyped = [e for e in entities if e not in entity_types]
    if untyped:
        logger.warning(f"Unrecognized entities (defaulting to Disease): {untyped}")
        disease_entities.extend(untyped)

    q = query.lower()
    has_drug_intent = any(k in q for k in ["药", "吃什么", "用药", "治疗", "能吃", "服用", "可以吃"])
    has_symptom_intent = any(k in q for k in ["症状", "表现", "征象", "有什么"])

    if disease_entities and drug_entities and any(k in q for k in ["可以吃", "能吃", "能用", "管用", "适合"]):
        drug_name = drug_entities[0]
        disease_name = disease_entities[0]
        indications = graph.query_by_template("indications", drug_name)
        if indications:
            if disease_name in indications:
                return f"【{disease_name}】可以吃{drug_name}。{indications} 具体用药请遵医嘱。"
            else:
                return f"【{disease_name}】图谱未明确记录{drug_name}可用于治疗{disease_name}。具体用药请遵医嘱。"
        else:
            return f"图谱中未记录{drug_name}的适应症信息，无法确认是否适用于{disease_name}。具体用药请遵医嘱。"

    if has_drug_intent:
        if disease_entities:
            all_drugs = []
            for ent in disease_entities:
                drugs = graph.query_disease_drugs(ent)
                if drugs:
                    all_drugs.extend(drugs)
            if all_drugs or drug_entities:
                logger.info("Fast path: disease drug query, skipping LLM")
                return _format_drug_response(disease_entities[0], all_drugs, drug_entities)
        elif drug_entities:
            indications_results = []
            for ent in drug_entities:
                ind = graph.query_by_template("indications", ent)
                if ind and "暂无记录" not in ind:
                    indications_results.append(ind)
            if indications_results:
                logger.info("Fast path: drug indications query, skipping LLM")
                return "\n".join(indications_results)

    if has_symptom_intent and disease_entities:
        all_symptoms = []
        for ent in disease_entities:
            symptoms = graph.query_disease_symptoms(ent)
            if symptoms:
                all_symptoms.extend(symptoms)
        if all_symptoms:
            logger.info("Fast path: symptom query, skipping LLM")
            return _format_symptom_response(disease_entities[0], all_symptoms)

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
            logger.warning(f"Subgraph query failed for {ent}: {e}")

    if has_drug_intent:
        for ent in disease_entities:
            try:
                drugs = graph.query_disease_drugs(ent)
                if drugs:
                    drug_str = ", ".join([d["name"] for d in drugs[:3]])
                    contexts.append(f"【{ent}】相关药物：{drug_str}")
            except Exception as e:
                logger.warning(f"Drug query failed for {ent}: {e}")
        for ent in drug_entities:
            try:
                diseases = graph.query_drug_diseases(ent)
                if diseases:
                    dis_str = ", ".join([d["name"] for d in diseases[:3]])
                    contexts.append(f"【{ent}】用于治疗：{dis_str}")
            except Exception as e:
                logger.warning(f"Drug-disease query failed for {ent}: {e}")

    if has_symptom_intent:
        for ent in disease_entities:
            try:
                symptoms = graph.query_disease_symptoms(ent)
                if symptoms:
                    contexts.append(f"【{ent}】症状：{', '.join(symptoms[:5])}")
            except Exception as e:
                logger.warning(f"Symptom query failed for {ent}: {e}")

    if not contexts:
        return "知识图谱中未找到相关信息。"

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
        logger.error(f"LLM summarization failed: {e}")
        return "图谱信息：" + "；".join(contexts[:3])

# ==================== 搜索模块 ====================

def _get_random_ua() -> str:
    return random.choice(SEARCH_CONFIG["user_agents"])

def _is_valid_result(title: str, abstract: str) -> bool:
    if len(title) < 3:
        return False
    junk_keywords = ["百度首页", "登录", "注册", "更多", "设置", "地图", "贴吧"]
    if any(kw in title for kw in junk_keywords):
        return False
    return len(abstract) > 10 or "电话" in title or "客服" in title

def _search_duckduckgo(query: str, num_results: int = 5, timeout: float = 5.0) -> List[Dict]:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            "User-Agent": _get_random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            logger.warning(f"DuckDuckGo status: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        for result in soup.select('.result'):
            title_elem = result.select_one('.result__a')
            abstract_elem = result.select_one('.result__snippet')

            if title_elem:
                title = title_elem.get_text(strip=True)
                abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                href = title_elem.get('href', '')

                if _is_valid_result(title, abstract):
                    results.append({
                        "title": title,
                        "abstract": abstract,
                        "url": href,
                        "source": "duckduckgo"
                    })
                    if len(results) >= num_results:
                        break

        logger.info(f"DuckDuckGo found {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo error: {e}")
        return []

def _search_bing(query: str, num_results: int = 5, timeout: float = 5.0) -> List[Dict]:
    try:
        url = f"https://www.bing.com/search?q={quote(query)}&count={num_results}"
        headers = {
            "User-Agent": _get_random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://www.bing.com/",
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            logger.warning(f"Bing status: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        for li in soup.select('.b_algo'):
            title_elem = li.select_one('h2 a')
            abstract_elem = li.select_one('.b_caption p') or li.select_one('.b_snippet')

            if title_elem:
                title = title_elem.get_text(strip=True)
                abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                href = title_elem.get('href', '')

                if _is_valid_result(title, abstract):
                    results.append({
                        "title": title,
                        "abstract": abstract,
                        "url": href,
                        "source": "bing"
                    })
                    if len(results) >= num_results:
                        break

        logger.info(f"Bing found {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.warning(f"Bing error: {e}")
        return []

def _search_baidu(query: str, num_results: int = 5, timeout: float = 5.0) -> List[Dict]:
    try:
        search_queries = [query, f"{query} 官方"]
        all_results = []

        for sq in search_queries[:2]:
            url = f"https://www.baidu.com/s?wd={quote(sq)}"
            headers = {
                "User-Agent": _get_random_ua(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Referer": "https://www.baidu.com/",
                "Connection": "keep-alive",
            }

            time.sleep(SEARCH_CONFIG.get("delay_between_requests", 0.5) * random.random())

            response = requests.get(url, headers=headers, timeout=timeout)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                logger.warning(f"Baidu status: {response.status_code}")
                continue

            if "verify" in response.text.lower() or "security" in response.text.lower():
                logger.warning("Baidu anti-bot detected, skipping...")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            selectors = [
                '.result',
                '.c-container',
                '[tpl]',
                '.c-result',
                '#content_left > div',
            ]

            containers = []
            for selector in selectors:
                containers = soup.select(selector)
                if containers:
                    logger.debug(f"Baidu using selector: {selector}")
                    break

            for container in containers:
                title_elem = (
                    container.select_one('h3 a') or
                    container.select_one('.t a') or
                    container.select_one('a[data-click]') or
                    container.select_one('.title_3M1bj a') or
                    container.select_one('a')
                )

                abstract_elem = (
                    container.select_one('.content-right_8Zs40') or
                    container.select_one('.c-abstract') or
                    container.select_one('.content-right') or
                    container.select_one('.abstract_3Qj1C') or
                    container.select_one('span[class*="abstract"]') or
                    container.select_one('p')
                )

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                    href = title_elem.get('href', '')

                    if _is_valid_result(title, abstract):
                        result = {
                            "title": title,
                            "abstract": abstract,
                            "url": href,
                            "source": "baidu"
                        }
                        if not any(r['title'] == title for r in all_results):
                            all_results.append(result)

                    if len(all_results) >= num_results:
                        break

            if len(all_results) >= num_results:
                break

        logger.info(f"Baidu found {len(all_results)} results for: {query}")
        return results

    except requests.exceptions.Timeout:
        logger.warning(f"Baidu timeout (> {timeout}s), skipping...")
        return []
    except Exception as e:
        logger.warning(f"Baidu error: {e}")
        return []

def _search_all(query: str, num_results: int = 5, timeout: float = 5.0) -> List[Dict]:
    engines = SEARCH_CONFIG.get("engines", ["duckduckgo", "bing", "baidu"])
    all_results = []
    seen_titles = set()

    for engine in engines:
        if len(all_results) >= num_results:
            break

        remaining = num_results - len(all_results)

        if engine == "duckduckgo":
            results = _search_duckduckgo(query, remaining, timeout)
        elif engine == "bing":
            results = _search_bing(query, remaining, timeout)
        elif engine == "baidu":
            results = _search_baidu(query, remaining, timeout)
        else:
            continue

        for r in results:
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                all_results.append(r)
                if len(all_results) >= num_results:
                    break

    logger.info(f"Total results from {len(set(r['source'] for r in all_results))} engines: {len(all_results)}")
    return all_results

_search_cache: Dict[str, tuple] = {}
_CACHE_TTL = 300

def cached_search(query: str, search_func_name: str = "all") -> List[Dict]:
    cache_key = f"{search_func_name}:{query}"
    now = time.time()
    if cache_key in _search_cache:
        result, ts = _search_cache[cache_key]
        if now - ts < _CACHE_TTL:
            logger.debug(f"Search cache hit: {cache_key}")
            return result

    results = _search_all(query) if search_func_name == "all" else []
    _search_cache[cache_key] = (results, now)
    return results

# ==================== 启动预加载 ====================
try:
    _load_entity_dict()
except Exception as e:
    logger.warning(f"Startup entity preload skipped: {e}")