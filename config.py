"""集中配置：提示词模板、工具描述、搜索配置、日志配置"""

from typing import Final

# ==================== 日志配置 ====================
LOG_CONFIG: Final[dict] = {
    "level": "INFO",           # DEBUG/INFO/WARNING/ERROR/CRITICAL
    "file": "./logs/medagent.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# ==================== 提示词模板（保持之前版本）====================
GENERIC_PROMPT_TPL: Final[str] = """\
1. 当你被人问起身份时，你必须用'我是一个医疗问诊机器人'回答。
 例如问题 [你好，你是谁，你是谁开发的，你和GPT有什么关系，你和OpenAI有什么关系]
2. 你必须拒绝讨论任何关于政治，色情，暴力相关的事件或者人物。
3. 请用中文回答用户问题，语气友好、简洁。
----------
用户问题: {query}
"""

RETRIEVAL_PROMPT_TPL: Final[str] = """\
你是寻医问药网的企业客服助手。请严格根据以下检索到的内部文档内容回答用户问题。
如果检索结果中没有相关信息，请回复"抱歉，暂时没有找到相关信息。"，不要编造。
--------
检索结果：{context}
--------
用户问题：{query}
"""

SEARCH_PROMPT_TPL: Final[str] = """\
请根据以下搜索引擎检索结果，用简洁语言回答用户问题。不要发散和联想。
如果检索结果中没有相关信息，请回复"暂时无法获取相关信息，请换个问法试试。"。
--------
检索结果：{query_result}
--------
用户问题：{query}
"""

SUMMARY_PROMPT_TPL: Final[str] = """\
请结合以下历史对话信息，和用户消息，总结出一个简洁、完整的用户消息。
直接给出总结好的消息，不需要其他信息，适当补全句子中的主语等信息。
如果和历史对话消息没有关联，直接输出用户原始消息。
注意，仅补充内容，不能改变原消息的语义和句式。

----------
历史对话：
{chat_history}
----------
用户消息：{query}
----------
输出：
"""

# ==================== 工具描述 ====================
TOOL_DESCRIPTIONS: Final[dict[str, str]] = {
    "generic_func": "可以解答通用领域的知识，例如打招呼等问题",
    "retrieval_func": "用于回答寻医问药网相关问题，基于企业内部文档",
    "kg_query_func": "用于回答疾病、症状、药物之间的医学关联关系，基于医疗知识图谱。当问题涉及'吃什么药'、'有什么症状'、'怎么治疗'时使用",
    "search_func": "其他工具没有正确答案时，通过搜索引擎回答通用类问题",
}

# ==================== 搜索配置 ====================
SEARCH_CONFIG: Final[dict] = {
    "engines": ["duckduckgo", "bing", "baidu"],
    "timeout": 5.0,
    "max_results": 5,
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ],
    "delay_between_requests": 0.5,
}

# ==================== 医疗图谱查询模板（保持之前版本）====================
GRAPH_TEMPLATE: Final[dict] = {
    "desc": {
        "slots": ["disease"],
        "question": "什么叫{disease}？/{disease}是一种什么病？",
        "cypher": """
        MATCH (d:Disease)
        WHERE d.name = $disease
        RETURN d.name AS name, coalesce(d.desc, '暂无描述') AS RES
        LIMIT 1
        """,
        "answer": "【{disease}】{RES}",
    },
    "cause": {
        "slots": ["disease"],
        "question": "{disease}一般是由什么引起的？",
        "cypher": """
        MATCH (n:Disease)
        WHERE n.name = $disease
        RETURN coalesce(n.cause, '具体病因需结合临床检查确定') AS RES
        LIMIT 1
        """,
        "answer": "【{disease}】的病因/诱因：{RES}。",
    },
    "symptom": {
        "slots": ["disease"],
        "question": "{disease}会有哪些症状？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_SYMPTOM]->(s:Symptom)
        WHERE d.name = $disease
        RETURN COLLECT(DISTINCT s.name) AS RES LIMIT 10
        """,
        "answer": "【{disease}】的症状包括：{RES}。",
    },
    "cure_way": {
        "slots": ["disease"],
        "question": "{disease}吃什么药好得快？/{disease}用什么药？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_DRUG]->(drug:Drug)
        WHERE d.name = $disease
        RETURN COLLECT(DISTINCT drug.name) AS RES LIMIT 10
        """,
        "answer": "【{disease}】可用药物：{RES}。具体用药请遵医嘱。",
    },
    "cure_method": {
        "slots": ["disease"],
        "question": "{disease}怎么治？/{disease}治疗方法有哪些？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_CUREWAY]->(c:Cureway)
        WHERE d.name = $disease
        RETURN COLLECT(DISTINCT c.name) AS RES LIMIT 10
        """,
        "answer": "【{disease}】治疗方法包括：{RES}。",
    },
    "check": {
        "slots": ["disease"],
        "question": "{disease}要做哪些检查？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_CHECK]->(c:Check)
        WHERE d.name = $disease
        RETURN COLLECT(DISTINCT c.name) AS RES LIMIT 10
        """,
        "answer": "【{disease}】的检查项目：{RES}。",
    },
    "department": {
        "slots": ["disease"],
        "question": "得了{disease}去医院挂什么科室？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_DEPARTMENT]->(dep:Department)
        WHERE d.name = $disease
        RETURN COLLECT(DISTINCT dep.name) AS RES LIMIT 10
        """,
        "answer": "【{disease}】建议就诊科室：{RES}。",
    },
    "cured_prob": {
        "slots": ["disease"],
        "question": "{disease}能治好吗？",
        "cypher": """
        MATCH (n:Disease)
        WHERE n.name = $disease
        RETURN coalesce(n.cured_prob, '治愈率信息暂无') AS RES
        LIMIT 1
        """,
        "answer": "【{disease}】的治愈率：{RES}。",
    },
    "indications": {
        "slots": ["drug"],
        "question": "{drug}能治哪些病？",
        "cypher": """
        MATCH (d:Disease)-[:DISEASE_DRUG]->(drug:Drug)
        WHERE drug.name = $drug
        RETURN COLLECT(DISTINCT d.name) AS RES LIMIT 10
        """,
        "answer": "【{drug}】能治疗的疾病：{RES}。",
    },
    "prevent": {
        "slots": ["disease"],
        "question": "{disease}怎么预防？",
        "cypher": """
        MATCH (n:Disease)
        WHERE n.name = $disease
        RETURN coalesce(n.prevent, '暂无预防信息') AS RES
        LIMIT 1
        """,
        "answer": "【{disease}】预防措施：{RES}。",
    },
}