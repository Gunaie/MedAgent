"""业务层：四层清晰路由 + 纯规则多轮补全"""

import os
import json
import re
from typing import Dict

from langchain_core.messages import HumanMessage, AIMessage

from models import get_llm_model
from memory import FileChatMessageHistory
from tools import (
    generic_func, retrieval_func, kg_query_func, search_func,
    _extract_entities_fast,
)
from utils import get_logger

logger = get_logger("medagent.service")

# 高频问答缓存
_HARDCODED_PATH = "./data/hardcoded_qa.json"
_hardcoded_cache: Dict[str, str] = {}

def _load_hardcoded() -> Dict[str, str]:
    global _hardcoded_cache
    if os.path.exists(_HARDCODED_PATH):
        try:
            with open(_HARDCODED_PATH, "r", encoding="utf-8") as f:
                _hardcoded_cache = json.load(f)
        except Exception:
            pass
    if not _hardcoded_cache:
        _hardcoded_cache = {
            "你好": "你好，我是一个医疗问诊机器人。",
            "你叫什么名字": "你好，我是一个医疗问诊机器人。",
            "你是谁": "你好，我是一个医疗问诊机器人。",
            "客服电话": "寻医问药网的客服电话是 400-859-1200（09:00-18:00）。",
            "寻医问药网客服": "寻医问药网的客服电话是 400-859-1200（09:00-18:00），客服微信号：xywygfkf。",
            "官网": "寻医问药网的官网地址是 https://www.xywy.com",
            "投诉邮箱": "投诉邮箱：wangjingying@xywy.com。",
            "联系地址": "地址：北京市朝阳区德外大街华严北里甲1号健翔山庄C10。",
        }
    return _hardcoded_cache

# ==================== 意图分类 ====================

_GENERIC_KEYWORDS = ["你好", "是谁", "名字", "gpt", "openai", "chatgpt", "开发",
                     "谢谢", "感谢", "再见", "拜拜", "在吗", "帮助", "能干", "介绍", "功能"]

_RETRIEVAL_KEYWORDS = ["寻医问药", "客服", "电话", "官网", "投资", "xywy", "投诉", "邮箱", "地址", "联系", "商务合作", "公司", "融资", "招聘", "加入我们"]

_KG_KEYWORDS = [
    "症状", "治疗", "怎么治", "吃什么药", "药", "疾病", "病", "科室", "检查", "治愈率", "并发症", "病因", "引起",
    "能吃", "忌口", "危害", "表现", "缓解", "自愈", "挂什么科", "怎么分", "怎么办", "能否", "可以", "哪些",
    "什么病", "什么叫", "什么是", "的定义", "是一种什么", "预防", "饮食", "注意", "禁忌", "传染", "遗传", "复发", "手术", "住院", "费用", "医保",
    "疼", "痛", "痒", "肿", "烧", "咳", "吐", "泻", "晕", "麻", "乏", "困", "失眠", "出血", "发烧", "感冒",
    "不舒服", "难受", "异常", "失调", "紊乱", "亢进", "减退", "肿大", "结节", "积水", "硬化", "萎缩", "坏死",
]

_NEGATION_PATTERNS = [
    r"我叫.*(?:名字|姓名)",
    r"我的名字是",
    r"我姓",
    r"我(?:爸|妈|爷|奶|哥|姐|弟|妹|老公|老婆|孩子|儿子|女儿).*有",
    r"(?:不要|不是|没有|别|没).*症状",
    r"(?:不|没).*疼",
    r"(?:不|没).*病",
]

def _has_negation_context(query: str) -> bool:
    for pattern in _NEGATION_PATTERNS:
        if re.search(pattern, query):
            return True
    return False

# 在 _score_intent 函数中，增加企业问题的最高优先级判断
def _score_intent(query: str) -> Dict[str, int]:
    q = query.lower()
    scores = {"generic": 0, "retrieval": 0, "kg": 0, "search": 0}

    # FIX: 企业信息精确模式（最高优先级）
    enterprise_patterns = [
        r"寻医问药网.*(?:投资|融资|客服|电话|官网|邮箱|地址)",
        r"(?:投资|融资|客服|电话|官网).*寻医问药网",
    ]
    for pattern in enterprise_patterns:
        if re.search(pattern, q):
            scores["retrieval"] += 200  # 压倒一切
            return scores

    has_negation = _has_negation_context(query)

    for kw in _GENERIC_KEYWORDS:
        if kw in q:
            scores["generic"] += 1 if not has_negation else 0.3

    for kw in _RETRIEVAL_KEYWORDS:
        if kw in q:
            scores["retrieval"] += 2

    for kw in _KG_KEYWORDS:
        if kw in q:
            scores["kg"] += 3

    if re.search(r".+是什么[病]?[？\?]", query):
        scores["kg"] += 5

    if re.search(r".+(?:怎么办|怎么处理|怎么缓解|严重吗|危险吗)[？\?]?", query):
        scores["kg"] += 5

    if re.match(r"^(你好|您好|嗨|hello|hi)[!！。]*$", query, re.IGNORECASE):
        scores["generic"] += 10

    if any(kw in q for kw in ["客服电话", "投诉邮箱", "联系地址", "公司官网"]):
        scores["retrieval"] += 10

    return scores

def _classify_intent(query: str) -> str:
    scores = _score_intent(query)
    logger.debug(f"Intent classification: '{query}' -> scores: {scores}")

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score < 1:
        return "search"

    return best

# ==================== 多轮对话补全 ====================

_PRONOUNS = ["它", "这个", "那种", "此", "上述", "这些", "那些", "这病", "该病", "此病", "这个药", "那种药", "上面说的"]

# FIX: 优化 KG 短查询模式，支持更多省略句式，包括带问号的变体
_KG_SHORT_PATTERNS = [
    r"一般?会有?哪些症状[？\?]?", r"有什么症状[？\?]?", r"症状有哪些[？\?]?",
    r"吃什么药[？\?]?", r"用什么药[？\?]?", r"一般?会?有什么症状[？\?]?",
    r"怎么治[？\?]?", r"怎么治疗[？\?]?", r"能治好吗[？\?]?", r"严重吗[？\?]?", r"多久能好[？\?]?", r"需要注意什么[？\?]?",
    r"可以.*吗[？\?]?", r"能吃.*吗[？\?]?", r"有什么危害[？\?]?", r"是什么病[？\?]?", r"怎么引起的[？\?]?",
    r"为什么会[？\?]?", r"有什么表现[？\?]?", r"如何缓解[？\?]?", r"需要忌口[？\?]?", r"能自愈[？\?]?",
    r"一般?有?哪些症状[？\?]?", r"通常?有?什么症状[？\?]?", r"主要?症状[？\?]?",
]

def _is_short_kg_query(text: str) -> bool:
    # FIX: 去除标点后再匹配，提高命中率
    text_clean = re.sub(r'[？\?。！，,\.]', '', text)
    for pattern in _KG_SHORT_PATTERNS:
        if re.search(pattern, text):
            return True
    # FIX: 额外检查：如果去除标点后是短句且包含症状/药物关键词
    if len(text_clean) <= 15:
        if any(kw in text_clean for kw in ["症状", "药", "治", "吃", "用"]):
            return True
    return False

# FIX: 从历史消息中正确提取实体，支持 BaseMessage 和 dict 两种格式
def _extract_topic_from_history(history: list) -> str:
    for msg in reversed(history):
        content = ""
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            # FIX: 如果是 Gradio 的 history dict 格式
            if not content and "role" in msg:
                content = msg.get("content", "")
        if content:
            entities = _extract_entities_fast(content)
            if entities:
                return entities[0]
    return ""

def _rule_summarize(message: str, history: list) -> str:
    if not history:
        return message

    entities_now = _extract_entities_fast(message)
    if entities_now:
        return message

    current_topic = _extract_topic_from_history(history)
    if not current_topic:
        return message

    # FIX: 指代词替换 - 支持更多变体
    message_clean = message
    for p in _PRONOUNS:
        if p in message_clean:
            replaced = message_clean.replace(p, current_topic, 1)
            logger.debug(f"Pronoun replacement: '{message}' -> '{replaced}' (topic: {current_topic})")
            return replaced

    # FIX: 省略主语补全 - 改进匹配逻辑
    if _is_short_kg_query(message):
        # FIX: 智能拼接，避免重复
        if message.startswith("一般") or message.startswith("通常"):
            enriched = f"{current_topic}{message}"
        elif message.startswith("有") or message.startswith("会"):
            enriched = f"{current_topic}{message}"
        else:
            enriched = f"{current_topic}{message}"
        logger.debug(f"Subject enrichment: '{message}' -> '{enriched}' (topic: {current_topic})")
        return enriched

    return message

class ChatService:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.history = FileChatMessageHistory(session_id)
        _load_hardcoded()

    def _summarize(self, message: str) -> str:
        recent = self.history.messages[-6:]
        if not recent:
            return message
        return _rule_summarize(message, recent)

    def answer(self, message: str) -> str:
        if self.history.messages:
            message = self._summarize(message)

        msg_lower = message.lower()

        hardcoded = _load_hardcoded()
        for key, value in hardcoded.items():
            if key in msg_lower:
                self.history.add_messages([
                    HumanMessage(content=message),
                    AIMessage(content=value),
                ])
                return value

        intent = _classify_intent(message)
        response = ""

        try:
            if intent == "generic":
                response = generic_func.invoke({"query": message})
            elif intent == "retrieval":
                response = retrieval_func.invoke({"query": message})
            elif intent == "kg":
                response = kg_query_func.invoke({"query": message})
            else:
                response = search_func.invoke({"query": message})

        except Exception as e:
            logger.error(f"Tool execution failed (intent={intent}): {e}")
            response = "抱歉，系统暂时繁忙，请稍后再试，或联系客服 400-859-1200。"

        self.history.add_messages([
            HumanMessage(content=message),
            AIMessage(content=response),
        ])

        return response