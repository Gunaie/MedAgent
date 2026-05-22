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


# ==================== 四层意图分类 ====================

_INTENT_RULES = {
    "generic": ["你好", "是谁", "名字", "gpt", "openai", "chatgpt", "开发", "谢谢", "再见", "在吗", "帮助", "能干"],
    "retrieval": ["寻医问药", "客服", "电话", "官网", "投资", "xywy", "投诉", "邮箱", "地址", "联系", "商务合作", "公司", "融资"],
    "kg": ["症状", "治疗", "怎么治", "吃什么药", "药", "疾病", "病", "科室", "检查", "治愈率", "并发症", "病因", "引起", "能吃", "忌口", "危害", "表现", "缓解", "自愈", "挂什么科", "怎么分", "怎么办", "能否", "可以", "哪些", "什么病", "什么叫"],
}


def _classify_intent(query: str) -> str:
    q = query.lower()
    for kw in _INTENT_RULES["generic"]:
        if kw in q:
            return "generic"
    for kw in _INTENT_RULES["retrieval"]:
        if kw in q:
            return "retrieval"
    if _extract_entities_fast(query):
        return "kg"
    for kw in _INTENT_RULES["kg"]:
        if kw in q:
            return "kg"
    return "search"


# ==================== 多轮对话补全（纯规则，零 LLM）====================

_PRONOUNS = ["它", "这个", "那种", "此", "上述", "这些", "那些"]

_KG_SHORT_PATTERNS = [
    r"一般?会有?哪些症状", r"有什么症状", r"症状有哪些", r"吃什么药", r"用什么药",
    r"怎么治", r"怎么治疗", r"能治好吗", r"严重吗", r"多久能好", r"需要注意什么",
    r"可以.*吗", r"能吃.*吗", r"有什么危害", r"是什么病", r"怎么引起的",
    r"为什么会", r"有什么表现", r"如何缓解", r"需要忌口", r"能自愈",
]


def _is_short_kg_query(text: str) -> bool:
    for pattern in _KG_SHORT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _extract_topic_from_history(history: list) -> str:
    for msg in reversed(history):
        content = ""
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        if content:
            entities = _extract_entities_fast(content)
            if entities:
                return entities[0]
    return ""


def _rule_summarize(message: str, history: list) -> str:
    """
    纯规则补全：
    1. 当前消息已有新实体 → 不补全（用户切换话题）
    2. 包含指代词 → 替换为历史话题实体
    3. 医疗短问句 → 在前面补上历史话题主语
    4. 其他 → 原样返回
    """
    if not history:
        return message

    # 当前消息已有新实体，说明在切换话题，直接返回
    entities_now = _extract_entities_fast(message)
    if entities_now:
        return message

    current_topic = _extract_topic_from_history(history)
    if not current_topic:
        return message

    # 指代词替换
    for p in _PRONOUNS:
        if p in message:
            replaced = message.replace(p, current_topic, 1)
            print(f"[Context] 指代替换: '{message}' → '{replaced}' (话题: {current_topic})")
            return replaced

    # 省略主语补全
    if _is_short_kg_query(message):
        enriched = f"{current_topic}{message}"
        print(f"[Context] 补全省略主语: '{message}' → '{enriched}' (话题: {current_topic})")
        return enriched

    return message


class ChatService:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.history = FileChatMessageHistory(session_id)
        _load_hardcoded()

    def _summarize(self, message: str) -> str:
        """多轮对话补全（纯规则，零 LLM，零超时风险）"""
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
            print(f"[ERROR] Tool execution failed (intent={intent}): {e}")
            response = "抱歉，系统暂时繁忙，请稍后再试，或联系客服 400-859-1200。"

        self.history.add_messages([
            HumanMessage(content=message),
            AIMessage(content=response),
        ])

        return response