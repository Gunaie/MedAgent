import os
import re
import json
import logging
from typing import Dict, List
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from config import CHAT_HISTORY_DIR
from tools import (
    kg_query_func, search_func, retrieval_func, generic_chat_func,
    _extract_entities_fast, _extract_entities, _is_disease, _is_symptom, _is_medicine
)

logger = logging.getLogger(__name__)

_PRONOUNS = ["它", "该病", "此疾病", "这种病", "这个", "此", "该", "他", "她"]

_KG_SHORT_PATTERNS = [
    r"一般?会有?哪些症状[？\?]?",
    r"主要?症状是?什么[？\?]?",
    r"主要?表现是?什么[？\?]?",
    r"怎么?治疗[？\?]?",
    r"怎么?用药[？\?]?",
    r"吃?什么药[？\?]?",
    r"需要?做?什么检查[？\?]?",
    r"应该?去?哪个科室[？\?]?",
    r"怎么?预防[？\?]?",
    r"怎么?引起[？\?]?",
    r"严重?吗[？\?]?",
    r"有?什么?危害[？\?]?",
    r"会?传染?吗[？\?]?",
    r"能?治?好吗[？\?]?",
    r"会?有?什么?症状[？\?]?",      # 新增
    r"通常?表现[为]?[？\?]?",       # 新增
    r"常见?症状[是]?[？\?]?",       # 新增
    r"有?哪些?不适[？\?]?",         # 新增
    r"怎么?办[？\?]?",              # 新增
    r"如何?处理[？\?]?",            # 新增
]

_KG_KEYWORDS = [
    "症状", "表现", "征象", "治疗", "药物", "药", "检查", "科室", "预防",
    "病因", "定义", "简介", "危害", "严重", "传染", "治愈", "并发症",
    "怎么办", "怎么处理", "如何", "怎么", "什么", "哪些", "不适", "感觉",
    "需要", "应该", "可以", "能", "会",
]

_RETRIEVAL_KEYWORDS = [
    "指南", "文献", "论文", "研究", "临床", "病例", "报告", "综述", "meta分析",
    "随机对照", "RCT", "cohort", "队列", "回顾性", "前瞻性", "系统评价",
]

_HARDCODED_PATH = os.path.join(os.path.dirname(__file__), "hardcoded.json")
_hardcoded_cache = None


def _load_hardcoded() -> Dict[str, str]:
    global _hardcoded_cache
    if _hardcoded_cache is not None:
        return _hardcoded_cache
    if not os.path.exists(_HARDCODED_PATH):
        _hardcoded_cache = {}
        return _hardcoded_cache
    try:
        with open(_HARDCODED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _hardcoded_cache = {k.lower(): v for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to load hardcoded: {e}")
        _hardcoded_cache = {}
    return _hardcoded_cache


# ========== FIX 1: 安全获取消息内容 ==========
def _get_message_content(msg) -> str:
    if hasattr(msg, 'content'):
        return msg.content or ""
    elif isinstance(msg, dict):
        if "content" in msg:
            return msg.get("content", "") or ""
        if "data" in msg and isinstance(msg["data"], dict):
            return msg["data"].get("content", "") or ""
    return ""


# ========== FIX 2: 改进历史话题提取，优先识别【实体】格式 ==========
def _extract_topic_from_history(history: list) -> str:
    for msg in reversed(history):
        content = _get_message_content(msg)
        if not content:
            continue
        # 优先从助手的 【实体】 格式中提取
        bracket_match = re.search(r'【([^】]{2,20})】', content)
        if bracket_match:
            candidate = bracket_match.group(1)
            if _extract_entities_fast(candidate):
                logger.info(f"Topic from bracket: {candidate}")
                return candidate
        # 常规实体提取
        entities = _extract_entities_fast(content)
        if entities:
            logger.info(f"Topic from entities: {entities[0]}")
            return entities[0]

    # FIX: 从用户提问中提取实体（兜底）
    for msg in reversed(history):
        content = _get_message_content(msg)
        if not content:
            continue
        # 只查用户消息（role=user 或没有 role 标记的）
        if hasattr(msg, 'type') and msg.type == 'human':
            entities = _extract_entities_fast(content)
            if entities:
                logger.info(f"Topic from user query: {entities[0]}")
                return entities[0]
        # 兼容字符串格式
        elif isinstance(msg, dict) and msg.get('role') == 'user':
            entities = _extract_entities_fast(content)
            if entities:
                logger.info(f"Topic from user query: {entities[0]}")
                return entities[0]
        elif isinstance(content, str):
            # 如果历史记录是纯字符串列表，尝试提取
            entities = _extract_entities_fast(content)
            if entities:
                logger.info(f"Topic from history string: {entities[0]}")
                return entities[0]

    return ""


def _is_short_kg_query(text: str) -> bool:
    text_clean = re.sub(r'[？\?。！，,\.]', '', text)
    for pattern in _KG_SHORT_PATTERNS:
        if re.search(pattern, text):
            return True
    if len(text_clean) <= 20:  # FIX: 放宽到 20 字
        for kw in _KG_KEYWORDS:
            if kw in text_clean:
                return True
    return False


# ========== FIX 3: 强化多轮对话补全 ==========
def _rule_summarize(message: str, history: list) -> str:
    if not history:
        return message

    entities_now = _extract_entities_fast(message)
    if entities_now:
        return message

    current_topic = _extract_topic_from_history(history)
    if not current_topic:
        logger.warning(f"No topic found in history for: {message}")
        return message

    # 指代词替换
    for p in _PRONOUNS:
        if p in message:
            replaced = message.replace(p, current_topic, 1)
            logger.info(f"Pronoun replace: '{message}' -> '{replaced}'")
            return replaced

    # 短 KG 查询补全
    if _is_short_kg_query(message):
        if current_topic not in message:
            enriched = f"{current_topic}{message}"
            logger.info(f"KG enrich: '{message}' -> '{enriched}'")
            return enriched

    # FIX: 兜底——短句含医疗关键词但无实体，强制补全
    text_clean = re.sub(r'[？\?。！，,\.]', '', message)
    if len(text_clean) <= 25:
        if any(kw in text_clean for kw in _KG_KEYWORDS):
            if current_topic not in message:
                enriched = f"{current_topic}{message}"
                logger.info(f"Forced enrich: '{message}' -> '{enriched}'")
                return enriched

    return message


# ========== FIX 4: 改进意图分类，实体+医疗关键词强制 kg ==========
def _score_intent(query: str) -> Dict[str, int]:
    q = query.lower()
    scores = {"generic": 0, "retrieval": 0, "kg": 0, "search": 0}

    for kw in _KG_KEYWORDS:
        if kw in q:
            scores["kg"] += 3

    # FIX: 如果查询中包含医疗实体 + 医疗关键词，强制给 kg 高分
    entities_in_q = _extract_entities_fast(q)
    if entities_in_q:
        medical_keywords = ["症状", "药", "治疗", "检查", "科室", "预防", "病因",
                           "定义", "危害", "严重", "传染", "治愈", "并发症", "表现",
                           "怎么办", "怎么", "如何", "什么", "哪些"]
        if any(kw in q for kw in medical_keywords):
            scores["kg"] += 15  # 强信号，确保不被 search 覆盖

    if re.search(r".+是什么[病]?[？\?]", q):
        scores["kg"] += 5
    elif re.search(r".+的?定义[是]?[？\?]", q):
        scores["kg"] += 5
    elif re.search(r".+的?简介[是]?[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+怎么[治医]?[？\?]", q):
        scores["kg"] += 5
    elif re.search(r".+怎么[办做]?[？\?]", q):
        scores["kg"] += 5
    elif re.search(r".+如何[治医]?[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+用[什么]?药[？\?]", q):
        scores["kg"] += 5
    elif re.search(r".+吃[什么]?药[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+去?哪个?科室[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+做[什么]?检查[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+预防[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+病因[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+症状[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+危害[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+严重[吗]?[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+传染[吗]?[？\?]", q):
        scores["kg"] += 5

    if re.search(r".+能[否]?治[愈好]?[？\?]", q):
        scores["kg"] += 5

    for kw in _RETRIEVAL_KEYWORDS:
        if kw in q:
            scores["retrieval"] += 3

    if re.search(r"搜索|查找|查一下|搜一下|网上|百度|google|bing", q):
        scores["search"] += 5

    if re.search(r"你好|在吗|谢谢|再见|拜拜|help|hello|hi", q):
        scores["generic"] += 10

    # FIX: 有实体时降低 search 权重，避免被搜索覆盖
    if entities_in_q and scores["kg"] > 0:
        scores["search"] = max(0, scores["search"] - 3)

    return scores


def _classify_intent(query: str) -> str:
    scores = _score_intent(query)
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return "generic"
    second_best = sorted(scores, key=scores.get, reverse=True)[1]
    second_score = scores[second_best]
    if best_score - second_score < 3 and best != "generic":
        return "generic"
    return best


class ChatService:
    def __init__(self, session_id: str = "anonymous"):
        self.user_id = session_id
        self.session_id = session_id
        # FIX: 自动创建历史记录目录
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        self.history = FileChatMessageHistory(
            file_path=os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        )

    def _summarize(self, message: str) -> str:
        recent = self.history.messages[-6:]
        if not recent:
            return message
        return _rule_summarize(message, recent)

    def answer(self, message: str) -> str:
        original_message = message
        logger.info(f"=== Turn start | Original: {original_message} | History: {len(self.history.messages)} ===")

        if self.history.messages:
            message = self._summarize(message)
            if message != original_message:
                logger.info(f"Enriched: '{original_message}' -> '{message}'")

        msg_lower = message.lower()

        hardcoded = _load_hardcoded()
        for key, value in hardcoded.items():
            if key in msg_lower:
                self.history.add_user_message(original_message)
                self.history.add_ai_message(value)
                return value

        intent = _classify_intent(message)
        logger.info(f"Intent: {intent} | Query: '{message}'")

        tool_map = {
            "kg": kg_query_func,
            "search": search_func,
            "retrieval": retrieval_func,
            "generic": generic_chat_func,
        }
        tool = tool_map.get(intent, generic_chat_func)

        try:
            result = tool.run(message)
            result_text = result.content if hasattr(result, "content") else str(result)
            logger.info(f"Result: {result_text[:200]}...")
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            result = "抱歉，系统暂时无法回答该问题。"

        # 确保 result 是字符串
        final_result = result.content if hasattr(result, "content") else str(result)
        self.history.add_user_message(original_message)
        self.history.add_ai_message(final_result)
        return final_result