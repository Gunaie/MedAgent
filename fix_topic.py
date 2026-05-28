# fix_topic.py
with open('service.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''    # 常规实体提取
        entities = _extract_entities_fast(content)
        if entities:
            logger.info(f"Topic from entities: {entities[0]}")
            return entities[0]
    return ""'''

new = '''    # 常规实体提取
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

    return ""'''

content = content.replace(old, new)

with open('service.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ _extract_topic_from_history 修复完成")