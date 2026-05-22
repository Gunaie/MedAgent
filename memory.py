"""长期会话记忆：文件存储"""

import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """基于 JSON 文件的持久化聊天历史"""

    def __init__(self, session_id: str, storage_dir: str = "./chat_history"):
        self.session_id = session_id
        self.storage_dir = storage_dir
        self.file_path = os.path.join(storage_dir, f"{session_id}.json")
        os.makedirs(storage_dir, exist_ok=True)

    @property
    def messages(self) -> list[BaseMessage]:
        """读取历史消息"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return messages_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """追加并保存消息"""
        all_messages = list(self.messages)
        all_messages.extend(messages)

        serialized = [message_to_dict(msg) for msg in all_messages]

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清空历史"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)