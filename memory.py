"""长期会话记忆：文件存储（带并发锁）"""

import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

try:
    from filelock import FileLock, Timeout
except ImportError:
    raise ImportError("filelock is required. Install: uv pip install filelock")

from utils import get_logger

logger = get_logger("medagent.memory")

class FileChatMessageHistory(BaseChatMessageHistory):
    """基于 JSON 文件的持久化聊天历史（线程/进程安全）"""

    def __init__(self, session_id: str, storage_dir: str = "./chat_history"):
        self.session_id = session_id
        self.storage_dir = storage_dir
        self.file_path = os.path.join(storage_dir, f"{session_id}.json")
        self.lock_path = self.file_path + ".lock"
        os.makedirs(storage_dir, exist_ok=True)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return messages_from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        lock = FileLock(self.lock_path, timeout=10)

        try:
            with lock:
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_messages = messages_from_dict(data)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_messages = []

                all_messages.extend(messages)
                serialized = [message_to_dict(msg) for msg in all_messages]

                temp_path = self.file_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(serialized, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, self.file_path)
                logger.debug(f"Saved {len(all_messages)} messages for session {self.session_id}")

        except Timeout:
            logger.error(f"File lock timeout: {self.lock_path}, messages may be lost")
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")

    def clear(self) -> None:
        lock = FileLock(self.lock_path, timeout=10)
        try:
            with lock:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
                logger.info(f"Cleared history for session {self.session_id}")
        except Timeout:
            logger.error(f"Clear history lock timeout: {self.lock_path}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")