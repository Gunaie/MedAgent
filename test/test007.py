import threading
from memory import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

def worker(session_id: str, worker_id: int):
    hist = FileChatMessageHistory(session_id)
    for i in range(50):
        hist.add_messages([
            HumanMessage(content=f"worker{worker_id}-msg{i}"),
            AIMessage(content=f"reply{i}"),
        ])
    print(f"Worker {worker_id} done")

# 模拟 4 个线程同时写入同一个 session
session = "concurrent_test"
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(session, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# 验证数据完整性
hist = FileChatMessageHistory(session)
msgs = hist.messages
print(f"总消息数: {len(msgs)}")  # 应为 400（4 workers * 50 * 2 messages）
print(f"最后一条: {msgs[-1].content if msgs else 'None'}")

# 清理
import os
os.remove(hist.file_path)
os.remove(hist.lock_path)