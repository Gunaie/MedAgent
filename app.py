"""Gradio 前端（用 State 保存 session_id）"""

import gradio as gr
import uuid
from cachetools import TTLCache

from service import ChatService

# FIX: TTL + LRU 缓存，1小时过期，最多1000个会话，防止内存泄漏
_service_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)


def get_service(session_id: str) -> ChatService:
    """获取或创建会话服务，自动清理过期会话"""
    if session_id not in _service_cache:
        _service_cache[session_id] = ChatService(session_id=session_id)
    return _service_cache[session_id]


def doctor_bot(message: str, history: list[dict], session_id: str):
    if not session_id:
        session_id = f"user_{uuid.uuid4().hex[:8]}"
        print(f"[Session] New session created: {session_id}")
    else:
        print(f"[Session] Using existing session: {session_id}")

    service = get_service(session_id)
    response = service.answer(message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, "", session_id


def clear_chat():
    return [], "", ""


with gr.Blocks() as demo:
    gr.Markdown("# 医疗问诊机器人")
    gr.Markdown("基于阿里云通义千问大模型")

    session_state = gr.State(value="")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="在此输入您的问题", container=False, scale=7)

    with gr.Row():
        submit_btn = gr.Button("提交", variant="primary")
        clear_btn = gr.Button("清空记录")

    gr.Examples(
        examples=[
            "你好，你叫什么名字？",
            "寻医问药网获得过哪些投资？",
            "寻医问药网的客服电话是多少？",
            "鼻炎是一种什么病？",
            "一般会有哪些症状？",
            "吃什么药好得快？可以吃阿莫西林吗？",
        ],
        inputs=msg,
    )

    submit_btn.click(doctor_bot, inputs=[msg, chatbot, session_state], outputs=[chatbot, msg, session_state])
    msg.submit(doctor_bot, inputs=[msg, chatbot, session_state], outputs=[chatbot, msg, session_state])
    clear_btn.click(clear_chat, inputs=None, outputs=[chatbot, msg, session_state])

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    demo = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")