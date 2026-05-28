"""Gradio 前端"""
from dotenv import load_dotenv
load_dotenv()

import os
# FIX: 在导入任何可能加载 ChromaDB 的模块之前，彻底禁用遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import gradio as gr
import uuid
from cachetools import TTLCache

from service import ChatService
from utils import get_logger

logger = get_logger("medagent.app")

_service_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)

def get_service(session_id: str) -> ChatService:
    if session_id not in _service_cache:
        _service_cache[session_id] = ChatService(session_id=session_id)
        logger.info(f"New session created: {session_id}")
    return _service_cache[session_id]

def doctor_bot(message: str, history: list[dict], session_id: str):
    if not session_id:
        session_id = f"user_{uuid.uuid4().hex[:8]}"
        logger.info(f"New session initialized: {session_id}")
    else:
        logger.debug(f"Using existing session: {session_id}")

    service = get_service(session_id)
    logger.info(f"User query: {message}")

    response = service.answer(message)
    logger.info(f"Bot response: {response[:100]}...")

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
    # FIX: 直接使用 Gradio 的 launch，不再用 FastAPI/uvicorn 包装
    logger.info("Starting MedAgent server on http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )