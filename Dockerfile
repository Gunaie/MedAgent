# 使用 Python 3.11 轻量镜像
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p /app/data/db /app/data/inputs /app/chat_history /app/logs

# 暴露 Gradio 端口
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]