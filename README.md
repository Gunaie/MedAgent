# MedAgent 医疗问诊机器人

基于 **Neo4j 医疗知识图谱 + Chroma 向量检索 + ReAct Agent** 的智能医疗问诊系统，使用阿里云 DashScope（通义千问）作为大模型，Gradio 作为前端界面。

---

## 项目架构

```
MedAgent/
├── app.py                 # Gradio 前端入口
├── agent.py               # ReAct Agent 定义
├── service.py             # 业务层：意图分类 + 多轮对话
├── models.py              # LLM & Embedding 模型配置
├── memory.py              # 文件持久化聊天历史（并发安全）
├── tools.py               # Agent 工具函数（搜索/图谱/检索）
├── vector_store.py        # Chroma 向量库封装
├── neo4j_store.py         # Neo4j 知识图谱封装
├── config.py              # 集中配置：提示词/模板/搜索配置
├── utils.py               # 工具函数：环境变量/统一日志
├── scripts/
│   ├── data_process.py    # 文档向量化处理
│   └── import_data.py     # Neo4j 数据导入
├── data/
│   ├── inputs/            # 企业文档输入目录
│   ├── db/                # Chroma 向量库持久化
│   └── hardcoded_qa.json  # 高频问答缓存
├── neo4j_data/            # Neo4j 节点/关系 CSV
├── chat_history/          # 会话历史文件
├── logs/                  # 统一日志输出
├── Dockerfile             # Docker 镜像构建
├── docker-compose.yml     # 一键编排部署
└── requirements.txt       # Python 依赖
```

---

## 快速启动（Docker 推荐）

### 1. 准备环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 构建并启动

```bash
docker-compose up --build
```

### 3. 访问服务

| 服务 | 地址 |
|------|------|
| Gradio 问诊界面 | http://localhost:7860 |
| Neo4j Browser | http://localhost:7474 |

### 4. 导入医疗数据（首次运行）

将 `medical.json` 放入 `neo4j_data/` 目录，然后执行：

```bash
# 导入知识图谱
docker exec medagent-app python scripts/import_data.py

# 构建向量库（需先放入企业文档到 data/inputs/）
docker exec medagent-app python scripts/data_process.py
```

---

## 本地开发

### 环境要求

- Python 3.11+
- Neo4j 5.x
- 阿里云 DashScope API Key

### 安装依赖

```bash
# 推荐使用 uv
uv pip install -r requirements.txt

# 或传统 pip
pip install -r requirements.txt
```

### 配置环境变量

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_PASSWORD="password"

# Linux/macOS
export DASHSCOPE_API_KEY=your_api_key_here
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=password
```

### 启动服务

```bash
# 1. 确保 Neo4j 已运行
# 2. 导入医疗数据
python scripts/import_data.py

# 3. 构建向量库（如有企业文档）
python scripts/data_process.py

# 4. 启动 Gradio 界面
python app.py
```

访问 http://127.0.0.1:7860

---

## 核心功能

### 四层意图路由

| 意图 | 触发条件 | 处理工具 |
|------|----------|----------|
| **generic** | 打招呼、身份询问 | `generic_func` |
| **retrieval** | 寻医问药网企业信息 | `retrieval_func` |
| **kg** | 医疗实体 + 症状/药物/治疗关键词 | `kg_query_func` |
| **search** | 通用问题兜底 | `search_func` |

### 医疗知识图谱查询

支持 10 类模板查询：
- 疾病描述 (`desc`)
- 病因 (`cause`)
- 症状 (`symptom`)
- 药物 (`cure_way`)
- 治疗方法 (`cure_method`)
- 检查项目 (`check`)
- 就诊科室 (`department`)
- 治愈率 (`cured_prob`)
- 药物适应症 (`indications`)
- 预防措施 (`prevent`)

### 多轮对话补全

- **指代词替换**："它有什么症状" → "[鼻炎]有什么症状"
- **省略主语补全**："吃什么药好得快" → "[鼻炎]吃什么药好得快"
- **话题切换检测**：新实体出现时不补全

### 实体抽取策略

1. **Trie 精确匹配**：18,000+ 医疗实体毫秒级命中
2. **滑动窗口模糊搜索**："阿司匹林" → "阿司匹林肠溶片"
3. **LLM 兜底**：JSON 格式抽取 + 自动映射到图谱实体

### 搜索增强

- **多引擎 fallback**：DuckDuckGo → Bing → Baidu
- **反爬策略**：随机 UA + 随机延迟 + 验证码检测
- **结果去重**：跨引擎去重，标注来源

---

## 配置说明

### 日志配置

```bash
# 日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
LOG_LEVEL=INFO

# 日志文件路径
LOG_FILE=./logs/medagent.log
```

日志输出格式：
```
[2024-01-01 12:00:00] [INFO] [medagent.service] Intent classification: '鼻炎吃什么药' -> scores: {'kg': 100}
```

### 搜索配置

```python
# config.py
SEARCH_CONFIG = {
    "engines": ["duckduckgo", "bing", "baidu"],  # 优先级排序
    "timeout": 5.0,
    "max_results": 5,
    "delay_between_requests": 0.5,  # 反爬延迟
}
```

### 向量检索配置

```python
# 默认相似度阈值（Chroma 余弦相似度）
score_threshold = 0.45  # 比默认 0.6 更宽松，召回更多
```

---

## 常见问题

### Q: 实体抽取失败，提示"未识别到医疗实体"

A: 检查 `data/entity_dict.json` 是否存在。如缺失，删除后重启服务会自动从 Neo4j 重新加载。

### Q: 向量检索返回空结果

A: 确认 `data/db/` 目录有数据且 `collection count > 0`。如为空，重新运行 `scripts/data_process.py`。

### Q: 百度搜索失败

A: 百度反爬机制可能触发。系统会自动 fallback 到 DuckDuckGo/Bing，无需手动处理。

### Q: 多轮对话补全不准确

A: 检查 `_PRONOUNS` 和 `_KG_SHORT_PATTERNS` 是否覆盖目标句式，可在 `service.py` 中扩展。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | 阿里云 DashScope (qwen3.5-flash) |
| 知识图谱 | Neo4j 5.x + Cypher |
| 向量库 | Chroma + text-embedding-v2 |
| Agent 框架 | LangChain ReAct |
| 前端 | Gradio 4.x |
| 部署 | Docker + Docker Compose |
| 日志 | Python logging + 彩色控制台 |

---



## 致谢

- 寻医问药网提供医疗数据支持
- 阿里云 DashScope 提供模型 API
