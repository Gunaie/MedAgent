"""ReAct Agent"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from models import get_llm_model
from tools import generic_func, retrieval_func, search_func, kg_query_func


def create_medical_agent(verbose: bool = False):
    """创建医疗问诊 Agent"""

    tools = [generic_func, retrieval_func, kg_query_func, search_func]

    # 优化 ReAct 提示词，明确告诉模型收到 Observation 后必须直接 Final Answer
    template = """\
请用中文回答问题！Final Answer 必须尊重 Observation 的结果，不能改变语义。
如果 Observation 已经包含答案，你必须立即输出 Final Answer，不要再次调用工具。

你有以下工具可以使用：
{tools}

使用以下格式：

Question: 你需要回答的问题
Thought: 你应该始终思考该做什么
Action: 要采取的行动，必须是以下之一 [{tool_names}]
Action Input: 行动的输入
Observation: 行动的结果
...（这个 Thought/Action/Action Input/Observation 可以重复 N 次）
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=get_llm_model(),
        tools=tools,
        prompt=prompt,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        verbose=verbose,
        max_iterations=2,        # 最多迭代 2 次（决策1次 + 总结1次）
        max_execution_time=20,   # 整个 Agent 执行不超过 20 秒
        early_stopping_method="generate",  # 超时或达到迭代上限时，让 LLM 直接生成答案
    )