"""LangChain ReAct Agent 定义"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from models import get_llm_model
from tools import generic_func, retrieval_func, kg_query_func, search_func
from config import TOOL_DESCRIPTIONS

# 定义 ReAct 提示词模板
AGENT_PROMPT = """尽你所能回答以下问题。你可以使用以下工具：

{tools}

请使用以下格式：

问题：你必须回答的输入问题
思考：你应该始终思考该怎么做
行动：要采取的行动，应该是 [{tool_names}] 之一
行动输入：行动的输入
观察：行动的结果
...（这个思考/行动/行动输入/观察可以重复 N 次）
思考：我现在知道最终答案了
最终答案：对原始问题的最终答案

开始！

问题：{input}
{agent_scratchpad}
"""

def get_agent():
    """创建 ReAct Agent"""
    tools = [generic_func, retrieval_func, kg_query_func, search_func]
    tool_names = [t.name for t in tools]

    prompt = PromptTemplate.from_template(AGENT_PROMPT).partial(
        tools="\n".join([f"{t.name}: {TOOL_DESCRIPTIONS.get(t.name, t.description)}" for t in tools]),
        tool_names=", ".join(tool_names),
    )

    llm = get_llm_model()
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=2,  # 限制迭代次数，防止无限循环
        max_execution_time=30,  # 30秒超时
        handle_parsing_errors=True,
    )