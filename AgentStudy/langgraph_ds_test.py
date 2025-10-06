import os

from typing import Annotated
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"  # 替换为您自己的 Tavily API 密钥


# 初始化状态图
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
graph_builder = StateGraph(State)


# 初始化 DeepSeek 大模型客户端
llm = ChatDeepSeek(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key="your_deepseek_api_key"  # 替换为您自己的 DeepSeek API 密钥
)

# 初始化 Tavily 搜索工具
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    # 调用 DeepSeek 大模型生成回复
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 添加节点到状态图
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
# 添加条件边
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")

# 使用内存保存器保存状态
memory = InMemorySaver()
 
# 编译状态图
graph = graph_builder.compile(checkpointer=memory)
 
def stream_graph_updates(user_input: str):
    # 构造对话历史，包含系统提示词和用户输入
    messages = [
        {"role": "system", "content": "你是一个有创意的助手，擅长根据用户问题提供有趣且相关的内容。输出内容长度不超过100个字。"},
        {"role": "user", "content": user_input},
    ]
    config = {"configurable": {"thread_id": "1"}}
    # 流式输出模型生成的回复
    for event in graph.stream({"messages": messages}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
 
while True:
    try:
        # 获取用户输入
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # 调用流式更新函数
        stream_graph_updates(user_input)
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"An error occurred: {e}")
        break