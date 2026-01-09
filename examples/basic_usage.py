#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础使用示例 - 最简单的 Agent 使用方式
"""

from langchain_agent_protocol import UniversalAgent
from langchain_core.tools import tool


# 定义一个简单的工具
@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def main():
    print("=" * 60)
    print("LangChain Agent Protocol - 基础使用示例")
    print("=" * 60)
    
    # 方式1: 最简单的快速启动
    print("\n[方式1] 快速启动 (无工具)")
    agent = UniversalAgent.quick_start(
        api_key="your-api-key-here",  # 替换为你的 API Key
        model="gpt-4",
        system_prompt="你是一个友好的助手"
    )
    
    response = agent.run("你好,介绍一下自己")
    print(f"回复: {response}\n")
    
    # 方式2: 带工具的完整示例
    print("\n[方式2] 带工具的Agent")
    from langchain_agent_protocol import AgentConfig
    from langchain_agent_protocol.adapters import OpenAIAdapter
    
    # 配置
    config = AgentConfig(
        system_prompt="你是一个智能助手,可以查询时间和进行数学计算",
        tool_name_map={
            "get_current_time": "时间查询",
            "calculate": "数学计算器"
        },
        temperature=0.7
    )
    
    # 创建Agent
    agent_with_tools = UniversalAgent(
        llm_adapter=OpenAIAdapter(api_key="your-api-key-here"),
        tools=[get_current_time, calculate],
        config=config
    )
    
    # 同步调用
    print("问题: 现在几点了?")
    response = agent_with_tools.run("现在几点了?")
    print(f"回复: {response}\n")
    
    print("问题: 计算 123 + 456")
    response = agent_with_tools.run("帮我算一下 123 加 456 等于多少")
    print(f"回复: {response}\n")
    
    # 流式调用
    print("\n[方式3] 流式输出")
    print("问题: 先告诉我现在时间,然后帮我算 100 乘以 50")
    print("回复: ", end="", flush=True)
    for chunk in agent_with_tools.run_stream("先告诉我现在时间,然后帮我算 100 乘以 50"):
        print(chunk, end="", flush=True)
    print("\n")
    
    print("=" * 60)
    print("示例结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
