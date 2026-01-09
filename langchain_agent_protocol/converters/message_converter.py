# -*-coding: utf-8 -*-
"""
消息格式转换模块
统一处理 LangChain 和 OpenAI 消息格式互转
"""

import json
from typing import List, Dict, Any, Optional

# 这里我们不直接导入 langchain,而是定义消息结构
# 这样可以减少依赖,提高灵活性

class MessageConverter:
    """消息格式转换器 - 统一处理 LangChain 和 OpenAI 消息格式互转"""
    
    @staticmethod
    def langchain_to_openai(messages: List[Any]) -> List[Dict[str, Any]]:
        """
        将 LangChain 消息列表转换为 OpenAI 格式
        
        Args:
            messages: LangChain 消息列表
            
        Returns:
            OpenAI 格式的消息列表
        """
        from langchain_core.messages import (
            BaseMessage,
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )
        
        formatted = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({
                    "role": "system",
                    "content": msg.content
                })
                
            elif isinstance(msg, HumanMessage):
                formatted.append({
                    "role": "user",
                    "content": msg.content
                })
                
            elif isinstance(msg, AIMessage):
                ai_msg = {
                    "role": "assistant",
                    "content": msg.content
                }
                
                # 处理工具调用 - 优先使用 tool_calls 属性
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    ai_msg["tool_calls"] = MessageConverter._convert_lc_tool_calls_to_openai(
                        msg.tool_calls
                    )
                elif hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                    ai_msg["tool_calls"] = msg.additional_kwargs["tool_calls"]
                    
                formatted.append(ai_msg)
                
            elif isinstance(msg, ToolMessage):
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
                
        return formatted
    
    @staticmethod
    def _convert_lc_tool_calls_to_openai(
        lc_tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将 LangChain 格式的 tool_calls 转换为 OpenAI 格式
        
        Args:
            lc_tool_calls: LangChain 格式的工具调用列表
            
        Returns:
            OpenAI 格式的工具调用列表
        """
        openai_tool_calls = []
        
        for tc in lc_tool_calls:
            openai_tool_calls.append({
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": (
                        json.dumps(tc["args"], ensure_ascii=False)
                        if isinstance(tc["args"], dict)
                        else tc["args"]
                    )
                }
            })
            
        return openai_tool_calls
    
    @staticmethod
    def openai_to_langchain(
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """
        将 OpenAI 格式的响应转换为 LangChain AIMessage
        
        Args:
            content: AI 响应内容
            tool_calls: OpenAI 格式的工具调用列表
            
        Returns:
            LangChain AIMessage 对象
        """
        from langchain_core.messages import AIMessage
        
        additional_kwargs = {}
        lc_tool_calls = []
        
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
            lc_tool_calls = MessageConverter._convert_openai_tool_calls_to_lc(
                tool_calls
            )
        
        return AIMessage(
            content=content or "",
            additional_kwargs=additional_kwargs,
            tool_calls=lc_tool_calls
        )
    
    @staticmethod
    def _convert_openai_tool_calls_to_lc(
        openai_tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将 OpenAI 格式的 tool_calls 转换为 LangChain 内部格式
        
        Args:
            openai_tool_calls: OpenAI 格式的工具调用列表
            
        Returns:
            LangChain 格式的工具调用列表
        """
        lc_tool_calls = []
        
        for tc in openai_tool_calls:
            if "function" not in tc:
                continue
                
            func = tc["function"]
            
            # 安全解析 JSON 参数
            try:
                if isinstance(func.get("arguments"), str):
                    args = json.loads(func["arguments"])
                else:
                    args = func.get("arguments", {})
            except (json.JSONDecodeError, KeyError):
                args = {}
            
            lc_tool_calls.append({
                "name": func["name"],
                "args": args or {},
                "id": tc.get("id"),
                "type": "tool_call"
            })
            
        return lc_tool_calls
