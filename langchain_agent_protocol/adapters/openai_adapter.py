# -*- coding: utf-8 -*-
"""
OpenAI Adapter - OpenAI API 适配器实现
"""

import os
import json
from typing import List, Dict, Any, Iterator, Optional

from ..core.base_llm_adapter import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI API 适配器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        初始化 OpenAI 适配器
        
        Args:
            api_key: OpenAI API Key (如果为 None,从环境变量读取)
            model: 模型名称
            base_url: API 基础 URL (可选,用于自定义端点)
            organization: 组织 ID (可选)
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install it with: "
                "pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it via api_key parameter or OPENAI_API_KEY env var"
            )
        
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        
        # 创建客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization
        )
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        同步聊天
        
        Args:
            messages: OpenAI 格式的消息列表
            tools: 工具列表
            **kwargs: 额外参数 (temperature, max_tokens 等)
            
        Returns:
            响应字典
        """
        try:
            # 准备请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            
            if tools:
                request_params["tools"] = tools
            
            # 调用 API
            response = self.client.chat.completions.create(**request_params)
            
            # 解析响应
            message = response.choices[0].message
            
            # 提取工具调用
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            return {
                "success": True,
                "response": message.content or "",
                "tool_calls": tool_calls,
                "metadata": {
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        流式聊天
        
        Args:
            messages: OpenAI 格式的消息列表
            tools: 工具列表
            **kwargs: 额外参数
            
        Yields:
            文本块或工具调用 JSON
        """
        try:
            # 准备请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            
            if tools:
                request_params["tools"] = tools
            
            # 流式调用
            stream = self.client.chat.completions.create(**request_params)
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 文本内容
                if delta.content:
                    yield delta.content
                
                # 工具调用
                elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls_data = [
                        {
                            "id": tc.id if hasattr(tc, 'id') else None,
                            "type": "function",
                            "function": {
                                "name": tc.function.name if hasattr(tc.function, 'name') else "",
                                "arguments": tc.function.arguments if hasattr(tc.function, 'arguments') else ""
                            }
                        }
                        for tc in delta.tool_calls
                    ]
                    yield json.dumps({"tool_calls_chunk": tool_calls_data}, ensure_ascii=False)
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model
    
    def validate_config(self) -> bool:
        """验证配置"""
        return bool(self.api_key and self.model)
    
    @classmethod
    def from_env(cls, model: Optional[str] = None) -> 'OpenAIAdapter':
        """
        从环境变量创建适配器
        
        环境变量:
            - OPENAI_API_KEY: API Key
            - OPENAI_MODEL: 模型名称 (默认 gpt-4)
            - OPENAI_BASE_URL: 自定义 API 端点 (可选)
            - OPENAI_ORGANIZATION: 组织 ID (可选)
        
        Args:
            model: 覆盖环境变量中的模型名称
            
        Returns:
            OpenAIAdapter 实例
        """
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            organization=os.getenv("OPENAI_ORGANIZATION")
        )
