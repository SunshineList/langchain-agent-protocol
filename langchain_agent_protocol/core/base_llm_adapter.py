# -*- coding: utf-8 -*-
"""
LLM Adapter Base Class - 定义统一的 LLM 调用协议

所有 LLM 提供商必须实现这个接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional


class BaseLLMAdapter(ABC):
    """LLM 适配器基类 - 定义统一的调用协议"""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        同步聊天接口
        
        Args:
            messages: OpenAI 格式的消息列表
                      [{"role": "user"|"assistant"|"system", "content": "..."}]
            tools: OpenAI 格式的工具列表 (可选)
            **kwargs: 额外参数 (temperature, max_tokens 等)
            
        Returns:
            Dict containing:
                - success: bool - 是否成功
                - response: str - AI 响应内容
                - tool_calls: Optional[List[Dict]] - 工具调用列表
                - metadata: Dict - 可选的元数据 (model, usage 等)
                - error: str - 错误信息 (仅当 success=False 时)
        
        Example:
            >>> adapter.chat([{"role": "user", "content": "Hello"}])
            {
                "success": True,
                "response": "Hi there!",
                "tool_calls": None,
                "metadata": {"model": "gpt-4", "usage": {...}}
            }
        """
        pass
    
    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        流式聊天接口
        
        Args:
            messages: OpenAI 格式的消息列表
            tools: OpenAI 格式的工具列表 (可选)
            **kwargs: 额外参数
            
        Yields:
            str: 文本块或包含工具调用的 JSON 字符串
                 - 普通文本: "Hello"
                 - 工具调用: '{"tool_calls_chunk": [...]}'
        
        Example:
            >>> for chunk in adapter.chat_stream([{"role": "user", "content": "Hi"}]):
            ...     print(chunk, end="", flush=True)
        """
        pass
    
    def get_model_name(self) -> str:
        """
        获取当前使用的模型名称
        
        Returns:
            模型名称字符串
        """
        return getattr(self, 'model', 'unknown')
    
    def validate_config(self) -> bool:
        """
        验证适配器配置是否有效
        
        Returns:
            是否配置有效
        """
        return True
