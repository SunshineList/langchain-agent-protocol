# -*- coding: utf-8 -*-
"""
流式事件处理模块
解析和格式化 LangChain Agent 流式输出事件
"""

import logging
from typing import Dict, Any, List, Generator

logger = logging.getLogger(__name__)


class StreamEventProcessor:
    """流式事件处理器 - 解析和格式化 LangChain Agent 流式输出事件"""
    
    def __init__(self, tool_name_map: Dict[str, str]):
        """
        初始化流式事件处理器
        
        Args:
            tool_name_map: 工具名称映射字典
        """
        from ..parsers.tool_call_parser import ToolCallParser
        
        self.tool_name_map = tool_name_map
        self.tool_parser = ToolCallParser()
    
    def process_event(self, event: Dict[str, Any]) -> Generator[str, None, None]:
        """
        处理单个流式事件
        
        Args:
            event: LangChain Agent 返回的事件字典
            
        Yields:
            格式化的输出文本
        """
        # 事件结构: {node_name: node_output, ...}
        # node_output 可能包含 "messages" 键
        
        for node_name, node_output in event.items():
            # 情况1: node_output 是包含 messages 的字典
            if isinstance(node_output, dict) and "messages" in node_output:
                yield from self._process_messages(node_output["messages"])
            
            # 情况2: node_name 本身就是 "messages" (全局 values 模式)
            elif node_name == "messages" and isinstance(node_output, list):
                yield from self._process_messages(node_output)
    
    def _process_messages(self, messages: List[Any]) -> Generator[str, None, None]:
        """
        处理消息列表
        
        Args:
            messages: 消息列表
            
        Yields:
            格式化的输出文本
        """
        from langchain_core.messages import AIMessage
        
        if not messages:
            return
        
        # 只处理最后一条消息 (最新的状态)
        last_msg = messages[-1]
        
        if not isinstance(last_msg, AIMessage):
            return
        
        # 优先处理工具调用
        tool_calls = self.tool_parser.extract_tool_calls_from_message(last_msg)
        
        if tool_calls:
            # 生成工具调用通知
            for tc in tool_calls:
                tool_name = tc.get("name")
                if tool_name:
                    notification = self.tool_parser.format_tool_call_notification(
                        tool_name,
                        self.tool_name_map
                    )
                    yield notification
        
        # 处理文本内容 (如果没有工具调用或工具调用之后的响应)
        elif last_msg.content:
            yield last_msg.content
    
    def process_error(self, error: Exception) -> str:
        """
        处理错误并格式化错误消息
        
        Args:
            error: 异常对象
            
        Returns:
            格式化的错误消息
        """
        logger.error(f"Stream processing error: {error}", exc_info=True)
        return f"❌ 执行过程中出现错误: {str(error)}"
