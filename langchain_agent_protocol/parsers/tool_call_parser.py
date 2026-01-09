# -*- coding: utf-8 -*-
"""
工具调用解析模块
统一处理工具调用的解析、验证和格式化
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ToolCallParser:
    """工具调用解析器 - 统一处理工具调用的解析、验证和格式化"""
    
    @staticmethod
    def extract_tool_calls_from_chunk(chunk: str) -> Optional[List[Dict[str, Any]]]:
        """
        从流式响应块中提取工具调用信息
        
        Args:
            chunk: 流式响应的文本块
            
        Returns:
            工具调用列表，如果不包含工具调用则返回 None
        """
        if not chunk.startswith('{') or '"tool_calls_chunk"' not in chunk:
            return None
        
        try:
            data = json.loads(chunk)
            return data.get("tool_calls_chunk")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call chunk: {e}")
            return None
    
    @staticmethod
    def format_tool_call_notification(
        tool_name: str,
        tool_name_map: Dict[str, str],
        template: Optional[str] = None
    ) -> str:
        """
        格式化工具调用通知消息
        
        Args:
            tool_name: 工具的英文标识符
            tool_name_map: 工具名称映射字典
            template: 自定义模板 (可选)
            
        Returns:
            格式化的通知消息
        """
        display_name = tool_name_map.get(tool_name, tool_name)
        
        if template:
            return template.format(tool_name=tool_name, display_name=display_name)
        
        # 默认模板 (Markdown 格式)
        return f"\n> [!NOTE]\n> **🔧 工具调用**: 正在执行 `{display_name}`...\n"
    
    @staticmethod
    def validate_tool_call(tool_call: Dict[str, Any]) -> bool:
        """
        验证工具调用数据的完整性
        
        Args:
            tool_call: 工具调用字典
            
        Returns:
            是否有效
        """
        if not isinstance(tool_call, dict):
            return False
        
        # 检查必需字段
        if "name" not in tool_call:
            logger.warning("Tool call missing 'name' field")
            return False
        
        # 检查 args 字段 (可选但应该是字典)
        if "args" in tool_call and not isinstance(tool_call.get("args"), dict):
            logger.warning(f"Tool call '{tool_call['name']}' has invalid args type")
            return False
        
        return True
    
    @staticmethod
    def extract_tool_calls_from_message(message: Any) -> List[Dict[str, Any]]:
        """
        从消息对象中提取工具调用列表
        
        Args:
            message: 消息对象 (通常是 AIMessage)
            
        Returns:
            工具调用列表
        """
        tool_calls = []
        
        # 优先使用 tool_calls 属性
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = message.tool_calls
        # 其次检查 additional_kwargs
        elif hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            tool_calls = message.additional_kwargs["tool_calls"]
        
        # 验证并过滤有效的工具调用
        validated_calls = [
            tc for tc in tool_calls
            if ToolCallParser.validate_tool_call(tc)
        ]
        
        if len(validated_calls) < len(tool_calls):
            logger.warning(
                f"Filtered out {len(tool_calls) - len(validated_calls)} "
                f"invalid tool calls"
            )
        
        return validated_calls
