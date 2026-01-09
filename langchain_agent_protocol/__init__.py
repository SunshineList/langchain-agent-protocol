# -*- coding: utf-8 -*-
"""
LangChain Agent Protocol - 通用框架无关的 Agent 封装库

这个库提供了一套标准化的接口和实现,用于在不同项目间移植 LangChain Agent。
"""

__version__ = "0.1.4"
__author__ = "Tuple"
__email__ = "wzw571603974@qq.com"

from .core.agent import UniversalAgent
from .core.config import AgentConfig
from .core.base_llm_adapter import BaseLLMAdapter
from .converters.message_converter import MessageConverter
from .parsers.tool_call_parser import ToolCallParser
from .processors.stream_processor import StreamEventProcessor

__all__ = [
    "UniversalAgent",
    "AgentConfig",
    "BaseLLMAdapter",
    "MessageConverter",
    "ToolCallParser",
    "StreamEventProcessor",
]
