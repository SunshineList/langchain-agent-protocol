# -*- coding: utf-8 -*-
"""
Universal Agent - 通用框架无关的 LangChain Agent

这是核心 Agent 类,提供统一的接口用于跨项目移植
"""

import logging
from typing import List, Any, Generator, Optional, Dict
from langchain.agents import create_agent

from .base_llm_adapter import BaseLLMAdapter
from .config import AgentConfig
from ..converters.message_converter import MessageConverter
from ..processors.stream_processor import StreamEventProcessor

logger = logging.getLogger(__name__)


class ChatLLMBridge:
    """
    LLM 适配器桥接器 - 将我们的 BaseLLMAdapter 转换为 LangChain 兼容的聊天模型
    """
    
    def __init__(self, llm_adapter: BaseLLMAdapter, config: AgentConfig):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
        
        self.llm_adapter = llm_adapter
        self.config = config
        self.message_converter = MessageConverter()
        self.bound_tools = []
        
        # 动态创建 BaseChatModel 子类
        class DynamicChatModel(BaseChatModel):
            """动态创建的聊天模型"""
            
            adapter = llm_adapter
            converter = MessageConverter()
            tools = []
            
            def bind_tools(inner_self, tools: List[Any], **kwargs) -> 'DynamicChatModel':
                """绑定工具"""
                from langchain_core.utils.function_calling import convert_to_openai_tool
                inner_self.tools = [convert_to_openai_tool(t) for t in tools]
                return inner_self
            
            def _generate(inner_self, messages, stop=None, run_manager=None, **kwargs):
                """同步生成"""
                formatted = inner_self.converter.langchain_to_openai(messages)
                tools = kwargs.get("tools") or inner_self.tools
                
                result = inner_self.adapter.chat(
                    formatted,
                    tools=tools if tools else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **kwargs
                )
                
                if not result.get("success"):
                    raise Exception(f"LLM Error: {result.get('error')}")
                
                ai_message = inner_self.converter.openai_to_langchain(
                    content=result.get("response", ""),
                    tool_calls=result.get("tool_calls")
                )
                
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            
            def _stream(inner_self, messages, stop=None, run_manager=None, **kwargs):
                """流式生成"""
                from ..parsers.tool_call_parser import ToolCallParser
                
                formatted = inner_self.converter.langchain_to_openai(messages)
                tools = kwargs.get("tools") or inner_self.tools
                
                for chunk in inner_self.adapter.chat_stream(
                    formatted,
                    tools=tools if tools else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **kwargs
                ):
                    tool_calls = ToolCallParser.extract_tool_calls_from_chunk(chunk)
                    
                    if tool_calls:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                additional_kwargs={"tool_calls": tool_calls},
                                tool_call_chunks=tool_calls
                            )
                        )
                    else:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            
            @property
            def _llm_type(inner_self) -> str:
                return "universal_agent_bridge"
        
        self.chat_model = DynamicChatModel()


class UniversalAgent:
    """
    通用 LangChain Agent - 框架无关,跨项目可移植
    
    这个类提供了统一的接口,可以在任何 Python 项目中使用,
    无论是 Django、Flask、FastAPI 还是纯 Python 脚本
    """
    
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        tools: List[Any],
        config: AgentConfig
    ):
        """
        初始化 Universal Agent
        
        Args:
            llm_adapter: LLM 适配器实例
            tools: LangChain 工具列表
            config: Agent 配置
        """
        self.llm_adapter = llm_adapter
        self.tools = tools
        self.config = config
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format=config.log_format
        )
        
        # 初始化组件
        self.stream_processor = StreamEventProcessor(
            tool_name_map=config.tool_name_map
        )
        
        self._agent = None
        
        logger.info("UniversalAgent initialized successfully")
    
    def _get_agent(self):
        """获取或创建 Agent 实例 (延迟初始化)"""
        if self._agent is None:
            # 创建桥接的聊天模型
            bridge = ChatLLMBridge(self.llm_adapter, self.config)
            
            # 创建 Agent
            self._agent = create_agent(
                model=bridge.chat_model,
                tools=self.tools,
                system_prompt=self.config.system_prompt
            )
            
            logger.info("LangChain Agent created successfully")
        
        return self._agent
    
    def run(self, message: str, **kwargs: Any) -> str:
        """
        同步运行 Agent
        
        Args:
            message: 用户输入消息
            **kwargs: 额外参数
            
        Returns:
            Agent 的最终响应
        """
        agent = self._get_agent()
        
        try:
            logger.info(f"Running agent with message: {message[:50]}...")
            
            response = agent.invoke({
                "messages": [{"role": "user", "content": message}]
            })
            
            # 提取最后一条消息
            if "messages" in response:
                last_msg = response["messages"][-1]
                return last_msg.content
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            return f"抱歉,执行过程中出现错误: {str(e)}"
    
    def run_stream(self, message: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        流式运行 Agent
        
        Args:
            message: 用户输入消息
            **kwargs: 额外参数
            
        Yields:
            Agent 的流式输出 (包括工具调用通知和响应内容)
        """
        agent = self._get_agent()
        
        try:
            logger.info(f"Running agent stream with message: {message[:50]}...")
            
            for event in agent.stream({
                "messages": [{"role": "user", "content": message}]
            }):
                yield from self.stream_processor.process_event(event)
                
        except Exception as e:
            yield self.stream_processor.process_error(e)
    
    @classmethod
    def quick_start(
        cls,
        api_key: str,
        model: str = "gpt-4",
        tools: Optional[List[Any]] = None,
        system_prompt: str = "你是一个智能助手。",
        **config_kwargs
    ) -> 'UniversalAgent':
        """
        快速启动方法 - 最简单的使用方式
        
        Args:
            api_key: OpenAI API Key
            model: 模型名称
            tools: 工具列表
            system_prompt: 系统提示词
            **config_kwargs: 额外配置
            
        Returns:
            UniversalAgent 实例
        """
        from ..adapters.openai_adapter import OpenAIAdapter
        
        llm_adapter = OpenAIAdapter(api_key=api_key, model=model)
        config = AgentConfig(system_prompt=system_prompt, **config_kwargs)
        
        return cls(
            llm_adapter=llm_adapter,
            tools=tools or [],
            config=config
        )
