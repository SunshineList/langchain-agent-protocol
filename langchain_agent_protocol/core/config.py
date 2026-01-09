# -*- coding: utf-8 -*-
"""
Agent Configuration - 配置管理

支持从字典、YAML、JSON、环境变量等多种来源加载配置
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class AgentConfig:
    """Agent 配置类 - 管理所有 Agent 相关配置"""
    
    # 系统提示词
    system_prompt: str = "你是一个智能助手,请帮助用户解决问题。"
    
    # 工具名称映射 (英文 -> 中文显示名)
    tool_name_map: Dict[str, str] = field(default_factory=dict)
    
    # LLM 参数
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Agent 行为配置
    max_iterations: int = 10  # 最大工具调用次数
    early_stopping_method: str = "force"  # 停止策略
    
    # 额外配置
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            AgentConfig 实例
        """
        # 提取已知字段
        known_fields = {
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }
        
        # 其余字段放入 extra_config
        extra = {
            k: v for k, v in config_dict.items()
            if k not in cls.__dataclass_fields__
        }
        
        if extra and 'extra_config' not in known_fields:
            known_fields['extra_config'] = extra
        
        return cls(**known_fields)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AgentConfig':
        """
        从 YAML 文件加载配置
        
        Args:
            yaml_path: YAML 文件路径
            
        Returns:
            AgentConfig 实例
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load from YAML. "
                "Install it with: pip install pyyaml"
            )
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'AgentConfig':
        """
        从 JSON 文件加载配置
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            AgentConfig 实例
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "AGENT_") -> 'AgentConfig':
        """
        从环境变量加载配置
        
        环境变量格式: {prefix}{FIELD_NAME}
        例如: AGENT_SYSTEM_PROMPT, AGENT_TEMPERATURE
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            AgentConfig 实例
        """
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                field_name = key[len(prefix):].lower()
                
                # 类型转换
                if field_name == 'temperature':
                    config_dict[field_name] = float(value)
                elif field_name == 'max_tokens':
                    config_dict[field_name] = int(value) if value else None
                elif field_name == 'tool_name_map':
                    config_dict[field_name] = json.loads(value)
                else:
                    config_dict[field_name] = value
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return asdict(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        保存为 YAML 文件
        
        Args:
            yaml_path: 输出文件路径
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    def to_json(self, json_path: str, indent: int = 2) -> None:
        """
        保存为 JSON 文件
        
        Args:
            json_path: 输出文件路径
            indent: 缩进空格数
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
    
    def get_tool_display_name(self, tool_name: str) -> str:
        """
        获取工具的显示名称
        
        Args:
            tool_name: 工具标识符
            
        Returns:
            工具的显示名称,如果未找到则返回原始名称
        """
        return self.tool_name_map.get(tool_name, tool_name)
