# LangChain Agent Protocol

一个通用、框架无关的 LangChain Agent 封装库，支持跨项目无缝移植。

## 特性

✨ **框架无关** - Django / Flask / FastAPI / 纯Python脚本都能用  
🔌 **插件化设计** - LLM适配器、配置、工具都可插拔  
📦 **零移植成本** - 配置和工具定义直接复制  
🚀 **生产就绪** - 内置日志、错误处理、性能监控  
🎯 **类型安全** - 完整的类型注解

## 快速开始

### 安装

```bash
pip install langchain-agent-protocol
```

### 最简单的使用

```python
from langchain_agent_protocol import UniversalAgent

# 一行代码创建 Agent
agent = UniversalAgent.quick_start(
    api_key="sk-xxx",
    model="gpt-4",
    system_prompt="你是一个智能助手"
)

# 使用
response = agent.run("你好")
print(response)

# 流式输出
for chunk in agent.run_stream("介绍一下自己"):
    print(chunk, end="", flush=True)
```

### 完整示例 (带工具)

```python
from langchain_agent_protocol import UniversalAgent, AgentConfig
from langchain_agent_protocol.adapters import OpenAIAdapter
from langchain_core.tools import tool

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city} 的天气是晴天"

@tool
def search_web(query: str) -> str:
    """搜索网页"""
    return f"搜索结果: {query}"

# 2. 配置
config = AgentConfig(
    system_prompt="你是一个智能助手,可以查天气和搜索",
    tool_name_map={
        "get_weather": "天气查询",
        "search_web": "网页搜索"
    },
    temperature=0.7
)

# 3. 创建适配器
llm_adapter = OpenAIAdapter(
    api_key="your-api-key",
    model="gpt-4"
)

# 4. 初始化 Agent
agent = UniversalAgent(
    llm_adapter=llm_adapter,
    tools=[get_weather, search_web],
    config=config
)

# 5. 使用
response = agent.run("北京今天天气怎么样?")
print(response)
```

## 配置管理

### 从 YAML 加载

```bash
# config.yaml
system_prompt: |
  你是一个专业的AI助手

tool_name_map:
  search_db: "数据库搜索"
  send_email: "发送邮件"

temperature: 0.7
max_tokens: 2000
```

```python
from langchain_agent_protocol import AgentConfig

config = AgentConfig.from_yaml('config.yaml')
```

### 从环境变量加载

```bash
export AGENT_SYSTEM_PROMPT="你是助手"
export AGENT_TEMPERATURE="0.7"
```

```python
config = AgentConfig.from_env()
```

## 框架集成示例

### Django 项目

```python
# myapp/ai_service.py
from langchain_agent_protocol import UniversalAgent, AgentConfig
from langchain_agent_protocol.adapters import OpenAIAdapter
from .tools import business_tools

# 使用 Django settings
from django.conf import settings

agent = UniversalAgent(
    llm_adapter=OpenAIAdapter(api_key=settings.OPENAI_API_KEY),
    tools=business_tools,
    config=AgentConfig.from_yaml('config/agent_config.yaml')
)

# views.py
from django.http import JsonResponse
from .ai_service import agent

def chat_api(request):
    message = request.POST.get('message')
    response = agent.run(message)
    return JsonResponse({'response': response})
```

### FastAPI 项目

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_agent_protocol import UniversalAgent
from langchain_agent_protocol.adapters import OpenAIAdapter

app = FastAPI()

# 初始化
agent = UniversalAgent.quick_start(
    api_key="sk-xxx",
    tools=my_tools,
    system_prompt="你是助手"
)

@app.post("/chat")
async def chat(message: str):
    result = agent.run(message)
    return {"response": result}

@app.get("/chat/stream")
async def chat_stream(message: str):
    return StreamingResponse(
        agent.run_stream(message),
        media_type="text/plain"
    )
```

## 自定义 LLM 适配器

```python
from langchain_agent_protocol.core import BaseLLMAdapter

class MyCustomAdapter(BaseLLMAdapter):
    def __init__(self, my_service):
        self.service = my_service
    
    def chat(self, messages, tools=None, **kwargs):
        # 调用你的自定义服务
        return self.service.call(messages, tools)
    
    def chat_stream(self, messages, tools=None, **kwargs):
        # 流式调用
        for chunk in self.service.stream(messages, tools):
            yield chunk

# 使用
agent = UniversalAgent(
    llm_adapter=MyCustomAdapter(your_service),
    tools=tools,
    config=config
)
```

## API 文档

### UniversalAgent

#### `__init__(llm_adapter, tools, config)`
创建 Agent 实例

#### `run(message: str) -> str`
同步执行

#### `run_stream(message: str) -> Generator[str]`
流式执行

#### `quick_start(api_key, model, tools, system_prompt, **kwargs)`
快速创建 Agent 的类方法

### AgentConfig

#### 从不同来源加载
- `from_dict(config_dict)` - 从字典
- `from_yaml(yaml_path)` - 从 YAML 文件
- `from_json(json_path)` - 从 JSON 文件  
- `from_env(prefix)` - 从环境变量

## 项目迁移

### 1. 导出配置

```python
# 旧项目
old_config = {
    'system_prompt': '...',
    'tool_name_map': {...}
}

# 保存
config = AgentConfig.from_dict(old_config)
config.to_yaml('agent_config.yaml')
```

### 2. 复制工具定义

工具定义直接复制，无需修改:

```python
# tools.py - 直接复制
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    return "result"
```

### 3. 新项目中使用

```python
from langchain_agent_protocol import UniversalAgent
from langchain_agent_protocol.adapters import OpenAIAdapter
from .tools import my_tools  # 直接导入

agent = UniversalAgent(
    llm_adapter=OpenAIAdapter.from_env(),
    tools=my_tools,
    config=AgentConfig.from_yaml('agent_config.yaml')
)
```

## 开发

```bash
# 克隆项目
git clone https://github.com/SunshineList/langchain-agent-protocol
cd langchain-agent-protocol

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black .
isort .
```

## 许可证

MIT License

## 贡献

欢迎 PR 和 Issues!
