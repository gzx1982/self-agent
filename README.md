# Self Agent Framework

一个基于配置文件的自定义 Agent 框架，支持连接多种大模型并自主执行任务。

## 项目结构

```
self-agents/
├── agent/                      # Agent 核心框架
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   ├── llm.py                 # LLM 抽象层
│   ├── loop.py                # Agent 执行循环
│   ├── memory.py              # 记忆系统
│   ├── tools.py               # 工具系统
│   └── types.py               # 类型定义
├── providers/                  # LLM Provider 实现
│   ├── __init__.py
│   ├── base.py                # Provider 基类
│   ├── openai_provider.py     # OpenAI 兼容 Provider
│   ├── anthropic_provider.py  # Anthropic Provider
│   ├── minimax_provider.py    # MiniMax Provider
│   └── ollama_provider.py     # Ollama 本地 Provider
├── config/
│   └── agent.yaml             # Agent 配置文件
├── main.py                    # 主入口
├── requirements.txt           # Python 依赖
└── README.md                  # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml openai anthropic requests
```

### 2. 配置

编辑 `config/agent.yaml`:

```yaml
# Agent 基本配置
agent:
  name: "my-agent"
  model: "openai/gpt-4"          # 默认模型
  max_tokens: 4096
  temperature: 0.7
  timeout: 60

# Provider 配置
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      - "gpt-4"
      - "gpt-3.5-turbo"
  
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - "claude-3-opus-20240229"
      - "claude-3-sonnet-20240229"
  
  minimax:
    api_key: "${MINIMAX_API_KEY}"
    base_url: "https://api.minimax.chat/v1"
    model: "MiniMax-Text-01"

  ollama:
    base_url: "http://localhost:11434"
    model: "llama3"
    # 无需 API key

# 工具配置
tools:
  enabled:
    - "web_search"      # 网页搜索
    - "file_read"        # 文件读取
    - "file_write"       # 文件写入
    - "exec"             # 命令执行
  
  web_search:
    engine: "duckduckgo"  # or "google", "bing"
  
  exec:
    allowed_commands:     # 允许执行的命令白名单
      - "ls"
      - "cat"
      - "grep"
      - "curl"
    max_output_size: 10240  # 最大输出字节

# 记忆配置
memory:
  type: "file"           # "file" 或 "sqlite"
  path: "./memory_db"
  max_items: 1000        # 最大记忆条数
  context_window: 10      # 发送给 LLM 的历史消息数
```

### 3. 运行

```bash
python main.py --config config/agent.yaml --model "openai/gpt-4"
```

或交互模式：

```bash
python main.py --interactive
```

### 4. API Key 配置

支持环境变量引用：

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"  # 自动从环境变量读取
```

或直接配置：

```yaml
providers:
  openai:
    api_key: "sk-xxxxyourkey"
```

---

## 核心设计

### 1. 配置管理 (config.py)

```python
class Config:
    """配置管理，支持 YAML 格式配置文件"""
    
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path))
    
    def get(self, key: str, default=None):
        """获取配置项，支持点号路径如 'providers.openai.api_key'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        return value
    
    def resolve_env_vars(self, value: str) -> str:
        """解析环境变量引用 ${VAR_NAME}"""
        import re
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            value = value.replace(f'${{{var_name}}}', os.getenv(var_name, ''))
        return value
```

### 2. LLM Provider 抽象 (llm.py)

```python
class LLMProvider(ABC):
    """LLM Provider 基类"""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """同步补全"""
        pass
    
    @abstractmethod
    async def complete_async(self, prompt: str, **kwargs) -> str:
        """异步补全"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """聊天补全"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容 Provider"""
    
    def __init__(self, config: Config):
        api_key = config.resolve_env_vars(config.get('providers.openai.api_key'))
        base_url = config.get('providers.openai.base_url', 'https://api.openai.com/v1')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get('model', 'gpt-4'),
            messages=messages,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 4096),
        )
        return response.choices[0].message.content
```

### 3. Agent 执行循环 (loop.py)

```python
class AgentLoop:
    """Agent 执行循环"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = self._create_llm_provider()
        self.memory = Memory(config)
        self.tools = Tools(config)
    
    def run(self, task: str) -> str:
        """执行单个任务"""
        # 1. 记忆检索
        context = self.memory.retrieve(task)
        
        # 2. 构建提示词
        prompt = self._build_prompt(task, context)
        
        # 3. 调用 LLM
        response = self.llm.chat(prompt)
        
        # 4. 处理工具调用
        while self._has_tool_calls(response):
            tool_results = self.tools.execute(response.tool_calls)
            response = self.llm.chat(
                prompt + [{"role": "assistant", "content": response},
                          {"role": "tool", "content": tool_results}]
            )
        
        # 5. 保存记忆
        self.memory.add(task, response)
        
        return response
    
    def _build_prompt(self, task: str, context: List[str]) -> str:
        """构建提示词"""
        system_prompt = f"""你是一个专业的 AI 助手，名叫 {self.config.get('agent.name')}。
你可以使用工具来完成任务。
可用工具: {self.tools.get_descriptions()}
"""
        context_str = "\n".join([f"相关记忆: {c}" for c in context])
        return f"{system_prompt}\n\n{context_str}\n\n任务: {task}"
```

### 4. 工具系统 (tools.py)

```python
class Tools:
    """工具管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled_tools = config.get('tools.enabled', [])
        self.tool_registry = {
            'web_search': self._web_search,
            'file_read': self._file_read,
            'file_write': self._file_write,
            'exec': self._exec,
        }
    
    def execute(self, tool_calls: List) -> List[Dict]:
        """执行工具调用"""
        results = []
        for call in tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            if tool_name in self.tool_registry:
                result = self.tool_registry[tool_name](**args)
                results.append({"tool": tool_name, "result": result})
        return results
    
    def _exec(self, command: str) -> str:
        """执行命令（受白名单限制）"""
        allowed = self.config.get('tools.exec.allowed_commands', [])
        cmd_parts = command.split()
        if not cmd_parts or cmd_parts[0] not in allowed:
            return f"Error: Command '{cmd_parts[0]}' not allowed"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout[:self.config.get('tools.exec.max_output_size', 10240)]
```

### 5. 记忆系统 (memory.py)

```python
class Memory:
    """记忆系统，支持文件存储和 SQLite"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_type = config.get('memory.type', 'file')
        self.max_items = config.get('memory.max_items', 1000)
        self.context_window = config.get('memory.context_window', 10)
        self.storage = self._create_storage()
    
    def add(self, task: str, response: str):
        """添加记忆"""
        entry = {
            "task": task,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.storage.append(entry)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """检索相关记忆（简单关键词匹配）"""
        all_entries = self.storage.get_all()
        # 简单实现：按关键词重叠度排序
        scored = []
        query_words = set(query.lower().split())
        for entry in all_entries[-100:]:  # 只检查最近100条
            entry_words = set(entry['task'].lower().split())
            score = len(query_words & entry_words)
            scored.append((score, entry['response']))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for _, s in scored[:top_k]]
```

---

## 使用示例

### 基础对话

```python
from agent import AgentLoop, Config

config = Config("config/agent.yaml")
agent = AgentLoop(config)

response = agent.run("帮我写一个 Hello World 程序")
print(response)
```

### 带工具调用的任务

```python
response = agent.run("帮我查找今天的天气情况，然后写入 weather.txt")
# Agent 会自动调用 web_search 工具，然后调用 file_write 工具
```

### 异步执行

```python
import asyncio
from agent import AgentLoop, Config

async def main():
    config = Config("config/agent.yaml")
    agent = AgentLoop(config)
    
    tasks = [
        agent.run_async("任务1"),
        agent.run_async("任务2"),
        agent.run_async("任务3"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

asyncio.run(main())
```

---

## 支持的 Provider

| Provider | 模型示例 | 认证方式 |
|----------|----------|----------|
| OpenAI 兼容 | gpt-4, gpt-3.5-turbo | API Key |
| Anthropic | claude-3-opus, claude-3-sonnet | API Key |
| MiniMax | MiniMax-Text-01 | API Key |
| Ollama | llama3, mistral | 本地无需 Key |
| vLLM | 各种开源模型 | 本地无需 Key |
| Azure OpenAI | gpt-4 | Azure AD |

---

## 扩展开发

### 添加新的 Provider

```python
from providers.base import LLMProvider

class MyProvider(LLMProvider):
    def __init__(self, config: Config):
        # 初始化逻辑
        pass
    
    def complete(self, prompt: str, **kwargs) -> str:
        # 实现补全逻辑
        pass
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        # 实现聊天逻辑
        pass
```

### 添加新的工具

```python
from agent.tools import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "执行特定任务"
    
    def execute(self, **kwargs) -> str:
        # 工具逻辑
        return result

# 注册工具
agent.tools.register(MyTool())
```

---

*生成时间: 2026-03-26*