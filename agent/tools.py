"""
Self Agent Framework - 工具系统

提供内置工具和自定义工具注册机制
"""

import os
import re
import json
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional, Type
from dataclasses import dataclass, field

from .types import ToolDefinition, ToolResult, ToolCall
from .config import Config

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """工具基类"""
    
    name: str = ""  # 工具名称
    description: str = ""  # 工具描述
    parameters: Dict[str, Any] = {}  # JSON Schema 格式参数定义
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """获取工具定义"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class WebSearchTool(BaseTool):
    """网页搜索工具"""
    
    name = "web_search"
    description = "搜索互联网获取信息。输入搜索关键词，返回搜索结果摘要。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "num_results": {
                "type": "integer",
                "description": "返回结果数量，默认5",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = config.get('tools.web_search.engine', 'duckduckgo')
    
    def execute(self, query: str, num_results: int = 5, **kwargs) -> str:
        """执行搜索"""
        try:
            if self.engine == 'duckduckgo':
                return self._duckduckgo_search(query, num_results)
            elif self.engine == 'google':
                return self._google_search(query, num_results)
            elif self.engine == 'bing':
                return self._bing_search(query, num_results)
            else:
                return f"Unknown search engine: {self.engine}"
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def _duckduckgo_search(self, query: str, num_results: int) -> str:
        """DuckDuckGo 搜索"""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return "No results found."
            
            output = []
            for i, r in enumerate(results, 1):
                output.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body'][:200]}...")
            return "\n\n".join(output)
        except ImportError:
            return "duckduckgo_search package not installed. Run: pip install duckduckgo-search"
    
    def _google_search(self, query: str, num_results: int) -> str:
        """Google 搜索（需要配置 API）"""
        # TODO: 实现 Google 搜索
        return "Google search not implemented yet. Please configure DuckDuckGo or Bing."
    
    def _bing_search(self, query: str, num_results: int) -> str:
        """Bing 搜索"""
        # TODO: 实现 Bing 搜索
        return "Bing search not implemented yet."


class FileReadTool(BaseTool):
    """文件读取工具"""
    
    name = "file_read"
    description = "读取文件内容。输入文件路径，返回文件内容。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（可选）",
            },
            "limit": {
                "type": "integer",
                "description": "读取行数限制（可选）",
            },
        },
        "required": ["path"],
    }
    
    def execute(self, path: str, offset: int = 0, limit: int = None, **kwargs) -> str:
        """读取文件"""
        try:
            if not os.path.exists(path):
                return f"File not found: {path}"
            
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if offset > 0:
                lines = lines[offset:]
            if limit:
                lines = lines[:limit]
            
            content = ''.join(lines)
            return content
        except Exception as e:
            return f"Failed to read file: {str(e)}"


class FileWriteTool(BaseTool):
    """文件写入工具"""
    
    name = "file_write"
    description = "写入内容到文件。如果文件存在则覆盖，不存在则创建。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "content": {
                "type": "string",
                "description": "要写入的内容",
            },
            "append": {
                "type": "boolean",
                "description": "是否追加模式，默认覆盖",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }
    
    def execute(self, path: str, content: str, append: bool = False, **kwargs) -> str:
        """写入文件"""
        try:
            # 确保目录存在
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote to {len(content)} characters to {path}"
        except Exception as e:
            return f"Failed to write file: {str(e)}"


class ExecTool(BaseTool):
    """命令执行工具"""
    
    name = "exec"
    description = "在本地执行Shell命令并返回输出。用于文件操作、系统命令等。"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令",
            },
            "timeout": {
                "type": "integer",
                "description": "超时时间（秒）",
                "default": 30,
            },
        },
        "required": ["command"],
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.allowed_commands = config.get('tools.exec.allowed_commands', ['ls', 'cat', 'grep', 'echo', 'pwd', 'cd'])
        self.max_output_size = config.get('tools.exec.max_output_size', 10240)
    
    def execute(self, command: str, timeout: int = 30, **kwargs) -> str:
        """执行命令"""
        try:
            # 安全检查：验证命令
            if not self._is_command_allowed(command):
                return f"Error: Command not in whitelist. Allowed: {', '.join(self.allowed_commands)}"
            
            # 执行命令
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR: " + result.stderr
            
            # 限制输出长度
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + f"\n... (output truncated, total {len(output)} bytes)"
            
            return output
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Command execution failed: {str(e)}"
    
    def _is_command_allowed(self, command: str) -> bool:
        """检查命令是否在白名单中"""
        # 提取第一个命令
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
        
        base_cmd = cmd_parts[0]
        return base_cmd in self.allowed_commands


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name = "calculator"
    description = "执行数学计算。输入数学表达式，返回计算结果。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 2' 或 'sqrt(16)'",
            },
        },
        "required": ["expression"],
    }
    
    def execute(self, expression: str, **kwargs) -> str:
        """执行计算"""
        try:
            # 使用 eval 进行计算（受限制）
            # 移除所有非安全字符
            safe_chars = set('0123456789+-*/(). ,%')
            if not all(c in safe_chars for c in expression.replace(' ', '')):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"


class Tools:
    """工具管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tools: Dict[str, BaseTool] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        # 工具配置
        enabled = self.config.get('tools.enabled', [])
        
        if 'web_search' in enabled:
            self.register(WebSearchTool(self.config))
        
        if 'file_read' in enabled:
            self.register(FileReadTool())
        
        if 'file_write' in enabled:
            self.register(FileWriteTool())
        
        if 'exec' in enabled:
            self.register(ExecTool(self.config))
        
        if 'calculator' in enabled:
            self.register(CalculatorTool())
    
    def register(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name)
    
    def execute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """执行工具调用列表"""
        results = []
        for call in tool_calls:
            result = self.execute_one(call)
            results.append(result)
        return results
    
    def execute_one(self, tool_call: ToolCall) -> ToolResult:
        """执行单个工具调用"""
        tool = self.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                result=f"Tool not found: {tool_call.name}",
                is_error=True,
            )
        
        try:
            result = tool.execute(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                result=str(result),
                is_error=False,
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                result=f"Tool execution failed: {str(e)}",
                is_error=True,
            )
    
    def get_definitions(self) -> List[ToolDefinition]:
        """获取所有工具定义"""
        return [tool.get_definition() for tool in self.tools.values()]
    
    def get_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self.tools.keys())
    
    def get_descriptions(self) -> str:
        """获取工具描述"""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)


# 便捷函数
def create_tools(config: Config) -> Tools:
    """创建工具管理器"""
    return Tools(config)
