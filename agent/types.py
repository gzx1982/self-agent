"""
Self Agent Framework - 类型定义
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """聊天消息"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List['ToolCall']] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """从字典创建"""
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])] if data.get("tool_calls") else None,
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ToolCall':
        """从字典创建"""
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
        )


@dataclass 
class ToolResult:
    """工具执行结果"""
    tool_call_id: str
    result: str
    is_error: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.result,
            "is_error": self.is_error,
        }


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    task: str
    response: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "task": self.task,
            "response": self.response,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            task=data.get("task", ""),
            response=data.get("response", ""),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str = "agent"
    model: str = "openai/gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    system_prompt: Optional[str] = None
    
    @classmethod
    def from_config(cls, config: 'Config') -> 'AgentConfig':
        """从配置对象创建"""
        return cls(
            name=config.get("agent.name", "agent"),
            model=config.get("agent.model", "openai/gpt-4"),
            max_tokens=config.get("agent.max_tokens", 4096),
            temperature=config.get("agent.temperature", 0.7),
            timeout=config.get("agent.timeout", 60),
            system_prompt=config.get("agent.system_prompt"),
        )


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    raw_response: Optional[Dict] = None
    usage: Optional[Dict] = None
    
    def to_message(self) -> Message:
        """转换为 Message 对象"""
        return Message(
            role=MessageRole.ASSISTANT,
            content=self.content,
            tool_calls=self.tool_calls,
        )
