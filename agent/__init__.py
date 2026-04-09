"""
Self Agent Framework - 核心模块
"""

from .config import Config, load_config
from .types import Message, MessageRole, ToolCall, ToolResult, MemoryEntry, AgentConfig, LLMResponse
from .llm import LLMProvider, ProviderFactory, create_llm
from .tools import BaseTool, Tools, create_tools
from .memory import Memory, create_memory
from .skill import Skill, SkillManager, create_skill_manager
from .loop import AgentLoop, MultiAgent, create_agent, run_agent_async

__version__ = "0.1.0"
__all__ = [
    'Config',
    'load_config',
    'Message',
    'MessageRole',
    'ToolCall',
    'ToolResult',
    'MemoryEntry',
    'AgentConfig',
    'LLMResponse',
    'LLMProvider',
    'ProviderFactory',
    'create_llm',
    'BaseTool',
    'Tools',
    'create_tools',
    'Memory',
    'create_memory',
    'Skill',
    'SkillManager',
    'create_skill_manager',
    'AgentLoop',
    'MultiAgent',
    'create_agent',
    'run_agent_async',
]
