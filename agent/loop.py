"""
Self Agent Framework - Agent 执行循环

核心 Agent 循环实现
"""

import os
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

from .config import Config
from .llm import create_llm, LLMProvider
from .types import Message, MessageRole, LLMResponse, ToolCall, ToolResult
from .memory import create_memory, Memory
from .tools import create_tools, Tools

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Agent 执行循环
    
    核心流程:
    1. 接收任务
    2. 记忆检索获取上下文
    3. 构建提示词
    4. 调用 LLM
    5. 处理工具调用
    6. 保存记忆
    7. 返回结果
    """
    
    def __init__(self, config: Config, model: Optional[str] = None):
        """
        初始化 Agent
        
        Args:
            config: 配置对象
            model: 指定模型（可选）
        """
        self.config = config
        self.model = model or config.get('agent.model', 'openai/gpt-4')
        
        # 初始化 LLM
        self.llm = create_llm(config, self.model)
        
        # 初始化记忆系统
        self.memory = create_memory(config)
        
        # 初始化工具系统
        self.tools = create_tools(config)
        
        # Agent 配置
        self.agent_config = {
            'name': config.get('agent.name', 'agent'),
            'temperature': config.get('agent.temperature', 0.7),
            'max_tokens': config.get('agent.max_tokens', 4096),
            'timeout': config.get('agent.timeout', 60),
        }
        
        # 系统提示词
        self.system_prompt = self._build_system_prompt()
        
        # 消息历史
        self.message_history: List[Dict] = []
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        custom_prompt = self.config.get('agent.system_prompt')
        if custom_prompt:
            return custom_prompt
        
        tools_desc = self.tools.get_descriptions()
        
        return f"""你是一个专业的 AI 助手，名叫 {self.agent_config['name']}。

你可以使用工具来帮助你完成任务。
可用工具:
{tools_desc}

使用工具时，请遵循以下格式:
- 调用工具: {{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}
- 当完成所有任务后，返回最终答案

记住:
1. 每次只调用一个工具
2. 仔细检查参数是否正确
3. 如果工具执行失败，尝试其他方法
"""
    
    def run(self, task: str, context: List[Dict] = None) -> str:
        """
        执行单个任务（同步）
        
        Args:
            task: 任务描述
            context: 额外的上下文消息
            
        Returns:
            Agent 的响应
        """
        logger.info(f"Agent task: {task}")
        
        # 1. 构建消息列表
        messages = self._build_messages(task, context)
        
        # 2. 调用 LLM
        response = self._call_llm(messages)
        
        # 3. 处理工具调用循环
        while response.tool_calls:
            # 添加助手消息和工具结果到历史
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [tc.to_dict() for tc in response.tool_calls],
            })
            
            # 执行工具调用
            tool_results = self.tools.execute(response.tool_calls)
            
            # 添加工具结果到消息
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": result.tool_call_id,
                })
            
            # 继续调用 LLM
            response = self._call_llm(messages)
        
        # 4. 保存记忆
        if self.config.get('memory.enabled', True):
            self.memory.add(task, response.content)
        
        return response.content
    
    async def run_async(self, task: str, context: List[Dict] = None) -> str:
        """
        执行单个任务（异步）
        
        Args:
            task: 任务描述
            context: 额外的上下文消息
            
        Returns:
            Agent 的响应
        """
        import asyncio
        
        # 1. 构建消息列表
        messages = self._build_messages(task, context)
        
        # 2. 调用 LLM（异步）
        response = await self._call_llm_async(messages)
        
        # 3. 处理工具调用循环
        max_iterations = self.config.get('agent.max_tool_iterations', 10)
        iteration = 0
        
        while response.tool_calls and iteration < max_iterations:
            iteration += 1
            
            # 添加助手消息
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [tc.to_dict() for tc in response.tool_calls],
            })
            
            # 执行工具调用
            loop = asyncio.get_event_loop()
            tool_results = await loop.run_in_executor(
                None,
                self.tools.execute,
                response.tool_calls
            )
            
            # 添加工具结果
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": result.tool_call_id,
                })
            
            # 继续调用 LLM
            response = await self._call_llm_async(messages)
        
        # 4. 保存记忆
        if self.config.get('memory.enabled', True):
            self.memory.add(task, response.content)
        
        return response.content
    
    def _build_messages(self, task: str, context: List[Dict] = None) -> List[Dict]:
        """构建消息列表"""
        messages = []
        
        # 系统提示词
        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })
        
        # 上下文
        if context:
            messages.extend(context)
        
        # 相关记忆
        if self.config.get('memory.enabled', True):
            memory_context = self.memory.get_context(task)
            messages.extend(memory_context)
        
        # 用户任务
        messages.append({
            "role": "user",
            "content": task,
        })
        
        return messages
    
    def _call_llm(self, messages: List[Dict]) -> LLMResponse:
        """调用 LLM"""
        try:
            return self.llm.chat(
                messages,
                model=self.model,
                temperature=self.agent_config['temperature'],
                max_tokens=self.agent_config['max_tokens'],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return LLMResponse(
                content=f"Error: LLM call failed: {str(e)}",
            )
    
    async def _call_llm_async(self, messages: List[Dict]) -> LLMResponse:
        """异步调用 LLM"""
        # 对于异步支持，可以使用 asyncio.to_thread
        import asyncio
        return await asyncio.to_thread(self._call_llm, messages)
    
    def chat(self, message: str) -> str:
        """
        聊天模式（多轮对话）
        
        Args:
            message: 用户消息
            
        Returns:
            Agent 响应
        """
        # 添加用户消息到历史
        self.message_history.append({
            "role": "user",
            "content": message,
        })
        
        # 构建消息
        messages = [{
            "role": "system",
            "content": self.system_prompt,
        }]
        messages.extend(self.message_history)
        
        # 调用 LLM
        response = self._call_llm(messages)
        
        # 处理工具调用
        while response.tool_calls:
            self.message_history.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [tc.to_dict() for tc in response.tool_calls],
            })
            
            tool_results = self.tools.execute(response.tool_calls)
            
            for result in tool_results:
                self.message_history.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": result.tool_call_id,
                })
            
            messages = [{
                "role": "system",
                "content": self.system_prompt,
            }]
            messages.extend(self.message_history)
            
            response = self._call_llm(messages)
        
        # 添加响应到历史
        self.message_history.append({
            "role": "assistant",
            "content": response.content,
        })
        
        return response.content
    
    def reset_history(self):
        """重置聊天历史"""
        self.message_history = []
    
    def get_history(self) -> List[Dict]:
        """获取聊天历史"""
        return self.message_history.copy()


class MultiAgent:
    """多 Agent 协作"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[str, AgentLoop] = {}
        self._init_agents()
    
    def _init_agents(self):
        """初始化多个 Agent"""
        agents_config = self.config.get('agents', {})
        for name, agent_config in agents_config.items():
            model = agent_config.get('model')
            self.agents[name] = AgentLoop(self.config, model)
    
    def get_agent(self, name: str) -> Optional[AgentLoop]:
        """获取指定 Agent"""
        return self.agents.get(name)
    
    def delegate(self, task: str, to: str) -> str:
        """委托任务给指定 Agent"""
        agent = self.get_agent(to)
        if not agent:
            return f"Agent not found: {to}"
        return agent.run(task)
    
    def broadcast(self, task: str) -> Dict[str, str]:
        """广播任务给所有 Agent"""
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.run(task)
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        return results


# 便捷函数
def create_agent(config: Config, model: str = None) -> AgentLoop:
    """创建 Agent"""
    return AgentLoop(config, model)


async def run_agent_async(config: Config, task: str, model: str = None) -> str:
    """异步运行 Agent"""
    agent = AgentLoop(config, model)
    return await agent.run_async(task)
