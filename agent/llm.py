"""
Self Agent Framework - LLM Provider 实现

支持多种大模型 Provider：
- OpenAI 兼容 (OpenAI, Azure, 本地代理等)
- Anthropic
- MiniMax
- Ollama (本地)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .types import Message, ToolCall, ToolDefinition, LLMResponse
from .config import Config

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """LLM Provider 抽象基类"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数如 model, temperature, max_tokens
            
        Returns:
            LLMResponse 对象
        """
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """补全请求（同步）"""
        pass
    
    def supports_tools(self) -> bool:
        """是否支持工具调用"""
        return False
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """获取工具定义列表"""
        return []


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容 Provider（支持 OpenAI、Azure、本地代理等）"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = self.config.resolve_env_vars(
            self.config.get('providers.openai.api_key', '')
        )
        base_url = self.config.resolve_env_vars(
            self.config.get('providers.openai.base_url', 'https://api.openai.com/v1')
        )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.get('agent.timeout', 60),
        )
        self.default_model = self.config.get('providers.openai.model', 'gpt-4')
        self.tools = self._load_tools()
    
    def _load_tools(self) -> Optional[List[Dict]]:
        """加载工具定义"""
        tools_config = self.config.get('providers.openai.tools')
        if tools_config:
            return tools_config
        return None
    
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        model = kwargs.get('model', self.default_model)
        temperature = kwargs.get('temperature', self.config.get('agent.temperature', 0.7))
        max_tokens = kwargs.get('max_tokens', self.config.get('agent.max_tokens', 4096))
        
        request_kwargs = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        # 添加工具调用支持
        if self.tools:
            request_kwargs['tools'] = self.tools
        
        response = self.client.chat.completions.create(**request_kwargs)
        response_dict = response.model_dump()
        
        choice = response_dict['choices'][0]
        message = choice['message']
        
        tool_calls = None
        if 'tool_calls' in message and message['tool_calls']:
            tool_calls = [ToolCall.from_dict(tc) for tc in message['tool_calls']]
        
        return LLMResponse(
            content=message.get('content', ''),
            tool_calls=tool_calls,
            raw_response=response_dict,
            usage=response_dict.get('usage'),
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全请求"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs).content
    
    def supports_tools(self) -> bool:
        """是否支持工具调用"""
        return self.tools is not None


class AnthropicProvider(LLMProvider):
    """Anthropic Provider (Claude 系列)"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = self.config.resolve_env_vars(
            self.config.get('providers.anthropic.api_key', '')
        )
        
        self.client = Anthropic(api_key=api_key)
        self.default_model = self.config.get('providers.anthropic.model', 'claude-3-opus-20240229')
    
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        model = kwargs.get('model', self.default_model)
        temperature = kwargs.get('temperature', self.config.get('agent.temperature', 0.7))
        max_tokens = kwargs.get('max_tokens', self.config.get('agent.max_tokens', 4096))
        
        # 转换消息格式
        anthropic_messages = []
        system_prompt = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                anthropic_messages.append({
                    "role": msg['role'],
                    "content": msg['content'],
                })
        
        request_kwargs = {
            'model': model,
            'messages': anthropic_messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        if system_prompt:
            request_kwargs['system'] = system_prompt
        
        response = self.client.messages.create(**request_kwargs)
        response_dict = response.model_dump()
        
        content = response_dict['content'][0]['text']
        
        return LLMResponse(
            content=content,
            raw_response=response_dict,
            usage={
                'input_tokens': response_dict.get('usage', {}).get('input_tokens'),
                'output_tokens': response_dict.get('usage', {}).get('output_tokens'),
            },
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全请求"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs).content


class MiniMaxProvider(LLMProvider):
    """MiniMax Provider"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = self.config.resolve_env_vars(
            self.config.get('providers.minimax.api_key', '')
        )
        base_url = self.config.resolve_env_vars(
            self.config.get('providers.minimax.base_url', 'https://api.minimax.chat/v1')
        )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.default_model = self.config.get('providers.minimax.model', 'MiniMax-Text-01')
    
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        model = kwargs.get('model', self.default_model)
        temperature = kwargs.get('temperature', self.config.get('agent.temperature', 0.7))
        max_tokens = kwargs.get('max_tokens', self.config.get('agent.max_tokens', 4096))
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_dict = response.model_dump()
        
        return LLMResponse(
            content=response_dict['choices'][0]['message']['content'],
            raw_response=response_dict,
            usage=response_dict.get('usage'),
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全请求"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs).content


class OllamaProvider(LLMProvider):
    """Ollama 本地 Provider"""

    def __init__(self, config: Config):
        super().__init__(config)
        import requests

        self.base_url = self.config.resolve_env_vars(
            self.config.get('providers.ollama.base_url', 'http://localhost:11434')
        )
        self.default_model = self.config.get('providers.ollama.model', 'llama3')
        self.session = requests.Session()

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        model = kwargs.get('model', self.default_model)
        # 去掉 provider 前缀 (e.g. "ollama/qwen3.5:35b" -> "qwen3.5:35b")
        if '/' in model:
            model = model.split('/')[1]
        temperature = kwargs.get('temperature', self.config.get('agent.temperature', 0.7))
        
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
        }
        
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.config.get('agent.timeout', 60),
        )
        response.raise_for_status()
        response_dict = response.json()
        
        return LLMResponse(
            content=response_dict['message']['content'],
            raw_response=response_dict,
        )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """补全请求"""
        model = kwargs.get('model', self.default_model)
        # 去掉 provider 前缀
        if '/' in model:
            model = model.split('/')[1]
        temperature = kwargs.get('temperature', self.config.get('agent.temperature', 0.7))

        payload = {
            'model': model,
            'prompt': prompt,
            'temperature': temperature,
            'stream': False,
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.config.get('agent.timeout', 60),
        )
        response.raise_for_status()
        response_dict = response.json()
        
        return response_dict['response']
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []


class ProviderFactory:
    """LLM Provider 工厂类"""
    
    _providers = {
        'openai': OpenAIProvider,
        'azure': OpenAIProvider,  # Azure 也使用 OpenAI 兼容接口
        'anthropic': AnthropicProvider,
        'minimax': MiniMaxProvider,
        'ollama': OllamaProvider,
    }
    
    @classmethod
    def create(cls, provider_name: str, config: Config) -> LLMProvider:
        """
        创建 Provider 实例
        
        Args:
            provider_name: Provider 名称 (openai, anthropic, minimax, ollama)
            config: 配置对象
            
        Returns:
            LLMProvider 实例
        """
        provider_class = cls._providers.get(provider_name.lower())
        if provider_class is None:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls._providers.keys())}")
        
        return provider_class(config)
    
    @classmethod
    def create_from_model(cls, model: str, config: Config) -> LLMProvider:
        """
        根据模型名称自动选择 Provider
        
        Args:
            model: 模型名称，格式如 "openai/gpt-4" 或 "anthropic/claude-3"
            config: 配置对象
        """
        if '/' in model:
            provider_name = model.split('/')[0]
        else:
            provider_name = 'openai'  # 默认使用 OpenAI
        
        return cls.create(provider_name, config)
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册新的 Provider"""
        cls._providers[name.lower()] = provider_class


def create_llm(config: Config, model: Optional[str] = None) -> LLMProvider:
    """
    创建 LLM Provider 的便捷函数
    
    Args:
        config: 配置对象
        model: 模型名称（可选，默认使用配置的模型）
        
    Returns:
        LLMProvider 实例
    """
    if model:
        return ProviderFactory.create_from_model(model, config)
    
    # 从配置获取默认模型和 provider
    default_model = config.get('agent.model', 'openai/gpt-4')
    return ProviderFactory.create_from_model(default_model, config)
