"""
Self Agent Framework - 配置文件管理
"""

import os
import re
import yaml
from typing import Any, Optional


class Config:
    """配置管理类，支持 YAML 格式配置文件和环境变量引用"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: YAML 配置文件路径
            config_dict: 配置字典（优先于文件）
        """
        if config_dict is not None:
            self.config = config_dict
        elif config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号路径
        
        Args:
            key: 配置路径，如 'providers.openai.api_key'
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def resolve_env_vars(self, value: str) -> str:
        """
        解析字符串中的环境变量引用 ${VAR_NAME}
        
        Args:
            value: 包含环境变量引用的字符串
            
        Returns:
            解析后的字符串
        """
        if not isinstance(value, str):
            return value
        
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name, '')
            value = value.replace(f'${{{var_name}}}', env_value)
        return value
    
    def get_resolved(self, key: str, default: Any = None) -> Any:
        """获取配置项并自动解析环境变量"""
        value = self.get(key, default)
        if isinstance(value, str):
            return self.resolve_env_vars(value)
        return value
    
    def merge(self, other_config: dict):
        """合并另一个配置字典"""
        self._deep_merge(self.config, other_config)
    
    def _deep_merge(self, base: dict, update: dict):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> dict:
        """导出配置为字典"""
        return self.config.copy()
    
    @staticmethod
    def from_env(prefix: str = "AGENT_") -> 'Config':
        """
        从环境变量创建配置
        
        环境变量格式: AGENT_PROVIDERS_OPENAI_API_KEY -> providers.openai.api_key
        """
        config_dict = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 转换 AGENT_PROVIDERS_OPENAI_API_KEY -> providers.openai.api_key
                parts = key[len(prefix):].lower().split('_')
                config = config_dict
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                config[parts[-1]] = value
        return Config(config_dict=config_dict)
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)


def load_config(config_path: str = None) -> Config:
    """加载配置的便捷函数"""
    if config_path is None:
        # 尝试默认路径
        default_paths = [
            'config/agent.yaml',
            'config/agent.yml',
            'agent.yaml',
            'agent.yml',
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            return Config()  # 返回空配置
    return Config(config_path)
