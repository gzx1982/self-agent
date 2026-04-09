"""
Self Agent Framework - Skill 系统

Skill 是一种预定义的 Agent 行为模式，包含特定的系统提示词和工具配置。
可以通过命令、关键词或文件模式触发。
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch

import yaml

from .config import Config
from .types import Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """
    Skill 定义

    Attributes:
        name: Skill 唯一标识名
        description: Skill 描述
        prompt: 系统提示词（会覆盖或扩展默认系统提示词）
        prompt_mode: prompt 应用模式 - "replace" 替换, "prefix" 前缀, "suffix" 后缀
        tools: 该 Skill 额外启用的工具列表
        disable_tools: 该 Skill 禁用的工具列表
        triggers: 触发条件列表
        examples: 使用示例
    """
    name: str
    description: str = ""
    prompt: str = ""
    prompt_mode: str = "replace"  # replace, prefix, suffix
    tools: List[str] = field(default_factory=list)
    disable_tools: List[str] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'Skill':
        """从字典创建 Skill"""
        return cls(
            name=name,
            description=data.get('description', ''),
            prompt=data.get('prompt', ''),
            prompt_mode=data.get('prompt_mode', 'replace'),
            tools=data.get('tools', []),
            disable_tools=data.get('disable_tools', []),
            triggers=data.get('triggers', []),
            examples=data.get('examples', []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'prompt': self.prompt,
            'prompt_mode': self.prompt_mode,
            'tools': self.tools,
            'disable_tools': self.disable_tools,
            'triggers': self.triggers,
            'examples': self.examples,
        }

    def match(self, message: str, context: Dict[str, Any] = None) -> float:
        """
        检查消息是否匹配此 Skill

        Args:
            message: 用户输入消息
            context: 额外上下文（如当前文件路径）

        Returns:
            匹配分数 (0.0-1.0)，0 表示不匹配
        """
        message = message.strip()
        context = context or {}

        for trigger in self.triggers:
            trigger_type = trigger.get('type', 'command')

            if trigger_type == 'command':
                # 命令触发: /skill_name 或 @skill_name
                patterns = trigger.get('patterns', [f'/{self.name}', f'@{self.name}'])
                for pattern in patterns:
                    if message.startswith(pattern):
                        return trigger.get('score', 1.0)

            elif trigger_type == 'keyword':
                # 关键词触发
                keywords = trigger.get('keywords', [])
                match_all = trigger.get('match_all', False)

                if match_all:
                    if all(kw.lower() in message.lower() for kw in keywords):
                        return trigger.get('score', 0.8)
                else:
                    if any(kw.lower() in message.lower() for kw in keywords):
                        return trigger.get('score', 0.6)

            elif trigger_type == 'regex':
                # 正则表达式触发
                pattern = trigger.get('pattern', '')
                if pattern and re.search(pattern, message, re.IGNORECASE):
                    return trigger.get('score', 0.7)

            elif trigger_type == 'file_pattern':
                # 文件模式触发（基于 context 中的 file_path）
                file_path = context.get('file_path', '')
                if file_path:
                    patterns = trigger.get('patterns', [])
                    for pattern in patterns:
                        if fnmatch(file_path, pattern):
                            return trigger.get('score', 0.9)

        return 0.0

    def extract_task(self, message: str) -> str:
        """
        从消息中提取实际任务内容（移除触发命令）

        Args:
            message: 原始消息

        Returns:
            清理后的任务内容
        """
        # 检查命令触发器
        for trigger in self.triggers:
            if trigger.get('type') == 'command':
                patterns = trigger.get('patterns', [f'/{self.name}', f'@{self.name}'])
                for pattern in patterns:
                    if message.startswith(pattern):
                        # 移除命令部分，返回剩余内容
                        task = message[len(pattern):].strip()
                        # 移除常见的分隔符
                        task = re.sub(r'^[\s:：,，]+', '', task)
                        return task

        return message


class SkillManager:
    """Skill 管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.skills: Dict[str, Skill] = {}
        self.default_skill: Optional[str] = None
        self._load_skills()

    def _load_skills(self):
        """加载所有 Skill"""
        # 从 skills/ 目录加载 skill 包
        self._load_skills_from_directory("skills")

        # 从配置加载自定义 skills (配置文件优先级更高，可覆盖目录中的 skill)
        custom_skills = self.config.get('skills.custom', {})

        for name, skill_data in custom_skills.items():
            if isinstance(skill_data, dict):
                skill = Skill.from_dict(name, skill_data)
                self.skills[name] = skill
                logger.info(f"[SkillManager] Loaded skill from config: {name}")

        # 加载内置 skills
        self._load_builtin_skills()

        # 设置默认 skill
        self.default_skill = self.config.get('skills.default')

        logger.info(f"[SkillManager] Total skills loaded: {len(self.skills)}")

    def _load_skills_from_directory(self, directory: str):
        """
        从 skills/ 目录加载 skill 包

        Skill 包结构：
        skills/
          skill-name/
            SKILL.md          # 必需，包含 YAML frontmatter 的 skill 定义
            _meta.json        # 可选，元数据
            scripts/          # 可选，脚本文件
        """
        if not os.path.isdir(directory):
            logger.debug(f"[SkillManager] Skills directory not found: {directory}")
            return

        for item in os.listdir(directory):
            skill_path = os.path.join(directory, item)
            if not os.path.isdir(skill_path):
                continue

            skill_file = os.path.join(skill_path, "SKILL.md")
            if not os.path.isfile(skill_file):
                continue

            try:
                skill = self._parse_skill_file(skill_file, skill_path)
                if skill:
                    # 使用 skill 文件中定义的 name，或从目录名推断
                    skill_name = skill.name or item.replace("-", "_").replace(" ", "_")
                    # 如果配置文件已定义同名 skill，跳过（配置优先级更高）
                    if skill_name not in self.skills:
                        self.skills[skill_name] = skill
                        logger.info(f"[SkillManager] Loaded skill from directory: {skill_name}")
            except Exception as e:
                logger.error(f"[SkillManager] Failed to load skill from {skill_path}: {e}")

    def _parse_skill_file(self, file_path: str, base_dir: str) -> Optional[Skill]:
        """
        解析 SKILL.md 文件

        格式：
        ---
        name: skill-name
        description: "描述"
        prompt: "系统提示词"
        prompt_mode: "replace"
        tools:
          - "exec"
        triggers:
          - type: "command"
            patterns: ["/command"]
        ---

        # 后续内容（可选）
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析 YAML frontmatter
        if not content.startswith('---'):
            logger.warning(f"[SkillManager] No YAML frontmatter found in {file_path}")
            return None

        # 提取 frontmatter
        parts = content.split('---', 2)
        if len(parts) < 3:
            logger.warning(f"[SkillManager] Invalid frontmatter format in {file_path}")
            return None

        yaml_content = parts[1].strip()

        try:
            data = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError as e:
            logger.error(f"[SkillManager] YAML parse error in {file_path}: {e}")
            return None

        if not isinstance(data, dict):
            logger.warning(f"[SkillManager] YAML frontmatter is not a dict in {file_path}")
            return None

        # 处理 prompt 中的 {baseDir} 占位符
        prompt = data.get('prompt', '')
        if prompt:
            prompt = prompt.replace('{baseDir}', base_dir)

        # 构建 Skill 对象
        skill = Skill.from_dict(data.get('name', ''), {
            'description': data.get('description', ''),
            'prompt': prompt,
            'prompt_mode': data.get('prompt_mode', 'replace'),
            'tools': data.get('tools', []),
            'disable_tools': data.get('disable_tools', []),
            'triggers': data.get('triggers', []),
            'examples': data.get('examples', []),
        })

        # 如果没有定义 triggers，添加默认的命令触发器
        if not skill.triggers and skill.name:
            skill.triggers = [
                {'type': 'command', 'patterns': [f'/{skill.name}', f'@{skill.name}']}
            ]

        return skill

    def _load_builtin_skills(self):
        """加载内置 skills"""
        # 检查是否启用内置 skills
        builtin_enabled = self.config.get('skills.builtin_enabled', True)
        if not builtin_enabled:
            return

        # 内置 skill: commit - Git 提交助手
        if 'commit' not in self.skills:
            self.skills['commit'] = Skill(
                name='commit',
                description='Git 提交助手，帮助生成规范的 commit message',
                prompt="""你是一个 Git 专家。帮助用户创建规范的 commit message。

工作流程：
1. 运行 `git status` 查看变更文件
2. 运行 `git diff --staged` 或 `git diff` 查看具体改动
3. 分析变更类型（feat/fix/docs/style/refactor/test/chore）
4. 生成符合 Conventional Commits 规范的提交信息
5. 询问用户确认或直接执行提交

Commit message 格式：
<type>(<scope>): <subject>

<body>

<footer>

类型说明：
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 代码格式调整（不影响功能）
- refactor: 重构（既不是新功能也不是修复 bug）
- test: 测试相关
- chore: 构建/工具/依赖更新""",
                triggers=[
                    {'type': 'command', 'patterns': ['/commit', '@commit']},
                    {'type': 'keyword', 'keywords': ['提交代码', 'git commit', 'commit message'], 'match_all': False},
                ],
                examples=[
                    '/commit 帮我提交当前修改',
                    '@commit 生成 commit message',
                ]
            )

        # 内置 skill: review - 代码审查
        if 'review' not in self.skills:
            self.skills['review'] = Skill(
                name='review',
                description='代码审查助手，检查代码质量和潜在问题',
                prompt="""你是一个资深代码审查专家。审查代码时关注：

1. **代码质量**
   - 代码可读性和可维护性
   - 命名是否清晰有意义
   - 函数/类设计是否合理

2. **潜在问题**
   - 空指针/边界条件
   - 资源泄露
   - 并发安全问题
   - 性能隐患

3. **最佳实践**
   - 是否符合语言/框架惯用法
   - 是否遵循项目规范
   - 是否有重复代码可以提取

4. **安全**
   - 注入漏洞
   - 敏感信息泄露
   - 不安全的依赖

输出格式：
- 🟢 好的实践（肯定）
- 🟡 建议改进（非阻塞）
- 🔴 必须修复（阻塞性问题）""",
                triggers=[
                    {'type': 'command', 'patterns': ['/review', '@review']},
                    {'type': 'keyword', 'keywords': ['审查代码', 'code review', 'review this'], 'match_all': False},
                    {'type': 'file_pattern', 'patterns': ['*.py', '*.js', '*.ts', '*.java', '*.go', '*.rs']},
                ],
                examples=[
                    '/review 请审查这个文件',
                    '@review agent/skill.py',
                ]
            )

        # 内置 skill: explain - 代码解释
        if 'explain' not in self.skills:
            self.skills['explain'] = Skill(
                name='explain',
                description='代码解释助手，用通俗易懂的方式解释代码',
                prompt="""你是一个技术讲解员。用通俗易懂的方式解释代码：

1. **整体概述** - 这段代码是做什么的
2. **关键逻辑** - 核心算法或业务逻辑
3. **数据流** - 数据如何流动和转换
4. **注意事项** - 需要特别关注的地方

解释风格：
- 先讲 "是什么"，再讲 "为什么"
- 用类比帮助理解复杂概念
- 对关键代码行添加注释说明
- 如果有设计模式或最佳实践，指出来""",
                triggers=[
                    {'type': 'command', 'patterns': ['/explain', '@explain']},
                    {'type': 'keyword', 'keywords': ['解释代码', 'explain code', '这段代码什么意思', 'how does this work'], 'match_all': False},
                ],
                examples=[
                    '/explain 解释这段代码',
                    '@explain 这个函数是做什么的',
                ]
            )

        # 内置 skill: test - 测试生成
        if 'test' not in self.skills:
            self.skills['test'] = Skill(
                name='test',
                description='测试生成助手，为代码生成单元测试',
                prompt="""你是一个测试专家。为代码生成全面的单元测试：

测试原则：
1. **覆盖率** - 覆盖正常路径、边界条件、异常路径
2. **可读性** - 测试名称清晰描述测试场景
3. **独立性** - 每个测试独立，不依赖执行顺序
4. **可维护性** - 使用参数化测试减少重复代码

测试结构（AAA）：
- Arrange: 准备测试数据
- Act: 执行被测方法
- Assert: 验证结果

包含的测试类型：
- 正常输入的输出验证
- 边界值测试
- 异常情况处理
- （可选）性能基准测试

使用项目现有的测试框架和风格。""",
                triggers=[
                    {'type': 'command', 'patterns': ['/test', '@test']},
                    {'type': 'keyword', 'keywords': ['生成测试', 'write test', 'unit test', '测试用例'], 'match_all': False},
                ],
                examples=[
                    '/test 为 skill.py 生成测试',
                    '@test 给这个函数写测试',
                ]
            )

    def register(self, skill: Skill):
        """注册 Skill"""
        self.skills[skill.name] = skill
        logger.info(f"[SkillManager] Registered skill: {skill.name}")

    def unregister(self, name: str):
        """注销 Skill"""
        if name in self.skills:
            del self.skills[name]

    def get(self, name: str) -> Optional[Skill]:
        """获取指定 Skill"""
        return self.skills.get(name)

    def list_skills(self) -> List[Skill]:
        """获取所有 Skill 列表"""
        return list(self.skills.values())

    def match(self, message: str, context: Dict[str, Any] = None) -> Optional[Skill]:
        """
        匹配最适合的 Skill

        Args:
            message: 用户输入
            context: 额外上下文

        Returns:
            最匹配的 Skill，如果没有匹配则返回 None
        """
        best_skill = None
        best_score = 0.0

        for skill in self.skills.values():
            score = skill.match(message, context)
            if score > best_score:
                best_score = score
                best_skill = skill

        # 阈值判断
        threshold = self.config.get('skills.match_threshold', 0.5)
        if best_score >= threshold:
            logger.info(f"[SkillManager] Matched skill '{best_skill.name}' with score {best_score}")
            return best_skill

        return None

    def apply_skill(self, base_prompt: str, skill: Skill) -> str:
        """
        应用 Skill 到系统提示词

        Args:
            base_prompt: 基础系统提示词
            skill: 要应用的 Skill

        Returns:
            应用后的系统提示词
        """
        if not skill or not skill.prompt:
            return base_prompt

        if skill.prompt_mode == 'replace':
            return skill.prompt
        elif skill.prompt_mode == 'prefix':
            return f"{skill.prompt}\n\n{'='*40}\n\n{base_prompt}"
        elif skill.prompt_mode == 'suffix':
            return f"{base_prompt}\n\n{'='*40}\n\n{skill.prompt}"
        else:
            return skill.prompt

    def get_skill_tools(self, skill: Skill, base_tools: List[str]) -> List[str]:
        """
        获取 Skill 应用后的工具列表

        Args:
            skill: 要应用的 Skill
            base_tools: 基础工具列表

        Returns:
            应用后的工具列表
        """
        if not skill:
            return base_tools

        tools = set(base_tools)

        # 添加 Skill 特定的工具
        if skill.tools:
            tools.update(skill.tools)

        # 移除禁用的工具
        if skill.disable_tools:
            tools -= set(skill.disable_tools)

        return list(tools)

    def extract_task(self, message: str, context: Dict[str, Any] = None, skill: Skill = None) -> tuple:
        """
        提取实际任务内容和匹配的 Skill

        Args:
            message: 原始消息
            context: 额外上下文（如当前文件路径）
            skill: 可选的指定 Skill

        Returns:
            (task, skill) 元组
        """
        # 如果没有指定 skill，先进行匹配
        if skill is None:
            skill = self.match(message, context)

        if skill:
            task = skill.extract_task(message)
            return task, skill

        return message, None


# 便捷函数
def create_skill_manager(config: Config) -> SkillManager:
    """创建 Skill 管理器"""
    return SkillManager(config)
