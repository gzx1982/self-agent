"""
Self Agent Framework - 记忆系统

提供基于文件存储和 SQLite 的记忆管理
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from abc import ABC, abstractmethod

from .types import MemoryEntry, Message
from .config import Config

logger = logging.getLogger(__name__)


class MemoryStorage(ABC):
    """记忆存储抽象基类"""
    
    @abstractmethod
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆"""
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[MemoryEntry]:
        """获取记忆"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """搜索记忆"""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """删除记忆"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空记忆"""
        pass


class FileMemoryStorage(MemoryStorage):
    """基于文件的记忆存储"""
    
    def __init__(self, config: Config):
        self.config = config
        self.storage_dir = config.get('memory.path', './memory_db')
        self.max_items = config.get('memory.max_items', 1000)
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
    
    def _get_file_path(self, id: str) -> str:
        """获取记忆文件路径"""
        return os.path.join(self.storage_dir, f"{id}.json")
    
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.storage_dir, "index.json")
    
    def _load_index(self) -> List[str]:
        """加载索引"""
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_index(self, index: List[str]):
        """保存索引"""
        index_path = self._get_index_path()
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False)
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆"""
        try:
            # 保存记忆条目
            file_path = self._get_file_path(entry.id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
            
            # 更新索引
            index = self._load_index()
            index.append(entry.id)
            
            # 保持最大条目限制
            if len(index) > self.max_items:
                # 删除最旧的记忆
                old_id = index.pop(0)
                old_path = self._get_file_path(old_id)
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            self._save_index(index)
            logger.debug(f"Added memory: {entry.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False
    
    def get(self, id: str) -> Optional[MemoryEntry]:
        """获取记忆"""
        try:
            file_path = self._get_file_path(id)
            if not os.path.exists(file_path):
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return MemoryEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to get memory {id}: {e}")
            return None
    
    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        entries = []
        for id in self._load_index():
            entry = self.get(id)
            if entry:
                entries.append(entry)
        return entries
    
    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """搜索记忆（简单关键词匹配）"""
        query_words = set(query.lower().split())
        all_entries = self.get_all()
        
        # 计算相似度分数
        scored = []
        for entry in all_entries:
            entry_words = set(entry.task.lower().split())
            score = len(query_words & entry_words)
            if score > 0:
                scored.append((score, entry))
        
        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]
    
    def delete(self, id: str) -> bool:
        """删除记忆"""
        try:
            file_path = self._get_file_path(id)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            index = self._load_index()
            if id in index:
                index.remove(id)
                self._save_index(index)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {id}: {e}")
            return False
    
    def clear(self) -> bool:
        """清空记忆"""
        try:
            for id in self._load_index():
                file_path = self._get_file_path(id)
                if os.path.exists(file_path):
                    os.remove(file_path)
            self._save_index([])
            return True
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False


class SQLiteMemoryStorage(MemoryStorage):
    """基于 SQLite 的记忆存储"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.get('memory.path', './memory.db')
        self.max_items = config.get('memory.max_items', 1000)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_index (
                id TEXT PRIMARY KEY,
                task_text TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO memories (id, task, response, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
                (entry.id, entry.task, entry.response, entry.timestamp, json.dumps(entry.metadata))
            )
            cursor.execute(
                'INSERT INTO memory_index (id, task_text) VALUES (?, ?)',
                (entry.id, entry.task)
            )
            
            # 保持最大条目限制
            cursor.execute('SELECT COUNT(*) FROM memories')
            count = cursor.fetchone()[0]
            if count > self.max_items:
                cursor.execute('SELECT id FROM memories ORDER BY timestamp ASC LIMIT ?', (count - self.max_items,))
                old_ids = cursor.fetchall()
                for (old_id,) in old_ids:
                    cursor.execute('DELETE FROM memories WHERE id = ?', (old_id,))
                    cursor.execute('DELETE FROM memory_index WHERE id = ?', (old_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False
    
    def get(self, id: str) -> Optional[MemoryEntry]:
        """获取记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE id = ?', (id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return MemoryEntry(
                    id=row[0],
                    task=row[1],
                    response=row[2],
                    timestamp=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {id}: {e}")
            return None
    
    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append(MemoryEntry(
                    id=row[0],
                    task=row[1],
                    response=row[2],
                    timestamp=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                ))
            return entries
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """搜索记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 简单关键词匹配
            keywords = query.lower().split()
            for kw in keywords:
                cursor.execute(
                    'SELECT id FROM memory_index WHERE task_text LIKE ?',
                    (f'%{kw}%',)
                )
            
            result_ids = set()
            for kw in keywords:
                cursor.execute(
                    'SELECT id FROM memory_index WHERE task_text LIKE ?',
                    (f'%{kw}%',)
                )
                result_ids.update([row[0] for row in cursor.fetchall()])
            
            entries = []
            for id in result_ids:
                entry = self.get(id)
                if entry:
                    entries.append(entry)
            
            conn.close()
            return entries[:top_k]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def delete(self, id: str) -> bool:
        """删除记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE id = ?', (id,))
            cursor.execute('DELETE FROM memory_index WHERE id = ?', (id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {id}: {e}")
            return False
    
    def clear(self) -> bool:
        """清空记忆"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories')
            cursor.execute('DELETE FROM memory_index')
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False


class Memory:
    """记忆系统主类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.context_window = config.get('memory.context_window', 10)
        
        # 选择存储后端
        storage_type = config.get('memory.type', 'file')
        if storage_type == 'sqlite':
            self.storage = SQLiteMemoryStorage(config)
        else:
            self.storage = FileMemoryStorage(config)
    
    def add(self, task: str, response: str, metadata: Dict = None) -> bool:
        """添加记忆"""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            task=task,
            response=response,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        return self.storage.add(entry)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """检索相关记忆"""
        entries = self.storage.search(query, top_k)
        return [entry.response for entry in entries]
    
    def get_context(self, query: str, top_k: int = None) -> List[Dict]:
        """
        获取上下文（用于发送给 LLM）
        
        Args:
            query: 查询字符串
            top_k: 返回条数，默认从配置读取
            
        Returns:
            消息列表格式的记忆
        """
        if top_k is None:
            top_k = self.context_window
        
        entries = self.storage.search(query, top_k)
        messages = []
        for entry in entries:
            messages.append({
                "role": "user",
                "content": f"相关任务: {entry.task}"
            })
            messages.append({
                "role": "assistant", 
                "content": entry.response
            })
        return messages
    
    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        all_entries = self.storage.get_all()
        return all_entries[-limit:]
    
    def clear(self) -> bool:
        """清空记忆"""
        return self.storage.clear()
    
    def count(self) -> int:
        """获取记忆条数"""
        return len(self.storage.get_all())


def create_memory(config: Config) -> Memory:
    """创建记忆系统"""
    return Memory(config)
