"""
Self Agent Framework - Web 服务

FastAPI 后端提供 REST API 访问 Agent 功能
"""

import os
import sys
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AgentLoop, Config, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Self Agent Web UI",
    description="Self Agent Framework 的 Web 界面",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Agent 实例
_config: Config = None
_agent: AgentLoop = None


def get_agent() -> AgentLoop:
    """获取或初始化 Agent 实例"""
    global _config, _agent
    if _agent is None:
        _config = load_config()
        _agent = AgentLoop(_config)
        logger.info("Agent initialized")
    return _agent


# 请求/响应模型
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class HistoryItem(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    history: List[HistoryItem]


class ResetResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    status: str
    agent_name: str
    model: str
    tools: List[str]


@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse("web/static/index.html")


@app.get("/status", response_model=StatusResponse)
async def status():
    """获取 Agent 状态"""
    agent = get_agent()
    return StatusResponse(
        status="running",
        agent_name=agent.agent_config['name'],
        model=agent.model,
        tools=agent.tools.get_names()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送消息给 Agent (非流式)

    Args:
        request: 包含 message 字段的请求体

    Returns:
        Agent 的响应
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        import asyncio
        agent = get_agent()
        logger.info(f"Chat request: {request.message[:100]}...")
        # 使用 asyncio.to_thread 将同步调用转为异步，避免阻塞事件循环
        response = await asyncio.to_thread(agent.chat, request.message)
        logger.info(f"Chat response: {response[:100]}...")
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    发送消息给 Agent (流式 SSE)

    Args:
        request: 包含 message 字段的请求体

    Returns:
        SSE 流式响应
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    async def event_generator():
        import asyncio
        agent = get_agent()
        logger.info(f"Stream chat request: {request.message[:100]}...")

        try:
            # 在后台线程运行 agent.chat 获取完整响应
            response = await asyncio.to_thread(agent.chat, request.message)
            logger.info(f"Stream chat response: {response[:100]}...")

            # 模拟流式输出：将响应分段发送
            chunk_size = 4  # 每4个字符发送一次
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i+chunk_size]
                # SSE 格式: data: <内容>


                # 对数据进行转义处理
                safe_data = chunk.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
                yield f"data: {safe_data}\n\n"
                # 小延迟让流式效果更自然
                await asyncio.sleep(0.01)

            # 发送结束标记
            yield f"event: done\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream chat error: {e}", exc_info=True)
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """获取对话历史"""
    agent = get_agent()
    history = agent.get_history()
    return HistoryResponse(history=history)


@app.post("/reset", response_model=ResetResponse)
async def reset():
    """重置对话历史"""
    agent = get_agent()
    agent.reset_history()
    logger.info("History reset")
    return ResetResponse(status="ok")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
