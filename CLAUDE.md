# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self Agent Framework - A configuration-based AI agent framework supporting multiple LLM providers with tool execution and memory capabilities.

## Build/Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single task
python main.py --config config/agent.yaml --model "openai/gpt-4" --task "your task"

# Interactive mode
python main.py --interactive

# Web UI mode
python main.py --web

# Web UI with custom port
python main.py --web --port 8080

# With verbose logging
python main.py --config config/agent.yaml --verbose
```

## Architecture

### Core Components (in `agent/`)

1. **llm.py** - LLM provider abstraction via `ProviderFactory`:
   - `OpenAIProvider` - OpenAI/Azure compatible (uses `openai` package)
   - `AnthropicProvider` - Claude series (uses `anthropic` package)
   - `MiniMaxProvider` - MiniMax API
   - `OllamaProvider` - Local Ollama server
   - Model format: `"provider/model"` (e.g., `"openai/gpt-4"`, `"anthropic/claude-3-sonnet"`)

2. **loop.py** - `AgentLoop` class:
   - Executes tasks via LLM with tool call loop
   - Supports async (`run_async`) and sync (`run`) execution
   - Multi-turn chat via `chat()` method
   - `MultiAgent` class for delegating tasks to named agents

3. **tools.py** - `Tools` manager with built-in tools:
   - `WebSearchTool` - DuckDuckGo search (enabled via `tools.enabled`)
   - `FileReadTool` / `FileWriteTool` - File operations
   - `ExecTool` - Shell commands (whitelist-controlled)
   - `CalculatorTool` - Safe mathematical evaluation
   - Register custom tools via `Tools.register(BaseTool subclass)`

4. **memory.py** - `Memory` system with two storage backends:
   - `FileMemoryStorage` - JSON files in configured directory
   - `SQLiteMemoryStorage` - SQLite database
   - Keyword-based retrieval via `memory.retrieve(query)`

5. **config.py** - `Config` class:
   - YAML configuration loading
   - Dot-notation access: `config.get('providers.openai.api_key')`
   - Environment variable substitution: `${VAR_NAME}` in YAML
   - `load_config()` auto-searches default paths

6. **types.py** - Core data classes:
   - `Message`, `MessageRole` - Chat messages
   - `ToolCall`, `ToolResult`, `ToolDefinition` - Tool system types
   - `MemoryEntry` - Memory records
   - `AgentConfig`, `LLMResponse` - Agent configuration and responses

### Configuration

Configuration file: `config/agent.yaml`

Key sections:
- `agent.*` - Agent name, model, temperature, max_tokens, timeout
- `providers.*` - Per-provider API keys and endpoints
- `tools.enabled` - List of enabled tool names
- `tools.exec.allowed_commands` - Whitelist for exec tool
- `memory.type` - `"file"` or `"sqlite"`

### Entry Point

`main.py` provides CLI with modes:
- Task mode: `--task "..."`
- Interactive mode: `--interactive`
- Web UI mode: `--web` (starts FastAPI server on port 8000)
- Default: shows help

### Web Interface

The web interface is provided by `web/app.py` (FastAPI) with a frontend at `web/static/index.html`:
- REST API at `/chat`, `/history`, `/reset`, `/status`
- Browser UI at `/`
- Supports streaming responses and conversation history

### Execution Flow

1. `AgentLoop.run(task)` builds messages from system prompt + memory context + task
2. `llm.chat()` returns `LLMResponse` with optional `tool_calls`
3. If tool calls exist, `Tools.execute()` runs them and loops back to LLM
4. Final response saved to memory and returned

### Important Notes

- LLM providers are instantiated in `agent/llm.py` (not in `providers/` directory despite imports)
- The `providers/__init__.py` references modules that don't exist as separate files
- All provider implementations are contained within `agent/llm.py`
- `create_llm(config, model)` auto-selects provider from model name prefix
