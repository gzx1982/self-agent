---
name: tavily-search
description: "Web search via Tavily API (alternative to Brave). Use when the user asks to search the web / look up sources / find links and Brave web_search is unavailable or undesired. Returns a small set of relevant results (title, url, snippet) and can optionally include short answer summaries."
prompt: |
  你是一个网络搜索助手，使用 Tavily API 帮助用户搜索网络信息。

  当用户需要搜索时：
  1. 使用 exec 工具运行 tavily_search.py 脚本
  2. 脚本路径: {baseDir}/scripts/tavily_search.py
  3. 分析搜索结果并回答用户问题

  命令格式：
  python3 {baseDir}/scripts/tavily_search.py --query "搜索词" --max-results 5

  可选参数：
  --include-answer: 包含简短答案
  --format brave: 输出格式兼容 web_search
  --format md: Markdown 格式输出

  注意：
  - 保持 max-results 较小(3-5)以减少 token 消耗
  - 优先返回 URLs + 摘要；只在需要时获取完整页面
tools:
  - "exec"
triggers:
  - type: "command"
    patterns: ["/tavily", "@tavily"]
  - type: "keyword"
    keywords: ["tavily搜索", "tavily search", "使用tavily"]
---

# Tavily Search

Use the bundled script to search the web with Tavily.

## Requirements

- Provide API key via either:
  - environment variable: `TAVILY_API_KEY`, or
  - `~/.openclaw/.env` line: `TAVILY_API_KEY=...`

## Commands

Run from the OpenClaw workspace:

```bash
# raw JSON (default)
python3 {baseDir}/scripts/tavily_search.py --query "..." --max-results 5

# include short answer (if available)
python3 {baseDir}/scripts/tavily_search.py --query "..." --max-results 5 --include-answer

# stable schema (closer to web_search): {query, results:[{title,url,snippet}], answer?}
python3 {baseDir}/scripts/tavily_search.py --query "..." --max-results 5 --format brave

# human-readable Markdown list
python3 {baseDir}/scripts/tavily_search.py --query "..." --max-results 5 --format md
```

## Output

### raw (default)
- JSON: `query`, optional `answer`, `results: [{title,url,content}]`

### brave
- JSON: `query`, optional `answer`, `results: [{title,url,snippet}]`

### md
- A compact Markdown list with title/url/snippet.

## Notes

- Keep `max-results` small by default (3–5) to reduce token/reading load.
- Prefer returning URLs + snippets; fetch full pages only when needed.
