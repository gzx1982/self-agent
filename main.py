#!/usr/bin/env python3
"""
Self Agent Framework - 主入口

Usage:
    python main.py --config config/agent.yaml --model "openai/gpt-4"
    python main.py --interactive
    python main.py --task "帮我写一个 Hello World"
"""

import os
import sys
import argparse
import logging
from typing import Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AgentLoop, Config, load_config


def run_web_server(port: int = 8000):
    """启动 Web 服务"""
    try:
        import uvicorn
        from web.app import app
    except ImportError:
        print("Error: Web 服务需要安装 uvicorn")
        print("请运行: pip install uvicorn")
        sys.exit(1)

    print(f"=== Self Agent Web Server ===")
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"Open http://localhost:{port} in your browser")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def interactive_mode(agent: AgentLoop):
    """交互模式"""
    print(f"=== Self Agent Interactive Mode ===")
    print(f"Agent: {agent.agent_config['name']}")
    print(f"Model: {agent.model}")
    print(f"Tools: {', '.join(agent.tools.get_names()) or 'None'}")
    print(f"Type 'exit' or 'quit' to exit, 'reset' to clear history")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_history()
                print("History cleared.")
                continue
            
            if user_input.lower() == 'history':
                for i, msg in enumerate(agent.get_history()):
                    role = msg['role']
                    content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                    print(f"[{i}] {role}: {content}")
                continue
            
            if user_input.lower() == 'help':
                print("""
Commands:
  exit/quit/q   - Exit interactive mode
  reset         - Clear conversation history
  history       - Show conversation history
  help          - Show this help
""")
                continue
            
            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def task_mode(agent: AgentLoop, task: str):
    """单任务模式"""
    print(f"Task: {task}")
    print("-" * 50)

    try:
        response = agent.run(task)
        print("\n" + "=" * 50)
        print("Agent Response:")
        print(response)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Self Agent Framework - 基于配置文件的自定义 Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/agent.yaml --model "openai/gpt-4"
  python main.py --interactive
  python main.py --task "帮我写一个 Hello World"
  
Environment variables:
  OPENAI_API_KEY      - OpenAI API Key
  ANTHROPIC_API_KEY   - Anthropic API Key
  MINIMAX_API_KEY     - MiniMax API Key
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='指定模型，如 openai/gpt-4, anthropic/claude-3'
    )
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        default=None,
        help='要执行的任务'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='交互模式'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='启动 Web 界面'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Web 服务端口 (默认: 8000)'
    )

    args = parser.parse_args()

    # Web 模式独立运行
    if args.web:
        run_web_server(args.port)
        return

    # 设置日志
    setup_logging(args.verbose)

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 创建 Agent
    agent = AgentLoop(config, model=args.model)

    # 根据模式运行
    if args.interactive:
        interactive_mode(agent)
    elif args.task:
        task_mode(agent, args.task)
    else:
        # 默认显示帮助
        parser.print_help()


if __name__ == '__main__':
    main()
