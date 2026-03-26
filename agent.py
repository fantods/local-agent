#!/usr/bin/env python3
import os
import sys
import json
import subprocess as sp
from datetime import datetime
from typing import Generator

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

console = Console()

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8000")

TOOL_KEYWORDS = [
    "search",
    "find",
    "look up",
    "google",
    "what time",
    "when do",
    "when is",
    "when does",
    "when are",
    "who do",
    "who is playing",
    "who plays",
    "who won",
    "what happened",
    "what is the score",
    "weather",
    "news",
    "latest",
    "schedule",
    "score",
    "tonight",
    "today",
    "tomorrow",
    "yesterday",
    "this week",
    "next game",
    "play next",
    "playing next",
    "results",
    "standings",
    "price",
    "stock",
    "market",
    "crypto",
    "bitcoin",
    "fetch",
    "download",
    "read file",
    "write file",
    "create file",
    "run",
    "execute",
    "list files",
    "show me",
    "open",
    "browse",
    "url",
    "http",
    "website",
    "how much",
    "where is",
    "directions",
    "recipe",
    "explore",
    "repo",
    "repository",
    "github",
    "tell me more",
    "more about",
    "what else",
    "continue",
    "go deeper",
]


def llm_stream(
    messages: list, max_tokens: int = 2048, temperature: float = 0.7
) -> Generator[str, None, None]:
    """Stream response from local LLM server."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        response = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield "[ERROR] Cannot connect to LLM server. Is llama-server running?"
    except Exception as e:
        yield f"[ERROR] {e}"


def llm_call(
    messages: list, max_tokens: int = 2048, temperature: float = 0.7
) -> tuple[str, dict]:
    """Non-streaming call to LLM, returns (content, timings)."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        timings = {
            "predicted_per_second": data.get("tokens_per_second", 0),
            "tokens_generated": data.get("tokens_generated", 0),
        }
        return content, timings
    except Exception as e:
        return f"Error: {e}", {}


def classify_intent(message: str) -> str:
    """Ask LLM to classify: 'search', 'shell', or 'chat'. One fast call (~1s)."""
    try:
        result, _ = llm_call(
            [
                {
                    "role": "system",
                    "content": """Classify the user's request into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- search: needs web search (news, scores, weather, prices, current events, looking up info online)
- shell: needs filesystem or command execution (find files, list directories, read/write files, run commands, look at desktop, explore folders, check disk space, anything involving the local computer)
- chat: general conversation, reasoning, math, coding questions, explanations (no tools needed)

Reply with ONLY one word: search, shell, or chat""",
                },
                {"role": "user", "content": message},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return result.strip().lower().split()[0]
    except Exception:
        return "chat"


def generate_shell_command(query: str, work_dir: str = ".") -> str:
    """Ask LLM to generate the right shell command for a file/system task."""
    home = os.path.expanduser("~")
    result, _ = llm_call(
        [
            {
                "role": "system",
                "content": f"""You are a macOS shell command generator. The user's home directory is {home}. Current working directory is {work_dir}.

Generate a single shell command that accomplishes the user's request. Output ONLY the command, nothing else. No explanation, no markdown, no backticks.

Examples:
- "find videos on my desktop" → find {home}/Desktop -type f \\( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \\)
- "what files are on my desktop" → ls -la {home}/Desktop
- "how much disk space do I have" → df -h /
- "show me python files in this project" → find . -name "*.py" -type f
- "read the readme" → cat README.md
- "what's running on port 8000" → lsof -i :8000
- "count lines of code" → find . -name "*.py" -exec wc -l {{}} +""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    return result.strip().strip("`").strip()


def run_smart_tool(query: str, work_dir: str = ".") -> Generator[str, None, None]:
    """Execute a shell command generated by the LLM, stream results back."""
    yield "[dim]Generating command...[/dim]"

    cmd = generate_shell_command(query, work_dir)

    yield f"\n[dim]Executing:[/dim] [cyan]{cmd}[/cyan]\n"

    try:
        result = sp.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=work_dir
        )
        output = result.stdout[:8000]
        if result.stderr:
            output += f"\n{result.stderr[:2000]}"
    except sp.TimeoutExpired:
        output = "Command timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {e}"

    yield "\n[dim]Summarizing results...[/dim]\n\n"

    today = datetime.now().strftime("%A, %B %d, %Y")
    for chunk in llm_stream(
        [
            {
                "role": "system",
                "content": f"Today is {today}. You ran a shell command and got results. Present the results clearly to the user. If it's a file listing, format it nicely. If it's code, use formatting. Be helpful and concise.",
            },
            {
                "role": "user",
                "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {query}",
            },
        ],
        max_tokens=1000,
    ):
        yield chunk


def print_header():
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]Local Agent[/bold cyan] [dim]• Chat with your local LLM[/dim]\n"
            "[dim]Commands: /clear, /quit, /help[/dim]",
            border_style="cyan",
        )
    )


def print_help():
    console.print(
        Panel(
            "[bold]Available Commands:[/bold]\n"
            "  /clear  - Clear conversation history\n"
            "  /quit   - Exit the agent\n"
            "  /help   - Show this help\n\n"
            "[bold]Features:[/bold]\n"
            "  • [cyan]chat[/cyan] - General conversation\n"
            "  • [cyan]shell[/cyan] - File system operations (auto-detected)\n"
            "  • [cyan]search[/cyan] - Web search (auto-detected)\n\n"
            "[dim]The agent automatically classifies your intent and routes accordingly.[/dim]",
            title="Help",
            border_style="dim",
        )
    )


def chat_loop():
    history = []
    work_dir = os.getcwd()

    print_header()

    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().strip("/")
            if cmd == "quit" or cmd == "exit":
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "clear":
                history = []
                print_header()
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif cmd == "help":
                print_help()
                continue
            else:
                console.print(f"[red]Unknown command: /{cmd}[/red]")
                continue

        history.append({"role": "user", "content": user_input})

        intent = classify_intent(user_input)

        console.print()

        if intent == "shell":
            console.print("[dim]Intent: shell operation[/dim]")
            response_text = ""
            with Live(console=console, refresh_per_second=30) as live:
                for chunk in run_smart_tool(user_input, work_dir):
                    response_text += chunk
                    live.update(Text(chunk if not chunk.startswith("[dim]") else chunk))

            history.append({"role": "assistant", "content": response_text})

        else:
            if intent == "search":
                console.print(
                    "[dim]Intent: search (not implemented, chatting instead)[/dim]"
                )

            response_text = ""
            console.print("[bold blue]Assistant[/bold blue]: ", end="")

            for chunk in llm_stream(history):
                response_text += chunk
                console.print(chunk, end="")

            console.print()
            history.append({"role": "assistant", "content": response_text})


def main():
    try:
        chat_loop()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
