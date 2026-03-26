# local-agent

Run a local coding agent on your MacBook via llama.cpp for free.

## Features

- **Chat** - General conversation and coding questions
- **Shell operations** - File system operations (auto-detected)
- **Web search** - Web search capability (auto-detected)
- **Metadata display** - Shows tokens/sec and context window size after each message

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- uv (`brew install uv`)
- Go (`brew install go`)

## Installation

```bash
./setup.sh
```

This will:
1. Check all prerequisites
2. Install llama.cpp via Homebrew
3. Create a virtual environment and install Python dependencies
4. Download the Qwen3.5 35B model

## Running

### 1. Start the LLM server

```bash
llama-server \
  --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --port 8000 --host 127.0.0.1 \
  --flash-attn on --ctx-size 12288 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

### 2. Run the agent

```bash
uv run python agent.py
```

## Usage

Start chatting with the agent. It automatically classifies your intent:

- **Shell operations**: "list files in current directory", "find python files", "show disk space"
- **Chat**: General questions, coding help, explanations
- **Search**: Web queries (currently falls back to chat)

### Commands

- `/clear` - Clear conversation history
- `/quit` - Exit the agent
- `/help` - Show help

## Dependencies

- `rich` - Terminal formatting and live display
- `requests` - HTTP client for LLM API calls
- `huggingface-hub` - Model downloading

## Environment Variables

- `LLM_URL` - LLM server URL (default: `http://127.0.0.1:8000`)
