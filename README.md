# local-agent

Run a local coding agent on your MacBook via llama.cpp for free.

## Features

- **Chat** - General conversation and coding questions
- **Shell operations** - File system operations (auto-detected)
- **Web search** - Web search capability (auto-detected)
- **Metadata display** - Shows tokens/sec and context window size after each message

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.13+
- Go 1.25+ (`brew install go`)

## Installation

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

### 2. Clone and set up Python environment

```bash
git clone <repo-url>
cd local-agent
uv init
uv venv
source .venv/bin/activate
uv sync
```

### 3. Download a model

Create the models directory and download a GGUF model:

```bash
mkdir -p ~/models

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

### 4. Start the LLM server

```bash
llama-server \
  --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --port 8000 \
  --host 127.0.0.1
```

### 5. Run the agent

```bash
source .venv/bin/activate
python agent.py
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
