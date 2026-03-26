# local-agent

Run a local coding agent on your MacBook via llama.cpp for free.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.10+
- Go 1.25+ (`brew install go`)

## Installation Steps

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

### 2. Install Python dependencies

```bash
pip3 install huggingface-hub rich --break-system-packages
```

### 3. Download the 35B MoE model (default — 30 tok/s via SSD paging)

```bash
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```