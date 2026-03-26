#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Local Agent Setup ==="
echo ""

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 not found."
        if [ -n "$2" ]; then
            echo "Install with: $2"
        fi
        return 1
    fi
    return 0
}

echo "[1/6] Checking prerequisites..."
MISSING=0

if ! command -v brew &> /dev/null; then
    echo "  ✗ Homebrew not found"
    MISSING=1
else
    echo "  ✓ Homebrew installed"
fi

if ! command -v uv &> /dev/null; then
    echo "  ✗ uv not found"
    MISSING=1
else
    echo "  ✓ uv installed"
fi

if ! command -v go &> /dev/null; then
    echo "  ✗ Go not found"
    MISSING=1
else
    echo "  ✓ Go installed"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Please install missing prerequisites and try again."
    exit 1
fi

echo ""
echo "[2/6] Installing llama.cpp..."
if ! command -v llama-server &> /dev/null; then
    brew install llama.cpp
    echo "  ✓ llama.cpp installed"
else
    echo "  ✓ llama.cpp already installed"
fi

echo ""
echo "[3/6] Setting up Python virtual environment and installing dependencies..."
if [ ! -d "$REPO_DIR/.venv" ]; then
    uv venv "$REPO_DIR/.venv"
    echo "  ✓ Virtual environment created"
fi
. "$REPO_DIR/.venv/bin/activate"
uv pip install huggingface-hub rich requests
echo "  ✓ Python dependencies installed"

echo ""
echo "[4/6] Downloading Qwen3.5 35B model..."
mkdir -p ~/models
MODEL_FILE="Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
MODEL_PATH="$HOME/models/$MODEL_FILE"

if [ -f "$MODEL_PATH" ]; then
    echo "  ✓ Model already exists at $MODEL_PATH"
else
    echo "  Downloading $MODEL_FILE (this may take a while)..."
    uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    '$MODEL_FILE', local_dir='$HOME/models/')
"
    echo "  ✓ Model downloaded"
fi

echo ""
echo "[5/6] Building PicoClaw..."
if [ -d "$REPO_DIR/picoclaw" ]; then
    echo "  PicoClaw directory exists, updating..."
    cd "$REPO_DIR/picoclaw" && git pull
else
    echo "  Cloning PicoClaw..."
    git clone https://github.com/sipeed/picoclaw.git "$REPO_DIR/picoclaw"
fi

cd "$REPO_DIR/picoclaw"
make deps
make build
echo "  ✓ PicoClaw built"

echo ""
echo "[6/6] Configuring PicoClaw..."
mkdir -p ~/.picoclaw/workspace

if [ -f "$REPO_DIR/config.example.json" ]; then
    if [ ! -f ~/.picoclaw/config.json ]; then
        cp "$REPO_DIR/config.example.json" ~/.picoclaw/config.json
        echo "  ✓ Config copied to ~/.picoclaw/config.json"
    else
        echo "  ✓ Config already exists"
    fi
else
    echo "  ⚠ config.example.json not found in repo, skipping config copy"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the LLM server:"
echo "  llama-server \\"
echo "    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \\"
echo "    --port 8000 --host 127.0.0.1 \\"
echo "    --flash-attn on --ctx-size 12288 \\"
echo "    --cache-type-k q4_0 --cache-type-v q4_0 \\"
echo "    --n-gpu-layers 99 --reasoning off -np 1 -t 4"
echo ""
echo "Then run the agent:"
echo "  uv run python $REPO_DIR/agent.py"
