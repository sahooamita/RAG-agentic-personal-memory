#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip wheel

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing Playwright browsers ==="
playwright install chromium

echo "=== Downloading Gemma 2 2B IT GGUF ==="
MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
MODEL_DIR="$PROJECT_ROOT/models"
MODEL_PATH="$MODEL_DIR/gemma-2-2b-it-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists — skipping download."
else
    echo "Downloading model (this may take a few minutes)..."
    curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
    echo "Download complete: $MODEL_PATH"
fi

echo "=== Verifying model load ==="
python3 -c "
from llama_cpp import Llama
from pathlib import Path
model_path = Path('$MODEL_PATH')
llm = Llama(str(model_path), n_ctx=512, verbose=False, n_gpu_layers=-1)
print('Model loaded successfully. Running quick test...')
out = llm.create_chat_completion(messages=[{'role':'user','content':'Hello'}], max_tokens=16)
print('Test output:', out['choices'][0]['message']['content'].strip())
"

echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run app with:   streamlit run app/streamlit_app.py"
