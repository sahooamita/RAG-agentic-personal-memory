"""Load the local Gemma 2 2B IT GGUF model via llama-cpp-python."""
import os
from pathlib import Path
from llama_cpp import Llama

from src.config import LLM_MODEL_PATH, DEFAULT_MAX_TOKENS


def load_llm(model_path: Path = LLM_MODEL_PATH, verbose: bool = False) -> Llama:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/setup.sh to download it."
        )

    # Metal on Apple Silicon: n_gpu_layers=-1 offloads all layers to GPU
    n_gpu = -1 if os.uname().machine == "arm64" else 0

    return Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=n_gpu,
        verbose=verbose,
        chat_format="gemma",
    )


def generate(
    llm: Llama,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<end_of_turn>"],
    )
    return response["choices"][0]["message"]["content"].strip()
