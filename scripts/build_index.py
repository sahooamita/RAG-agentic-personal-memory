"""One-shot script: generate synthetic data + build vector index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CORPUS_PATH
from src.data_generator import generate_corpus, save_corpus
from src.model_loader import load_llm
from src.rag_engine import RAGEngine


def main():
    print("Generating synthetic corpus...")
    corpus = generate_corpus()
    save_corpus(CORPUS_PATH, corpus)
    print(f"Saved {len(corpus)} documents to {CORPUS_PATH}")

    print("Loading LLM (this may take a moment)...")
    llm = load_llm()

    print("Building index...")
    engine = RAGEngine(llm)
    n_chunks = engine.index_corpus(corpus)
    print(f"Indexed {len(corpus)} docs into {n_chunks} chunks.")
    print("Done.")


if __name__ == "__main__":
    main()
