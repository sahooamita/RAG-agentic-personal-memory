"""Smoke tests for the RAG pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from src.config import CORPUS_PATH
from src.data_generator import generate_corpus, save_corpus
from src.embedder import Embedder
from src.vector_store import VectorStore


def test_data_generation():
    corpus = generate_corpus(n_per_type=2)
    assert len(corpus) == 10  # 5 types * 2
    for doc in corpus:
        assert "id" in doc
        assert "type" in doc
        assert "content" in doc
    save_corpus(CORPUS_PATH, corpus)
    loaded = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert len(loaded) == 10


def test_embedder():
    emb = Embedder()
    texts = ["hello world", "this is a test"]
    vectors = emb.embed(texts)
    assert vectors.shape == (2, 384)
    q = emb.embed_query("hello")
    assert q.shape == (384,)


def test_vector_store():
    store = VectorStore(collection_name="test_collection")
    store.reset()
    emb = Embedder()
    texts = ["chunk one", "chunk two"]
    vectors = emb.embed(texts).tolist()
    store.add_chunks(
        ids=["c1", "c2"],
        texts=texts,
        embeddings=vectors,
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    assert store.count() == 2
    results = store.query(vectors[0], n_results=2)
    assert len(results["documents"][0]) == 2


if __name__ == "__main__":
    test_data_generation()
    test_embedder()
    test_vector_store()
    print("All smoke tests passed.")
