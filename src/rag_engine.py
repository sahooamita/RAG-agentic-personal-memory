"""End-to-end RAG pipeline: retrieve → generate."""
from typing import List, Dict, Any
from llama_cpp import Llama

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_RETRIEVAL_K
from src.embedder import Embedder
from src.vector_store import VectorStore


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-based sliding window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


class RAGEngine:
    def __init__(self, llm: Llama, embedder: Embedder = None, store: VectorStore = None):
        self.llm = llm
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def index_corpus(self, corpus: List[Dict[str, Any]]) -> int:
        """Chunk, embed, and store all documents. Returns total chunks."""
        self.store.reset()
        all_ids, all_texts, all_embeddings, all_metas = [], [], [], []

        for doc in corpus:
            chunks = _chunk_text(doc["content"])
            embeddings = self.embedder.embed(chunks)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                all_ids.append(f"{doc['id']}_{i}")
                all_texts.append(chunk)
                all_embeddings.append(emb.tolist())
                all_metas.append({
                    "doc_id": doc["id"],
                    "doc_type": doc["type"],
                    "title": doc["title"],
                    "chunk_index": i,
                })

        if all_ids:
            self.store.add_chunks(all_ids, all_texts, all_embeddings, all_metas)
        return len(all_ids)

    def retrieve(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Dict[str, Any]]:
        emb = self.embedder.embed_query(query)
        results = self.store.query(emb.tolist(), n_results=k)
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def generate(self, query: str, chunks: List[Dict[str, Any]], temperature: float = 0.7, top_p: float = 0.9) -> str:
        context = "\n\n".join(
            f"[{i+1}] {c['metadata']['title']} ({c['metadata']['doc_type']}): {c['text']}"
            for i, c in enumerate(chunks)
        )
        system_msg = (
            "You are a helpful personal memory assistant. Use the provided context to answer. "
            "If the context does not contain the answer, say so honestly. Cite sources like [1], [2]."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        from src.model_loader import generate
        return generate(self.llm, messages, temperature=temperature, top_p=top_p)

    def chat(self, query: str, k: int = DEFAULT_RETRIEVAL_K, temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        chunks = self.retrieve(query, k=k)
        answer = self.generate(query, chunks, temperature=temperature, top_p=top_p)
        return {"query": query, "answer": answer, "sources": chunks}
