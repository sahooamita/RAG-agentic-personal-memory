"""ChromaDB persistent vector store wrapper."""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, EMBEDDING_MODEL


class VectorStore:
    def __init__(self, collection_name: str = "personal_memory"):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: List[float], n_results: int = 4) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
