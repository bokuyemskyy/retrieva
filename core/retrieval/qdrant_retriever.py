from typing import List, Dict, Any

from core.retrieval.base_retriever import BaseRetriever


class QdrantRetriever(BaseRetriever):
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
    ) -> List[Dict[str, Any]]:

        query_vector = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=fetch_k,
            collection=self.vector_store.chunks_collection,
        )

        chunks = []

        for result in results:
            chunks.append(
                {
                    "chunk_id": str(result.id),
                    "document_id": result.payload.get("document_id"),
                    "text": result.payload.get("text"),
                    "score": result.score,
                    "metadata": result.payload,
                }
            )

        return chunks[:top_k]
