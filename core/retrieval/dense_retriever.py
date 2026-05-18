from core.retrieval.base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int):
        vec = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_vector=vec,
            top_k=top_k,
            collection=self.vector_store.chunks_collection,
        )

        return [
            {
                "text": r.payload["text"],
                "score": r.score,
                "document_id": r.payload["document_id"],
            }
            for r in results
        ]
