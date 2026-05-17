from core.retrieval.base_retriever import BaseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, dense: BaseRetriever, bm25: BaseRetriever, alpha: float = 0.7):
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int):
        dense_results = self.dense.retrieve(query, top_k * 2)
        bm25_results = self.bm25.retrieve(query, top_k * 2)

        scores = {}

        for r in dense_results:
            scores[r["document_id"]] = (
                scores.get(r["document_id"], 0) + self.alpha * r["score"]
            )

        for r in bm25_results:
            scores[r["document_id"]] = (
                scores.get(r["document_id"], 0) + (1 - self.alpha) * r["score"]
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {"document_id": doc_id, "score": score} for doc_id, score in ranked[:top_k]
        ]
