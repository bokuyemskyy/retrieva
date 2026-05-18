from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from core.retrieval.reranking.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]

        scores = self.model.predict(pairs)

        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked.append(
                {
                    **chunk,
                    "rerank_score": float(score),
                }
            )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
