from abc import ABC, abstractmethod
from typing import List
from models import SearchResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        workspace: str,
        top_k: int,
    ) -> List[SearchResult]:
        pass


class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        workspace: str,
        top_k: int,
    ) -> List[SearchResult]:
        query_vector = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            workspace=workspace,
        )

        return results


class BM25Retriever(BaseRetriever):
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        workspace: str,
        top_k: int,
    ):

        results = self.vector_store.text_search(
            query_text=query,
            workspace=workspace,
            limit=top_k,
        )

        return results


# class HybridRetriever(BaseRetriever):
#     def __init__(self, dense: BaseRetriever, bm25: BaseRetriever, alpha: float = 0.7):
#         self.dense = dense
#         self.bm25 = bm25
#         self.alpha = alpha

#     def retrieve(self, query: str, top_k: int):
#         dense_results = self.dense.retrieve(query, top_k * 2)
#         bm25_results = self.bm25.retrieve(query, top_k * 2)

#         scores = {}

#         for r in dense_results:
#             scores[r["document_id"]] = (
#                 scores.get(r["document_id"], 0) + self.alpha * r["score"]
#             )

#         for r in bm25_results:
#             scores[r["document_id"]] = (
#                 scores.get(r["document_id"], 0) + (1 - self.alpha) * r["score"]
#             )

#         ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#         return [
#             {"document_id": doc_id, "score": score} for doc_id, score in ranked[:top_k]
#         ]


# class RerankingRetriever:
#     def __init__(self, base_retriever, reranker):
#         self.base_retriever = base_retriever
#         self.reranker = reranker

#     def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 30):
#         candidates = self.base_retriever.retrieve(query, top_k=fetch_k)

#         reranked = self.reranker.rerank(
#             query=query,
#             chunks=candidates,
#             top_k=top_k,
#         )

#         return reranked
