from core.retrieval.base_retriever import BaseRetriever
from rank_bm25 import BM25Okapi  # type: ignore


class BM25Retriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents
        self.tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return [
            {
                "text": self.documents[i],
                "score": score,
                "document_id": i,
            }
            for i, score in ranked[:top_k]
        ]
